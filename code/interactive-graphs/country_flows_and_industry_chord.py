from itertools import count
from pathlib import Path
from types import SimpleNamespace

import branca.colormap as cm
import folium
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
from branca.element import MacroElement
from folium import FeatureGroup, LayerControl
from holoviews import dim
from jinja2 import Template


RAW = "<raw folder>"
config = SimpleNamespace(
    RELATIONSHIP_FILE=f"{RAW}/Factset_Revere/ENT/ent_scr_relationships.parquet",
    PROCESSED_DATA_DIR="<output folder>/",
    firm_i="SOURCE_FACTSET_ENTITY_ID",
    firm_j="TARGET_FACTSET_ENTITY_ID",
    METADATA_FILES={
        "coverage_ent": f"{RAW}/Factset_Revere/ENT/ent_entity_coverage.dta",
        "geo_coordinates": f"{RAW}/Factset_Revere/ENT/ent_entity_address_coord.dta",
        "address_id": f"{RAW}/Factset_Revere/ENT/ent_scr_address.dta",
        "sym_entity_1": f"{RAW}/Factset_Revere/SYM/sym_entity.dta",
        "sym_entity_2": f"{RAW}/Factset_Revere/SYM (New Data)/sym_entity.dta",
    },
)


# The competitor records, collapsed into episodes per directed pair
df_rel = pd.read_parquet(config.RELATIONSHIP_FILE)
df_comp = df_rel[df_rel["REL_TYPE"] == "COMPETITOR"]
df_comp = df_comp.drop(columns=["index", "REVENUE_PCT"], errors="ignore")
df_comp["END_DATE"] = pd.to_datetime(df_comp["END_DATE"])
df_comp["START_DATE"] = pd.to_datetime(df_comp["START_DATE"])
df_comp["END_DATE"] = df_comp["END_DATE"].fillna(pd.Timestamp("2021-12-31"))
df_comp = df_comp.sort_values(by=[config.firm_i, config.firm_j, "START_DATE", "END_DATE"])
df_comp = (
    df_comp
    .groupby([config.firm_i, config.firm_j, "START_DATE", "END_DATE"], as_index=False)
    .agg({
        "ID": lambda x: "|".join(sorted(set(x))),
        "REL_TYPE": "first"
    })
)
df_comp = (
    df_comp
    .groupby([config.firm_i, config.firm_j, "START_DATE"], as_index=False)
    .agg({
        "END_DATE": "max",
        "ID": lambda x: "|".join(sorted(set(x))),
        "REL_TYPE": "first"
    })
)
df_comp = df_comp.sort_values(by=[config.firm_i, config.firm_j, "START_DATE"])
group_id_counter = count()
group_ids = []
prev_pair = None
current_end = pd.Timestamp.min
group_id = next(group_id_counter)
for idx, row in df_comp.iterrows():
    pair = (row[config.firm_i], row[config.firm_j])
    start = row["START_DATE"]
    end = row["END_DATE"]

    if pair != prev_pair:
        group_id = next(group_id_counter)
        current_end = end
    else:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            group_id = next(group_id_counter)
            current_end = end

    group_ids.append(group_id)
    prev_pair = pair

df_comp["overlap_group"] = group_ids
df_comp_collapsed = (
    df_comp
    .groupby([config.firm_i, config.firm_j, "overlap_group"], as_index=False)
    .agg({
        "START_DATE": "min",
        "END_DATE": "max",
        "ID": lambda x: "|".join(sorted(set(x))),
        "REL_TYPE": "first"
    })
)
df_comp_collapsed = df_comp_collapsed.sort_values(
    by=[config.firm_i, config.firm_j, "START_DATE"]
)
df_comp_collapsed["prev_end_date"] = df_comp_collapsed.groupby(
    [config.firm_i, config.firm_j]
)["END_DATE"].shift(1)
df_comp_collapsed["inter_episode_gap_days"] = (
    (df_comp_collapsed["START_DATE"] - df_comp_collapsed["prev_end_date"])
    .dt.days
)
df_comp_collapsed = df_comp_collapsed.sort_values(
    by=[config.firm_i, config.firm_j, "START_DATE"]
)
merge_group_ids = []
group_counter = count()
current_group = next(group_counter)
prev_pair = None
for idx, row in df_comp_collapsed.iterrows():
    pair = (row[config.firm_i], row[config.firm_j])
    gap = row["inter_episode_gap_days"]

    if pair != prev_pair:
        current_group = next(group_counter)
    else:
        if pd.isna(gap) or gap > 30:
            current_group = next(group_counter)

    merge_group_ids.append(current_group)
    prev_pair = pair

df_comp_collapsed["merge_gap_group"] = merge_group_ids
df_comp_cleaned = (
    df_comp_collapsed
    .groupby([config.firm_i, config.firm_j, "merge_gap_group"], as_index=False)
    .agg({
        "START_DATE": "min",
        "END_DATE": "max",
        "ID": lambda x: "|".join(sorted(set(x))),
        "REL_TYPE": "first"
    })
)
df_comp_cleaned["episode_count"] = df_comp_cleaned.groupby(
    [config.firm_i, config.firm_j]
)["START_DATE"].transform("count")
df_comp_cleaned["ID_count"] = df_comp_cleaned["ID"].str.split("|").apply(len)
df_comp_cleaned.drop(columns=['merge_gap_group'], inplace=True)
df_comp_cleaned["relationship_duration_days"] = ((df_comp_cleaned["END_DATE"]) - (df_comp_cleaned["START_DATE"])).dt.total_seconds() / (60 * 60 * 24)
df_comp_cleaned = df_comp_cleaned.drop(columns=['REL_TYPE']).copy()
df_comp_cleaned.to_parquet(config.PROCESSED_DATA_DIR + "relationships_cleaned.parquet", compression="snappy")
df_comp_cleaned_id = df_comp_cleaned.reset_index(drop=True).copy()
df_comp_cleaned_id["row_id"] = df_comp_cleaned_id.index
df_comp_cleaned_id.to_parquet(config.PROCESSED_DATA_DIR + "relationships_cleaned_rowid.parquet", index=False)
df_comp_cleaned = pd.read_parquet(config.PROCESSED_DATA_DIR + "relationships_cleaned_rowid.parquet")

# Countries, industries and coordinates of both firms of each pair
META = getattr(config, "METADATA_FILES", None)
if META is None:
    META = getattr(config, "METDATA_FILES")
p_address_id      = Path(META['address_id'])
p_geo_coordinates = Path(META['geo_coordinates'])
p_coverage_ent    = Path(META['coverage_ent'])
p_sym_entity_1    = Path(META['sym_entity_1'])
p_sym_entity_2    = Path(META['sym_entity_2'])

def read_stata_filtered(path, usecols, filter_col=None, keep_values=None, chunksize=200_000):
    if keep_values is None or filter_col is None:
        return pd.read_stata(path, columns=usecols)
    keep_values = set(keep_values)
    out = []
    for chunk in pd.read_stata(path, columns=usecols, chunksize=chunksize):
        out.append(chunk[chunk[filter_col].isin(keep_values)])
    if not out:
        return pd.DataFrame(columns=usecols)
    return pd.concat(out, ignore_index=True)

def coalesce_series(*series):
    out = series[0].copy()
    for s in series[1:]:
        out = out.fillna(s)
    return out

ids_i = pd.Index(df_comp_cleaned[config.firm_i].dropna().unique())
ids_j = pd.Index(df_comp_cleaned[config.firm_j].dropna().unique())
entity_ids_needed = pd.Index(np.unique(np.concatenate([ids_i.values, ids_j.values])))
cov_cols = ['FACTSET_ENTITY_ID', 'ISO_COUNTRY_INCORP', 'PRIMARY_SIC_CODE', 'ISO_COUNTRY']
cov = read_stata_filtered(p_coverage_ent, cov_cols, filter_col='FACTSET_ENTITY_ID', keep_values=entity_ids_needed)

# Namibia's code NA, missing in the .dta copies, from the 2023 delivery of FactSet's entity table; a blank code is missing
export = pd.read_csv(f"{RAW}/Factset_Revere/SYM (New Data)/CSV/sym_entity.csv", sep="\t", encoding="utf-16",
                     usecols=["factset_entity_id", "iso_country"], dtype=str, keep_default_na=False)
namibia = cov["FACTSET_ENTITY_ID"].isin(export.loc[export["iso_country"].eq("NA"), "factset_entity_id"])
for c in ("ISO_COUNTRY_INCORP", "ISO_COUNTRY"):
    blank = cov[c].eq("")
    cov[c] = cov[c].mask(blank & namibia, "NA").mask(blank & ~namibia)
sym1_cols = ['FACTSET_ENTITY_ID', 'ISO_COUNTRY']
sym1 = read_stata_filtered(p_sym_entity_1, sym1_cols, filter_col='FACTSET_ENTITY_ID', keep_values=entity_ids_needed)
sym1 = sym1.rename(columns={'ISO_COUNTRY': 'ISO_COUNTRY_SYM1'})
sym2_cols = ['FACTSET_ENTITY_ID', 'ISO_COUNTRY']
sym2 = read_stata_filtered(p_sym_entity_2, sym2_cols, filter_col='FACTSET_ENTITY_ID', keep_values=entity_ids_needed)
sym2 = sym2.rename(columns={'ISO_COUNTRY': 'ISO_COUNTRY_SYM2'})
ent = (
    cov
    .merge(sym1, on='FACTSET_ENTITY_ID', how='outer')
    .merge(sym2, on='FACTSET_ENTITY_ID', how='outer')
)
ent = ent.rename(columns={'ISO_COUNTRY': 'ISO_COUNTRY_COV'})
ent['ISO_COUNTRY_COALESCED'] = coalesce_series(ent['ISO_COUNTRY_SYM1'], ent['ISO_COUNTRY_SYM2'], ent['ISO_COUNTRY_COV'])

def _distinct_nonnull(values):
    vals = pd.Series(values).dropna().unique()
    return [v for v in vals if pd.notna(v)]

ent['_iso_vals'] = ent[['ISO_COUNTRY_SYM1','ISO_COUNTRY_SYM2','ISO_COUNTRY_COV']].apply(_distinct_nonnull, axis=1)
ent['ISO_COUNTRY_CONFLICT'] = ent['_iso_vals'].apply(lambda xs: len(set(xs)) > 1)
ent['ISO_COUNTRY_DISTINCT'] = ent['_iso_vals'].apply(lambda xs: ','.join(map(str, xs)) if len(xs) else np.nan)
ent = ent.drop(columns=['_iso_vals'])
addr_cols = ['FACTSET_ENTITY_ID', 'ADDRESS_ID']
addr = read_stata_filtered(p_address_id, addr_cols, filter_col='FACTSET_ENTITY_ID', keep_values=entity_ids_needed)
geo_cols = ['ADDRESS_ID', 'LATITUDE', 'LONGITUDE', 'RESOLUTION_CODE']
geo = read_stata_filtered(p_geo_coordinates, geo_cols)
ent_i = ent.rename(columns={
    'FACTSET_ENTITY_ID': config.firm_i,
    'ISO_COUNTRY_COALESCED': 'ISO_COUNTRY_I',
    'ISO_COUNTRY_INCORP': 'ISO_COUNTRY_INCORP_I',
    'PRIMARY_SIC_CODE': 'PRIMARY_SIC_CODE_I',
    'ISO_COUNTRY_SYM1': 'ISO_COUNTRY_SYM1_I',
    'ISO_COUNTRY_SYM2': 'ISO_COUNTRY_SYM2_I',
    'ISO_COUNTRY_COV':  'ISO_COUNTRY_COV_I',
    'ISO_COUNTRY_CONFLICT': 'ISO_COUNTRY_CONFLICT_I',
    'ISO_COUNTRY_DISTINCT': 'ISO_COUNTRY_DISTINCT_I',
})
ent_j = ent.rename(columns={
    'FACTSET_ENTITY_ID': config.firm_j,
    'ISO_COUNTRY_COALESCED': 'ISO_COUNTRY_J',
    'ISO_COUNTRY_INCORP': 'ISO_COUNTRY_INCORP_J',
    'PRIMARY_SIC_CODE': 'PRIMARY_SIC_CODE_J',
    'ISO_COUNTRY_SYM1': 'ISO_COUNTRY_SYM1_J',
    'ISO_COUNTRY_SYM2': 'ISO_COUNTRY_SYM2_J',
    'ISO_COUNTRY_COV':  'ISO_COUNTRY_COV_J',
    'ISO_COUNTRY_CONFLICT': 'ISO_COUNTRY_CONFLICT_J',
    'ISO_COUNTRY_DISTINCT': 'ISO_COUNTRY_DISTINCT_J',
})
addr_i = addr.rename(columns={'FACTSET_ENTITY_ID': config.firm_i, 'ADDRESS_ID': 'ADDRESS_ID_I'})
addr_j = addr.rename(columns={'FACTSET_ENTITY_ID': config.firm_j, 'ADDRESS_ID': 'ADDRESS_ID_J'})
df_enriched = (
    df_comp_cleaned
      .merge(ent_i, on=config.firm_i, how='left')
      .merge(ent_j, on=config.firm_j, how='left')
      .merge(addr_i, on=config.firm_i, how='left')
      .merge(addr_j, on=config.firm_j, how='left')
)
geo_i = geo.rename(columns={
    'ADDRESS_ID': 'ADDRESS_ID_I',
    'LATITUDE': 'LATITUDE_I',
    'LONGITUDE': 'LONGITUDE_I',
    'RESOLUTION_CODE': 'RESOLUTION_CODE_I'
})
geo_j = geo.rename(columns={
    'ADDRESS_ID': 'ADDRESS_ID_J',
    'LATITUDE': 'LATITUDE_J',
    'LONGITUDE': 'LONGITUDE_J',
    'RESOLUTION_CODE': 'RESOLUTION_CODE_J'
})
df_enriched = (
    df_enriched
      .merge(geo_i, on='ADDRESS_ID_I', how='left')
      .merge(geo_j, on='ADDRESS_ID_J', how='left')
)

# Country-to-country flows; published as country_flows.html
ISO_I, ISO_J = "ISO_COUNTRY_INCORP_I", "ISO_COUNTRY_INCORP_J"
LAT_I, LON_I = "LATITUDE_I", "LONGITUDE_I"
LAT_J, LON_J = "LATITUDE_J", "LONGITUDE_J"
FIRM_I = config.firm_i
FIRM_J = config.firm_j
OUTFILE = config.PROCESSED_DATA_DIR + "country_flows.html"

def country_centroids(df, iso_col, lat_col, lon_col):
    d = df[[iso_col, lat_col, lon_col]].dropna().copy()
    if d.empty:
        return pd.DataFrame(columns=["ISO2","lat","lon"])
    g = (d.groupby(iso_col, as_index=False)
           .agg(lat=(lat_col, "median"), lon=(lon_col, "median"))
           .rename(columns={iso_col: "ISO2"}))
    return g

cent_i = country_centroids(df_enriched, ISO_I, LAT_I, LON_I)
cent_j = country_centroids(df_enriched, ISO_J, LAT_J, LON_J)
centroids = (
    pd.concat([cent_i, cent_j], ignore_index=True)
      .dropna()
      .drop_duplicates(subset=["ISO2"])
      .set_index("ISO2")
)
pairs_full = df_enriched[[FIRM_I, FIRM_J, ISO_I, ISO_J]].dropna().copy()
pairs_full[ISO_I] = pairs_full[ISO_I].astype(str).str.upper().str.strip()
pairs_full[ISO_J] = pairs_full[ISO_J].astype(str).str.upper().str.strip()
uniq_pairs = pairs_full.drop_duplicates(subset=[FIRM_I, FIRM_J])
flows = (
    uniq_pairs
      .groupby([ISO_I, ISO_J], as_index=False)
      .size()
      .rename(columns={ISO_I: "ISO2_I", ISO_J: "ISO2_J", "size": "count"})
)
flows = flows[flows["ISO2_I"] != flows["ISO2_J"]]
flows = flows[
    flows["ISO2_I"].isin(centroids.index) &
    flows["ISO2_J"].isin(centroids.index)
].copy()
flows = flows.sort_values("count", ascending=False).reset_index(drop=True)
flows["rank"] = np.arange(1, len(flows) + 1)
flows["rank_pct"] = flows["rank"] / len(flows)
total_flows = float(flows["count"].sum())
outdeg = flows.groupby("ISO2_I")["count"].sum().rename("out")
indeg  = flows.groupby("ISO2_J")["count"].sum().rename("in")
deg = pd.concat([outdeg, indeg], axis=1).fillna(0)
deg["total"] = deg["out"] + deg["in"]
nodes = centroids.copy().join(deg, how="left").fillna(0)
nodes["share_total_flows"] = np.where(
    total_flows > 0, (nodes["total"] / (2.0 * total_flows)) * 100.0, 0.0
)
cmin, cmax = (flows["count"].min(), flows["count"].max()) if not flows.empty else (1, 1)
cmap = cm.LinearColormap(colors=["#b3cde3", "#6497b1", "#03396c"], vmin=cmin, vmax=cmax)

def edge_width(count, min_w=0.8, max_w=5.0):
    if count <= 0: return min_w
    v = np.log1p(count)
    vmin, vmax = np.log1p(cmin), np.log1p(cmax)
    if vmax <= vmin: return (min_w + max_w)/2
    return float(min_w + (v - vmin) * (max_w - min_w) / (vmax - vmin))

features = []
for _, r in flows.iterrows():
    i, j, cnt = r["ISO2_I"], r["ISO2_J"], int(r["count"])
    lat1, lon1 = float(nodes.loc[i, "lat"]), float(nodes.loc[i, "lon"])
    lat2, lon2 = float(nodes.loc[j, "lat"]), float(nodes.loc[j, "lon"])
    color = cmap(cnt)
    width = edge_width(cnt)
    share = (cnt / total_flows * 100.0) if total_flows > 0 else 0.0

    feat = {
        "type": "Feature",
        "properties": {
            "i": i, "j": j,
            "count": cnt,
            "rank_pct": float(r["rank_pct"]),
            "color": color,
            "width": width,
            "tooltip": f"{i} → {j} — {cnt:,} unique rivalries"
        },
        "geometry": {
            "type": "LineString",
            "coordinates": [[lon1, lat1], [lon2, lat2]]
        }
    }
    features.append(feat)

edge_geojson = {"type": "FeatureCollection", "features": features}
center = [nodes["lat"].median(), nodes["lon"].median()] if not nodes.empty else [20, 0]
m = folium.Map(location=center, zoom_start=2, tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
               attr="Tiles: Esri, HERE, Garmin, FAO, NOAA, USGS")
edges_layer = folium.GeoJson(
    data=edge_geojson,
    name="Flows (use slider)",
    style_function=lambda feat: {
        "color": feat["properties"]["color"],
        "weight": feat["properties"]["width"],
        "opacity": 0.6,
    },
    tooltip=folium.features.GeoJsonTooltip(fields=["tooltip"], aliases=[""], labels=False)
).add_to(m)
node_fg = FeatureGroup(name="Country nodes (degree)", overlay=True, show=True)
if not nodes.empty:
    tmin, tmax = nodes["total"].min(), nodes["total"].max()
    for iso2, r in nodes.iterrows():
        lat, lon = float(r["lat"]), float(r["lon"])
        total = float(r["total"]); outv = int(r["out"]); inv = int(r["in"])
        share_involved = float(r["share_total_flows"])
        radius = 4 if tmax <= tmin else 4 + 10 * (total - tmin) / (tmax - tmin)
        folium.CircleMarker(
            location=(lat, lon),
            radius=radius,
            color="#444",
            fill=True, fill_color="#444", fill_opacity=0.7,
            tooltip=(
                f"{iso2} — degree={int(total):,} (out={outv:,}, in={inv:,}) • "
                f"involved in {share_involved:.1f}% of flows"
            ),
        ).add_to(node_fg)
node_fg.add_to(m)
cmap.caption = "Unique rivalries (edge color)"
m.add_child(cmap)
slider = MacroElement()
slider._template = Template("""
{% macro script(this, kwargs) %}
// Add a simple Top-% slider
var sliderCtrl = L.control({position:'topright'});
sliderCtrl.onAdd = function(map){
  var div = L.DomUtil.create('div', 'info legend');
  div.style.background = 'white';
  div.style.padding = '10px';
  div.style.border = '1px solid #ccc';
  div.style.borderRadius = '6px';
  div.style.boxShadow = '0 1px 4px rgba(0,0,0,0.2)';
  div.innerHTML = `
    <label style="font-weight:600; font-size:12px;">
      Top % edges: <span id="pctVal">100</span>%
    </label><br/>
    <input id="edgeSlider" type="range" min="1" max="100" value="100" step="1" style="width:160px;">
    <div style="font-size:11px; color:#555; margin-top:4px;">Heaviest flows first</div>
  `;
  L.DomEvent.disableClickPropagation(div);
  return div;
};
sliderCtrl.addTo({{this.map_name}});

// Hook up behavior
var sliderEl = document.getElementById('edgeSlider');
var pctEl    = document.getElementById('pctVal');

// Get the GeoJson layer by the injected name
var edgesLayer = {{this.layer_name}};

function updateEdges(percent){
  pctEl.textContent = percent;
  // Show edges with rank_pct*100 <= percent
  edgesLayer.eachLayer(function(l){
    var rp = (l.feature && l.feature.properties && l.feature.properties.rank_pct)
               ? (l.feature.properties.rank_pct * 100.0) : 100.0;
    var w  = (l.feature && l.feature.properties && l.feature.properties.width)
               ? l.feature.properties.width : 1.0;
    if (rp <= percent){
      l.setStyle({opacity:0.6, weight:w});
    } else {
      l.setStyle({opacity:0.0, weight:0});
    }
  });
}

sliderEl.addEventListener('input', function(e){
  updateEdges(parseInt(this.value));
});

updateEdges(100); // initial state
{% endmacro %}
""")
slider.map_name   = m.get_name()
slider.layer_name = edges_layer.get_name()
m.get_root().add_child(slider)
LayerControl(collapsed=False).add_to(m)
m.save(OUTFILE)
print(f"Saved: {OUTFILE}")
print(f"Flows (country→country): {len(flows):,} | Nodes: {len(nodes):,}")

# Industry-to-industry chord diagram; published as industry_chord.html
hv.extension('bokeh')
pn.extension('bokeh')
firm_i, firm_j = config.firm_i, config.firm_j
sic_i,  sic_j  = "PRIMARY_SIC_CODE_I", "PRIMARY_SIC_CODE_J"
lab_i,  lab_j  = "SIC2_LABEL_I", "SIC2_LABEL_J"
sic_sector_full_map = {
    '01':'Agriculture','02':'Agriculture','07':'Agriculture','08':'Agriculture','09':'Agriculture',
    '10':'Mining','11':'Mining','12':'Mining','13':'Oil & Gas','14':'Mining','19':'Mining',
    '15':'Construction','16':'Construction','17':'Construction','65':'Real Estate',
    '20':'Food & Kindred Products','21':'Tobacco','22':'Textiles','23':'Apparel','24':'Lumber & Wood',
    '25':'Furniture','26':'Paper & Allied','27':'Printing & Publishing','28':'Chemicals',
    '29':'Petroleum & Coal','30':'Rubber & Plastics','31':'Leather Products','32':'Stone, Clay, Glass',
    '33':'Primary Metals','34':'Fabricated Metals','35':'Machinery','36':'Electrical Equipment',
    '37':'Transportation Equipment','38':'Measuring Instruments','39':'Misc. Manufacturing',
    '40':'Railroad Transportation','41':'Local Transit','42':'Trucking & Warehousing','44':'Water Transportation',
    '45':'Air Transportation','46':'Pipelines','47':'Transportation Services','48':'Communications',
    '49':'Electric, Gas & Sanitary','50':'Wholesale Durable','51':'Wholesale Nondurable',
    '52':'Retail Building Materials','53':'Retail General Merchandise','54':'Retail Food Stores',
    '55':'Retail Auto Dealers','56':'Retail Apparel','57':'Retail Furniture','58':'Restaurants & Bars',
    '59':'Retail Miscellaneous','60':'Depository Institutions','61':'Nondepository Credit',
    '62':'Security Brokers','63':'Insurance Carriers','64':'Insurance Agents','67':'Investment Offices',
    '70':'Hotels & Lodging','72':'Personal Services','73':'Business Services','74':'Veterinary Svcs',
    '75':'Automotive Repair','76':'Misc. Repair','78':'Motion Pictures','79':'Amusement & Recreation',
    '80':'Healthcare Services','81':'Legal Services','82':'Educational Services','83':'Social Services',
    '84':'Museums/Botanical/Zoo','85':'Nonprofit Charitable','86':'Membership Orgs',
    '87':'Engineering & Management','88':'Private Households','89':'Services - Misc.',
    '91':'Executive/Legislative','92':'Justice/Public Order','93':'Public Finance','94':'Administration',
    '95':'Environmental Quality','96':'Economic Programs','97':'National Security','99':'Nonclassifiable'
}
base = (
    df_enriched[[firm_i, firm_j, sic_i, sic_j]]
    .dropna(subset=[sic_i, sic_j])
    .copy()
)
base["SIC2_I"] = base[sic_i].astype(str).str[:2]
base["SIC2_J"] = base[sic_j].astype(str).str[:2]
uniq_pairs = base.drop_duplicates(subset=[firm_i, firm_j]).copy()
if lab_i in df_enriched.columns and lab_j in df_enriched.columns:
    lab_i_df = df_enriched[[firm_i, lab_i]].dropna().drop_duplicates()
    lab_j_df = df_enriched[[firm_j, lab_j]].dropna().drop_duplicates()
    uniq_pairs = (
        uniq_pairs
        .merge(lab_i_df, on=firm_i, how="left")
        .merge(lab_j_df, on=firm_j, how="left")
        .rename(columns={lab_i: "LAB_I", lab_j: "LAB_J"})
    )
else:
    uniq_pairs["LAB_I"] = uniq_pairs["SIC2_I"].map(sic_sector_full_map)
    uniq_pairs["LAB_J"] = uniq_pairs["SIC2_J"].map(sic_sector_full_map)
uniq_pairs = uniq_pairs.dropna(subset=["LAB_I","LAB_J"])
A = np.minimum(uniq_pairs["LAB_I"].to_numpy(), uniq_pairs["LAB_J"].to_numpy())
B = np.maximum(uniq_pairs["LAB_I"].to_numpy(), uniq_pairs["LAB_J"].to_numpy())
links_full = pd.DataFrame({"A": A, "B": B}).value_counts().reset_index(name="value")
links_full = links_full.sort_values("value", ascending=False).reset_index(drop=True)
links_full["rank"] = np.arange(1, len(links_full)+1)
links_full["rank_pct"] = links_full["rank"] / len(links_full)

def make_chord(top_percent=100):
    mask = (links_full["rank_pct"] * 100.0) <= float(top_percent)
    links = links_full.loc[mask].copy()
    if links.empty:
        return hv.Div("No edges at this threshold.")

    names = sorted(pd.unique(pd.concat([links["A"], links["B"]], ignore_index=True)))
    nodes = pd.DataFrame({"name": names})
    nodes["index"] = range(len(names))
    idx = dict(zip(names, nodes["index"]))

    edges = links.copy()
    edges["source"] = edges["A"].map(idx)
    edges["target"] = edges["B"].map(idx)

    edges_ds = hv.Dataset(
        edges[["source","target","value","A","B"]],
        kdims=["source","target"],
        vdims=["value","A","B"]
    )
    nodes_ds = hv.Dataset(nodes, kdims=["index"], vdims=["name"])

    chord = hv.Chord((edges_ds, nodes_ds)).opts(
        cmap="Category20",
        edge_color=dim("source"),
        node_color=dim("index"),
        labels="name",
        edge_alpha=0.75,
        node_size=10,
        tools=["hover"],
        width=950, height=950,
        title=f"Industry ↔ Industry Rivalries (unique firm-pair, undirected) — Top {int(top_percent)}%",
    )
    return chord

slider = pn.widgets.IntSlider(name="Show Top % of edges", start=10, end=100, step=10, value=100)
view = pn.bind(make_chord, top_percent=slider)
app = pn.Column(slider, view)
pn.io.save.save(app, config.PROCESSED_DATA_DIR + "industry_chord.html", embed=True)
print("Saved: industry_chord.html")
