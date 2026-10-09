import folium
import pandas as pd

from projlib import config

E = "FACTSET_ENTITY_ID"
Y0, Y1 = config.YEAR_MIN, config.YEAR_MAX
OUT = config.out("09_appendix/web", "x").parent
ent = pd.read_parquet(config.out("01_universe", "entities.parquet")).set_index(E)
xy = pd.read_parquet(config.out("07_panel", "entity_coordinates.parquet")).set_index(E)
pw = pd.read_parquet(config.out("07_panel", "rival_pair_year.parquet"), columns=["focal", "rival", "year"]).drop_duplicates()
pw = pw[pw["year"].between(Y0, Y1)]
names_from = pw.groupby("focal")["year"].agg(["min", "max"])
named_by = pw.groupby("rival")["focal"].nunique()
ids = sorted(set(pw["focal"]) | set(pw["rival"]))
d = pd.DataFrame(index=pd.Index(ids, name=E))
d["lat"] = d.index.map(xy["latitude"]); d["lon"] = d.index.map(xy["longitude"])
d["name"] = d.index.map(ent["ENTITY_PROPER_NAME"]); d["country"] = d.index.map(ent["ISO_COUNTRY"])
d["names"] = d.index.isin(names_from.index); d["y0"] = d.index.map(names_from["min"]); d["y1"] = d.index.map(names_from["max"])
d["named_by"] = d.index.map(named_by).fillna(0).astype(int)
d = d.dropna(subset=["lat", "lon"])
print(f"entities in the rivalry network {len(ids):,}; with coordinates {len(d):,}: naming competitors {int(d['names'].sum()):,}, only named {int((~d['names']).sum()):,}")


def features(sub, kind):
    out = []
    for r in sub.itertuples():
        yrs = f"names rivals {int(r.y0)}–{int(r.y1)}" if r.names else "does not name competitors"
        out.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(float(r.lon), 4), round(float(r.lat), 4)]},
                    "properties": {"t": f"{r.name} ({r.country}): {yrs}; named by {r.named_by} firm{'s' if r.named_by != 1 else ''}"}})
    return {"type": "FeatureCollection", "features": out}


ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
m = folium.Map(location=[25, 10], zoom_start=2, tiles=ESRI, attr="Tiles: Esri, HERE, Garmin, FAO, NOAA, USGS", prefer_canvas=True, control_scale=True)
for kind, sub, color, label in (("named", d[~d["names"]], "#e08a2e", "entities only named as competitors"),
                                ("source", d[d["names"]], "#2b5d8a", "source firms naming at least one competitor")):
    fg = folium.FeatureGroup(name=f"{label} ({len(sub):,})", show=True)
    folium.GeoJson(features(sub, kind), marker=folium.CircleMarker(radius=2, weight=0, fill=True, fill_opacity=0.55),
                   style_function=lambda _f, c=color: {"fillColor": c, "color": c}, tooltip=folium.GeoJsonTooltip(fields=["t"], labels=False)).add_to(fg)
    fg.add_to(m)
from folium.plugins import FastMarkerCluster
for sub, color, label in ((d[~d["names"]], "#e08a2e", "entities only named, clustered"), (d[d["names"]], "#2b5d8a", "source firms naming a competitor, clustered")):
    cb = ("function (row) { var m = L.circleMarker(new L.LatLng(row[0], row[1]), {radius: 4, weight: 0, fillOpacity: 0.8, fillColor: '%s'}); m.bindTooltip(row[2]); return m; }" % color)
    FastMarkerCluster(sub[["lat", "lon"]].assign(t=sub.apply(lambda r: f"{r['name']} ({r['country']})", axis=1)).to_numpy().tolist(), callback=cb, name=label, show=False).add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
OUT.mkdir(parents=True, exist_ok=True)
m.save(str(OUT / "firm_map.html"))
print(f"written {OUT / 'firm_map.html'} ({(OUT / 'firm_map.html').stat().st_size / 1e6:.1f} MB)")
