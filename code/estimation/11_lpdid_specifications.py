import hashlib
import pathlib

import numpy as np
import pandas as pd

from projlib import config

ROOT = config.out("08_estimation/clean_runs_ladders", "x").parent
OUT = {"ihs_rnd": "asinh(R&D), missing kept", "ln_rnd": "log R&D (positive only)", "rnd_lat_w": "R&D over lagged assets (winsorised)", "rnd_lsl_w": "R&D over lagged sales (winsorised)",
       "ihs_fam": "asinh(patent families originated, priority year)", "ihs_grt": "asinh(granted patents, dated by their filing year)"}
GROWTH = {"ln_at": "log assets", "ln_sales": "log sales", "ln_emp": "log employees", "cash_ratio_w": "cash over assets (winsorised)", "lev_w": "leverage (winsorised)", "capx_at_w": "capital expenditure over assets (winsorised)"}
SETS = {"d": ("rival", "", "nriv_ca", "norec_ca", "the designated rivals", "ca"), "u": ("undir", "u_", "u_n_ca", "u_norec_ca", "the undirected set (designated rivals and namers)", "ca"),
        "m": ("mutual", "m_", "m_n_ca", "m_norec_ca", "the mutual rivals", "ca")}

p = pd.read_parquet(config.out("08_estimation", "estimation_panel_v3.parquet"))
e = pd.read_parquet(config.out("07_panel", "v2_exposure_calendar.parquet"), columns=["focal", "fyear"] + [f"{s}_{k}" for s in ("rival", "undir", "mutual") for k in ("share", "acqd", "acqs", "hor")])
o = pd.read_parquet(config.out("07_panel", "outcomes_focal_fyear.parquet"), columns=["focal", "fyear", "patents_missing", "families_originated", "granted", "emp", "sic", "intan_usd", "families_v", "cpc_subclasses", "fam_new_subclass"])
p = p.merge(e, on=["focal", "fyear"], how="left").merge(o, on=["focal", "fyear"], how="left").sort_values(["focal", "fyear"]).reset_index(drop=True)
g = p.groupby("focal")
p["consec"] = g["fyear"].shift(1).eq(p["fyear"] - 1)
p["cov_before"] = g["covered"].cumsum() - p["covered"]
p["pat_ok"] = (p["patents_missing"].eq(False)).astype(int)
p["ihs_fam"] = np.arcsinh(p["families_originated"].where(p["pat_ok"].eq(1)))
p["ihs_grt"] = np.arcsinh(p["granted"].where(p["pat_ok"].eq(1)))
p["ln_emp"] = np.log(pd.to_numeric(p["emp"], errors="coerce").where(lambda v: v > 0))
okp = p["pat_ok"].eq(1)
fv = p["families_v"].where(okp)
p["new_share"] = (p["fam_new_subclass"] / fv.where(fv > 0)).where(okp)
p["ihs_subcl"] = np.arcsinh(p["cpc_subclasses"].where(okp))
_spf = (p["cpc_subclasses"] / fv.where(fv > 0)).where(okp)
p["subcl_pf"] = _spf.clip(upper=_spf.groupby(p["fyear"]).transform(lambda x: x.quantile(0.99)))
p["subcl_net"] = (np.arcsinh(p["cpc_subclasses"]) - np.arcsinh(fv)).where(okp)
p["sic4"] = pd.to_numeric(p["sic"], errors="coerce")
lev = pd.read_parquet(config.out("07_panel", "panel_focal_fyear.parquet"), columns=["focal", "fyear", "leverage", "country"])
p = p.merge(lev, on=["focal", "fyear"], how="left")
g = p.groupby("focal")
_region = pd.read_csv(pathlib.Path(__file__).resolve().parent / "region_coverage_start.csv", dtype=str, keep_default_na=False, na_values=[""]).set_index("country")["region"]
p["reg_year"] = pd.factorize(p["country"].map(_region).fillna("other") + "_" + p["fyear"].astype(int).astype(str))[0] + 1
p["lev_w"] = p["leverage"].clip(lower=0); _q = p.groupby("fyear")["lev_w"].transform(lambda s: s.quantile(0.99)); p["lev_w"] = p["lev_w"].clip(upper=_q)
p["intan_at"] = pd.to_numeric(p["intan_usd"], errors="coerce").clip(lower=0) / np.exp(p["ln_at"])
samp = p["in_window"].eq(1) & p["own_accounts"].eq(1) & p["covered"].eq(1) & p["own_consol_post2"].eq(0) & p["cov_before"].ge(1)

# For each rival set: the share of its members in a control-changing deal (sh_), the stock of that share over t-5 to t-1 (s5_),
# and the event (sw_): a deal at t after a year without one, in consecutive fiscal years
for s, (ex, pre, nvar, nrec, title, r) in SETS.items():
    any_ = f"{pre}any_{r}"
    p[f"sh_{s}"] = p[f"{ex}_share" if r == "ca" else f"{ex}_share_{r}"].where(p["covered"].eq(1)).fillna(0.0)
    p[f"s5_{s}"] = sum(g[f"sh_{s}"].shift(k).fillna(0.0) for k in range(1, 6))
    p[f"s7_{s}"] = sum(g[f"sh_{s}"].shift(k).fillna(0.0) for k in range(1, 8))
    pos = p.loc[samp & p[f"s5_{s}"].gt(0), f"s5_{s}"]; t1, t2 = float(pos.quantile(1 / 3)), float(pos.quantile(2 / 3))
    p[f"dbin_{s}"] = np.select([p[f"s5_{s}"].eq(0), p[f"s5_{s}"].le(t1), p[f"s5_{s}"].le(t2)], [0, 1, 2], 3)
    p[f"sw_{s}"] = (p[any_].eq(1) & g[any_].shift(1).eq(0) & p["consec"]).astype(int)
for r in ("ca",):
    a = p[f"any_{r}_m"]
    p[f"sw_m{r}"] = (a.eq(1) & g[f"any_{r}_m"].shift(1).eq(0) & p["consec"]).astype(float).where(a.notna())
p["rd_stock5"] = sum((0.85 ** (k - 1)) * g["rd"].shift(k).fillna(0.0) for k in range(1, 6))
p["rd_stock5"] = p["rd_stock5"].where(sum(g["rd"].shift(k).notna().astype(int) for k in range(1, 6)) > 0)
TER = {"sales": g["ln_sales"].shift(1), "at": g["ln_at"].shift(1), "intanat": g["intan_at"].shift(1),
       "cash": g["cash_ratio_w"].shift(1), "age": g["ln_age"].shift(1), "lev": g["lev_w"].shift(1)}
for k, v in TER.items():
    q = v[samp].quantile([1 / 3, 2 / 3])
    p[f"ter_{k}"] = np.select([v.isna(), v <= q.iloc[0], v <= q.iloc[1]], [np.nan, 1, 2], 3)
p["prev_ca"] = g["any_ca"].cummax().shift(1).fillna(0)
ev = pd.read_parquet(config.out("07_panel", "exposure_events.parquet"), columns=["focal", "related", "kind", "cls", "year"])
ev = ev[ev["kind"].eq("rival") & ev["cls"].str.startswith("consolidation")]
tr = pd.read_parquet(config.out("07_panel", "rival_pair_traits.parquet"), columns=["focal", "rival", "year", "mutual"]).rename(columns={"rival": "related"})
ev = ev.merge(tr, on=["focal", "related", "year"], how="left")
cov = pd.read_stata(config.raw("entity_coverage"), convert_categoricals=False, columns=["FACTSET_ENTITY_ID", "PRIMARY_SIC_CODE"])
esic = pd.to_numeric(cov.drop_duplicates("FACTSET_ENTITY_ID").set_index("FACTSET_ENTITY_ID")["PRIMARY_SIC_CODE"], errors="coerce") // 100
ev["same_ind"] = (ev["related"].map(esic) == ev["focal"].map(esic)) & ev["related"].map(esic).notna()
sm = ev.groupby(["focal", "year"]).agg(sameind=("same_ind", "max")).reset_index().rename(columns={"year": "fyear"})
p = p.merge(sm, on=["focal", "fyear"], how="left")
p["sameind"] = p["sameind"].astype("float64")
esic4 = pd.to_numeric(cov.drop_duplicates("FACTSET_ENTITY_ID").set_index("FACTSET_ENTITY_ID")["PRIMARY_SIC_CODE"], errors="coerce")
ev["same_ind4"] = (ev["related"].map(esic4) == ev["focal"].map(esic4)) & ev["related"].map(esic4).notna()
ev["nonmut"] = ev["mutual"].fillna(0).ne(1)
sm4 = ev.groupby(["focal", "year"]).agg(sameind4=("same_ind4", "max"), nm_any=("nonmut", "max")).reset_index().rename(columns={"year": "fyear"})
p = p.merge(sm4, on=["focal", "fyear"], how="left")
p["sameind4"] = p["sameind4"].astype("float64")
p["nm_any"] = p["nm_any"].astype("float64").fillna(0.0)
g = p.groupby("focal")
fin = pd.read_parquet(config.out("03_financials", "panel_entity_year.parquet"), columns=["FACTSET_ENTITY_ID", "fyear", "sale_usd", "rd_net_usd"]).rename(columns={"FACTSET_ENTITY_ID": "related"})
fin["year"] = fin["fyear"] + 1
rv = ev[["focal", "related", "year"]].drop_duplicates().merge(fin[["related", "year", "sale_usd", "rd_net_usd"]], on=["related", "year"], how="left")
rv = rv.groupby(["focal", "year"]).agg(riv_sales=("sale_usd", "max"), riv_rnd=("rd_net_usd", "max")).reset_index().rename(columns={"year": "fyear"})
p = p.merge(rv, on=["focal", "fyear"], how="left")
own_sales = np.exp(g["ln_sales"].shift(1)); own_rnd = g["rd"].shift(1)
for k, num, den in (("sales", own_sales, p["riv_sales"]), ("rnd", own_rnd, p["riv_rnd"])):
    ratio = (num / den.where(den > 0)).where(num > 0)
    p[f"pos_{k}"] = np.select([ratio.isna(), ratio < 0.5, ratio <= 2.0], [np.nan, 1, 2], 3)
g = p.groupby("focal")
p["acqd_t"] = p["rival_acqd"].fillna(0).gt(0).astype(int)
p["acqs_t"] = p["rival_acqs"].fillna(0).gt(0).astype(int)
p["hor_t"] = p["rival_hor"].fillna(0).gt(0).astype(int)
drop = [c for c in p.columns if c.startswith(("rival_", "undir_", "mutual_"))] + ["families_originated", "granted", "patents_missing", "emp", "sic", "intan_usd"]
_sw = p["sw_d"].eq(1) & p["in_window"].eq(1) & p["own_accounts"].eq(1) & p["covered"].eq(1)
for flag, keys in (("cs1", ["fyear", "country", "sic2"]), ("cs2", ["fyear", "country", "sic2", "ter_sales"])):
    strata = p.loc[_sw, keys].drop_duplicates().assign(_ok=1)
    p[flag] = p[keys].merge(strata, on=keys, how="left")["_ok"].fillna(0).astype(int).to_numpy()
_cm = pd.read_parquet(config.out("08_estimation", "rival_communities.parquet"))
p = p.merge(_cm[["focal", "comm_id"]], on="focal", how="left"); g = p.groupby("focal")
p["comm_id"] = p["comm_id"].fillna(-1).astype(int)
p["own_pre3"] = ((p["own_consol"].fillna(0).eq(1)) | g["own_consol"].shift(1).fillna(0).eq(1) | g["own_consol"].shift(2).fillna(0).eq(1)).astype(int)
_fr = pd.read_parquet(config.out("08_estimation", "frontier_focal_fyear.parquet"), columns=["focal", "fyear", "dist_p90"])
p = p.merge(_fr, on=["focal", "fyear"], how="left"); g = p.groupby("focal")
for k in ("p90",):
    p[f"ldist_{k}"] = g[f"dist_{k}"].shift(1).where(p["consec"])
    _ev = p.loc[p["in_window"].eq(1) & p["own_accounts"].eq(1) & p["covered"].eq(1) & p["ln_rnd"].notna() & p["sw_d"].eq(1), f"ldist_{k}"].dropna()
    _t1, _t2 = float(_ev.quantile(1 / 3)), float(_ev.quantile(2 / 3))
    p[f"fter_{k}"] = np.select([p[f"ldist_{k}"].isna(), p[f"ldist_{k}"].le(_t1), p[f"ldist_{k}"].le(_t2)], [np.nan, 1, 2], 3)
p["ihs_rnd_k"] = np.arcsinh(np.sinh(p["ihs_rnd"]) * 1e3)
p["ihs_rnd_d"] = np.arcsinh(np.sinh(p["ihs_rnd"]) * 1e6)
for v, a, b in (("sup", "sup_acqd_ca", "sup_acqs_ca"), ("par", "par_acqd_ca", "par_acqs_ca"), ("nam", "nam_acqd_ca", "nam_acqs_ca"), ("rr", "rr_acqd_ca", "rr_acqs_ca")):
    p[f"{v}_any"] = (p[a].eq(1) | p[b].eq(1)).astype(float).where(p[a].notna() | p[b].notna())
for v, col in (("sup", "sup_any"), ("par", "par_any"), ("nam", "nam_any"), ("rr", "rr_any"), ("wave", "wave_w"), ("eff", "eff_ca"), ("wdr", "wdr_ca")):
    x = p[col]
    gx = x.groupby(p["id"])
    p[f"s5c_{v}"] = sum(gx.shift(k).fillna(0.0) for k in range(1, 6))
    p[f"swc_{v}"] = (x.eq(1) & gx.shift(1).eq(0) & p["consec"]).astype(int)
_it = pd.read_parquet(config.out("08_estimation", "intensity_focal_fyear.parquet"), columns=["focal", "fyear", "step_max"])
p = p.merge(_it, on=["focal", "fyear"], how="left")
p["step_max"] = pd.to_numeric(p["step_max"], errors="coerce").astype("float64"); p["ldist_p90"] = pd.to_numeric(p["ldist_p90"], errors="coerce").astype("float64")
_mev = p["in_window"].eq(1) & p["own_accounts"].eq(1) & p["covered"].eq(1) & p["ln_rnd"].notna() & p["sw_d"].eq(1)
_med_i = float(p.loc[_mev, "step_max"].median()); _med_f = float(p.loc[_mev, "ldist_p90"].median())
p["int_g"] = np.select([p["step_max"].isna(), p["step_max"].gt(_med_i)], [0, 2], 1)
p["fh"] = np.select([p["ldist_p90"].isna(), p["ldist_p90"].le(_med_f)], [np.nan, 1], 2)
_t1i, _t2i = float(p.loc[_mev, "step_max"].quantile(1 / 3)), float(p.loc[_mev, "step_max"].quantile(2 / 3))
p["int_t"] = np.select([p["step_max"].isna(), p["step_max"].le(_t1i), p["step_max"].le(_t2i)], [0, 1, 2], 3)
p["nn_ca"] = (p["u_n_ca"] - p["nriv_ca"]).clip(lower=0) + p["m_n_ca"]
p["nn_norec"] = (p["nn_ca"].eq(0) & p["covered"].eq(1)).astype(int)
p["nn_any"] = (p["nam_any"].fillna(0).eq(1) | p["m_any_ca"].fillna(0).eq(1)).astype(float).where(p["any_ca"].notna())
p["nm_d"] = p["nm_any"].fillna(0).astype(int)
p["na_d"] = p["nam_any"].fillna(0).astype(int)
_gn = p.groupby("id")
p["s5_nn"] = sum(_gn["nn_any"].shift(k).fillna(0.0) for k in range(1, 6))
p["sw_nn"] = (p["nn_any"].eq(1) & _gn["nn_any"].shift(1).eq(0) & p["consec"]).astype(int)
_ai = pd.read_parquet(config.out("08_estimation", "attention_inputs.parquet")).rename(columns={"age_min": "cage_min", "age_max": "cage_max"})
p = p.merge(_ai, on=["focal", "fyear"], how="left")
for c in ("cage_min", "cage_max", "lr_od", "lr_on", "lr_mu", "sim_max", "sim_obs", "n_new"):
    p[c] = pd.to_numeric(p[c], errors="coerce").astype("float64")
p["lnriv"] = np.log1p(p["nriv_ca"]).where(p["covered"].eq(1))
p["ihs_new"] = np.arcsinh(p["n_new"].fillna(0)).where(p["covered"].eq(1))
p["mu_d"] = p["m_any_ca"].fillna(0).astype(int)
p["od_d"] = (p["any_ca"].eq(1) & (p["m_any_ca"].ne(1) | p["nm_d"].eq(1))).astype(float).where(p["any_ca"].notna())
p["od_c"] = p["od_d"].fillna(0)
p["on_t"] = p["na_d"].astype(float).where(p["any_ca"].notna())
p["od_n"] = (p["nriv_ca"] - p["m_n_ca"]).clip(lower=0); p["on_n"] = (p["u_n_ca"] - p["nriv_ca"]).clip(lower=0)
p["od_norec"] = (p["od_n"].eq(0) & p["covered"].eq(1)).astype(int); p["on_norec"] = (p["on_n"].eq(0) & p["covered"].eq(1)).astype(int)
_ga = p.groupby("id")
for v, d_ in (("od", "od_d"), ("on", "on_t")):
    p[f"s5_{v}"] = sum(_ga[d_].shift(k).fillna(0.0) for k in range(1, 6))
    p[f"sw_{v}"] = (p[d_].eq(1) & _ga[d_].shift(1).eq(0) & p["consec"]).astype(int)
_ev = p["in_window"].eq(1) & p["own_accounts"].eq(1) & p["covered"].eq(1) & p["ln_rnd"].notna() & p["sw_d"].eq(1)
_sm = float(p.loc[_ev & p["sim_obs"].eq(1) & p["sim_max"].gt(0), "sim_max"].median())
p["sim_g"] = np.select([p["sim_obs"].ne(1), p["sim_max"].le(0), p["sim_max"].le(_sm)], [0, 1, 2], 3)
_ni = pd.read_parquet(config.out("08_estimation", "network_inputs.parquet"), columns=["focal", "fyear", "sh_larger", "has_whole", "all_whole"])
p = p.merge(_ni, on=["focal", "fyear"], how="left")
for c in ("sh_larger", "has_whole", "all_whole"):
    p[c] = pd.to_numeric(p[c], errors="coerce").astype("float64")
_gw = p.groupby("id")
p["lsh_larger"] = _gw["sh_larger"].shift(1).where(p["consec"])
p["lprom"] = (np.log1p(_gw["nn_ca"].shift(1)) - np.log1p(_gw["nriv_ca"].shift(1))).where(p["consec"])
_evw = p["in_window"].eq(1) & p["own_accounts"].eq(1) & p["covered"].eq(1) & p["ln_rnd"].notna() & p["sw_d"].eq(1)
p["npos"] = np.select([p["lsh_larger"].isna(), p["lsh_larger"].lt(0.5), p["lsh_larger"].lt(1.0)], [np.nan, 1, 2], 3)
for v, nm in (("lprom", "nprom"),):
    _x = p.loc[_evw, v].dropna(); _a, _b = float(_x.quantile(1 / 3)), float(_x.quantile(2 / 3))
    p[nm] = np.select([p[v].isna(), p[v].le(_a), p[v].le(_b)], [np.nan, 1, 2], 3)
_gp = p.groupby("id")
_fam = p["families_originated"].where(p["pat_ok"].eq(1))
_lrd = _gp["rd"].shift(1).where(p["consec"])
_r = (_fam / _lrd.where(_lrd > 0))
p["fam_lrd_w"] = _r.clip(upper=_r.groupby(p["fyear"]).transform(lambda x: x.quantile(0.99)))
_se = pd.read_parquet(config.out("08_estimation", "patents_same_entity.parquet"))
p = p.merge(_se, on=["focal", "fyear"], how="left")
p["ihs_famse"] = np.arcsinh(pd.to_numeric(p["fam_se"], errors="coerce").fillna(0.0)).where(p["pat_ok"].eq(1))
_gq = p.groupby("id")
samp = p["in_window"].eq(1) & p["own_accounts"].eq(1) & p["covered"].eq(1) & p["own_consol_post2"].eq(0) & p["cov_before"].ge(1)
p["rdsa"] = p["rd_stock5"] / np.exp(_gq["ln_at"].shift(1))
_q = p.loc[samp, "rdsa"].quantile([1 / 3, 2 / 3])
p["ter_rdsa"] = np.select([p["rdsa"].isna(), p["rdsa"] <= _q.iloc[0], p["rdsa"] <= _q.iloc[1]], [np.nan, 1, 2], 3)
_fam5 = pd.concat([_gq["families_originated"].shift(k) for k in range(1, 6)], axis=1).sum(axis=1, min_count=1)
p["pat_stock5"] = _fam5.where(p["pat_ok"].eq(1))
_m = p.loc[samp & p["pat_stock5"].gt(0), "pat_stock5"].median()
p["pst_bin"] = np.select([p["pat_stock5"].isna(), p["pat_stock5"].eq(0), p["pat_stock5"].le(_m)], [np.nan, 1, 2], 3)
p["full5"] = pd.concat([_gq["any_ca"].shift(k).notna() for k in range(1, 6)], axis=1).all(axis=1).astype(int)
_ev5 = samp & p["sw_d"].eq(1) & p["full5"].eq(1)
_mx = p.loc[_ev5 & p["s5_d"].gt(0), "s5_d"].median()
p["pex"] = np.select([p["full5"].ne(1), p["s5_d"].eq(0), p["s5_d"].le(_mx)], [np.nan, 1, 2], 3)
_last = pd.Series(np.nan, index=p.index)
for _k in (5, 4, 3, 2):
    _last = _last.mask(_gq["any_ca"].shift(_k).eq(1), _k)
p["prec"] = np.select([p["full5"].ne(1), _last.eq(2), _last.isin([3, 4])], [np.nan, 1, 2], 3)
_own = pd.read_parquet(config.out("08_estimation", "estimation_panel.parquet"), columns=["focal", "fyear", "own_n_consolidation_acquired"])
p = p.merge(_own, on=["focal", "fyear"], how="left")
p["own_tgt"] = p["own_n_consolidation_acquired"].fillna(0).gt(0).astype(int)
_ok4 = p["sic4"].notna()
_tot = p[_ok4].groupby(["sic4", "fyear"])["own_consol"].transform("sum")
p["ind_n"] = (_tot - p.loc[_ok4, "own_consol"]).reindex(p.index)
_deal = p.loc[_ok4 & p["own_consol"].eq(1), ["focal", "fyear", "sic4"]].rename(columns={"focal": "rival", "sic4": "rsic4"})
_pairs = pd.read_parquet(config.out("07_panel", "rival_pair_year.parquet"), columns=["focal", "rival", "year"]).rename(columns={"year": "fyear"}).drop_duplicates()
_nm = _pairs.merge(_deal, on=["rival", "fyear"], how="inner").merge(p[["focal", "fyear", "sic4"]], on=["focal", "fyear"], how="inner")
_nm = _nm[_nm["rsic4"].eq(_nm["sic4"])].groupby(["focal", "fyear"]).size().rename("ind_named_n").reset_index()
p = p.merge(_nm, on=["focal", "fyear"], how="left")
p["ind_named_n"] = p["ind_named_n"].fillna(0).where(p["ind_n"].notna())
p["ind_any"] = p["ind_n"].gt(0).astype(float).where(p["ind_n"].notna())
_gi = p.groupby("focal")
p["s5_ind"] = sum(_gi["ind_any"].shift(k).fillna(0.0) for k in range(1, 6))
p["sw_ind"] = (p["ind_any"].eq(1) & _gi["ind_any"].shift(1).eq(0) & p["consec"]).astype(int)
p["csic4"] = p["sic4"].fillna(-1)

# The panel the Stata runner reads
st = p.drop(columns=drop).copy()
for c in st.columns:
    if str(st[c].dtype) in ("Int64", "Int32", "boolean"):
        st[c] = st[c].astype("float")
    elif st[c].dtype == bool:
        st[c] = st[c].astype(int)
    elif st[c].dtype == object or str(st[c].dtype) == "string":
        st[c] = st[c].astype("string").fillna("").astype(str)
st.to_stata(config.out("08_estimation", "estimation_panel_ladders.dta"), write_index=False, version=118)
print(f"ladders panel: {len(st):,} rows, {st.shape[1]} columns")


def base_sample(y):
    return f"in_window==1 & own_accounts==1 & covered==1 & L2.{y}!=. & L2.ln_at!=. & L2.ln_sales!=." + (" & pat_ok==1" if y in ("ihs_fam", "ihs_grt") or y in PAT else "")


PAT = {"subcl_pf": "distinct CPC subclasses per family of the year", "subcl_net": "asinh(subclasses) - asinh(families): breadth net of volume", "grant_share": "the share of the year's originated families with a grant", "new_share": "the share of the year's families in subclasses new to the firm",
       "ihs_subcl": "asinh(distinct CPC subclasses of the year's families)", "ln_cit5pf": "log(1 + forward citations within five years per family)"}
OUTS = {"ihs_rnd_k": "asinh(R&D in thousands of dollars)", "ihs_rnd_d": "asinh(R&D in dollars)", "lnriv": "log(1 + live designated rivals)", "ihs_famse": "asinh(patent families originated under the firm's own names)", "fam_lrd_w": "patent families originated per million dollars of lagged R&D (winsorised)", "ihs_rcin": "asinh(family-subclass counts in the named rivals' core fields)", "ihs_rcout": "asinh(family-subclass counts outside the rivals' core fields)", "sh_rc": "share of family-subclass counts in the named rivals' core fields", "ihs_new": "asinh(rivals named for the first time in the year)"}
OUT_ALL = {**OUT, **GROWTH, **PAT, **OUTS}


rows = []


# One LP-DiD per row of the grid (stata/lpdid_run.do). The sample holds the events and the comparison firm-years; in the main
# comparison (zerostock) these have no deal in the rival set at t and none from t-5 to t-1, and lpdid's
# nonabsorbing(1, oneoff) drops those with a deal before t+h
def cell(ladder, row, y, s="d", treat=None, cut="", bin_=9, cgroup="zerostock", extra="", title="", ctrl_rows_override=None, cluster="unit_id", absorb="sic2_year",
         ctrl_str=None, pre=4, post_=None):
    treat = treat or f"sw_{s}"
    post = 8 if y in ("ihs_fam", "ihs_grt") or y in PAT else 5
    post = post_ if post_ is not None else post
    any_, stock, dbin = f"{SETS[s][1]}any_{SETS[s][5]}", f"s5_{s}", f"dbin_{s}"
    tr = f"{treat}==1" + (f" & {dbin}=={bin_}" if bin_ != 9 else "") + cut
    ctrl_rows = ctrl_rows_override or {"zerostock": f"{stock}==0 & {any_}==0", "never": f"{stock}==0 & {any_}==0", "allclean": f"{any_}==0 & L.{any_}==0", "notyet": f"{stock}==0 & {any_}==0"}[cgroup]
    sample = f"{base_sample(y)}{extra} & (({tr}) | ({ctrl_rows}))"
    opt = {"zerostock": "nonabsorbing(1, oneoff)", "allclean": "nonabsorbing(1, oneoff)", "never": "nonabsorbing(1, oneoff) nevertreated", "notyet": "nonabsorbing(1, oneoff notyet)"}[cgroup]
    options = opt + " rw pooled dylags(1)"
    ctrl = ctrl_str
    key = "|".join([y, treat, sample, options, ctrl, absorb, cluster, str(pre), str(post)])
    rows.append({"run_id": hashlib.sha1(key.encode()).hexdigest()[:10], "ladder": ladder, "row": row, "outcome": y, "set": s, "bin": bin_, "cgroup": cgroup, "design": "isolated",
                 "stab_lag": 1, "y": y, "treat": treat, "sample": sample, "options": options, "controls": ctrl, "absorb": absorb, "cluster": cluster, "pre": pre, "post": post,
                 "title": f"{ladder} / {row} / {y}: {title}; {OUT_ALL[y]}"})


BINS = {0: "zero", 1: "low", 2: "mid", 3: "high"}
PAPER_BLOCKS = [("c0", ""), ("c1", "L.ln_at L.ln_sales L.ln_age L.cash_ratio_w L.capx_at_w"), ("c2", "L.D.ln_at L.D.ln_sales"), ("c3", "L.nriv_ca norec_ca L.norec_ca"), ("c4", "own_consol L.own_consol L2.own_consol")]

# 42_paper: the main specification, controls added one block at a time, against the five-year-quiet comparison (zerostock) and the
# estimator's standard comparison (allclean); the excl rows drop the firm-years with an own deal at t, t-1 or t-2
for y in ("ln_rnd", "ihs_rnd"):
    for cg in ("zerostock", "allclean"):
        acc = []
        for name, blk in PAPER_BLOCKS:
            acc.append(blk)
            cell("42_paper", f"{name}_{cg}", y, cgroup=cg, ctrl_str=" ".join(a for a in acc if a), absorb="sic2_year cty_year", title=f"the paper's specification, controls through block {name}; industry-by-year and country-by-year effects; controls {cg}")
        cell("42_paper", f"excl_{cg}", y, cgroup=cg, ctrl_str=" ".join(b for _, b in PAPER_BLOCKS[:4] if b), extra=" & own_pre3==0", absorb="sic2_year cty_year", title=f"the paper's specification with the firm-years carrying an own deal at t, t-1 or t-2 excluded from events and comparisons; controls {cg}")
# 42_paper: the main specification on the undirected and the mutual rival sets
for y in ("ln_rnd", "ihs_rnd"):
    for s_ in ("u", "m"):
        nvar, nrec = SETS[s_][2], SETS[s_][3]
        ctl = f"L.ln_at L.ln_sales L.ln_age L.cash_ratio_w L.capx_at_w L.D.ln_at L.D.ln_sales L.{nvar} {nrec} L.{nrec} own_consol L.own_consol L2.own_consol"
        cell("42_paper", f"set_{s_}_zerostock", y, s=s_, cgroup="zerostock", ctrl_str=ctl, absorb="sic2_year cty_year", title=f"the paper's specification on {SETS[s_][4]}; five-year-quiet comparison on that set")
# 76_decomp: the rival-set controls taken apart
_C2 = " ".join(b for _, b in PAPER_BLOCKS[:3] if b)
for y in ("ln_rnd", "ihs_rnd"):
    for name, part in (("size", "L.nriv_ca"), ("norec", "norec_ca L.norec_ca"), ("norec_t", "norec_ca"), ("norec_t1", "L.norec_ca")):
        cell("76_decomp", f"c2_{name}", y, cgroup="zerostock", ctrl_str=f"{_C2} {part}", absorb="sic2_year cty_year", title=f"column 3 of the paper's ladder plus {part} only")
# 77_audit: the main specification with one change at a time
_FULL = " ".join(b for _, b in PAPER_BLOCKS if b)
assert _FULL.count("norec_ca L.norec_ca") == 1
for y in ("ln_rnd", "ihs_rnd"):
    cell("77_audit", "fe_ind", y, cgroup="zerostock", ctrl_str=_FULL, absorb="sic2_year", title="the paper's specification with industry-by-year effects only")
    cell("77_audit", "fe_region", y, cgroup="zerostock", ctrl_str=_FULL, absorb="sic2_year reg_year", title="the paper's specification with industry-by-year and region-by-year effects")
    cell("77_audit", "drop_norec", y, cgroup="zerostock", ctrl_rows_override="s5_d==0 & any_ca==0 & norec_ca==0", ctrl_str=_FULL.replace("norec_ca L.norec_ca", "L.norec_ca"),
         absorb="sic2_year cty_year", title="the paper's specification with the comparison firm-years without a rival on record at t dropped, not flagged")
    cell("77_audit", "pool03", y, cgroup="zerostock", ctrl_str=_FULL, absorb="sic2_year cty_year", post_=3, title="the paper's specification with the pooled window h = 0..3")
# 43_sens: five placebo years; own deals at t-1 and t-2 only; not-yet and never exposed comparisons; seven quiet years
for y in ("ln_rnd", "ihs_rnd"):
    full = " ".join(b for _, b in PAPER_BLOCKS if b)
    cell("43_sens", "pre5", y, cgroup="zerostock", ctrl_str=full, absorb="sic2_year cty_year", pre=5, title="the paper's specification with five placebo horizons")
    cell("43_sens", "ownlags", y, cgroup="zerostock", ctrl_str=" ".join(b for _, b in PAPER_BLOCKS[:4] if b) + " L.own_consol L2.own_consol", absorb="sic2_year cty_year", title="the paper's specification with the own-deal indicators at t-1 and t-2 only (the same-year indicator out)")
    cell("43_sens", "notyet", y, cgroup="notyet", ctrl_str=full, absorb="sic2_year cty_year", title="the paper's specification, comparison firm-years of firms not yet exposed (five-year-quiet rows, the estimator's notyet option)")
    cell("43_sens", "never", y, cgroup="never", ctrl_str=full, absorb="sic2_year cty_year", title="the paper's specification, comparison firm-years of firms never exposed (five-year-quiet rows, the estimator's nevertreated option)")
    cell("43_sens", "quiet7", y, cgroup="zerostock", ctrl_rows_override="s7_d==0 & any_ca==0", ctrl_str=full, absorb="sic2_year cty_year", title="the paper's specification, comparison firm-years with no rival deal in the seven years before t")
# 43_sens: the inverse hyperbolic sine of R&D in thousands and in dollars
for y in ("ihs_rnd_k", "ihs_rnd_d"):
    cell("43_sens", "scale", y, cgroup="zerostock", ctrl_str=" ".join(b for _, b in PAPER_BLOCKS if b), absorb="sic2_year cty_year", title="the paper's specification, asinh of R&D in finer units")
# 43_sens: years without a rival on record excluded rather than coded as no deal
for y in ("ln_rnd", "ihs_rnd"):
    cell("43_sens", "missing", y, treat="sw_mca", cgroup="zerostock", ctrl_rows_override="s5_d==0 & any_ca_m==0", ctrl_str=" ".join(b for _, b in PAPER_BLOCKS if b), absorb="sic2_year cty_year", title="the paper's specification with the unrecorded years excluded")
# 44_ident: events, then also comparison firm-years, with a rival on record at t and t-1
for y in ("ln_rnd", "ihs_rnd"):
    full = " ".join(b for _, b in PAPER_BLOCKS if b)
    cell("44_ident", "onsets", y, cgroup="zerostock", cut=" & L.norec_ca==0", ctrl_str=full, absorb="sic2_year cty_year", title="identification: switches with a recorded year before; main comparison group")
    cell("44_ident", "both", y, cgroup="zerostock", cut=" & L.norec_ca==0", ctrl_rows_override="s5_d==0 & any_ca==0 & norec_ca==0 & L.norec_ca==0", ctrl_str=full, absorb="sic2_year cty_year", title="identification: switches and comparison firm-years recorded at t and t-1")
# 45_comp: the firm's characteristics as outcomes
for y in ("ln_at", "ln_sales", "ln_emp", "cash_ratio_w", "lev_w", "capx_at_w"):
    cell("45_comp", "paths", y, cgroup="zerostock", ctrl_str=" ".join(b for _, b in PAPER_BLOCKS if b), absorb="sic2_year cty_year", title="identification: the firm's characteristics as outcomes in the main design")
    cell("45_comp", "paths_rd", y, cgroup="zerostock", extra=" & ln_rnd!=. & L.ln_rnd!=. & L2.ln_rnd!=.", ctrl_str=" ".join(b for _, b in PAPER_BLOCKS if b), absorb="sic2_year cty_year", title="identification: the firm's characteristics as outcomes, on the firm-years of the log R&D sample")
# 45_comp: comparison firm-years on the common support of the events
for y in ("ln_rnd", "ihs_rnd"):
    for flag in ("cs1", "cs2"):
        cell("45_comp", f"support_{flag}", y, cgroup="zerostock", ctrl_rows_override=f"s5_d==0 & any_ca==0 & {flag}==1", ctrl_str=" ".join(b for _, b in PAPER_BLOCKS if b), absorb="sic2_year cty_year", title=f"identification: comparison firm-years on the common support ({flag})")
# 47_channels: the deals of suppliers and customers, partners, firms naming the firm, rivals of rivals, and industry waves, as
# controls and as events; deals dated by completion; withdrawn deals
L3 = lambda v: f"{v} L.{v} L2.{v}"
LL = lambda v: f"L.{v} L2.{v}"
full = " ".join(b for _, b in PAPER_BLOCKS if b)
CH = {"sup": L3("sup_acqd_ca") + " " + L3("sup_acqs_ca"), "par": L3("par_acqd_ca") + " " + L3("par_acqs_ca"), "nam": L3("nam_acqd_ca") + " " + L3("nam_acqs_ca"),
      "rr": L3("rr_acqd_ca") + " " + L3("rr_acqs_ca") + " " + L3("rrno_ca"), "rr_lag": LL("rr_acqd_ca") + " " + LL("rr_acqs_ca") + " " + LL("rrno_ca"),
      "wave": L3("wave_w"), "wave_lag": LL("wave_w")}
CH["all"] = " ".join(CH[k] for k in ("sup", "par", "nam", "rr", "wave"))
EVT = {"sup": ("swc_sup", "sup_any"), "par": ("swc_par", "par_any"), "nam": ("swc_nam", "nam_any"), "rr": ("swc_rr", "rr_any"), "wave": ("swc_wave", "wave_w")}
for y in ("ln_rnd", "ihs_rnd"):
    for k, c in CH.items():
        cell("47_channels", f"ctrl_{k}", y, cgroup="zerostock", ctrl_str=full + " " + c, absorb="sic2_year cty_year", title=f"identification: the {k} channel among the controls")
    for k, (tr, v) in EVT.items():
        cell("47_channels", f"evt5_{k}", y, treat=tr, cgroup="zerostock", cut=" & any_ca==0", ctrl_rows_override=f"s5c_{k}==0 & {v}==0 & any_ca==0", ctrl_str=full, absorb="sic2_year cty_year", title=f"identification: the {k} channel as the event, five-year quiet comparison")
        cell("47_channels", f"evt2_{k}", y, treat=tr, cgroup="allclean", cut=" & any_ca==0", ctrl_rows_override=f"{v}==0 & L.{v}==0 & any_ca==0", ctrl_str=full, absorb="sic2_year cty_year", title=f"identification: the {k} channel as the event, two-year quiet comparison")
    cell("47_channels", "completion", y, treat="swc_eff", cgroup="zerostock", ctrl_rows_override="s5c_eff==0 & eff_ca==0", ctrl_str=full, absorb="sic2_year cty_year", title="identification: events dated by the completion year")
    cell("47_channels", "withdrawn", y, treat="swc_wdr", cgroup="zerostock", cut=" & any_ca==0", ctrl_rows_override="s5_d==0 & any_ca==0 & wdr_ca==0", ctrl_str=full, absorb="sic2_year cty_year", title="identification: a rival's withdrawn or rumoured deal as a placebo event")
# 48_mech: the events by the share of the set in deals, the deal's type, industry proximity, waves, and the firm's position
fullm = " ".join(b for _, b in PAPER_BLOCKS if b)
MECH = [("dose_high", " & thr_ca==1"), ("dose_low", " & thr_ca==0"), ("hor", " & hor_t==1"), ("nonhor", " & hor_t==0"), ("acqd", " & acqd_t==1"), ("acqs", " & acqs_t==1"),
        ("prox_same4", " & sameind4==1"), ("prox_same2", " & sameind4==0 & sameind==1"), ("prox_other", " & sameind==0"), ("wave_in", " & wave_w==1"), ("wave_out", " & wave_w==0")]
MECH += [(f"pos_{k}_{pos}", f" & pos_{k}=={pos}") for k in ("sales", "rnd") for pos in (1, 2, 3)]
for y in ("ln_rnd", "ihs_rnd"):
    for row, cut in MECH:
        cell("48_mech", row, y, cgroup="zerostock", cut=cut, ctrl_str=fullm, absorb="sic2_year cty_year", title=f"mechanism: {row}")
    ctl_m = f"L.ln_at L.ln_sales L.ln_age L.cash_ratio_w L.capx_at_w L.D.ln_at L.D.ln_sales L.{SETS['m'][2]} {SETS['m'][3]} L.{SETS['m'][3]} own_consol L.own_consol L2.own_consol"
    cell("48_mech", "mutual_ctrl", y, s="m", cgroup="zerostock", ctrl_str=ctl_m + " nm_any L.nm_any L2.nm_any", absorb="sic2_year cty_year", title="mechanism: the mutual rivals' switch, the other designated rivals' deals controlled at t..t-2")
# 49_het: the events by the firm's characteristics at t-1 and by its prior exposure
for y in ("ln_rnd", "ihs_rnd"):
    for k in ("at", "age", "cash", "lev"):
        for ter in (1, 2, 3):
            cell("49_het", f"ter_{k}_{ter}", y, cgroup="zerostock", cut=f" & ter_{k}=={ter}", ctrl_rows_override=f"s5_d==0 & any_ca==0 & ter_{k}=={ter}", ctrl_str=fullm, absorb="sic2_year cty_year", title=f"heterogeneity: {k} tercile {ter}")
    for b, bt in BINS.items():
        cell("49_het", f"prior_total_{bt}", y, bin_=b, cgroup="zerostock", ctrl_str=fullm, absorb="sic2_year cty_year", title=f"heterogeneity: prior exposure {bt}, against the five-year-quiet comparison")
        if b > 0:
            cell("49_het", f"prior_incr_{bt}", y, bin_=b, cgroup="allclean", ctrl_rows_override=f"any_ca==0 & L.any_ca==0 & dbin_d=={b}", ctrl_str=fullm, absorb="sic2_year cty_year", title=f"heterogeneity: prior exposure {bt}, against the two-year-quiet firm-years of the same bin")
# 49_het: the tercile events against the main comparison group
for y in ("ln_rnd", "ihs_rnd"):
    for k in ("at", "age", "cash", "lev"):
        for ter in (1, 2, 3):
            cell("49_het", f"termain_{k}_{ter}", y, cgroup="zerostock", cut=f" & ter_{k}=={ter}", ctrl_str=fullm, absorb="sic2_year cty_year", title=f"heterogeneity: {k} tercile {ter} events, main comparison group")
# 55_naming: the designated, naming and mutual rival sets, with the other direction's deals controlled
L3_ = lambda v: f"{v} L.{v} L2.{v}"
LL_ = lambda v: f"L.{v} L2.{v}"
FUT_ = lambda v: "".join(f" & F{k}.{v}!=1" for k in range(1, 6)).replace("F1.", "F.")
for y in ("ln_rnd", "ihs_rnd"):
    ctl_n = "L.ln_at L.ln_sales L.ln_age L.cash_ratio_w L.capx_at_w L.D.ln_at L.D.ln_sales L.nn_ca nn_norec L.nn_norec own_consol L.own_consol L2.own_consol"
    ctl_m = f"L.ln_at L.ln_sales L.ln_age L.cash_ratio_w L.capx_at_w L.D.ln_at L.D.ln_sales L.{SETS['m'][2]} {SETS['m'][3]} L.{SETS['m'][3]} own_consol L.own_consol L2.own_consol"
    for tag, f in (("ctrl", L3_), ("lag", LL_)):
        cell("55_naming", f"des_{tag}", y, cgroup="zerostock", ctrl_str=fullm + " " + f("na_d"), absorb="sic2_year cty_year", title=f"naming direction: designated set, one-way namers' deals controlled ({tag})")
        cell("55_naming", f"nam_{tag}", y, treat="sw_nn", cgroup="zerostock", ctrl_rows_override="s5_nn==0 & nn_any==0", ctrl_str=ctl_n + " " + f("nm_d"), absorb="sic2_year cty_year", title=f"naming direction: naming set, one-way designated rivals' deals controlled ({tag})")
        cell("55_naming", f"mut_{tag}", y, s="m", cgroup="zerostock", ctrl_str=ctl_m + " " + f("nm_d") + " " + f("na_d"), absorb="sic2_year cty_year", title=f"naming direction: mutual set, both one-way directions' deals controlled ({tag})")
    cell("55_naming", "nam_plain", y, treat="sw_nn", cgroup="zerostock", ctrl_rows_override="s5_nn==0 & nn_any==0", ctrl_str=ctl_n, absorb="sic2_year cty_year", title="naming direction: naming set, no cross-direction control")
    cell("55_naming", "des_pure", y, cgroup="zerostock", cut=FUT_("na_d"), ctrl_str=fullm + " " + L3_("na_d"), absorb="sic2_year cty_year", title="naming direction: designated set, no one-way namer deal through t+5 (reserve)")
    cell("55_naming", "nam_pure", y, treat="sw_nn", cgroup="zerostock", cut=FUT_("nm_d"), ctrl_rows_override="s5_nn==0 & nn_any==0", ctrl_str=ctl_n + " " + L3_("nm_d"), absorb="sic2_year cty_year", title="naming direction: naming set, no one-way designated deal through t+5 (reserve)")
    cell("55_naming", "mut_pure", y, s="m", cgroup="zerostock", cut=FUT_("nm_d") + FUT_("na_d"), ctrl_str=ctl_m + " " + L3_("nm_d") + " " + L3_("na_d"), absorb="sic2_year cty_year", title="naming direction: mutual set, no one-way deal of either direction through t+5 (reserve)")
# 57_intensity: the deal's size relative to the consolidating rival, and the distance to the productivity frontier
for y in ("ln_rnd", "ihs_rnd"):
    for row, cut in (("int_high", " & int_g==2"), ("int_low", " & int_g==1"), ("int_none", " & int_g==0"),
                     ("int_high_near", " & int_g==2 & fh==1"), ("int_high_far", " & int_g==2 & fh==2"),
                     ("int_low_near", " & int_g==1 & fh==1"), ("int_low_far", " & int_g==1 & fh==2")):
        cell("57_intensity", row, y, cgroup="zerostock", cut=cut, ctrl_str=fullm, absorb="sic2_year cty_year", title=f"intensity: {row}")
    for h_, hl in ((1, "near"), (2, "far")):
        for t_ in (1, 2, 3):
            cell("57_intensity", f"intt{t_}_{hl}", y, cgroup="zerostock", cut=f" & int_t=={t_} & fh=={h_}", ctrl_str=fullm, absorb="sic2_year cty_year", title=f"intensity: step tercile {t_}, frontier half {hl}")
    for g_, gl in ((2, "high"), (1, "low")):
        cell("57_intensity", f"int_{gl}_nearest", y, cgroup="zerostock", cut=f" & int_g=={g_} & fter_p90==1", ctrl_str=fullm, absorb="sic2_year cty_year", title=f"intensity: {gl} step, nearest frontier tercile")
# 58_rdint: R&D over lagged assets and over lagged sales
for y in ("rnd_lat_w", "rnd_lsl_w"):
    cell("58_rdint", "main", y, cgroup="zerostock", ctrl_str=fullm, absorb="sic2_year cty_year", title="R&D intensity: the main specification")
    ctl_n8 = "L.ln_at L.ln_sales L.ln_age L.cash_ratio_w L.capx_at_w L.D.ln_at L.D.ln_sales L.nn_ca nn_norec L.nn_norec own_consol L.own_consol L2.own_consol"
    ctl_m8 = f"L.ln_at L.ln_sales L.ln_age L.cash_ratio_w L.capx_at_w L.D.ln_at L.D.ln_sales L.{SETS['m'][2]} {SETS['m'][3]} L.{SETS['m'][3]} own_consol L.own_consol L2.own_consol"
    cell("58_rdint", "des_ctrl", y, cgroup="zerostock", ctrl_str=fullm + " " + L3_("na_d"), absorb="sic2_year cty_year", title="R&D intensity: designated set, other direction controlled")
    cell("58_rdint", "nam_ctrl", y, treat="sw_nn", cgroup="zerostock", ctrl_rows_override="s5_nn==0 & nn_any==0", ctrl_str=ctl_n8 + " " + L3_("nm_d"), absorb="sic2_year cty_year", title="R&D intensity: naming set, other direction controlled")
    cell("58_rdint", "mut_ctrl", y, s="m", cgroup="zerostock", ctrl_str=ctl_m8 + " " + L3_("nm_d") + " " + L3_("na_d"), absorb="sic2_year cty_year", title="R&D intensity: mutual set, both directions controlled")
    for t_ in (1, 2, 3):
        cell("58_rdint", f"front_p90_{t_}", y, cgroup="zerostock", cut=f" & fter_p90=={t_}", ctrl_str=fullm, absorb="sic2_year cty_year", title=f"R&D intensity: frontier tercile {t_}")
    for row, cut in (("int_high", " & int_g==2"), ("int_low", " & int_g==1"), ("int_high_near", " & int_g==2 & fh==1"), ("int_high_far", " & int_g==2 & fh==2"),
                     ("int_low_near", " & int_g==1 & fh==1"), ("int_low_far", " & int_g==1 & fh==2")):
        cell("58_rdint", row, y, cgroup="zerostock", cut=cut, ctrl_str=fullm, absorb="sic2_year cty_year", title=f"R&D intensity: {row}")
    for k in ("at", "age", "cash", "lev"):
        for t_ in (1, 2, 3):
            cell("58_rdint", f"termain_{k}_{t_}", y, cgroup="zerostock", cut=f" & ter_{k}=={t_}", ctrl_str=fullm, absorb="sic2_year cty_year", title=f"R&D intensity: {k} tercile {t_}")
    for row, cut in (("dose_high", " & thr_ca==1"), ("dose_low", " & thr_ca==0")):
        cell("58_rdint", row, y, cgroup="zerostock", cut=cut, ctrl_str=fullm, absorb="sic2_year cty_year", title=f"R&D intensity: {row}")
    for b, bt in BINS.items():
        cell("58_rdint", f"prior_total_{bt}", y, bin_=b, cgroup="zerostock", ctrl_str=fullm, absorb="sic2_year cty_year", title=f"R&D intensity: prior exposure {bt}")
# 59_balanced: the firm-years whose outcome is observed in every year from t+1 to t+5
for y in ("ln_rnd", "ihs_rnd", "rnd_lat_w"):
    BAL_ = "".join(f" & F{k}.{y}!=." for k in range(1, 6)).replace("F1.", "F.")
    ctl_n9 = "L.ln_at L.ln_sales L.ln_age L.cash_ratio_w L.capx_at_w L.D.ln_at L.D.ln_sales L.nn_ca nn_norec L.nn_norec own_consol L.own_consol L2.own_consol"
    ctl_m9 = f"L.ln_at L.ln_sales L.ln_age L.cash_ratio_w L.capx_at_w L.D.ln_at L.D.ln_sales L.{SETS['m'][2]} {SETS['m'][3]} L.{SETS['m'][3]} own_consol L.own_consol L2.own_consol"
    cell("59_balanced", "main", y, cgroup="zerostock", ctrl_str=fullm, extra=BAL_, absorb="sic2_year cty_year", title="balanced sample: the main specification")
    cell("59_balanced", "des_ctrl", y, cgroup="zerostock", ctrl_str=fullm + " " + L3_("na_d"), extra=BAL_, absorb="sic2_year cty_year", title="balanced sample: designated set")
    cell("59_balanced", "nam_ctrl", y, treat="sw_nn", cgroup="zerostock", ctrl_rows_override="s5_nn==0 & nn_any==0", ctrl_str=ctl_n9 + " " + L3_("nm_d"), extra=BAL_, absorb="sic2_year cty_year", title="balanced sample: naming set")
    cell("59_balanced", "mut_ctrl", y, s="m", cgroup="zerostock", ctrl_str=ctl_m9 + " " + L3_("nm_d") + " " + L3_("na_d"), extra=BAL_, absorb="sic2_year cty_year", title="balanced sample: mutual set")
# 61_attn: how long the firm has named the rival, the disjoint direction sets, relative size, product similarity
for y in ("ln_rnd", "ihs_rnd", "rnd_lat_w"):
    cell("61_attn", "pre_old", y, cgroup="zerostock", cut=" & cage_min>=3 & cage_min<.", ctrl_str=fullm, absorb="sic2_year cty_year", title="attention: every consolidating rival named 3+ years before")
    cell("61_attn", "pre_new", y, cgroup="zerostock", cut=" & cage_max<3", ctrl_str=fullm, absorb="sic2_year cty_year", title="attention: every consolidating rival named within 2 years")
    ctl_od = f"L.ln_at L.ln_sales L.ln_age L.cash_ratio_w L.capx_at_w L.D.ln_at L.D.ln_sales L.od_n od_norec L.od_norec own_consol L.own_consol L2.own_consol {L3_('na_d')} {L3_('mu_d')}"
    ctl_on = f"L.ln_at L.ln_sales L.ln_age L.cash_ratio_w L.capx_at_w L.D.ln_at L.D.ln_sales L.on_n on_norec L.on_norec own_consol L.own_consol L2.own_consol {L3_('od_c')} {L3_('mu_d')}"
    cell("61_attn", "od", y, treat="sw_od", cgroup="zerostock", ctrl_rows_override="s5_od==0 & od_d==0", ctrl_str=ctl_od, absorb="sic2_year cty_year", title="attention: rivals named one way (disjoint)")
    cell("61_attn", "on", y, treat="sw_on", cgroup="zerostock", ctrl_rows_override="s5_on==0 & on_t==0", ctrl_str=ctl_on, absorb="sic2_year cty_year", title="attention: firms naming the focal firm one way (disjoint)")
    cell("61_attn", "od_sz", y, treat="sw_od", cgroup="zerostock", cut=" & lr_od>0 & lr_od<=2.3026", ctrl_rows_override="s5_od==0 & od_d==0", ctrl_str=ctl_od, absorb="sic2_year cty_year", title="attention: one-way named, consolidator 1-10x the firm")
    cell("61_attn", "on_sz", y, treat="sw_on", cgroup="zerostock", cut=" & lr_on>0 & lr_on<=2.3026", ctrl_rows_override="s5_on==0 & on_t==0", ctrl_str=ctl_on, absorb="sic2_year cty_year", title="attention: one-way namers, consolidator 1-10x the firm")
    cell("61_attn", "mu_sz", y, s="m", cgroup="zerostock", cut=" & lr_mu>0 & lr_mu<=2.3026", ctrl_str=ctl_m9 + " " + L3_("nm_d") + " " + L3_("na_d"), absorb="sic2_year cty_year", title="attention: mutual, consolidator 1-10x the firm")
    for g_, gl in ((3, "high"), (2, "low"), (1, "unrelated"), (0, "unobservable")):
        cell("61_attn", f"sim_{gl}", y, cgroup="zerostock", cut=f" & sim_g=={g_}", ctrl_str=fullm, absorb="sic2_year cty_year", title=f"attention: product similarity {gl}")
    cell("61_attn", "od_big", y, treat="sw_od", cgroup="zerostock", cut=" & lr_od>2.3026 & lr_od<.", ctrl_rows_override="s5_od==0 & od_d==0", ctrl_str=ctl_od, absorb="sic2_year cty_year", title="attention: one-way named, consolidator more than 10x the firm")
    cell("61_attn", "on_big", y, treat="sw_on", cgroup="zerostock", cut=" & lr_on>2.3026 & lr_on<.", ctrl_rows_override="s5_on==0 & on_t==0", ctrl_str=ctl_on, absorb="sic2_year cty_year", title="attention: one-way namers, consolidator more than 10x the firm")
    cell("61_attn", "main_setgrowth", y, cgroup="zerostock", ctrl_str=fullm + " L.D.nriv_ca L2.D.nriv_ca", absorb="sic2_year cty_year", title="attention: main, the rival set's growth at t-1 and t-2 controlled")
# 61_attn: the rival set's size and its newly named rivals as outcomes
_ctl_nd = "L.ln_at L.ln_sales L.ln_age L.cash_ratio_w L.capx_at_w L.D.ln_at L.D.ln_sales own_consol L.own_consol L2.own_consol"
for y in ("lnriv", "ihs_new"):
    cell("61_attn", "naming_dyn", y, cgroup="zerostock", ctrl_str=_ctl_nd, extra=" & ln_rnd!=. & L2.ln_rnd!=.", absorb="sic2_year cty_year", title="attention: naming dynamics around the event")
# 62_more: the age of the naming tie, the firm's position in its set, whole-firm deals, the direction sets on the balanced sample
for y in ("ln_rnd", "ihs_rnd", "rnd_lat_w"):
    for row, cut in (("dur_01", " & cage_max<=1"), ("dur_24", " & cage_max>=2 & cage_max<=4"), ("dur_5p", " & cage_max>=5 & cage_max<.")):
        cell("62_more", row, y, cgroup="zerostock", cut=cut, ctrl_str=fullm, absorb="sic2_year cty_year", title=f"strength of attention: {row}")
    for t_ in (1, 2, 3):
        cell("62_more", f"npos_{t_}", y, cgroup="zerostock", cut=f" & npos=={t_}", ctrl_str=fullm, absorb="sic2_year cty_year", title=f"network position: share of named rivals larger, tercile {t_}")
        cell("62_more", f"nprom_{t_}", y, cgroup="zerostock", cut=f" & nprom=={t_}", ctrl_str=fullm, absorb="sic2_year cty_year", title=f"network position: attention received over given, tercile {t_}")
    cell("62_more", "whole_any", y, cgroup="zerostock", cut=" & has_whole==1", ctrl_str=fullm, absorb="sic2_year cty_year", title="robustness: events with a whole-firm deal")
    cell("62_more", "whole_all", y, cgroup="zerostock", cut=" & all_whole==1", ctrl_str=fullm, absorb="sic2_year cty_year", title="robustness: events with whole-firm deals only")
    BAL6 = "".join(f" & F{k}.{y}!=." for k in range(1, 6)).replace("F1.", "F.")
    cell("62_more", "bal_od", y, treat="sw_od", cgroup="zerostock", ctrl_rows_override="s5_od==0 & od_d==0", ctrl_str=ctl_od, extra=BAL6, absorb="sic2_year cty_year", title="balanced: rivals named one way")
    cell("62_more", "bal_on", y, treat="sw_on", cgroup="zerostock", ctrl_rows_override="s5_on==0 & on_t==0", ctrl_str=ctl_on, extra=BAL6, absorb="sic2_year cty_year", title="balanced: firms naming the focal firm one way")
# 62_more: rivals of rivals and industry waves as events, with no named rival's deal through t+5
FUTD = "".join(f" & F{k}.any_ca!=1" for k in range(1, 6)).replace("F1.", "F.")
for y in ("ln_rnd", "ihs_rnd"):
    for k, (tr, v) in (("rr", ("swc_rr", "rr_any")), ("wave", ("swc_wave", "wave_w"))):
        cell("62_more", f"evt5_{k}_pure", y, treat=tr, cgroup="zerostock", cut=" & any_ca==0" + FUTD, ctrl_rows_override=f"s5c_{k}==0 & {v}==0 & any_ca==0", ctrl_str=fullm, absorb="sic2_year cty_year", title=f"identification: the {k} channel as the event, no designated rival deal through t+5")
# 62_more: patents, to five and to eight years
for y in ("ihs_fam", "ihs_grt", "ihs_subcl", "new_share"):
    for w in (5, 8):
        cell("62_more", f"pat_main_h{w}", y, cgroup="zerostock", ctrl_str=fullm, absorb="sic2_year cty_year", post_=w, title=f"patents: the main specification to h = {w}")
# 66_check: comparisons clean of every direction from t-5 to t+5; tie ages on firms covered five years; supplier and partner
# events; patents under the firm's own names; longer patent windows
FUT6 = lambda v: "".join(f" & F{k}.{v}!=1" for k in range(1, 6)).replace("F1.", "F.")
CLEANC = "s5_d==0 & any_ca==0 & s5_on==0 & on_t==0" + FUT6("any_ca") + FUT6("on_t")
for y in ("ln_rnd", "ihs_rnd", "rnd_lat_w"):
    ctl_od6 = f"L.ln_at L.ln_sales L.ln_age L.cash_ratio_w L.capx_at_w L.D.ln_at L.D.ln_sales L.od_n od_norec L.od_norec own_consol L.own_consol L2.own_consol {L3_('na_d')} {L3_('mu_d')}"
    ctl_on6 = f"L.ln_at L.ln_sales L.ln_age L.cash_ratio_w L.capx_at_w L.D.ln_at L.D.ln_sales L.on_n on_norec L.on_norec own_consol L.own_consol L2.own_consol {L3_('od_c')} {L3_('mu_d')}"
    ctl_mu6 = f"L.ln_at L.ln_sales L.ln_age L.cash_ratio_w L.capx_at_w L.D.ln_at L.D.ln_sales L.{SETS['m'][2]} {SETS['m'][3]} L.{SETS['m'][3]} own_consol L.own_consol L2.own_consol {L3_('od_c')} {L3_('na_d')}"
    pure_od = " & on_t!=1 & mu_d!=1" + FUT6("on_t") + FUT6("mu_d"); pure_on = " & any_ca!=1" + FUT6("any_ca"); pure_mu = " & od_d!=1 & on_t!=1" + FUT6("od_d") + FUT6("on_t")
    for nm, tr, cut, ctl in (("od", "sw_od", pure_od, ctl_od6), ("on", "sw_on", pure_on, ctl_on6), ("mu", "sw_m", pure_mu, ctl_mu6)):
        cell("66_check", f"{nm}_cc", y, treat=tr, cgroup="zerostock", cut=cut, ctrl_rows_override=CLEANC, ctrl_str=ctl, absorb="sic2_year cty_year", title=f"audit: {nm}, events pure and comparisons clean of every direction t-5..t+5")
        if nm != "mu":
            v = "lr_od" if nm == "od" else "lr_on"
            cell("66_check", f"{nm}_big_cc", y, treat=tr, cgroup="zerostock", cut=cut + f" & {v}>2.3026 & {v}<.", ctrl_rows_override=CLEANC, ctrl_str=ctl, absorb="sic2_year cty_year", title=f"audit: {nm}, consolidator >10x, clean")
    cell("66_check", "mu_ctrl2", y, s="m", cgroup="zerostock", ctrl_str=ctl_mu6, absorb="sic2_year cty_year", title="audit: mutual set, controls od_c and na_d")
    for row, cut in (("dur_01", " & cage_max<=1"), ("dur_24", " & cage_max>=2 & cage_max<=4"), ("dur_5p", " & cage_max>=5 & cage_max<.")):
        cell("66_check", f"{row}_cov5", y, cgroup="zerostock", cut=cut, ctrl_str=fullm, extra=" & cov_before>=5", absorb="sic2_year cty_year", title=f"audit: strength of attention, {row}, five covered years before t")
for y in ("ln_rnd", "ihs_rnd"):
    for k, (tr, v) in (("sup", ("swc_sup", "sup_any")), ("par", ("swc_par", "par_any"))):
        cell("66_check", f"evt5_{k}_cc", y, treat=tr, cgroup="zerostock", cut=" & any_ca==0" + FUT6("any_ca"), ctrl_rows_override=f"s5c_{k}==0 & {v}==0 & s5_d==0 & any_ca==0" + FUT6("any_ca"), ctrl_str=fullm, absorb="sic2_year cty_year", title=f"audit: the {k} channel as the event, pure events, clean comparisons")
for w in (5, 8):
    cell("66_check", f"famse_h{w}", "ihs_famse", cgroup="zerostock", ctrl_str=fullm, extra=" & pat_ok==1", absorb="sic2_year cty_year", post_=w, title=f"audit: patent families under the firm's own names, to h = {w}")
for y in ("ihs_fam", "ihs_grt", "ihs_subcl"):
    for w in (6, 7):
        cell("66_check", f"pat_h{w}", y, cgroup="zerostock", ctrl_str=fullm, absorb="sic2_year cty_year", post_=w, title=f"audit: patents to h = {w}")
# 67_cohort7: the firm-years with patent families observed to t+7
BAL7 = "".join(f" & F{k}.ihs_fam!=." for k in range(1, 8)).replace("F1.", "F.")
for y in ("ihs_fam", "ihs_grt", "ihs_subcl", "ln_rnd", "rnd_lat_w", "fam_lrd_w"):
    cell("67_cohort7", "bal7", y, cgroup="zerostock", ctrl_str=fullm, extra=" & pat_ok==1" + BAL7, absorb="sic2_year cty_year", post_=7, title="the seven-year cohort: patent families observed t+1..t+7")
    NOSW = "".join(f" & F{k}.sw_d!=1" for k in range(1, 8)).replace("F1.", "F."); NODEAL = "".join(f" & F{k}.any_ca!=1" for k in range(1, 8)).replace("F1.", "F.")
    cell("67_cohort7", "fixed7", y, cgroup="zerostock", cut=NOSW, ctrl_rows_override="s5_d==0 & any_ca==0" + NODEAL, ctrl_str=fullm, extra=" & pat_ok==1" + BAL7, absorb="sic2_year cty_year", post_=7, title="the seven-year cohort, fixed sample")
# 72_het2: the events by innovation assets, patent stock and prior exposure
for y in ("ln_rnd", "ihs_rnd", "rnd_lat_w"):
    for k in ("intanat", "rdsa"):
        for t_ in (1, 2, 3):
            cell("72_het2", f"{k}_{t_}", y, cgroup="zerostock", cut=f" & ter_{k}=={t_}", ctrl_str=fullm, absorb="sic2_year cty_year", title=f"heterogeneity: {k} tercile {t_}")
    for b_, bl in ((1, "none"), (2, "low"), (3, "high")):
        cell("72_het2", f"patst_{bl}", y, cgroup="zerostock", cut=f" & pst_bin=={b_}", ctrl_str=fullm, absorb="sic2_year cty_year", title=f"heterogeneity: five-year patent stock {bl}")
        cell("72_het2", f"pex_{bl}", y, cgroup="zerostock", cut=f" & pex=={b_}", ctrl_str=fullm, absorb="sic2_year cty_year", title=f"heterogeneity: prior exposure stock {bl}, full five-year history")
    for b_, bl in ((1, "t2"), (2, "t34"), (3, "t5p")):
        cell("72_het2", f"prec_{bl}", y, cgroup="zerostock", cut=f" & prec=={b_}", ctrl_str=fullm, absorb="sic2_year cty_year", title=f"heterogeneity: years since the last rival deal {bl}, full five-year history")
    cell("72_het2", "full5", y, cgroup="zerostock", cut=" & full5==1", ctrl_str=fullm, absorb="sic2_year cty_year", title="heterogeneity: every event with a full five-year history")
# 73_ind: four-digit industry peers as the rival set
FUT5 = lambda v: "".join(f" & F{k}.{v}!=1" for k in range(1, 6)).replace("F1.", "F.")
for y in ("ln_rnd", "ihs_rnd", "rnd_lat_w"):
    cell("73_ind", "ind_all", y, treat="sw_ind", cgroup="zerostock", ctrl_rows_override="s5_ind==0 & ind_any==0", ctrl_str=fullm, absorb="sic2_year cty_year", title="industry peers as the rival set")
    cell("73_ind", "ind_all_cl4", y, treat="sw_ind", cgroup="zerostock", ctrl_rows_override="s5_ind==0 & ind_any==0", ctrl_str=fullm, absorb="sic2_year cty_year", cluster="csic4", title="industry peers as the rival set, clustered by industry")
    cell("73_ind", "ind_named", y, treat="sw_ind", cgroup="zerostock", cut=" & ind_named_n>0", ctrl_rows_override="s5_ind==0 & ind_any==0", ctrl_str=fullm, absorb="sic2_year cty_year", title="industry onsets with a named rival among the dealing peers")
    cell("73_ind", "ind_unnamed_pure", y, treat="sw_ind", cgroup="zerostock", cut=" & ind_named_n==0 & any_ca!=1" + FUT5("any_ca"), ctrl_rows_override="s5_ind==0 & ind_any==0 & s5_d==0 & any_ca==0" + FUT5("any_ca"), ctrl_str=fullm, absorb="sic2_year cty_year", title="industry onsets with no named rival among the dealing peers, pure")
# 74_sellout: without the firm-years of firms acquired in t+1 to t+5
NOTGT = "".join(f" & F{k}.own_tgt!=1" for k in range(1, 6)).replace("F1.", "F.")
for y in ("ln_rnd", "ihs_rnd", "rnd_lat_w"):
    cell("74_sellout", "not_acquired", y, cgroup="zerostock", ctrl_str=fullm, extra=NOTGT, absorb="sic2_year cty_year", title="main specification, firms not acquired in t+1..t+5")
# 75_breadth: CPC subclasses per patent family
for y in ("subcl_pf", "subcl_net"):
    for w in (5, 6, 7, 8):
        cell("75_breadth", f"h{w}", y, cgroup="zerostock", ctrl_str=fullm, absorb="sic2_year cty_year", post_=w, title=f"breadth net of volume, to h = {w}")
    cell("75_breadth", "bal7", y, cgroup="zerostock", ctrl_str=fullm, extra=" & pat_ok==1" + BAL7, absorb="sic2_year cty_year", post_=7, title="breadth net of volume, the firm-years observed seven years")
    cell("75_breadth", "fixed7", y, cgroup="zerostock", cut=NOSW, ctrl_rows_override="s5_d==0 & any_ca==0" + NODEAL, ctrl_str=fullm, extra=" & pat_ok==1" + BAL7, absorb="sic2_year cty_year", post_=7, title="breadth net of volume, the fixed cohort")
# 50_cluster: other clusterings of the main cell
for y in ("ln_rnd", "ihs_rnd"):
    for cl in ("sic2", "sic4", "comm_id"):
        cell("50_cluster", f"cl_{cl}", y, cgroup="zerostock", ctrl_str=fullm, absorb="sic2_year cty_year", cluster=cl, title=f"inference: the main cell clustered by {cl}")
# 69_prepool: the main cells with the pre-period pooled over h = -4 and -3
PREPOOL = [("42_paper", f"c{k}_zerostock", y) for k in range(5) for y in ("ln_rnd", "ihs_rnd")] + [("58_rdint", "main", "rnd_lat_w")] + [("62_more", "pat_main_h5", y) for y in ("ihs_fam", "ihs_grt", "ihs_subcl")]
for lad, row, y in PREPOOL:
    src = [r for r in rows if r["ladder"] == lad and r["row"] == row and r["outcome"] == y]
    assert len(src) == 1, (lad, row, y)
    r = dict(src[0]); r["options"] = r["options"] + " pre_pooled(3 4)"; r["ladder"] = "69_prepool"; r["row"] = f"{lad}__{row}"
    r["run_id"] = hashlib.sha1("|".join([r["y"], r["treat"], r["sample"], r["options"], r["controls"], r["absorb"], r["cluster"], str(r["pre"]), str(r["post"])]).encode()).hexdigest()[:10]
    r["title"] = r["title"].replace(f"{lad} / {row}", f"69_prepool / {lad}__{row}") + "; pooled pre-period over h = -4 and -3"
    rows.append(r)
grid = pd.DataFrame(rows)
ROOT.mkdir(parents=True, exist_ok=True); (ROOT / "runs").mkdir(exist_ok=True); (ROOT / "valid").mkdir(exist_ok=True)
grid.to_csv(ROOT / "grid_ladders.csv", index=False)
print(f"grid ladders: {len(grid)} cells, {grid['run_id'].nunique()} runs; by outcome {grid.groupby('outcome').size().to_dict()}; by ladder {grid.groupby('ladder').size().to_dict()}")
