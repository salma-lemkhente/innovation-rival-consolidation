import numpy as np
import pandas as pd

from projlib import config

E = "FACTSET_ENTITY_ID"
u = pd.read_parquet(config.out("03_financials", "factset_universe_usd.parquet"), columns=[E, "fyear", "sale_usd", "emp"])
p = pd.read_parquet(config.out("03_financials", "panel_entity_year.parquet"), columns=[E, "fyear", "sale_usd", "emp"]).drop_duplicates([E, "fyear"])
x = u.merge(p, on=[E, "fyear"], how="outer", suffixes=("", "_p"))
x["sale"] = x["sale_usd"].where(x["sale_usd"] > 0, x["sale_usd_p"])
x["empl"] = x["emp"].where(x["emp"] > 0, x["emp_p"])
x = x[(x["sale"] > 0) & (x["empl"] > 0)].copy()
x["lp"] = np.log(x["sale"] * 1e6 / x["empl"])
cov = pd.read_stata(config.raw("entity_coverage"), convert_categoricals=False, columns=[E, "PRIMARY_SIC_CODE"]).drop_duplicates(E)
sic = pd.to_numeric(cov.set_index(E)["PRIMARY_SIC_CODE"], errors="coerce")
x["sic4"] = x[E].map(sic); x = x.dropna(subset=["sic4"])
x["sic3"] = x["sic4"] // 10; x["sic2"] = x["sic4"] // 100
q = x.groupby("fyear")["lp"].transform(lambda s: s.quantile(0.005)), x.groupby("fyear")["lp"].transform(lambda s: s.quantile(0.995))
x = x[(x["lp"] >= q[0]) & (x["lp"] <= q[1])]
fr = {}
for lev in ("sic4", "sic3", "sic2"):
    g = x.groupby([lev, "fyear"])["lp"]
    fr[lev] = pd.DataFrame({"n": g.size(), "p90": g.quantile(0.9), "mx": g.max()})
out = x[[E, "fyear", "lp", "sic4", "sic3", "sic2"]].copy()
out["level"] = None; out["front_p90"] = np.nan; out["front_max"] = np.nan
for lev in ("sic4", "sic3", "sic2"):
    f = fr[lev].reindex(list(zip(out[lev], out["fyear"])))
    take = out["level"].isna().values & (f["n"].values >= 10)
    out.loc[take, "front_p90"] = f["p90"].values[take]; out.loc[take, "front_max"] = f["mx"].values[take]; out.loc[take, "level"] = lev
out["dist_p90"] = (out["front_p90"] - out["lp"]).clip(lower=0)
out["dist_max"] = (out["front_max"] - out["lp"]).clip(lower=0)
focal = set(pd.read_parquet(config.out("01_universe", "entities.parquet")).query("is_focal")[E])
res = out[out[E].isin(focal)].rename(columns={E: "focal"})[["focal", "fyear", "lp", "front_p90", "front_max", "dist_p90", "dist_max", "level"]]
res.to_parquet(config.out("08_estimation", "frontier_focal_fyear.parquet"), index=False)
print(f"population firm-years with productivity: {len(x):,} ({x[E].nunique():,} firms); focal firm-years with a distance: {res['dist_p90'].notna().sum():,}")
print("frontier level used:", res["level"].value_counts(normalize=True).round(3).to_dict())
print(res[["dist_p90", "dist_max"]].describe().round(2).to_string())
