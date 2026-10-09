import numpy as np
import pandas as pd

from projlib import config

E = "FACTSET_ENTITY_ID"
rp = pd.read_parquet(config.out("07_panel", "rival_pair_traits.parquet"), columns=["focal", "rival", "year", "first_year", "rivalry_age", "mutual"]).drop_duplicates(["focal", "rival", "year"])
ev = pd.read_parquet(config.out("07_panel", "exposure_events.parquet"), columns=["focal", "related", "kind", "cls", "year"])
ev = ev[ev["kind"].eq("rival") & ev["cls"].str.startswith("consolidation")].drop_duplicates(["focal", "related", "year"]).rename(columns={"related": "rival"})
ev = ev.merge(rp[["focal", "rival", "year", "rivalry_age", "mutual"]], on=["focal", "rival", "year"], how="left")
rv = pd.read_parquet(config.out("07_panel", "reverse_events.parquet"), columns=["focal", "namer", "cls", "year", "only"])
rv = rv[rv["cls"].astype(str).str.startswith("consolidation") & rv["only"].eq(True)].drop_duplicates(["focal", "namer", "year"])

u = pd.read_parquet(config.out("03_financials", "factset_universe_usd.parquet"), columns=[E, "fyear", "sale_usd"]).drop_duplicates([E, "fyear"])
p = pd.read_parquet(config.out("03_financials", "panel_entity_year.parquet"), columns=[E, "fyear", "sale_usd"]).drop_duplicates([E, "fyear"])
su = u.set_index([E, "fyear"])["sale_usd"].where(lambda s: s > 0); sp = p.set_index([E, "fyear"])["sale_usd"].where(lambda s: s > 0)


def sales(ids, years):
    k = list(zip(ids, years - 1))
    a = su.reindex(k).values; b = sp.reindex(k).values
    return np.where(np.isnan(a), b, a)


ev["lr"] = np.log(sales(ev["rival"], ev["year"]) / sales(ev["focal"], ev["year"]))
rv["lr"] = np.log(sales(rv["namer"], rv["year"]) / sales(rv["focal"], rv["year"]))
g_age = ev.groupby(["focal", "year"]).agg(age_min=("rivalry_age", "min"), age_max=("rivalry_age", "max"))
g_od = ev[ev["mutual"].ne(True)].groupby(["focal", "year"])["lr"].max().rename("lr_od")
g_mu = ev[ev["mutual"].eq(True)].groupby(["focal", "year"])["lr"].max().rename("lr_mu")
g_on = rv.groupby(["focal", "year"])["lr"].max().rename("lr_on")
hp = pd.read_parquet(config.out("09_appendix/hp_validation", "hp_named_pairs_etnic3.parquet"), columns=["focal", "rival", "year", "score"])
e2 = ev.merge(hp, on=["focal", "rival", "year"], how="inner")
e2["sim"] = e2["score"].fillna(0.0)
g_sim = e2.groupby(["focal", "year"]).agg(sim_max=("sim", "max")).assign(sim_obs=1)
g_new = rp[rp["first_year"].eq(rp["year"])].groupby(["focal", "year"]).size().rename("n_new")
out = pd.concat([g_age, g_od, g_mu, g_on, g_sim, g_new], axis=1).reset_index().rename(columns={"year": "fyear"})
out.to_parquet(config.out("08_estimation", "attention_inputs.parquet"), index=False)
print(f"firm-years: {len(out):,}")
print("rivalry age of consolidating rivals at the deal (pair level):", ev["rivalry_age"].describe(percentiles=[0.25, 0.5, 0.75]).round(2).to_dict(),
      "| share named 3+ years before:", round(float(ev["rivalry_age"].ge(3).mean()), 3))
print("largest log size ratio, quantiles: od", out["lr_od"].quantile([0.1, 0.5, 0.9]).round(2).to_dict(), "| on", out["lr_on"].quantile([0.1, 0.5, 0.9]).round(2).to_dict(),
      "| mu", out["lr_mu"].quantile([0.1, 0.5, 0.9]).round(2).to_dict())
print("ETNIC-observable firm-years:", int(out["sim_obs"].sum()), "| similarity > 0:", int(out["sim_max"].gt(0).sum()))
