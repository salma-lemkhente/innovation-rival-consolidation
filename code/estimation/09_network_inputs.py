import numpy as np
import pandas as pd

from projlib import config

E = "FACTSET_ENTITY_ID"
u = pd.read_parquet(config.out("03_financials", "factset_universe_usd.parquet"), columns=[E, "fyear", "sale_usd"]).drop_duplicates([E, "fyear"])
p = pd.read_parquet(config.out("03_financials", "panel_entity_year.parquet"), columns=[E, "fyear", "sale_usd"]).drop_duplicates([E, "fyear"])
su = u.set_index([E, "fyear"])["sale_usd"].where(lambda s: s > 0); sp = p.set_index([E, "fyear"])["sale_usd"].where(lambda s: s > 0)


def sales(ids, years):
    k = list(zip(ids, years))
    a = su.reindex(k).values; b = sp.reindex(k).values
    return np.where(np.isnan(a), b, a)


rp = pd.read_parquet(config.out("07_panel", "rival_pair_traits.parquet"), columns=["focal", "rival", "year"]).drop_duplicates()
rp["sr"] = sales(rp["rival"], rp["year"]); rp["sf"] = sales(rp["focal"], rp["year"])
rp = rp.dropna(subset=["sr", "sf"])
pos = rp.assign(larger=rp["sr"] > rp["sf"]).groupby(["focal", "year"]).agg(sh_larger=("larger", "mean"), n_sized=("larger", "size"))

ev = pd.read_parquet(config.out("07_panel", "exposure_events.parquet"), columns=["focal", "related", "kind", "cls", "year", "MASTER_DEAL_NO"])
ev = ev[ev["kind"].eq("rival") & ev["cls"].str.startswith("consolidation")]
dl = pd.read_parquet(config.out("06_sdc", "deals.parquet"), columns=["MASTER_DEAL_NO", "asset_deal", "target_division"]).drop_duplicates("MASTER_DEAL_NO")
ev = ev.merge(dl, on="MASTER_DEAL_NO", how="left")
ev["whole"] = ~(ev["asset_deal"].fillna(False).astype(bool) | ev["target_division"].fillna(False).astype(bool))
wh = ev.groupby(["focal", "year"])["whole"].agg(has_whole="max", all_whole="min")
out = pd.concat([pos, wh], axis=1).reset_index().rename(columns={"year": "fyear"})
out[["has_whole", "all_whole"]] = out[["has_whole", "all_whole"]].astype("float64")
out.to_parquet(config.out("08_estimation", "network_inputs.parquet"), index=False)
print(f"firm-years: {len(out):,}; with a position: {out['sh_larger'].notna().sum():,}, median share of larger rivals {out['sh_larger'].median():.2f}")
print(f"deal rows that are whole-firm deals: {ev['whole'].mean():.1%}; firm-years with a deal: at least one whole-firm deal {out['has_whole'].mean():.1%}, only whole-firm deals {out['all_whole'].mean():.1%}")
