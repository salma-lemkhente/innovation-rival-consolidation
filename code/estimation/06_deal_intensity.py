import numpy as np
import pandas as pd

from projlib import config

E = "FACTSET_ENTITY_ID"
ev = pd.read_parquet(config.out("07_panel", "exposure_events.parquet"), columns=["focal", "related", "kind", "cls", "year", "MASTER_DEAL_NO", "role", "deal_value"])
ev = ev[ev["kind"].eq("rival") & ev["cls"].str.startswith("consolidation") & ev["role"].eq("acquiror")].drop_duplicates(["focal", "related", "year", "MASTER_DEAL_NO"])
u = pd.read_parquet(config.out("03_financials", "factset_universe_usd.parquet"), columns=[E, "fyear", "at_usd"]).drop_duplicates([E, "fyear"])
p = pd.read_parquet(config.out("03_financials", "panel_entity_year.parquet"), columns=[E, "fyear", "at_usd"]).drop_duplicates([E, "fyear"])
at = u.set_index([E, "fyear"])["at_usd"].where(lambda s: s > 0)
at_p = p.set_index([E, "fyear"])["at_usd"].where(lambda s: s > 0)
key = list(zip(ev["related"], ev["year"] - 1))
ev["rival_at"] = at.reindex(key).values
ev["rival_at"] = ev["rival_at"].where(ev["rival_at"].notna(), at_p.reindex(key).values)
ev["step"] = np.log1p(pd.to_numeric(ev["deal_value"], errors="coerce") / ev["rival_at"])
out = ev.groupby(["focal", "year"]).agg(step_max=("step", "max"), n_valued=("step", "count"), n_deals=("MASTER_DEAL_NO", "nunique")).reset_index().rename(columns={"year": "fyear"})
out.to_parquet(config.out("08_estimation", "intensity_focal_fyear.parquet"), index=False)
print(f"focal-years with a rival deal: {len(out):,}; with a valued deal: {out['step_max'].notna().mean():.1%}")
print("largest step per focal-year, quantiles:", out["step_max"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3).to_dict())
