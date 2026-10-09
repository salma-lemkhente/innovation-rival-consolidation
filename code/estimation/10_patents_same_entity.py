import glob

import pandas as pd

from projlib import config

E = "FACTSET_ENTITY_ID"
parts = sorted(glob.glob(str(config.out("05_patents/attached", "x").parent / "part_*.parquet")))
a = pd.concat((pd.read_parquet(p, columns=[E, "family_id", "priority_date", "relation"]) for p in parts), ignore_index=True)
print(f"attached publications {len(a):,}; by relation {a['relation'].value_counts().to_dict()}")
a = a[a["relation"].eq("same_entity")].drop_duplicates([E, "family_id"])
a["priority_year"] = a["priority_date"] // 10000
acq = pd.read_parquet(config.out("05_patents", "acquired.parquet"), columns=[E, "family_id", "acquired"])
a = a.merge(acq[acq["acquired"].eq(True)][[E, "family_id"]].assign(acq=1), on=[E, "family_id"], how="left")
a = a[a["acq"].isna()]
out = a.groupby([E, "priority_year"]).size().rename("fam_se").reset_index().rename(columns={E: "focal", "priority_year": "fyear"})
out.to_parquet(config.out("08_estimation", "patents_same_entity.parquet"), index=False)
tot = pd.read_parquet(config.out("05_patents", "outcomes_entity_year.parquet"), columns=[E, "year", "families_originated"]).rename(columns={E: "focal", "year": "fyear"})
m = tot.merge(out, on=["focal", "fyear"], how="left").fillna({"fam_se": 0})
print(f"firm-years {len(out):,}; same-entity families as a share of all originated families: {m['fam_se'].sum() / m['families_originated'].sum():.2f}")
