import hashlib
import sys

import pandas as pd

from projlib import config

N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 500
lad = pd.read_csv(config.out("08_estimation/clean_runs_ladders", "grid_ladders.csv"), dtype=str)
out = config.out("08_estimation/clean_runs_placebo", "grid_placebo.csv")
rows = []
for y in ("ln_rnd", "ihs_rnd"):
    t = lad[lad["ladder"].eq("42_paper") & lad["row"].eq("c4_zerostock") & lad["outcome"].eq(y)].iloc[0]
    assert t["treat"] == "sw_d" and "((sw_d==1) | (s5_d==0 & any_ca==0))" in t["sample"], t["sample"]
    for d in range(1, N + 1):
        r = t.copy()
        quiet = f"q{d}==0 & L.q{d}!=1 & L2.q{d}!=1 & L3.q{d}!=1 & L4.q{d}!=1 & L5.q{d}!=1"
        r["treat"] = f"psw_q{d}"
        r["sample"] = t["sample"].replace("((sw_d==1) | (s5_d==0 & any_ca==0))", f"((psw_q{d}==1) | ({quiet}))")
        key = "|".join([r["y"], r["treat"], r["sample"], r["options"], r["controls"], r["absorb"], r["cluster"], r["pre"], r["post"]])
        r["run_id"] = hashlib.sha1(key.encode()).hexdigest()[:10]
        r["ladder"] = "46_placebo_paper"; r["row"] = f"q{d}_paper"; r["draw"] = str(d); r["kind"] = "paper"
        r["title"] = f"46_placebo_paper / q{d} / {y}: the main specification with an industry-matched random rival set, draw {d}"
        rows.append(r)
grid = pd.DataFrame(rows)
grid.to_csv(out, index=False)
print(f"grid_placebo: {len(grid):,} cells, of which 46_placebo_paper {len(rows):,}")
