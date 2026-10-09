import sys
import time

import numpy as np
import pandas as pd

from projlib import config, io

N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 500
E = "FACTSET_ENTITY_ID"
t0 = time.time()

cov = pd.read_stata(config.raw("entity_coverage"), convert_categoricals=False, columns=[E, "PRIMARY_SIC_CODE"])
cov["sic4"] = pd.to_numeric(cov["PRIMARY_SIC_CODE"], errors="coerce")
sic = cov.dropna(subset=["sic4"]).drop_duplicates(E).set_index(E)["sic4"].astype(int)
LEVELS = {"sic4": lambda s: s, "sic3": lambda s: s // 10, "sic2": lambda s: s // 100}

rp = pd.read_parquet(config.out("07_panel", "rival_pair_year.parquet"), columns=["focal", "rival", "year"]).drop_duplicates()
rp["rsic4"] = rp["rival"].map(sic)
sizes = rp.groupby(["focal", "year"])["rival"].nunique().rename("n").reset_index()
sizes["fsic4"] = sizes["focal"].map(sic)
for lv, f in LEVELS.items():
    sizes[f"f{lv}"] = f(sizes["fsic4"])
    rp[f"r{lv}"] = f(rp["rsic4"])
    same = rp[rp[f"r{lv}"].eq(rp["focal"].map(sic).pipe(f))].groupby(["focal", "year"])["rival"].nunique().rename(f"own_{lv}")
    sizes = sizes.merge(same.reset_index(), on=["focal", "year"], how="left"); sizes[f"own_{lv}"] = sizes[f"own_{lv}"].fillna(0).astype(int)
pool_all = rp.groupby("year")["rival"].agg(lambda s: np.array(sorted(set(s)))).to_dict()
pools = {lv: rp.dropna(subset=[f"r{lv}"]).astype({f"r{lv}": int}).groupby(["year", f"r{lv}"])["rival"].agg(lambda s: np.array(sorted(set(s)))).to_dict() for lv in LEVELS}
sizes["level"] = "all"
for lv in ("sic2", "sic3", "sic4"):
    key = list(zip(sizes["year"], sizes[f"f{lv}"]))
    psize = np.array([len(pools[lv].get((y, int(s)), ())) if pd.notna(s) else 0 for y, s in key])
    ok = (psize - sizes[f"own_{lv}"].to_numpy() - 1) >= 2 * sizes["n"].to_numpy() + 5
    sizes.loc[ok, "level"] = lv
print(f"firm-years with rivals {len(sizes):,}; median set {sizes['n'].median():.0f}; level: {sizes['level'].value_counts(normalize=True).round(3).to_dict()}  [{time.time() - t0:.0f}s]")

ev = pd.read_parquet(config.out("07_panel", "events.parquet"), columns=[E, "MASTER_DEAL_NO", "cls", "date_announced"])
cons = ev[ev["cls"].isin(["consolidation_acquired", "consolidation_acquires"])].rename(columns={E: "other"})[["other", "date_announced"]].dropna().drop_duplicates()
cons["year"] = cons["date_announced"].dt.year
per = pd.read_parquet(config.out("07_panel", "panel_focal_fyear.parquet"), columns=["focal", "fyear", "period_start", "period_end"])
per = per.assign(y0=per["period_start"].dt.year, y1=per["period_end"].dt.year)
per_year = pd.concat([per[per["y0"] + k <= per["y1"]].assign(year=lambda d, k=k: d["y0"] + k) for k in range(3)], ignore_index=True)[["focal", "year", "fyear", "period_start", "period_end"]]
real = rp[["focal", "rival", "year"]].rename(columns={"rival": "other"}).assign(_real=True)


GROUPS = []
for (y, lv), g in sizes.groupby(["year", "level"]):
    for s, gg in ([(None, g)] if lv == "all" else g.groupby(f"f{lv}")):
        cand = pool_all[y] if lv == "all" else pools[lv][(y, int(s))]
        n = gg["n"].to_numpy(); m = 2 * n + 3
        GROUPS.append((y, cand, np.repeat(gg["focal"].to_numpy(), m), np.repeat(n, m), int(m.sum())))


def draw(d: int) -> pd.Series:
    rng = np.random.default_rng(1000 + d)
    parts = [pd.DataFrame({"focal": foc, "other": cand[rng.integers(0, len(cand), tot)], "year": y, "n": nn}) for y, cand, foc, nn, tot in GROUPS]
    r = pd.concat(parts, ignore_index=True)
    r = r[r["focal"] != r["other"]].drop_duplicates(["focal", "other", "year"])
    r = r.merge(real, on=["focal", "other", "year"], how="left"); r = r[r["_real"].isna()].drop(columns="_real")
    r = r[r.groupby(["focal", "year"]).cumcount() < r["n"]]
    a = r.merge(cons, on=["other", "year"]).merge(per_year, on=["focal", "year"])
    a = a[(a["date_announced"] >= a["period_start"]) & (a["date_announced"] <= a["period_end"])]
    return a.groupby(["focal", "fyear"]).size().gt(0).astype("int8")


p = pd.read_stata(config.out("08_estimation", "estimation_panel_ladders.dta"), convert_categoricals=False)
p = p.sort_values(["id", "fyear"]).reset_index(drop=True)
g = p.groupby("id")
idx = pd.MultiIndex.from_frame(p[["focal", "fyear"]])
unk = p["covered"].eq(0)
draws = p[["focal", "fyear"]].copy()
new = {}
for d in range(1, N + 1):
    q = draw(d).reindex(idx).fillna(0).astype("float32").to_numpy()
    q = pd.Series(q, index=p.index).where(~unk)
    lag = q.groupby(p["id"]).shift(1)
    sw = (q.eq(1) & lag.eq(0) & p["consec"].astype(bool)).astype("int8")
    new[f"q{d}"] = q.astype("float32"); new[f"psw_q{d}"] = sw
    if d % 25 == 0 or d == 1:
        print(f"  draw {d}: shocked rows {int(q.eq(1).sum()):,}; switches {int(sw.sum()):,}  [{time.time() - t0:.0f}s]", flush=True)
new = pd.DataFrame(new, index=p.index)
draws = pd.concat([draws, new[[f"q{d}" for d in range(1, N + 1)]]], axis=1)
io.write(draws.merge(sizes[["focal", "year", "level"]].rename(columns={"year": "fyear"}), on=["focal", "fyear"], how="left"), config.out("08_estimation", "placebo_industry_draws.parquet"), key=["focal", "fyear"])
p = pd.concat([p, new], axis=1)
assert all(len(c) <= 32 for c in p.columns)
p.to_stata(config.out("08_estimation", "estimation_panel_placebo.dta"), write_index=False, version=118)
print(f"panel {len(p):,} rows, {p.shape[1]} columns written  [{time.time() - t0:.0f}s]")
