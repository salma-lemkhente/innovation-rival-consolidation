import numpy as np
import pandas as pd

from projlib import config

OUT = config.out("08_estimation/attention", "x").parent
OUT.mkdir(parents=True, exist_ok=True)
Y0, Y1 = config.YEAR_MIN, config.YEAR_MAX
MEAS = {"sale_usd": "sales", "at_usd": "total assets", "emp": "employees", "rd_net_usd": "R&D"}
BINS = np.linspace(-8, 8, 65)

fin = pd.read_parquet(config.out("03_financials", "panel_entity_year.parquet"), columns=["FACTSET_ENTITY_ID", "fyear", *MEAS]).drop_duplicates(["FACTSET_ENTITY_ID", "fyear"]).set_index(["FACTSET_ENTITY_ID", "fyear"])
rp = pd.read_parquet(config.out("07_panel", "rival_pair_traits.parquet"), columns=["focal", "rival", "year", "mutual"])
rp = rp[rp["year"].between(Y0, Y1)].drop_duplicates(["focal", "rival", "year"])


def ratios(df, a, b, yr):
    out = {}
    for v in MEAS:
        s = fin[v].where(fin[v] > 0)
        out[v] = np.log(s.reindex(list(zip(df[b], df[yr]))).values / s.reindex(list(zip(df[a], df[yr]))).values)
    return out


def summ(x):
    x = x[~np.isnan(x)]
    return {"pairs": len(x), "rival larger": np.mean(x > 0), "rival more than twice": np.mean(x > np.log(2)), "rival less than half": np.mean(x < -np.log(2)),
            "within a factor of two": np.mean(np.abs(x) <= np.log(2)), "median log ratio": np.median(x), "median absolute log ratio": np.median(np.abs(x))}


R = ratios(rp, "focal", "rival", "year")
ow = rp["mutual"].ne(1).values
rows, hist = [], []
for v, lab in MEAS.items():
    for d, x in (("one-way", R[v][ow]), ("reverse", -R[v][ow]), ("mutual", R[v][~ow]), ("designated", R[v])):
        rows.append({"measure": lab, "direction": d, **summ(x)})
        h, _ = np.histogram(x[~np.isnan(x)], bins=BINS)
        hist += [{"measure": lab, "direction": d, "bin_mid": (BINS[i] + BINS[i + 1]) / 2, "share": h[i] / max(h.sum(), 1)} for i in range(len(h))]
pd.DataFrame(rows).to_csv(OUT / "attention_pairs.csv", index=False)
pd.DataFrame(hist).to_csv(OUT / "attention_hist.csv", index=False)

ev = pd.read_parquet(config.out("07_panel", "exposure_events.parquet"), columns=["focal", "related", "kind", "cls", "year"])
ev = ev[ev["cls"].str.startswith("consolidation")].drop_duplicates(["focal", "related", "year", "kind"])
d = pd.read_stata(config.out("08_estimation", "estimation_panel_ladders.dta"), columns=["focal", "fyear", "sw_u", "ln_rnd", "in_window", "own_accounts", "covered"])
sw = d[(d["sw_u"] == 1) & d["ln_rnd"].notna() & (d["in_window"] == 1) & (d["own_accounts"] == 1) & (d["covered"] == 1)][["focal", "fyear"]].rename(columns={"fyear": "year"})
riv = ev[ev["kind"].eq("rival")].merge(sw, on=["focal", "year"]).merge(rp[["focal", "rival", "year", "mutual"]].rename(columns={"rival": "related"}), on=["focal", "related", "year"], how="left")
riv["direction"] = np.where(riv["mutual"].eq(1), "mutual", "one-way")
rv = pd.read_parquet(config.out("07_panel", "reverse_events.parquet"))
erows = []
Re = ratios(riv, "focal", "related", "year")
for v, lab in MEAS.items():
    for dname in ("one-way", "mutual"):
        erows.append({"measure": lab, "direction": dname, **summ(Re[v][riv["direction"].eq(dname).values])})
rr = rv[rv["cls"].astype(str).str.startswith("consolidation") & rv["only"].eq(True)].rename(columns={"namer": "related"}).drop_duplicates(["focal", "related", "year"]).merge(sw, on=["focal", "year"])
Rr = ratios(rr, "focal", "related", "year")
for v, lab in MEAS.items():
    erows.append({"measure": lab, "direction": "reverse", **summ(Rr[v])})
    erows.append({"measure": lab, "direction": "designated", **summ(Re[v])})
    erows.append({"measure": lab, "direction": "undirected", **summ(np.concatenate([Re[v], Rr[v]]))})
pd.DataFrame(erows).to_csv(OUT / "attention_events.csv", index=False)
print(pd.DataFrame(rows).round(2).to_string(index=False))
print(pd.DataFrame(erows).round(2).to_string(index=False))
