import pandas as pd

from projlib import config, io

RULES = {"calendar": "ca", "gapfill": "gf", "exact": "ex", "bridged": "br"}
RULE_LABEL = {"ca": "calendar-year edges", "gf": "calendar-year edges, a pair's gap filled when its source's records of the type lapse",
              "ex": "exact record dates", "br": "exact record dates, a pair's records less than a year apart joined"}
p = pd.read_parquet(config.out("08_estimation", "estimation_panel_v2.parquet")).sort_values(["focal", "fyear"]).reset_index(drop=True)
v1 = pd.read_parquet(config.out("08_estimation", "estimation_panel.parquet"), columns=["focal", "fyear", "rd_missing", "ln1_rnd0", "ihs_rnd0", "rnd_lat_w", "rnd_lat0_w", "rnd_lsl_w"])
p = p.merge(v1, on=["focal", "fyear"], how="left")
dico = dict(pd.read_csv(config.here(__file__, "v2_dictionary.csv"), dtype=str).values.tolist())
dico.update({"rd_missing": "1 where accounts exist but R&D is not reported (the missingness dummy, at t)", "ln1_rnd0": "log(1 + R&D) with missing R&D set to zero",
             "ihs_rnd0": "asinh(R&D) with missing R&D set to zero", "rnd_lat_w": "R&D over the previous fiscal year's assets, winsorised 1/99 by year", "rnd_lat0_w": "the same with missing R&D set to zero", "rnd_lsl_w": "R&D over the previous fiscal year's sales, winsorised 1/99 by year"})
unk = p["covered"].eq(0)


def shock(mask, none):
    z = mask.astype(float).where(~unk)
    return z, z.where(~none)


for rule, r in RULES.items():
    e = pd.read_parquet(config.out("07_panel", f"v2_exposure_{rule}.parquet"))
    cols = [c for c in e.columns if c.startswith(("undir_", "mutual_", "rnd", "rival_n", "rival_ents", "rival_hor", "rival_share", "rival_wdrawn"))]
    p = p.merge(e[["focal", "fyear"] + cols], on=["focal", "fyear"], how="left")
    cnt = [c for c in cols if not c.endswith("_share")]
    p[cnt] = p[cnt].fillna(0).astype(int)
    lab = RULE_LABEL[r]
    sets = {"": ("rival", "the designated rivals"), "u_": ("undir", "the undirected set (designated rivals and namers)"), "m_": ("mutual", "the mutual rivals (named and naming back)")}
    for pre, (ch, slab) in sets.items():
        n, ents, hor, share = p[f"{ch}_n"], p[f"{ch}_ents"], p[f"{ch}_hor"], p[f"{ch}_share"].fillna(0)
        none = p["covered"].eq(1) & n.eq(0)
        if pre:
            p[f"{pre}n_{r}"] = n; p[f"{pre}norec_{r}"] = none.astype(int)
            dico[f"{pre}n_{r}"] = f"the number of live members of {slab} in the fiscal period; edges: {lab}"
            dico[f"{pre}norec_{r}"] = f"1 in a covered period in which {slab} has no live member (the reporting dummy, at t); edges: {lab}"
        for tag, mask, what in (("any", ents.gt(0), "a live member in a completed control-changing deal, either role"), ("hor", hor.gt(0), "a live member acquires another live member of the same firm's set"),
                                ("thr", share.ge(0.5), "at least half of the live set in such deals (0 without a live member)")):
            z, m = shock(mask, none)
            name = f"{pre}{tag}_{r}"
            p[name], p[f"{name}_m"] = z, m
            dico[name] = f"{what}; {slab}; edges: {lab}; missing before coverage, zero in a covered period without a live member"
            dico[f"{name}_m"] = f"{what}; {slab}; edges: {lab}; missing before coverage AND in a covered period without a live member"
    none_d = p["covered"].eq(1) & p["rival_n"].eq(0)
    p[f"wdr_{r}_m"] = p[f"wdr_{r}"].where(~none_d); dico[f"wdr_{r}_m"] = dico.get(f"wdr_{r}", "placebo: a rival's withdrawn or rumoured deal") + "; missing in a covered period without a live rival"
    for d in (range(1, 21) if "rnd1_ents" in cols else ()):
        for tag, mask, what in (("any", p[f"rnd{d}_ents"].gt(0), "any deal"), ("thr", p[f"rnd{d}_share"].fillna(0).ge(0.5), "at least half the set in deals")):
            if tag == "thr" and d > 1:
                continue
            z, m = shock(mask, none_d)
            name = f"r{d}_{tag}_{r}"
            p[name], p[f"{name}_m"] = z, m
            dico[name] = f"placebo: {what} among a RANDOM rival set (draw {d}: as many entities as the firm has live rivals, drawn from the entities that are a live rival of some focal firm that year); edges: {lab}"
            dico[f"{name}_m"] = dico[name] + "; missing in a covered period without a live rival"
    p = p.drop(columns=cols)

assert all(len(c) <= 32 for c in p.columns), [c for c in p.columns if len(c) > 32]
io.write(p, config.out("08_estimation", "estimation_panel_v3.parquet"), key=["focal", "fyear"])
pd.DataFrame({"name": list(dico), "definition": list(dico.values())}).to_csv(config.here(__file__, "v3_dictionary.csv"), index=False)
st = p.copy()
for c in st.columns:
    if str(st[c].dtype) in ("Int64", "Int32", "boolean"):
        st[c] = st[c].astype("float")
    elif st[c].dtype == bool:
        st[c] = st[c].astype(int)
    elif st[c].dtype == object or str(st[c].dtype) == "string":
        st[c] = st[c].astype("string").fillna("").astype(str)
st.to_stata(config.out("08_estimation", "estimation_panel_v3.dta"), write_index=False, version=118)
w = p[p["in_window"].eq(1) & p["own_accounts"].eq(1) & p["covered"].eq(1)]
print(f"panel v3: {len(p):,} rows, {p.shape[1]} columns; covered window rows with own accounts {len(w):,}")
for r in RULES.values():
    rnd = f"| random any {int(w[f'r1_any_{r}'].sum()):,} thr {int(w[f'r1_thr_{r}'].sum()):,} " if f"r1_any_{r}" in w.columns else "| no random sets "
    print(f"  {r}: any {int(w[f'any_{r}'].sum()):,} thr {int(w[f'thr_{r}'].sum()):,} hor {int(w[f'hor_{r}'].sum()):,} | undirected any {int(w[f'u_any_{r}'].sum()):,} thr {int(w[f'u_thr_{r}'].sum()):,} hor {int(w[f'u_hor_{r}'].sum()):,} "
          f"| mutual any {int(w[f'm_any_{r}'].sum()):,} thr {int(w[f'm_thr_{r}'].sum()):,} hor {int(w[f'm_hor_{r}'].sum()):,} (mutual set present {int(w[f'm_n_{r}'].gt(0).sum()):,}) "
          f"{rnd}| missing handling drops {int(w[f'any_{r}_m'].isna().sum()):,} rows")
