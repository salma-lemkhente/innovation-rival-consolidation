import pandas as pd

from projlib import config, io

RULES = {"exact": "ex", "calendar": "ca", "gapfill": "gf", "pairfill": "pf", "setfill": "sf", "bridged": "br"}
RULE_LABEL = {"ex": "exact record dates", "br": "exact record dates, a pair's records less than a year apart joined", "ca": "calendar-year edges", "gf": "calendar-year edges, a pair's gap filled when its source's records of the type lapse",
              "pf": "pair gaps filled", "sf": "pair gaps filled and the set carried through empty years"}
KEEP = ["focal", "fyear", "id", "unit_id", "sic2", "sic2_year", "cty_year", "in_window", "own_accounts", "has_accounts", "unit_dup", "crisis", "cal_year",
        "ihs_rnd", "ln1_rnd", "ln_rnd", "rd_w", "rd", "ln_at", "ln_sales", "ln_age", "cash_ratio_w", "capx_at_w", "own_consol", "own_consol_post2", "tc_at", "tc_sales", "isd4_q4", "isdc4_q4"]

v1 = pd.read_parquet(config.out("08_estimation", "estimation_panel.parquet"), columns=KEEP).sort_values(["focal", "fyear"]).reset_index(drop=True)
cov = pd.read_parquet(config.out("07_panel", "v2_coverage.parquet"))[["focal", "coverage_start"]].rename(columns={"coverage_start": "cov_start"})
p = v1.merge(cov, on="focal", how="left")
p["covered"] = (p["fyear"] >= p["cov_start"]).astype(int)
p = p.rename(columns={"isd4_q4": "wave_w", "isdc4_q4": "wave_c"})
dico = {"cov_start": "the firm's coverage start: its first record year as the disclosing company",
        "covered": "1 in the fiscal years from the coverage start on; every shock is missing where covered = 0",
        "wave_w": "the four-digit SDC deal wave of the firm's industry worldwide (the period's deal intensity in the top quartile of the industry's own years), the deals in the fiscal period",
        "wave_c": "the same wave in the firm's own country",
        "tc_at": "tercile within the fiscal year of log assets at t-1 (own-accounts window rows with R&D)", "tc_sales": "the same for log sales at t-1",
        "own_consol": "1 in a fiscal period in which the firm itself acquires or is acquired in a completed control-changing deal", "own_consol_post2": "1 in the year of an own consolidating deal and the two years after (the exclusion window)",
        "ihs_rnd": "asinh of R&D (million USD)", "ln1_rnd": "log(1 + R&D)", "ln_rnd": "log R&D (positive only)", "rd_w": "R&D, million USD, capped at the 99th percentile by year",
        "ln_at": "log total assets", "ln_sales": "log sales", "ln_age": "log(1 + years since founding)", "cash_ratio_w": "cash over assets, winsorised 1/99 by year", "capx_at_w": "capital expenditure over assets, winsorised 1/99 by year",
        "id": "firm", "unit_id": "reporting unit (the cluster)", "sic2_year": "two-digit industry by fiscal year (the absorbed effects)", "in_window": "the fiscal year touches the firm's window", "own_accounts": "the accounts are the firm's own record"}

own_rows = p["in_window"].eq(1) & p["own_accounts"].eq(1) & p["rd"].notna()
for rule, r in RULES.items():
    e = pd.read_parquet(config.out("07_panel", f"v2_exposure_{rule}.parquet"))
    p = p.merge(e, on=["focal", "fyear"], how="left")
    cnt = [c for c in e.columns if c not in ("focal", "fyear") and not c.endswith("_share")]
    p[cnt] = p[cnt].fillna(0).astype(int)
    unk = p["covered"].eq(0)
    def shock(mask):
        return mask.astype(float).where(~unk)
    p[f"any_{r}"] = shock(p["rival_ents"].gt(0))
    p[f"acqs_{r}"] = shock(p["rival_acqs"].gt(0)); p[f"acqd_{r}"] = shock(p["rival_acqd"].gt(0)); p[f"hor_{r}"] = shock(p["rival_hor"].gt(0))
    p[f"thr_{r}"] = shock(p["rival_share"].fillna(0).ge(0.5))
    p[f"und_{r}"] = shock((p["rival_ents"] + p["namer_ents"]).gt(0))
    und_n = p["rival_n"] + p["namer_n"]
    p[f"undthr_{r}"] = shock(((p["rival_ents"] + p["namer_ents"]) / und_n.where(und_n > 0)).fillna(0).ge(0.5))
    p[f"rr_{r}"] = shock(p["rr_ents"].gt(0)); p[f"sup_{r}"] = shock(p["supply_ents"].gt(0)); p[f"par_{r}"] = shock(p["partner_ents"].gt(0))
    p[f"wdr_{r}"] = shock(p["rival_wdrawn"].gt(0)); p[f"div_{r}"] = shock(p["rival_divest"].gt(0)); p[f"eff_{r}"] = shock((p["rival_acqd_eff"] + p["rival_acqs_eff"]).gt(0))
    p[f"nriv_{r}"] = p["rival_n"]
    p[f"norec_{r}"] = (p["covered"].eq(1) & p["rival_n"].eq(0)).astype(int)
    p[f"nam_acqd_{r}"] = p["namer_acqd"].gt(0).astype(int); p[f"nam_acqs_{r}"] = p["namer_acqs"].gt(0).astype(int)
    p[f"rr_acqd_{r}"] = p["rr_acqd"].gt(0).astype(int); p[f"rr_acqs_{r}"] = p["rr_acqs"].gt(0).astype(int); p[f"rrno_{r}"] = p["rr_known"].eq(0).astype(int)
    p[f"sup_acqd_{r}"] = p["supply_acqd"].gt(0).astype(int); p[f"sup_acqs_{r}"] = p["supply_acqs"].gt(0).astype(int)
    p[f"par_acqd_{r}"] = p["partner_acqd"].gt(0).astype(int); p[f"par_acqs_{r}"] = p["partner_acqs"].gt(0).astype(int)
    p["_s"] = p["rival_share"].fillna(0).where(p["covered"].eq(1) & p["in_window"].eq(1), 0.0)
    p["_o"] = (p["covered"].eq(1) & p["in_window"].eq(1)).astype(int)
    g = p.groupby("focal")
    cum_s = g["_s"].cumsum() - p["_s"]; cum_o = g["_o"].cumsum() - p["_o"]
    ok = own_rows & cum_o.ge(3)
    ranks = cum_s[ok].groupby(p.loc[ok, "fyear"]).rank(pct=True)
    p[f"tc_exp_{r}"] = 0
    p.loc[ok, f"tc_exp_{r}"] = pd.cut(ranks, [0, 1 / 3, 2 / 3, 1.0001], labels=False, include_lowest=True).fillna(-1).astype(int) + 1
    p = p.drop(columns=cnt + [c for c in e.columns if c.endswith("_share")] + ["_s", "_o"])
    lab = RULE_LABEL[r]
    dico.update({f"any_{r}": f"a live designated rival in a completed control-changing deal, either role, in the fiscal period; edges: {lab}; missing before coverage, 0 without a live rival",
                 f"acqs_{r}": f"a live designated rival acquires; edges: {lab}", f"acqd_{r}": f"a live designated rival is acquired; edges: {lab}", f"hor_{r}": f"a live rival buys another live rival of the firm; edges: {lab}",
                 f"thr_{r}": f"at least half of the live rival set in such deals (0 without a live rival); edges: {lab}", f"und_{r}": f"any_{r}, or an entity naming the firm as competitor (not among its rivals) in such a deal; edges: {lab}", f"undthr_{r}": f"at least half of the undirected set (live rivals and namers) in such deals; edges: {lab}",
                 f"rr_{r}": f"a rival of a rival (through the rivals that disclose) in such a deal; 0 where no rival discloses; edges: {lab}", f"sup_{r}": f"a supplier or customer in such a deal; edges: {lab}", f"par_{r}": f"a partner in such a deal; edges: {lab}",
                 f"wdr_{r}": f"placebo: a live rival's withdrawn or rumoured deal; edges: {lab}", f"div_{r}": f"placebo: a live rival's divestiture; edges: {lab}", f"eff_{r}": f"any_{r} placed on the completion date; edges: {lab}",
                 f"nriv_{r}": f"the number of rivals live on any day of the fiscal period; edges: {lab}", f"norec_{r}": f"1 in a covered fiscal period without a live rival (the no-record dummy); edges: {lab}",
                 f"nam_acqd_{r}": f"an entity naming the firm as competitor (not among its rivals) is acquired; edges: {lab}", f"nam_acqs_{r}": f"such a namer acquires; edges: {lab}",
                 f"rr_acqd_{r}": f"a rival of a rival is acquired; edges: {lab}", f"rr_acqs_{r}": f"a rival of a rival acquires; edges: {lab}", f"rrno_{r}": f"no live rival of the period discloses its rivals (the rival-of-rival set is unknown); edges: {lab}",
                 f"sup_acqd_{r}": f"a supplier or customer is acquired; edges: {lab}", f"sup_acqs_{r}": f"a supplier or customer acquires; edges: {lab}", f"par_acqd_{r}": f"a partner is acquired; edges: {lab}", f"par_acqs_{r}": f"a partner acquires; edges: {lab}",
                 f"tc_exp_{r}": f"tercile within the fiscal year of the cumulative share of the rival set in such deals over the covered window years to t-1 (firms with three or more covered years; 0 otherwise); edges: {lab}"})

assert all(len(c) <= 32 for c in p.columns), [c for c in p.columns if len(c) > 32]
io.write(p, config.out("08_estimation", "estimation_panel_v2.parquet"), key=["focal", "fyear"])
pd.DataFrame({"name": list(dico), "definition": list(dico.values())}).to_csv(config.here(__file__, "v2_dictionary.csv"), index=False)
st = p.copy()
for c in st.columns:
    if str(st[c].dtype) in ("Int64", "Int32", "boolean"):
        st[c] = st[c].astype("float")
    elif st[c].dtype == bool:
        st[c] = st[c].astype(int)
    elif st[c].dtype == object or str(st[c].dtype) == "string":
        st[c] = st[c].astype("string").fillna("").astype(str)
st.to_stata(config.out("08_estimation", "estimation_panel_v2.dta"), write_index=False, version=118)
w = p[p["in_window"].eq(1) & p["own_accounts"].eq(1)]
print(f"panel v2: {len(p):,} rows, {p.shape[1]} columns; window rows with own accounts {len(w):,}, of which covered {int(w['covered'].sum()):,}")
for r in RULES.values():
    c = w[w["covered"].eq(1)]
    print(f"  {r}: any {int(c[f'any_{r}'].sum()):,}  thr {int(c[f'thr_{r}'].sum()):,}  und {int(c[f'und_{r}'].sum()):,}  acqs {int(c[f'acqs_{r}'].sum()):,}  acqd {int(c[f'acqd_{r}'].sum()):,}  hor {int(c[f'hor_{r}'].sum()):,}  "
          f"rr {int(c[f'rr_{r}'].sum()):,}  sup {int(c[f'sup_{r}'].sum()):,}  par {int(c[f'par_{r}'].sum()):,}  wdr {int(c[f'wdr_{r}'].sum()):,}  div {int(c[f'div_{r}'].sum()):,}  eff {int(c[f'eff_{r}'].sum()):,}  "
          f"norec {int(c[f'norec_{r}'].sum()):,}  tc_exp set {int(c[f'tc_exp_{r}'].gt(0).sum()):,}")
