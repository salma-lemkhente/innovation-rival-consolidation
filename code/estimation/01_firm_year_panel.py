import numpy as np
import pandas as pd

from projlib import config, io

E = "FACTSET_ENTITY_ID"
p = pd.read_parquet(config.out("07_panel", "panel_focal_fyear.parquet"))
rr = pd.read_parquet(config.out("07_panel", "rival_of_rival_year.parquet")).rename(columns={"year": "cal_year"})
ind = pd.read_parquet(config.out("07_panel", "industry_network_year.parquet")).rename(columns={"year": "cal_year"})
rev = pd.read_parquet(config.out("07_panel", "reverse_rival_year.parquet")).rename(columns={"year": "cal_year"})
p = p.merge(rr, on=["focal", "cal_year"], how="left").merge(ind, on=["focal", "cal_year"], how="left").merge(rev, on=["focal", "cal_year"], how="left")
for c in [c for c in rev.columns if c.startswith("rev_") and "share" not in c]:
    p[c] = p[c].fillna(0).astype(int)
p = p.sort_values(["focal", "fyear"]).reset_index(drop=True)

p["d_acqd"] = p["n_consolidation_acquired"].gt(0).astype(int)
p["d_acqs"] = p["n_consolidation_acquires"].gt(0).astype(int)
p["d_any"] = (p["d_acqd"] | p["d_acqs"]).astype(int)
p["d_hor"] = p["n_consolidation_horizontal"].gt(0).astype(int)
p["dx_acqd"] = p["exact_n_consolidation_acquired"].gt(0).astype(int)
p["dx_acqs"] = p["exact_n_consolidation_acquires"].gt(0).astype(int)
p["dx_hor"] = p["exact_n_consolidation_horizontal"].gt(0).astype(int)
p["dx_any"] = (p["dx_acqd"] | p["dx_acqs"]).astype(int)
p["d_wdrawn"] = p["n_withdrawn"].gt(0).astype(int)
p["d_partial"] = p["n_partial"].gt(0).astype(int)
p["d_divest"] = p["n_divests"].gt(0).astype(int)
per = pd.read_parquet(config.out("07_panel", "panel_focal_fyear.parquet"), columns=["focal", "fyear", "period_start", "period_end"])
rv = pd.read_parquet(config.out("07_panel", "reverse_events.parquet"))
rv = rv[rv["only"]].merge(per, on="focal"); rv = rv[(rv["date_announced"] >= rv["period_start"]) & (rv["date_announced"] <= rv["period_end"])]
rvc = rv.groupby(["focal", "fyear", "cls"])["MASTER_DEAL_NO"].nunique().unstack("cls").fillna(0)
p = p.merge(rvc.rename(columns={"consolidation_acquired": "revf_n_cacqd", "consolidation_acquires": "revf_n_cacqs"}), left_on=["focal", "fyear"], right_index=True, how="left")
p[["revf_n_cacqd", "revf_n_cacqs"]] = p[["revf_n_cacqd", "revf_n_cacqs"]].fillna(0).astype(int)
rh = pd.read_parquet(config.out("07_panel", "reverse_horizontal_events.parquet")).merge(per, on="focal")
rh = rh[(rh["date_announced"] >= rh["period_start"]) & (rh["date_announced"] <= rh["period_end"])].groupby(["focal", "fyear"])["MASTER_DEAL_NO"].nunique().rename("symf_n_chor")
p = p.merge(rh, left_on=["focal", "fyear"], right_index=True, how="left"); p["symf_n_chor"] = p["symf_n_chor"].fillna(0).astype(int)
p["rev_only_n_consolidation_acquired"] = p["revf_n_cacqd"]
p["rev_only_n_consolidation_acquires"] = p["revf_n_cacqs"]
p["d_sym"] = (p["d_any"].eq(1) | (p["revf_n_cacqd"] + p["revf_n_cacqs"]).gt(0)).astype(int)
p["d_sym_acqs"] = (p["d_acqs"].eq(1) | p["revf_n_cacqs"].gt(0)).astype(int)
p["d_sym_acqd"] = (p["d_acqd"].eq(1) | p["revf_n_cacqd"].gt(0)).astype(int)
p["sym_n_chor"] = p["sym_n_chor"].fillna(0).astype(int)
p["d_sym_hor"] = (p["d_hor"].eq(1) | p["symf_n_chor"].gt(0)).astype(int)
p["d_ind"] = p["ind_n_consolidation_acquired"].fillna(0).gt(0).astype(int)
p["d_ind_cty"] = p["ind_cty_n_consolidation_acquired"].fillna(0).gt(0).astype(int)
p["d_ind_excl"] = p["ind_excl_n_consolidation_acquired"].fillna(0).gt(0).astype(int)
DOSE = {"d_acqd": "share_consolidation_acquired", "d_acqs": "share_consolidation_acquires", "d_any": None, "d_hor": "share_consolidation_horizontal", "d_sym": None, "d_sym_acqs": None, "d_sym_acqd": None, "d_sym_hor": None}
for d, dose in DOSE.items():
    first = p[p["in_window"] & p[d].eq(1)].groupby("focal")["fyear"].min()
    p[f"first_{d[2:]}"] = p["focal"].map(first)
    if dose:
        at_first = p[p["fyear"] == p[f"first_{d[2:]}"]].set_index("focal")[dose]
        p[f"dose1_{d[2:]}"] = p["focal"].map(at_first)

p["own_consol"] = (p["own_n_consolidation_acquired"].gt(0) | p["own_n_consolidation_acquires"].gt(0)).astype(int)
own_cols = [c for c in p.columns if c.startswith("own_n_")]
p["own_anyma"] = p[own_cols].gt(0).any(axis=1).astype(int)
for k in (1, 2, 3):
    s = p.groupby("focal")["own_consol"].transform(lambda x: x.rolling(2 * k + 1, center=True, min_periods=1).max())
    p[f"own_consol_pm{k}"] = s.astype(int)
lag1 = p[["focal", "fyear", "own_consol"]].assign(fyear=lambda d: d["fyear"] + 1).rename(columns={"own_consol": "_l1"})
lag2 = p[["focal", "fyear", "own_consol"]].assign(fyear=lambda d: d["fyear"] + 2).rename(columns={"own_consol": "_l2"})
p = p.merge(lag1, on=["focal", "fyear"], how="left").merge(lag2, on=["focal", "fyear"], how="left")
p["own_consol_post2"] = (p["own_consol"].eq(1) | p["_l1"].eq(1) | p["_l2"].eq(1)).astype(int)
p = p.drop(columns=["_l1", "_l2"])
for k in range(1, 6):
    lead = p[["focal", "fyear", "own_consol"]].assign(fyear=lambda d: d["fyear"] - k).rename(columns={"own_consol": f"_f{k}"})
    p = p.merge(lead, on=["focal", "fyear"], how="left")
p["own_consol_fwd5"] = (p["own_consol"].eq(1) | p[[f"_f{k}" for k in range(1, 6)]].eq(1).any(axis=1)).astype(int)
p["own_consol_win5"] = (p["own_consol_fwd5"].eq(1) | p["own_consol_pm3"].eq(1) | p.groupby("focal")["own_consol"].shift(4).eq(1) | p.groupby("focal")["own_consol"].shift(5).eq(1)).astype(int)
p = p.drop(columns=[f"_f{k}" for k in range(1, 6)])
p["own_consol_back5"] = (p["own_consol"].eq(1) | pd.concat([p.groupby("focal")["own_consol"].shift(k).eq(1) for k in range(1, 6)], axis=1).any(axis=1)).astype(int)
first_own = p[p["own_n_consolidation_acquired"].gt(0)].groupby("focal")["fyear"].min()
p["own_acqd_first"] = p["focal"].map(first_own)
p["post_own_acqd"] = (p["fyear"] >= p["own_acqd_first"]).fillna(False).astype(int)

p["rd"] = p["rd_net_usd"]
base1 = p[p["years_before_window"].eq(1)].sort_values("fyear").groupby("focal")["at_usd"].last()
base2 = p[p["years_before_window"].isin([1, 2])].groupby("focal")["at_usd"].mean()
p["at_base1"] = p["focal"].map(base1)
p["at_base2"] = p["focal"].map(base2)
p["rnd_at_base1"] = p["rd"] / p["at_base1"].where(p["at_base1"] > 0)
p["rnd_at_base2"] = p["rd"] / p["at_base2"].where(p["at_base2"] > 0)
p["ihs_rnd"] = np.arcsinh(p["rd"])
p["rnd_pos"] = p["rd"].gt(0).where(p["rd"].notna()).astype("float")
p["ln_rnd"] = np.log(p["rd"].where(p["rd"] > 0))
p["rnd_sales"] = p["rd_intensity_sales"]
p["rnd_at"] = p["rd_intensity_assets"]
p["n_pat"] = p["families_originated"].where(~p["patents_missing"].fillna(True).astype(bool))
p["n_fam"] = p["families"].where(~p["patents_missing"].fillna(True).astype(bool))
p["ln_at"] = np.log(p["at_usd"].where(p["at_usd"] > 0))
p["ln_sales"] = np.log(p["sale_usd"].where(p["sale_usd"] > 0))
p["ln_emp"] = np.log1p(p["emp"].where(p["emp"] >= 0))
p["ln_age"] = np.log1p(p["age"].where(p["age"] >= 0))
prev = p[["focal", "fyear", "at_usd"]].assign(fyear=lambda d: d["fyear"] + 1).rename(columns={"at_usd": "at_lag1"})
p = p.merge(prev, on=["focal", "fyear"], how="left")
p["rnd_lat"] = p["rd"] / p["at_lag1"].where(p["at_lag1"] > 0)
prev_s = p[["focal", "fyear", "sale_usd"]].assign(fyear=lambda d: d["fyear"] + 1).rename(columns={"sale_usd": "sale_lag1"})
p = p.merge(prev_s, on=["focal", "fyear"], how="left")
p["rnd_lsl"] = p["rd"] / p["sale_lag1"].where(p["sale_lag1"] > 0)
p["capx_at"] = p["capx_usd"] / p["at_usd"].where(p["at_usd"] > 0)
p["roa"] = p["oibdp_usd"] / p["at_usd"].where(p["at_usd"] > 0)
p["intan_at"] = p["intan_usd"] / p["at_usd"].where(p["at_usd"] > 0)
p["ln1_rnd"] = np.log1p(p["rd"].where(p["rd"] >= 0))
own_rows = p["in_window"].astype(bool) & p["level"].isin(["same_cleared", "same_reviewed", "factset_entity"])
for c in ("rnd_at", "rnd_lat", "rnd_lsl", "rnd_sales", "leverage", "cash_ratio", "capx_at", "roa", "intan_at"):
    q = p.loc[own_rows].groupby("fyear")[c].quantile([0.01, 0.99]).unstack()
    lo, hi = p["fyear"].map(q[0.01]), p["fyear"].map(q[0.99])
    p[f"{c}_w"] = p[c].clip(lower=lo, upper=hi)
p["ln_rnd_lat"] = np.log(p["rnd_lat_w"].where(p["rnd_lat_w"] > 0))
p["rd_w"] = p["rd"].clip(upper=p["fyear"].map(p.loc[own_rows].groupby("fyear")["rd"].quantile(0.99)))
p["rd_missing"] = (p["has_accounts"].astype(bool) & p["rd"].isna()).astype(int)
p["rd0"] = p["rd"].where(p["rd"].notna(), 0.0).where(p["has_accounts"].astype(bool))
p["rnd_lat0"] = p["rd0"] / p["at_lag1"].where(p["at_lag1"] > 0)
q0 = p.loc[own_rows].groupby("fyear")["rnd_lat0"].quantile([0.01, 0.99]).unstack()
p["rnd_lat0_w"] = p["rnd_lat0"].clip(lower=p["fyear"].map(q0[0.01]), upper=p["fyear"].map(q0[0.99]))
p["rd0_w"] = p["rd0"].clip(upper=p["fyear"].map(p.loc[own_rows].groupby("fyear")["rd0"].quantile(0.99)))
for base, src in (("rnd_lat0", "rnd_lat0_w"), ("rnd0", "rd0_w")):
    p[f"ln_{base}"] = np.log(p[src].where(p[src] > 0)); p[f"ln1_{base}"] = np.log1p(p[src]); p[f"ihs_{base}"] = np.arcsinh(p[src])
lag_prev = p[["focal", "fyear", "rnd_lat_w", "rnd_at_w"]].assign(fyear=lambda d: d["fyear"] + 1).rename(columns={"rnd_lat_w": "_lat1", "rnd_at_w": "_at1"})
lag_prev2 = p[["focal", "fyear", "rnd_at_w"]].assign(fyear=lambda d: d["fyear"] + 2).rename(columns={"rnd_at_w": "_at2"})
p = p.merge(lag_prev[["focal", "fyear", "_lat1"]], on=["focal", "fyear"], how="left").merge(lag_prev2, on=["focal", "fyear"], how="left")
for name, f in (("rnd_lat_w", lambda s: s), ("ln_rnd_lat", lambda s: np.log(s.where(s > 0))), ("ln1_rnd_lat", np.log1p), ("ihs_rnd_lat", np.arcsinh)):
    p[f"dl_{name}"] = f(p["_lat1"]) - f(p["_at2"])
p = p.drop(columns=["_lat1", "_at2"])
p["ln1_rnd_lat"] = np.log1p(p["rnd_lat_w"])
p["ihs_rnd_lat"] = np.arcsinh(p["rnd_lat_w"])

dy = pd.read_parquet(config.out("07_panel", "exposure_events.parquet"), columns=["focal", "related", "kind", "cls", "date_announced"])
dy = dy[dy["kind"].eq("rival") & dy["cls"].isin(["consolidation_acquired", "consolidation_acquires"])]
per = pd.read_parquet(config.out("07_panel", "panel_focal_fyear.parquet"), columns=["focal", "fyear", "period_start", "period_end"])
dy = dy.merge(per, on="focal"); dy = dy[(dy["date_announced"] >= dy["period_start"]) & (dy["date_announced"] <= dy["period_end"])]
any_r = dy.groupby(["focal", "fyear"])["related"].nunique().rename("rivals_any")
p = p.merge(any_r, left_on=["focal", "fyear"], right_index=True, how="left")
p["rivals_any"] = p["rivals_any"].fillna(0).astype(int)
p["share_any"] = p["rivals_any"] / p["n_rivals"].where(p["n_rivals"] > 0)

p["id"] = pd.factorize(p["focal"])[0] + 1
unit = p["gvkey_group"].astype("string").fillna(p["gvkey"].astype("string")).fillna("E:" + p["focal"].astype("string"))
p["unit_id"] = pd.factorize(unit)[0] + 1
p["sic2"] = pd.to_numeric(p["sic"], errors="coerce").floordiv(100).astype("Int64")
p["sic2_year"] = (p["sic2"].astype("float") * 10000 + p["fyear"]).astype("Int64")
p["cty_year"] = pd.factorize(p["country"].fillna("??") + "_" + p["fyear"].astype(str))[0] + 1
p["std_year"] = pd.factorize(p["accounting_standard"].fillna("NA") + "_" + p["fyear"].astype(str))[0] + 1
p["own_accounts"] = p["level"].isin(["same_cleared", "same_reviewed", "factset_entity", "orbis_same_cleared", "orbis_same_reviewed"]).astype(int)
p["in_window"] = p["in_window"].astype(int); p["has_accounts"] = p["has_accounts"].astype(int)
p["unit_dup"] = p["unit_duplicate"].astype(int); p["pat_missing"] = p["patents_missing"].fillna(True).astype(int)
p["trunc_years"] = p["patent_truncation_years"]

p["supply_acqd"] = p["supply_n_consolidation_acquired"].gt(0).astype(int); p["supply_acqs"] = p["supply_n_consolidation_acquires"].gt(0).astype(int)
p["partner_acqd"] = p["partner_n_consolidation_acquired"].gt(0).astype(int); p["partner_acqs"] = p["partner_n_consolidation_acquires"].gt(0).astype(int)
p["rr_acqd"] = p["rr_n_consolidation_acquired"].fillna(0).gt(0).astype(int); p["rr_acqs"] = p["rr_n_consolidation_acquires"].fillna(0).gt(0).astype(int)
p["rr_noset"] = p["rr_size"].isna().astype(int)
p["rev_acqd"] = p["revf_n_cacqd"].gt(0).astype(int); p["rev_acqs"] = p["revf_n_cacqs"].gt(0).astype(int)

p = p.sort_values(["focal", "fyear"])
for y in ("ihs_rnd", "ln1_rnd", "ln_rnd", "rd_w"):
    for k in range(1, 5):
        a = p[["focal", "fyear", y]].assign(fyear=lambda d: d["fyear"] + k).rename(columns={y: "_a"})
        b = p[["focal", "fyear", y]].assign(fyear=lambda d: d["fyear"] + k + 1).rename(columns={y: "_b"})
        p = p.merge(a, on=["focal", "fyear"], how="left").merge(b, on=["focal", "fyear"], how="left")
        d = p["_a"] - p["_b"]
        p[f"dym{k}_{y}"] = d.isna().astype(int)
        p[f"dy{k}_{y}"] = d.fillna(0.0)
        p = p.drop(columns=["_a", "_b"])

dy = pd.read_parquet(config.out("07_panel", "exposure_events.parquet"),
                     columns=["focal", "related", "kind", "MASTER_DEAL_NO", "cls", "year", "date_announced", "date_effective", "deal_value", "distance_km", "cross_border"])
dy = dy[dy["kind"].eq("rival") & dy["cls"].isin(["consolidation_acquired", "consolidation_acquires"])]
traits = pd.read_parquet(config.out("07_panel", "rival_pair_traits.parquet")).rename(columns={"rival": "related"})
dy = dy.merge(traits[["focal", "related", "year", "mutual", "rivalry_age"]], on=["focal", "related", "year"], how="left")
indeg = pd.read_parquet(config.out("07_panel", "indegree_entity_year.parquet")).rename(columns={"entity": "related"})
dy = dy.merge(indeg, on=["related", "year"], how="left")
dy["indegree"] = dy["indegree"].fillna(0)
dy["big"] = dy["deal_value"] >= dy.groupby("year")["deal_value"].transform("median")
dy["small"] = dy["deal_value"] < dy.groupby("year")["deal_value"].transform("median")
dy["pop"] = dy["indegree"] >= dy.groupby("year")["indegree"].transform(lambda s: s.quantile(2 / 3))
dy["unpop"] = ~dy["pop"]
dy["mut"] = dy["mutual"].fillna(False).astype(bool); dy["one"] = ~dy["mut"]
dy["near"] = dy["distance_km"] <= 500; dy["far"] = dy["distance_km"] > 500
dy["dom"] = dy["cross_border"].eq(False); dy["xb"] = dy["cross_border"].eq(True)
dy["old"] = dy["rivalry_age"] >= 3; dy["new"] = dy["rivalry_age"] < 3
TYPES = ["mut", "one", "near", "far", "big", "small", "dom", "xb", "old", "new", "pop", "unpop"]
dyf = dy.merge(per, on="focal")
ann = dyf[(dyf["date_announced"] >= dyf["period_start"]) & (dyf["date_announced"] <= dyf["period_end"])]
eff = dyf[dyf["date_effective"].notna() & (dyf["date_effective"] >= dyf["period_start"]) & (dyf["date_effective"] <= dyf["period_end"])]
typ = ann.groupby(["focal", "fyear"])[TYPES].max()
p = p.merge(typ.rename(columns={c: f"d_{c}" for c in TYPES}), left_on=["focal", "fyear"], right_index=True, how="left")
for c in TYPES:
    p[f"d_{c}"] = p[f"d_{c}"].fillna(False).astype(int)
p = p.sort_values(["focal", "fyear"])
PAIRS = (("mut", "one"), ("near", "far"), ("big", "small"), ("dom", "xb"), ("old", "new"), ("pop", "unpop"))
for a, b in PAIRS:
    for c, other in ((a, b), (b, a)):
        pure = (p[f"d_{c}"].eq(1) & p[f"d_{other}"].eq(0)).astype(int)
        p[f"w_{c}5"] = pure.groupby(p["focal"]).transform(lambda x: x.rolling(11, center=True, min_periods=1).max()).astype(int)
e = eff.groupby(["focal", "fyear"])["MASTER_DEAL_NO"].nunique().rename("n_any_eff")
p = p.merge(e, left_on=["focal", "fyear"], right_index=True, how="left"); p["d_any_eff"] = p["n_any_eff"].fillna(0).gt(0).astype(int); p = p.drop(columns="n_any_eff")
cal = pd.read_parquet(config.out("07_panel", "exposure_focal_year.parquet"), columns=["focal", "year", "n_consolidation_acquired", "n_consolidation_acquires"])
cal["d_any_cal"] = (cal["n_consolidation_acquired"] + cal["n_consolidation_acquires"]).gt(0).astype(int)
p = p.merge(cal[["focal", "year", "d_any_cal"]].rename(columns={"year": "cal_year"}), on=["focal", "cal_year"], how="left"); p["d_any_cal"] = p["d_any_cal"].fillna(0).astype(int)
p["_deals"] = p["n_consolidation_acquired"].fillna(0) + p["n_consolidation_acquires"].fillna(0)
p["_share"] = p["share_any"].fillna(0)
p["_obs"] = p["in_window"].eq(1).astype(int)
g = p.groupby("focal")
for name, src in (("sy", "d_any"), ("deals", "_deals"), ("share", "_share"), ("hist", "_obs")):
    cum = g[src].cumsum() - p[src]
    p[f"_c_{name}"] = cum
    lag = p[["focal", "fyear", f"_c_{name}"]].assign(fyear=lambda d: d["fyear"] + 5).rename(columns={f"_c_{name}": f"pe_{name}"})
    p = p.merge(lag, on=["focal", "fyear"], how="left")
for name in ("sy", "deals", "share"):
    ok = p["pe_hist"].ge(3) & p["in_window"].eq(1)
    ranks = p.loc[ok].groupby("fyear")[f"pe_{name}"].rank(pct=True)
    p[f"pb_{name}"] = 0
    p.loc[ok, f"pb_{name}"] = pd.cut(ranks, [0, 1 / 3, 2 / 3, 1.0001], labels=False, include_lowest=True).fillna(-1).astype(int) + 1
p = p.drop(columns=["_deals", "_share", "_obs", "_c_sy", "_c_deals", "_c_share", "_c_hist", "pe_hist"])
first = p[p["d_any"].eq(1) & p["in_window"].eq(1)].groupby("focal")["fyear"].min().rename("first_any_f")
p = p.merge(first, left_on="focal", right_index=True, how="left")
d1 = p[p["fyear"].eq(p["first_any_f"])][["focal", "fyear", "share_any"]].rename(columns={"share_any": "dose1_any"})
d1 = d1[d1["dose1_any"].notna()]
d1["fb_any"] = d1.groupby("fyear")["dose1_any"].rank(pct=True).pipe(lambda r: pd.cut(r, [0, 1 / 3, 2 / 3, 1.0001], labels=False, include_lowest=True)).fillna(-1).astype(int) + 1
p = p.merge(d1[["focal", "dose1_any", "fb_any"]], on="focal", how="left"); p["fb_any"] = p["fb_any"].fillna(0).astype(int)
own_rows_l = p["in_window"].eq(1) & p["level"].isin(["same_cleared", "same_reviewed", "factset_entity"]) & p["rd"].notna()
for name, src in (("at", "ln_at"), ("sales", "ln_sales"), ("cash", "cash_ratio_w"), ("nriv", "n_rivals"), ("named", "rev_all_size")):
    lagv = p[["focal", "fyear", src]].assign(fyear=lambda d: d["fyear"] + 1).rename(columns={src: "_l"})
    p = p.merge(lagv, on=["focal", "fyear"], how="left")
    p[f"tc_{name}"] = 0
    ok = own_rows_l & p["_l"].notna() & (p["_l"].gt(0) if name == "named" else True)
    ranks = p.loc[ok].groupby("fyear")["_l"].rank(pct=True)
    p.loc[ok, f"tc_{name}"] = pd.cut(ranks, [0, 1 / 3, 2 / 3, 1.0001], labels=False, include_lowest=True).fillna(-1).astype(int) + 1
    p = p.drop(columns="_l")
agel = p[["focal", "fyear", "age"]].assign(fyear=lambda d: d["fyear"] + 1).rename(columns={"age": "_agel"})
p = p.merge(agel, on=["focal", "fyear"], how="left"); p["young"] = p["_agel"].lt(10).astype(int); p = p.drop(columns="_agel")
p["ind5"] = p["sic2"].isin([28, 73, 36, 38, 35]).astype(int)
p["rd_pos"] = p["rd0"].gt(0).astype(int)
posl = p[["focal", "fyear", "rd_pos"]].assign(fyear=lambda d: d["fyear"] + 1).rename(columns={"rd_pos": "rd_pos_l1"})
p = p.merge(posl, on=["focal", "fyear"], how="left")
p["crisis"] = p["fyear"].isin([2008, 2009]).astype(int)
p["d_any_cov"] = p["d_any"].where(p["fyear"] >= 2003)
p["d_any_riv"] = p["d_any"].where((p["fyear"] >= 2003) & (p["n_rivals"] > 0))

isd = pd.read_parquet(config.out("07_panel", "industry_sdc_year.parquet"))
deals = pd.read_parquet(config.out("07_panel", "industry_sdc_deals.parquet"))
deals["date_announced"] = pd.to_datetime(deals["date_announced"])
p["sic3"] = (p["sic4"] // 10).astype("Int64")
per = pd.read_parquet(config.out("07_panel", "panel_focal_fyear.parquet"), columns=["focal", "fyear", "period_start", "period_end"])
pp = p[["focal", "fyear", "sic3", "sic4", "country"]].merge(per, on=["focal", "fyear"], how="left")


def fiscal_count(level, by_nation):
    out = pd.Series(0, index=pp.index, dtype="int64")
    keys = [level] + (["nation"] if by_nation else [])
    grp = deals.sort_values("date_announced").groupby(keys)["date_announced"]
    dates = {(k[0] if isinstance(k, tuple) and len(k) == 1 else k): v.to_numpy() for k, v in grp}
    codes = pp[level].astype("float")
    for i, (c, n, s0, e0) in enumerate(zip(codes, pp["country"], pp["period_start"], pp["period_end"])):
        if pd.isna(c) or pd.isna(s0):
            continue
        k = (int(c), n) if by_nation else int(c)
        v = dates.get(k)
        if v is None:
            continue
        out.iat[i] = int(np.searchsorted(v, np.datetime64(e0), side="right") - np.searchsorted(v, np.datetime64(s0), side="left"))
    return out


for level in ("sic3", "sic4"):
    lv = level[-1]
    for by_nation, pre in ((False, f"isd{lv}"), (True, f"isdc{lv}")):
        t = isd[isd["level"].eq(level) & (isd["nation"].ne("ALL") if by_nation else isd["nation"].eq("ALL"))].copy()
        thr = t.groupby(["code"] + (["nation"] if by_nation else []))["intensity"].agg(q75=lambda s: s.quantile(0.75), med="median").reset_index()
        size = t.drop_duplicates("code").set_index("code")["size"]
        n = fiscal_count(level, by_nation)
        p[f"{pre}_n"] = n.to_numpy()
        p[f"{pre}_int"] = (n / pp[level].astype("float").map(size).where(lambda s: s > 0)).fillna(0).to_numpy()
        key = pp[[level] + (["country"] if by_nation else [])].astype({level: "float"}).rename(columns={level: "code", "country": "nation"})
        thr["code"] = thr["code"].astype("float")
        m = key.merge(thr, on=["code"] + (["nation"] if by_nation else []), how="left")
        p[f"{pre}_q4"] = (p[f"{pre}_int"] > 0) & (p[f"{pre}_int"] >= m["q75"].to_numpy())
        p[f"{pre}_2x"] = (p[f"{pre}_int"] > 0) & (p[f"{pre}_int"] > 2 * m["med"].to_numpy())
        med_year = t.groupby("year")["intensity"].median()
        p[f"{pre}_high"] = (p[f"{pre}_int"] > p["cal_year"].map(med_year).fillna(0).to_numpy()).astype(int)
        for c in (f"{pre}_q4", f"{pre}_2x"):
            p[c] = p[c].astype(int); p[f"d_{c}"] = p[c]
tcal = isd[isd["level"].eq("sic4") & isd["nation"].eq("ALL")].rename(columns={"year": "cal_year", "code": "sic4", "wave_q4": "isd4_q4_cal", "wave_2x": "isd4_2x_cal"})[["cal_year", "sic4", "isd4_q4_cal", "isd4_2x_cal"]]
tcal["sic4"] = tcal["sic4"].astype("float"); p["sic4"] = p["sic4"].astype("float")
p = p.merge(tcal, on=["cal_year", "sic4"], how="left")
for c in ("isd4_q4_cal", "isd4_2x_cal"):
    p[c] = p[c].fillna(0).astype(int); p[f"d_{c}"] = p[c]

short = {"consolidation_acquired": "cacqd", "consolidation_acquires_group": "cacqg", "consolidation_acquires": "cacqs", "consolidation_horizontal": "chor",
         "withdrawn": "wdrawn", "partial": "partial", "divests": "divest", "pending": "pend", "rumour": "rumour"}
drop = ["period_start", "period_end", "source", "currency", "rd_raw_usd", "rd_net_filled_usd", "has_inprocess_rd", "gvkey", "gvkey_group", "sic", "unit_duplicate",
        "patents_missing", "patent_truncation_years", "families_originated", "families", "rd_intensity_sales", "rd_intensity_assets", "level", "window_days"]
out = p.drop(columns=[c for c in drop if c in p.columns])


def stata_name(c: str) -> str:
    n = c
    for k, v in sorted(short.items(), key=lambda kv: -len(kv[0])):
        n = n.replace(k, v)
    n = n.replace("_usd", "")
    return n[:32]


names = {c: stata_name(c) for c in out.columns}
assert len(set(names.values())) == len(names), "two columns collide under Stata names"
pd.DataFrame({"stata_name": list(names.values()), "panel_name": list(names.keys())}).to_csv(config.here(__file__, "stata_names.csv"), index=False)
io.write(out, config.out("08_estimation", "estimation_panel.parquet"), key=["focal", "fyear"])
st = out.rename(columns=names)
for c in st.columns:
    if str(st[c].dtype) in ("Int64", "Int32", "boolean"):
        st[c] = st[c].astype("float")
    elif st[c].dtype == bool:
        st[c] = st[c].astype(int)
    elif st[c].dtype == object:
        vals = set(st[c].dropna().unique().tolist())
        if vals <= {True, False}:
            st[c] = st[c].map({True: 1.0, False: 0.0})
        else:
            st[c] = st[c].astype("string").fillna("").astype(str)
    elif str(st[c].dtype) == "string":
        st[c] = st[c].fillna("").astype(str)
st.to_stata(config.out("08_estimation", "estimation_panel.dta"), write_index=False, version=118)
w = out[out["in_window"].eq(1)]
print(f"estimation panel: {len(out):,} rows, {out['focal'].nunique():,} firms, {out.shape[1]} columns; in window {len(w):,}")
for d in ("d_acqd", "d_acqs", "d_any", "d_hor", "dx_acqd", "d_wdrawn", "d_divest", "d_ind", "d_ind_cty"):
    print(f"  {d:10s} treated firm-years {int(w[d].sum()):>7,}  firms ever {w.loc[w[d].eq(1), 'focal'].nunique():>6,}")
print(f"frozen base: at_base1 for {int(w['at_base1'].notna().groupby(w['focal']).any().sum()):,} firms, at_base2 for {int(w['at_base2'].notna().groupby(w['focal']).any().sum()):,}; "
      f"rnd_at_base1 on {int(w['rnd_at_base1'].notna().sum()):,} window rows; own consolidation within 2 years: {int(w['own_consol_pm2'].sum()):,} rows")
