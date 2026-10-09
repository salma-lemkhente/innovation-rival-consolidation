import numpy as np
import pandas as pd

from projlib import config

OUTCOMES = {"ihs_rnd": "asinh(R&D), missing R&D kept", "ln1_rnd": "log(1 + R&D), missing R&D kept", "rnd_lat_w": "R&D over lagged assets (winsorised), missing kept",
            "ihs_rnd0": "asinh(R&D) with missing R&D set to zero (with rd_missing, the R&D reporting dummy)",
            "rnd_lsl_w": "R&D over lagged sales (winsorised), missing kept", "ihs_fam": "asinh(patent families originated), firms with a validated name",
            "rd_lvl": "R&D in dollars, missing kept (the Poisson outcome)", "fam_cnt": "patent families originated, the count, firms with a validated name"}

p = pd.read_parquet(config.out("08_estimation", "estimation_panel_v3.parquet"))
e = pd.read_parquet(config.out("07_panel", "v2_exposure_calendar.parquet"), columns=["focal", "fyear", "rival_ents", "rival_share", "mutual_share", "undir_share", "mutual_ents"])
p = p.merge(e, on=["focal", "fyear"], how="left").sort_values(["focal", "fyear"]).reset_index(drop=True)
fam = pd.read_parquet(config.out("07_panel", "outcomes_focal_fyear.parquet"), columns=["focal", "fyear", "families_originated", "patents_missing"])
p = p.merge(fam, on=["focal", "fyear"], how="left")
p["ihs_fam"] = np.arcsinh(p["families_originated"]).where(p["patents_missing"].eq(False))
p["rd_lvl"] = p["rd"]
p["fam_cnt"] = p["families_originated"].where(p["patents_missing"].eq(False))
p = p[p["covered"].eq(1)].copy()
p["share"] = p["rival_share"].fillna(0.0)
p["nrivd"] = p["rival_ents"].fillna(0).astype(float)
p["shock"] = p["share"].gt(0).astype(int)
first_riv = p["fyear"].where(pd.to_numeric(p["nriv_ca"], errors="coerce").gt(0)).groupby(p["focal"]).transform("min")
p["rc"] = ((p["fyear"] >= first_riv)).astype(int)
p["share_r"] = p["share"].where(p["rc"].eq(1)); p["shock_r"] = p["shock"].where(p["rc"].eq(1))
p["share_m"] = p["mutual_share"].fillna(0.0)
p["share_u"] = p["undir_share"].fillna(0.0)
p["nrivd_m"] = p["mutual_ents"].fillna(0).astype(float)
p["nrivd_r"] = p["nrivd"].where(p["rc"].eq(1)); p["share_m_r"] = p["share_m"].where(p["rc"].eq(1))
live = p["rival_share"].notna()
p["share_mis"] = p["share"].where(live); p["nrivd_mis"] = p["nrivd"].where(live); p["shock_mis"] = p["shock"].where(live)


def lag(v, k):
    s = p[["focal", "fyear", v]].copy(); s["fyear"] = s["fyear"] + k
    return p[["focal", "fyear"]].merge(s, on=["focal", "fyear"], how="left")[v].to_numpy()


for v in ("ln_at", "ln_sales", "ln_age", "cash_ratio_w", "capx_at_w", "nriv_ca", "norec_ca", "rd_missing"):
    p[f"L_{v}"] = lag(v, 1)
for v in ("ln_at", "ln_sales"):
    p[f"LD_{v}"] = lag(v, 1) - lag(v, 2)
INTER = ["nam_acqd_ca", "nam_acqs_ca", "rr_acqd_ca", "rr_acqs_ca", "rrno_ca", "sup_acqd_ca", "sup_acqs_ca", "par_acqd_ca", "par_acqs_ca", "wave_w"]
for v in INTER:
    p[f"L_{v}"] = lag(v, 1); p[f"L2_{v}"] = lag(v, 2)
CONTROLS = " ".join([f"L_{v}" for v in ("ln_at", "ln_sales", "ln_age", "cash_ratio_w", "capx_at_w", "nriv_ca")] + ["LD_ln_at", "LD_ln_sales"] + [f"{v} L_{v} L2_{v}" for v in INTER])

p["s5"] = sum(pd.Series(lag("share", k)).fillna(0).to_numpy() for k in range(1, 6))
p["sall"] = p.groupby("focal")["share"].cumsum() - p["share"]
T1, T2 = 0.533, 1.233
p["bin5"] = np.select([p["s5"].eq(0), p["s5"].le(T1), p["s5"].le(T2)], [0, 1, 2], 3)
w = p["in_window"].eq(1) & p["own_accounts"].eq(1)
pos = p.loc[w & p["sall"].gt(0), "sall"]; A1, A2 = float(pos.quantile(1 / 3)), float(pos.quantile(2 / 3))
p["binall"] = np.select([p["sall"].eq(0), p["sall"].le(A1), p["sall"].le(A2)], [0, 1, 2], 3)
p["y_ok"] = (w & p["own_consol_post2"].eq(0)).astype(int)
p["clus"] = p.groupby("focal")["unit_id"].transform("first")
for y in OUTCOMES:
    p[f"{y}_b"] = p[y].where(p["y_ok"].eq(1))
    p[f"{y}_k"] = p[y].where(w)
    p[f"{y}_a"] = p[y].where(p["own_accounts"].eq(1))
p["own_consol"] = p["own_consol"].fillna(0).astype(float)
p["L_own_consol"] = lag("own_consol", 1); p["L2_own_consol"] = lag("own_consol", 2)
keep = ["focal", "fyear", "id", "unit_id", "clus", "sic2", "share", "nrivd", "shock", "share_m", "share_u", "nrivd_m", "s5", "sall", "bin5", "binall", "y_ok", "own_consol", "L_own_consol", "L2_own_consol"] + [f"{y}_b" for y in OUTCOMES] + [f"{y}_k" for y in OUTCOMES] + [f"{y}_a" for y in OUTCOMES] + ["in_window", "rc", "share_r", "shock_r", "nrivd_r", "share_m_r", "share_mis", "nrivd_mis", "shock_mis", "ln_at", "ln_sales", "ln_age", "cash_ratio_w", "capx_at_w", "norec_ca", "L_norec_ca", "rd_missing", "L_rd_missing", "nriv_ca", "wave_w", "L_wave_w", "L2_wave_w"] + CONTROLS.split()
keep = list(dict.fromkeys(keep))
st = p[keep].copy()
for c in st.columns:
    if str(st[c].dtype) in ("Int64", "Int32", "boolean"):
        st[c] = st[c].astype("float")
    elif st[c].dtype == bool:
        st[c] = st[c].astype(int)
    elif st[c].dtype == object or str(st[c].dtype) == "string":
        st[c] = st[c].astype("string").fillna("").astype(str)
st.to_stata(config.out("08_estimation", "estimation_panel_b.dta"), write_index=False, version=118)
print(f"stage-B panel: {len(st):,} covered rows, {st['focal'].nunique():,} firms; outcome rows {int(p['y_ok'].sum()):,}; whole-history cuts {A1:.2f} / {A2:.2f}; "
      f"shocked rows {int(p['shock'].sum()):,}; share among shocked: median {p.loc[p['shock'].eq(1), 'share'].median():.2f}")
