version 18
clear all
set more off
global B "<data folder>/08_estimation"
global O "$B/clean_runs_ladders/balance"
cap mkdir "$O"
use "$B/estimation_panel_ladders.dta", clear
xtset id fyear

local c_main "L.ln_at L.ln_sales L.ln_age L.cash_ratio_w L.capx_at_w L.D.ln_at L.D.ln_sales L.nriv_ca norec_ca L.norec_ca own_consol L.own_consol L2.own_consol"
gen byte tgt15 = (F.own_tgt == 1) | (F2.own_tgt == 1) | (F3.own_tgt == 1) | (F4.own_tgt == 1) | (F5.own_tgt == 1)
gen byte tgt05 = (own_tgt == 1) | tgt15

tempname M
postfile `M' str10 y str8 sample str6 outcome double b se p mean0 n events using "$O/sellout.dta", replace
foreach y in ln_rnd ihs_rnd {
    foreach s in all small {
        cap drop smp
        gen byte smp = in_window == 1 & own_accounts == 1 & covered == 1 & fyear <= 2017 & `y' != . & L.`y' != . & L2.`y' != . & L2.ln_at != . & L2.ln_sales != . ///
            & (sw_d == 1 | (s5_d == 0 & any_ca == 0))
        if "`s'" == "small" replace smp = smp & ter_at == 1
        foreach o in tgt15 tgt05 {
            qui reghdfe `o' sw_d L.D.`y' `c_main' if smp, absorb(sic2_year cty_year) vce(cluster unit_id)
            local b = _b[sw_d]
            local se = _se[sw_d]
            local p = 2 * ttail(e(df_r), abs(`b' / `se'))
            qui sum `o' if e(sample) & sw_d == 0
            local m0 = r(mean)
            qui count if e(sample) & sw_d == 1
            local ev = r(N)
            post `M' ("`y'") ("`s'") ("`o'") (`b') (`se') (`p') (`m0') (e(N)) (`ev')
            di as text "`y' `s' `o': b = " %7.4f `b' " (" %6.4f `se' "), comparison mean " %6.4f `m0' ", events " `ev' ", N " e(N)
        }
    }
}
postclose `M'
use "$O/sellout.dta", clear
export delimited using "$O/sellout.csv", replace
