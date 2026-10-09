version 18
clear all
set more off
global B "<data folder>/08_estimation"
global O "$B/clean_runs_ladders/balance"
cap mkdir "$O"
use "$B/estimation_panel_ladders.dta", clear
xtset id fyear

local base "L.ln_at L.ln_sales L.ln_age L.cash_ratio_w L.capx_at_w L.D.ln_at L.D.ln_sales"
local own "own_consol L.own_consol L2.own_consol"
local c_main "`base' L.nriv_ca norec_ca L.norec_ca `own'"
local c_od "`base' L.od_n od_norec L.od_norec `own' na_d L.na_d L2.na_d mu_d L.mu_d L2.mu_d"
local c_mut "`base' L.m_n_ca m_norec_ca L.m_norec_ca `own' nm_d L.nm_d L2.nm_d na_d L.na_d L2.na_d"
local c_on "`base' L.on_n on_norec L.on_norec `own' od_c L.od_c L2.od_c mu_d L.mu_d L2.mu_d"
local t_main "sw_d"
local t_od "sw_od"
local t_mut "sw_m"
local t_on "sw_on"
local k_main "s5_d == 0 & any_ca == 0"
local k_od "s5_od == 0 & od_d == 0"
local k_mut "s5_m == 0 & m_any_ca == 0"
local k_on "s5_on == 0 & on_t == 0"

tempname M
postfile `M' str10 y str6 design str5 outcome double b se p mean0 n events using "$O/attrition.dta", replace
foreach y in ln_rnd ihs_rnd rnd_lat_w {
    cap drop bal acc5
    gen byte bal = F.`y' != . & F2.`y' != . & F3.`y' != . & F4.`y' != . & F5.`y' != .
    gen byte acc5 = F5.own_accounts == 1
    foreach d in main od mut on {
        local tr "`t_`d''"
        cap drop smp
        gen byte smp = in_window == 1 & own_accounts == 1 & covered == 1 & `y' != . & L.`y' != . & L2.`y' != . & L2.ln_at != . & L2.ln_sales != . ///
            & (`tr' == 1 | (`k_`d''))
        foreach o in bal acc5 {
            qui reghdfe `o' `tr' L.D.`y' `c_`d'' if smp, absorb(sic2_year cty_year) vce(cluster unit_id)
            local b = _b[`tr']
            local se = _se[`tr']
            local p = 2 * ttail(e(df_r), abs(`b' / `se'))
            qui sum `o' if e(sample) & `tr' == 0
            local m0 = r(mean)
            qui count if e(sample) & `tr' == 1
            local ev = r(N)
            post `M' ("`y'") ("`d'") ("`o'") (`b') (`se') (`p') (`m0') (e(N)) (`ev')
            di as text "`y' `d' `o': b = " %7.4f `b' " (" %6.4f `se' "), comparison mean " %5.3f `m0' ", events " `ev'
        }
    }
}
postclose `M'
use "$O/attrition.dta", clear
export delimited using "$O/attrition.csv", replace
