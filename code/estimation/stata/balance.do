version 18
clear all
set more off
global B "<data folder>/08_estimation"
global O "$B/clean_runs_ladders/balance"
cap mkdir "$O"
use "$B/estimation_panel_ladders.dta", clear
xtset id fyear

foreach v in ln_at ln_sales ln_age ln_emp cash_ratio_w capx_at_w lev_w nriv_ca norec_ca {
    gen double b_`v' = L.`v'
}
gen double b_rnd_lvl = L.ln_rnd
gen double b_rnd_ihs = L.ihs_rnd
gen double b_drnd_ln = L.ln_rnd - L2.ln_rnd
gen double b_drnd_ihs = L.ihs_rnd - L2.ihs_rnd
gen double b_dln_at = L.ln_at - L2.ln_at
gen double b_dln_sales = L.ln_sales - L2.ln_sales

foreach y in ln_rnd ihs_rnd {
    preserve
    local rl = cond("`y'" == "ln_rnd", "b_rnd_lvl b_drnd_ln", "b_rnd_ihs b_drnd_ihs")
    gen byte ev = sw_d == 1
    gen byte keepme = in_window == 1 & own_accounts == 1 & covered == 1 & `y' != . & L.`y' != . & L2.`y' != . & L2.ln_at != . & L2.ln_sales != . ///
        & (sw_d == 1 | (s5_d == 0 & any_ca == 0)) & !missing(L.ln_at, L.ln_sales, L.ln_age, L.cash_ratio_w, L.capx_at_w, L.nriv_ca, norec_ca, L.norec_ca, own_consol, L.own_consol, L2.own_consol)
    keep if keepme
    local grow = cond("`y'" == "ln_rnd", "b_drnd_ln b_dln_at b_dln_sales", "b_drnd_ihs b_dln_at b_dln_sales")
    local chars "b_ln_at b_ln_sales b_ln_age b_ln_emp b_cash_ratio_w b_capx_at_w b_lev_w `rl' b_dln_at b_dln_sales b_nriv_ca b_norec_ca"
    tempname M
    postfile `M' str24 var double m1 m0 sd1 sd0 n1 n0 stdiff cdiff cse cp using "$O/balance_`y'.dta", replace
    foreach v of local chars {
        qui sum `v' if ev == 1
        local m1 = r(mean)
        local s1 = r(sd)
        local n1 = r(N)
        qui sum `v' if ev == 0
        local m0 = r(mean)
        local s0 = r(sd)
        local n0 = r(N)
        local sdf = (`m1' - `m0') / sqrt((`s1'^2 + `s0'^2) / 2)
        cap noisily reghdfe `v' ev, absorb(sic2_year cty_year) cluster(unit_id)
        if !_rc {
            local cd = _b[ev]
            local cs = _se[ev]
            local cp = 2 * ttail(e(df_r), abs(_b[ev] / _se[ev]))
        }
        else {
            local cd = .
            local cs = .
            local cp = .
        }
        post `M' ("`v'") (`m1') (`m0') (`s1') (`s0') (`n1') (`n0') (`sdf') (`cd') (`cs') (`cp')
    }
    postclose `M'
    cap noisily reghdfe ev `chars', absorb(sic2_year cty_year) cluster(unit_id)
    local F = .
    local pF = .
    local NN = .
    local r2 = .
    if !_rc {
        qui test `chars'
        local F = r(F)
        local pF = r(p)
        local NN = e(N)
        local r2 = e(r2_within)
        qui test `grow'
        local Fg = r(F)
        local pg = r(p)
    }
    else {
        local Fg = .
        local pg = .
    }
    cap noisily reghdfe ev `grow', absorb(sic2_year cty_year) cluster(unit_id)
    local Fo = .
    local po = .
    if !_rc {
        qui test `grow'
        local Fo = r(F)
        local po = r(p)
    }
    use "$O/balance_`y'.dta", clear
    export delimited using "$O/balance_`y'.csv", replace
    clear
    set obs 1
    gen double F = `F'
    gen double p = `pF'
    gen double N = `NN'
    gen double r2_within = `r2'
    gen double F_growth_in_joint = `Fg'
    gen double p_growth_in_joint = `pg'
    gen double F_growth_only = `Fo'
    gen double p_growth_only = `po'
    export delimited using "$O/joint_`y'.csv", replace
    restore
}
di as result "balance: done"
