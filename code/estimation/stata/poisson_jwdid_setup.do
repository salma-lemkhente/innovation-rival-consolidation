version 18
clear all
set more off
set maxvar 32767
global B "<data folder>/08_estimation"
global O "$B/clean_runs_stage_b/etwfe5"
cap mkdir "$O"
use "$B/estimation_panel_b.dta", clear
xtset id fyear
gen double rd_m = rd_lvl_a
gen byte own3 = (own_consol == 1) | (L_own_consol == 1) | (L2_own_consol == 1)

foreach h in shock shock_mis {
    tempvar fy
    gen `fy' = fyear if `h' == 1
    bysort id: egen g_`h' = min(`fy')
    replace g_`h' = 0 if g_`h' == .
    drop `fy'
    tempvar og
    gen byte `og' = own3 if fyear == g_`h'
    bysort id: egen byte owng_`h' = max(`og')
    drop `og'
}
keep if in_window == 1 & rd_m != .
foreach x in ln_at ln_sales ln_age cash_ratio_w capx_at_w {
    bysort id (fyear): gen double b_`x' = `x'[1]
    qui sum b_`x'
    gen double z_`x' = (b_`x' - r(mean)) / r(sd)
}
global ZX "z_ln_at z_ln_sales z_ln_age z_cash_ratio_w z_capx_at_w"

cap program drop export_t
program define export_t
    syntax , file(string) tag(string)
    tempname T
    matrix `T' = r(table)'
    local names : rownames `T'
    preserve
    clear
    qui svmat double `T', names(col)
    gen str60 term = ""
    local i 1
    foreach n of local names {
        qui replace term = "`n'" in `i'
        local ++i
    }
    gen name = "`tag'"
    export delimited using "`file'", replace
    restore
end

cap program drop cell5
program define cell5
    syntax , name(string) d(varname) g(varname) [norec cov excl(varname) never window(string) trend leads bjs xgt xtonly]
    di as text _n(2) "==== etwfe5_`name'"
    cap confirm file "$O/`name'_info.csv"
    if !_rc {
        di as text " (already estimated, skipped)"
        exit
    }
    cap drop trd_*
    cap drop ld1 ld2 ld3 ld4 ld5
    local ex = cond("`norec'" != "", "exovar(norec_ca)", "")
    local xv = cond("`cov'" != "", "b_ln_at b_ln_sales b_ln_age b_cash_ratio_w b_capx_at_w", "")
    local cond "`d' != ."
    if "`excl'" != "" local cond "`d' != . & !(`g' > 0 & `excl' == 1) & !((`g' == 0 | fyear < `g') & own3 == 1)"
    local trs ""
    if "`trend'" != "" {
        qui levelsof `g' if `g' > 0 & `cond', local(gl)
        foreach c of local gl {
            qui gen double trd_`c' = (`g' == `c') * (fyear - 2003)
            local trs "`trs' trd_`c'"
        }
    }
    local lds ""
    local k0 = cond("`bjs'" != "", 1, 2)
    if "`leads'`bjs'" != "" {
        forvalues k = `k0'/5 {
            qui gen byte ld`k' = (`g' > 0 & fyear == `g' - `k')
            local lds "`lds' ld`k'"
        }
    }
    local xopt ""
    local xlev ""
    if "`xgt'" != "" {
        local xlev "$ZX"
        local xopt "xgvar($ZX) xtvar($ZX)"
    }
    if "`xtonly'" != "" {
        local xlev "$ZX"
        local xopt "xtvar($ZX)"
    }
    local exl = trim("`=cond("`norec'" != "", "norec_ca", "")' `trs' `lds' `xlev'")
    if "`exl'" != "" local ex "exovar(`exl')"
    cap noisily jwdid rd_m `xv' if `cond', ivar(id) tvar(fyear) gvar(`g') method(poisson) cluster(clus) `ex' `xopt' `never'
    if _rc {
        di as error "etwfe5_`name' failed, rc=" _rc
        exit
    }
    local N = e(N)
    local NC = e(N_clust)
    local conv = e(converged)
    local ptr = .
    local ntr = 0
    if "`trend'" != "" {
        cap testparm `trs'
        if !_rc {
            local ptr = r(p)
            local ntr = r(df)
        }
    }
    local plead = .
    forvalues k = 1/5 {
        local lb`k' = .
        local ls`k' = .
    }
    if "`leads'`bjs'" != "" {
        cap testparm `lds'
        if !_rc local plead = r(p)
        forvalues k = `k0'/5 {
            cap local lb`k' = _b[ld`k']
            cap local ls`k' = _se[ld`k']
        }
    }
    cap noisily estat simple
    if !_rc export_t, file("$O/`name'_simple.csv") tag("`name'")
    cap noisily estat simple, predict(xb)
    if !_rc export_t, file("$O/`name'_simplexb.csv") tag("`name'")
    local pp = .
    if "`never'" != "" {
        local win = cond("`window'" != "", "window(`window')", "")
        cap noisily estat event, predict(xb) pretrend `win'
        if !_rc {
            local pp = r(pre_p)
            export_t, file("$O/`name'_eventxb.csv") tag("`name'")
        }
    }
    else {
        cap noisily estat event, predict(xb)
        if !_rc export_t, file("$O/`name'_eventxb.csv") tag("`name'")
    }
    preserve
    clear
    qui set obs 1
    gen name = "`name'"
    gen long N = `N'
    gen long N_clust = `NC'
    gen converged = `conv'
    gen double p_pretrend = `pp'
    gen double p_trendtest = `ptr'
    gen int trend_df = `ntr'
    gen double p_leads = `plead'
    forvalues k = 1/5 {
        gen double lead`k'_b = `lb`k''
        gen double lead`k'_se = `ls`k''
    }
    export delimited using "$O/`name'_info.csv", replace
    restore
    cap drop trd_*
    cap drop ld1 ld2 ld3 ld4 ld5
end
