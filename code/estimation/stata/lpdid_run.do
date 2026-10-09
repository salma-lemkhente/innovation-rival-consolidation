do "`1'"
local K = cond("`3'"=="", 1, real("`3'"))
local k = cond("`2'"=="", 1, real("`2'"))
do "$DO/lpdid_setup.do"
cap mkdir "$OUT/runs"
cap mkdir "$OUT/valid"

* the events counted at h = 0 and at the last horizon: the outcome observed at t-1 and t+h, the event variable observed and
* zero from t-lag to t-1 and from t+1 to t+h
cap program drop count_events
program define count_events, rclass
    syntax varname, treat(varname) lag(integer) post(integer) sample(string)
    tempvar clean
    qui gen byte `clean' = `treat'==1 & (`sample')
    forvalues j = 1/`lag' {
        qui replace `clean' = 0 if L`j'.`treat'==1 | L`j'.`treat'==.
    }
    foreach h in 0 `post' {
        tempvar ok
        qui gen byte `ok' = `clean'==1 & F`h'.`varlist'!=. & L.`varlist'!=.
        forvalues j = 1/`h' {
            qui replace `ok' = 0 if F`j'.`treat'==1 | F`j'.`treat'==.
        }
        qui count if `ok'==1
        return scalar sw`h' = r(N)
        drop `ok'
    }
end

frame create grid
frame grid: import delimited "$OUT/$GRID", clear varnames(1) stringcols(_all) bindquote(strict) maxquotedrows(unlimited)
frame grid: qui count
local N = r(N)
local done = 0
local run = 0
local failed = 0
forvalues i = 1/`N' {
    if mod(`i', `K') != `k' - 1 continue
    frame grid: local run_id = run_id[`i']
    local V "$OUT/valid/`run_id'.csv"
    cap confirm file "`V'"
    if !_rc {
        local ++done
        continue
    }
    frame grid: local y = y[`i']
    frame grid: local treat = treat[`i']
    frame grid: local sample = sample[`i']
    frame grid: local options = options[`i']
    frame grid: local controls = controls[`i']
    frame grid: local absorb = absorb[`i']
    frame grid: local cluster = cluster[`i']
    frame grid: local pre = pre[`i']
    frame grid: local post = post[`i']
    frame grid: local L = stab_lag[`i']
    frame grid: local title = title[`i']
    local ++run
    di as text _n "==== `run_id' | `title'"
    cap noisily count_events `y', treat(`treat') lag(`L') post(`post') sample(`sample')
    local sw0 = cond(_rc, ., r(sw0))
    local swH = cond(_rc, ., r(sw`post'))
    local ctrlopt = cond("`controls'"=="", "", "controls(`controls')")
    * the LP-DiD of one grid row
    cap noisily lpdid `y' if `sample', unit(id) time(fyear) treat(`treat') pre_window(`pre') post_window(`post') `options' ///
        `ctrlopt' absorb(`absorb') cluster(`cluster') nograph
    local rc = _rc
    local cg ""
    local tg ""
    local obs0 = .
    local obsH = .
    if `rc' == 0 {
        local cg = e(control_group)
        local tg = e(treated_group)
        tempname R
        matrix `R' = e(results)
        local rn : rownames `R'
        local j = 1
        foreach r of local rn {
            if "`r'"=="tau0" local obs0 = `R'[`j', 7]
            if "`r'"=="tau`post'" local obsH = `R'[`j', 7]
            local ++j
        }
        export_lpdid, file("$OUT/runs/`run_id'.csv") tag("`run_id'")
    }
    else {
        di as error "lpdid failed: `run_id' rc=`rc'"
        local ++failed
    }
    preserve
    clear
    qui set obs 1
    gen str10 run_id = "`run_id'"
    gen rc = `rc'
    gen str80 control_group = "`cg'"
    gen str80 treated_group = "`tg'"
    gen switches_h0 = `sw0'
    gen switches_hpost = `swH'
    gen obs_h0 = `obs0'
    gen obs_hpost = `obsH'
    gen post = `post'
    export delimited "`V'", replace
    restore
}
di as result "lpdid_run shard `k' of `K' done: `run' estimated (`failed' failed), `done' already present"
