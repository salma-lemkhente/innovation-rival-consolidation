version 18
clear all
set more off
set maxvar 32767
use "$DATA", clear
xtset id fyear

cap program drop export_lpdid
program define export_lpdid
    syntax , file(string) tag(string)
    tempname R P
    matrix `R' = e(results)
    local rn : rownames `R'
    preserve
    clear
    svmat double `R'
    rename (`R'1 `R'2 `R'3 `R'4 `R'5 `R'6 `R'7) (coefficient se t p ci_low ci_high obs)
    gen str8 horizon = ""
    local i = 1
    foreach r of local rn {
        replace horizon = "`r'" in `i'
        local ++i
    }
    gen str120 spec = "`tag'"
    order spec horizon
    cap confirm matrix e(pooled_results)
    if !_rc {
        matrix `P' = e(pooled_results)
        local pn : rownames `P'
        local n = _N
        set obs `=_N + rowsof(`P')'
        local i = 1
        foreach r of local pn {
            local row = `n' + `i'
            replace horizon = "pooled_`r'" in `row'
            replace spec = "`tag'" in `row'
            replace coefficient = `P'[`i', 1] in `row'
            replace se = `P'[`i', 2] in `row'
            replace t = `P'[`i', 3] in `row'
            replace p = `P'[`i', 4] in `row'
            replace ci_low = `P'[`i', 5] in `row'
            replace ci_high = `P'[`i', 6] in `row'
            replace obs = `P'[`i', 7] in `row'
            local ++i
        }
    }
    cap confirm file "`file'"
    if !_rc {
        tempfile new
        save `new'
        import delimited "`file'", clear varnames(1) stringcols(1 2)
        append using `new'
    }
    export delimited "`file'", replace
    restore
end
