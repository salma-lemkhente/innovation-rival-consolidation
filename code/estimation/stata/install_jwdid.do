version 18
cap net install jwdid, from("https://friosavila.github.io/stpackages") replace
if _rc {
    di as text "GitHub install failed, rc=" _rc ", trying SSC"
    ssc install jwdid, replace
}
cap which hdfe
if _rc ssc install hdfe, replace
which hdfe
which jwdid
which ppmlhdfe
di as result _n "jwdid install, done"
