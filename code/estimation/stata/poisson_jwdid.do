do "<code folder>/estimation/stata/poisson_jwdid_setup.do"

* zero: years without a rival on record coded as no deal; mis: those years excluded
* xgt: baseline covariates by cohort and by year; excl: own deals excluded; bjs: leads test; never_w: never-exposed comparison, window -5 to 5
cell5, name(zero_base) d(shock) g(g_shock) norec
cell5, name(zero_xgt) d(shock) g(g_shock) norec xgt
cell5, name(zero_xgt_excl) d(shock) g(g_shock) norec xgt excl(owng_shock)
cell5, name(mis_base) d(shock_mis) g(g_shock_mis)
cell5, name(mis_xgt) d(shock_mis) g(g_shock_mis) xgt
cell5, name(mis_xgt_excl) d(shock_mis) g(g_shock_mis) xgt excl(owng_shock_mis)
cell5, name(zero_bjs) d(shock) g(g_shock) norec bjs
cell5, name(zero_xgt_bjs) d(shock) g(g_shock) norec xgt bjs
cell5, name(zero_xgt_excl_bjs) d(shock) g(g_shock) norec xgt excl(owng_shock) bjs
cell5, name(mis_bjs) d(shock_mis) g(g_shock_mis) bjs
cell5, name(mis_xgt_bjs) d(shock_mis) g(g_shock_mis) xgt bjs
cell5, name(mis_xgt_excl_bjs) d(shock_mis) g(g_shock_mis) xgt excl(owng_shock_mis) bjs
cell5, name(zero_never_w) d(shock) g(g_shock) norec never window(-5 5)
cell5, name(zero_xgt_never_w) d(shock) g(g_shock) norec xgt never window(-5 5)
cell5, name(zero_xgt_excl_never_w) d(shock) g(g_shock) norec xgt excl(owng_shock) never window(-5 5)
cell5, name(mis_never_w) d(shock_mis) g(g_shock_mis) never window(-5 5)
cell5, name(mis_xgt_never_w) d(shock_mis) g(g_shock_mis) xgt never window(-5 5)
cell5, name(mis_xgt_excl_never_w) d(shock_mis) g(g_shock_mis) xgt excl(owng_shock_mis) never window(-5 5)
