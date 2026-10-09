# Code

The estimation code of the paper and the code of its interactive graphs. The data are licensed and are not included.

Python 3.12 with pandas, numpy, pyarrow, networkx and tabulate (folium, branca, holoviews, panel and jinja2 for the graphs);
Stata 18 or later with lpdid 1.1.0, reghdfe 6.13.1, ftools 2.50.0, ppmlhdfe 2.3.3 and jwdid 2.201 (`estimation/stata/install_jwdid.do`).

Copy `config.example.toml` to `config.toml` and set the folder of the raw data and the Stata executable. In the do-files of
`estimation/stata`, replace `<data folder>` with the data folder of `config.toml` and `<code folder>` with this folder. Run the
scripts from this folder, with this folder on the Python path.

`estimation/`, in order:

1. `01` to `10` build the inputs of the estimation (`01`, `02` and `03` in this order, `07` before `08`).
2. `11_lpdid_specifications.py` writes the panel Stata reads and one row per LP-DiD estimate; `12_run_lpdid.py k K` runs part k of K
   of those rows (`stata/lpdid_run.do`).
3. Randomization inference: `13_placebo_draws.py`, `14_placebo_specifications.py`, then `12_run_lpdid.py --placebo k K`.
4. `15_poisson_panel.py`, then `16_run_stata.py poisson_jwdid`; `16_run_stata.py` also runs `balance`, `attrition` and `sellout`.
5. `17_rival_relative_size.py`.

`interactive-graphs/`: `firm_map.py` writes the map of the firms; `country_flows_and_industry_chord.py` writes the country flows and
the industry chord diagram (set the two folders at its top).
