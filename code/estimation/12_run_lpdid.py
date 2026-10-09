import pathlib
import subprocess
import sys

from projlib import config

here = pathlib.Path(__file__).resolve().parent
do = here / "stata" / "lpdid_run.do"
PLACEBO = "--placebo" in sys.argv
sys.argv = [a for a in sys.argv if a != "--placebo"]
out = config.out("08_estimation/clean_runs_placebo" if PLACEBO else "08_estimation/clean_runs_ladders", "x").parent
(out / "logs").mkdir(parents=True, exist_ok=True)
suffix = "_".join(["lpdid_run", *sys.argv[1:]])
work = out / "logs" / suffix
work.mkdir(parents=True, exist_ok=True)
paths = work / "paths.do"
data = config.out("08_estimation", "estimation_panel_placebo.dta" if PLACEBO else "estimation_panel_ladders.dta").as_posix()
paths.write_text(f'global DATA "{data}"\nglobal OUT "{out.as_posix()}"\nglobal RUN "lpdid_run"\nglobal DO "{(here / "stata").as_posix()}"\n'
                 f'global GRID "{"grid_placebo.csv" if PLACEBO else "grid_ladders.csv"}"\n', encoding="utf-8")
cmd = [config.CFG["tools"]["stata"], "/e", "do", str(do), str(paths), *sys.argv[1:]]
print("running:", " ".join(cmd))
subprocess.run(cmd, cwd=str(work), check=False)
logs = sorted(work.glob("*.log"), key=lambda f: f.stat().st_mtime)
produced = logs[-1] if logs else work / "missing.log"
if produced.exists():
    import time
    for attempt in range(10):
        try:
            produced = produced.replace(out / "logs" / f"{suffix}.log"); break
        except PermissionError:
            time.sleep(6)
text = produced.read_text(encoding="utf-8", errors="replace") if produced.exists() else ""
lines = text.splitlines()
errors = [l for l in lines if l.strip().startswith("r(") and l.strip().endswith(");")]
print("\n".join(lines[-12:]))
if errors:
    sys.exit(f"Stata reported an error: {errors[-1]}  (log: {produced})")
print(f"done: {suffix}; log {produced}")
