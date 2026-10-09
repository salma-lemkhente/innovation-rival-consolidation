import pathlib
import subprocess
import sys
import time

from projlib import config

here = pathlib.Path(__file__).resolve().parent
NAME = sys.argv[1]
do = here / "stata" / f"{NAME}.do"
out = config.out("08_estimation/clean_runs_stage_b", "x").parent
work = out / "logs" / NAME
work.mkdir(parents=True, exist_ok=True)
cmd = [config.CFG["tools"]["stata"], "/e", "do", str(do)]
print("running:", " ".join(cmd))
subprocess.run(cmd, cwd=str(work), check=False)
logs = sorted(work.glob("*.log"), key=lambda f: f.stat().st_mtime)
produced = logs[-1] if logs else work / "missing.log"
if produced.exists():
    for attempt in range(10):
        try:
            produced = produced.replace(out / "logs" / f"{NAME}.log"); break
        except PermissionError:
            time.sleep(6)
text = produced.read_text(encoding="utf-8", errors="replace") if produced.exists() else ""
lines = text.splitlines()
errors = [l for l in lines if l.strip().startswith("r(") and l.strip().endswith(");")]
print("\n".join(lines[-12:]))
if errors:
    print(f"Stata reported an error: {errors[-1]}")
print(f"done; log {produced}")
