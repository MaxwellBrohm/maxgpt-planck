"""Mutation check: break the harness on purpose and confirm the step-3 self-tests go red.

For each mutant in mutants.py: copy the harness .py files to a temp dir, apply the one
string replacement, run the suite there (CPU, 2 threads per worker), and record which
test files failed. A mutant is KILLED when pytest exits 1 (tests failed). Exit codes 2-5
(collection or usage errors) count as INVALID, never as killed. The unmutated copy is run
first and must be green. Also reports whether the STARTUP self-test (train.py's gate,
test_model_mask.py::test_selftest_passes_clean_model) caught each mask/causal mutant.
A mutant with a tests list (mutants.py; the screen flags' groups) runs those files instead of
the suite, and the baseline runs every file the selected mutants use.

  python mutation_check.py --group mask          # one group (keeps each call short)
  python mutation_check.py --id opt_wrong_sign   # one mutant
  python mutation_check.py --list
  --overfit adds test_s3_overfit.py (slower); --workers N (default 4)
Exit 0 only if every non-equivalent mutant is killed and every equivalent one survives.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from mutants import M

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = sorted(os.path.basename(p) for p in glob.glob(os.path.join(HERE, "test_s3_*.py")))
STARTUP = "test_model_mask.py"


def make_copy(mutant: dict | None) -> str:
    d = tempfile.mkdtemp(prefix="planck_mut_")
    for p in glob.glob(os.path.join(HERE, "*.py")):
        shutil.copy(p, d)
    if mutant is not None:
        path = os.path.join(d, mutant["file"])
        src = open(path).read()
        n = src.count(mutant["old"])
        if n != 1:
            shutil.rmtree(d)
            raise ValueError(f"{mutant['id']}: target string occurs {n} times in {mutant['file']}")
        open(path, "w").write(src.replace(mutant["old"], mutant["new"]))
    return d


def run_suite(d: str, files: list[str], timeout: int = 110) -> tuple[int, list[str], float]:
    env = {**os.environ, "PLANCK_TEST_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1"}
    args = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=no", "-rf", *files]
    t0 = time.time()
    try:
        r = subprocess.run(args, cwd=d, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return -9, ["TIMEOUT"], time.time() - t0
    failed = re.findall(r"^FAILED (\S+?)(?: - .*)?$", r.stdout, flags=re.M)
    if r.returncode not in (0, 1):
        failed = failed or [r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]]
    return r.returncode, failed, time.time() - t0


def check(mutant: dict, files: list[str]) -> dict:
    d = make_copy(mutant)
    try:
        code, failed, dt = run_suite(d, mutant.get("tests") or files)   # tests: a screen flag's own files
    finally:
        shutil.rmtree(d, ignore_errors=True)
    by_file = sorted({f.split("::")[0] for f in failed if "::" in f})
    startup = any(f.startswith(f"{STARTUP}::test_selftest_passes_clean_model") for f in failed)
    status = {0: "SURVIVED", 1: "killed"}.get(code, f"INVALID({code})")
    ok = (status == "SURVIVED") if mutant["equivalent"] else (status == "killed")
    return {**mutant, "status": status, "ok": ok, "n_failed": len(failed), "files": by_file,
            "startup_gate": startup, "seconds": round(dt, 1), "detail": failed[:1] if code not in (0, 1) else []}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group")
    ap.add_argument("--id")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    todo = [m for m in M if (not a.group or m["group"] == a.group) and (not a.id or m["id"] == a.id)]
    if a.list:
        for m in M:
            print(f"{m['group']:9s} {m['id']:34s} {m['file']:17s} {m['why']}")
        return 0
    files = SUITE + [STARTUP]
    if not a.overfit:
        files = [f for f in files if f != "test_s3_overfit.py"]
    for m in todo:                                   # fail fast on a stale target string
        shutil.rmtree(make_copy(m), ignore_errors=True)
    base = make_copy(None)
    base_files = list(dict.fromkeys(f for m in todo for f in (m.get("tests") or files))) or files
    try:
        code, failed, dt = run_suite(base, base_files)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    print(f"baseline (unmutated copy): exit {code}, {len(failed)} failed, {dt:.1f}s", flush=True)
    if code != 0:
        print("baseline is not green; stopping:", failed[:5])
        return 1
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(lambda m: check(m, files), todo))
    bad = 0
    for r in results:
        flag = "ok " if r["ok"] else "BAD"
        eq = " (equivalent, expected to survive)" if r["equivalent"] else ""
        gate = "  startup-gate:yes" if r["startup_gate"] else ""
        print(f"{flag} {r['status']:9s} {r['group']:9s} {r['id']:34s} {r['n_failed']:3d} failed "
              f"{r['seconds']:5.1f}s  {','.join(f.replace('test_', '').replace('.py', '') for f in r['files'])}"
              f"{gate}{eq} {r['detail']}", flush=True)
        bad += not r["ok"]
    print(f"{len(results) - bad}/{len(results)} mutants behaved as expected")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
