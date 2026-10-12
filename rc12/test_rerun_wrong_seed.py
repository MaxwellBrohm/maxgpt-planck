"""rerun_tie.py --wrong-seed and --stage report (rc12/notes.txt STEP 14S; cuts audit S13, optional; DECISIONS_LEVEL_R
decision 3 (a)). Fake dev runs (fake:SEEDED, whose sampled replies depend on the seed, plain, greedy / 1 / 2) built
with dev_batch.py in a temp dir, then the three rerun_tie stages run directly with a stub margin. No model.
  python3 -B test_rerun_wrong_seed.py      (exit 1 on any failure; mutants: mutation_rerun_ws.py)"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FAILS = []


def check(ok, msg):
    if not ok:
        FAILS.append(msg)
        print("FAIL", msg)


def stage(root, d, name, extra=()):
    cmd = [sys.executable, "-B", "rerun_tie.py", "--stage", name, "--dir", d, "--model", "SEEDED", "--render", "plain"]
    if name == "replay":
        cmd += ["--root", root, "--rerun", "3"]
    if name == "margins":
        cmd += ["--tie-margin-stub", "0.7"]
    return subprocess.run(cmd + list(extra), cwd=HERE, capture_output=True, text=True, timeout=300)


def three(root, extra=()):
    d = tempfile.mkdtemp(prefix="rc12_ws_")
    outs = [stage(root, d, s, extra if s == "replay" else ()) for s in ("replay", "margins", "report")]
    rep = json.load(open(os.path.join(d, "replay.json"))) if os.path.exists(os.path.join(d, "replay.json")) else None
    shutil.rmtree(d, ignore_errors=True)
    return outs, rep


def main():
    base = tempfile.mkdtemp(prefix="rc12_wsruns_")
    try:
        eng = os.path.join(base, "engines.json")
        json.dump({"models": {"SEEDED": {"engine": "fake", "dtype": "float32"}}}, open(eng, "w"))
        p = subprocess.run([sys.executable, "-B", "dev_batch.py", "--responder", "fake:SEEDED", "--render", "plain",
                            "--seeds", "greedy,1,2", "--root", base, "--engines", eng],
                           cwd=HERE, capture_output=True, text=True, timeout=300)
        if p.returncode != 0:
            print(p.stdout[-1500:], p.stderr[-1500:])
            check(False, "dev_batch failed")
            return 1
        (rp, mg, rt), rep = three(base, ["--wrong-seed"])
        check(rp.returncode == 0 and mg.returncode == 0 and rt.returncode == 0,
              f"stages exit {rp.returncode} {mg.returncode} {rt.returncode}: {rt.stdout[-300:]}{rt.stderr[-300:]}")
        tags = [c["tag"] for c in rep["comparisons"]] if rep else []
        check(tags[:2] == ["R1 seed 1 lockstep replay vs stored", "R1w seed 101 (wrong seed) replay vs stored seed 1"],
              f"comparisons {tags}")
        ws = next((c for c in (rep or {}).get("comparisons", []) if c["tag"].startswith("R1w")), None)
        check(ws is not None and ws["conversations"] == 3 and ws["identical"] < 3 and len(ws["divergences"]) >= 1,
              f"the wrong seed must diverge: {ws and {k: ws[k] for k in ('conversations', 'identical')}}")
        right = next((c for c in (rep or {}).get("comparisons", []) if c["tag"].startswith("R1 seed")), None)
        check(right is not None and right["identical"] == 3, f"the right seed replays exactly on fake:SEEDED {right}")
        if ws and ws["divergences"]:
            d0 = ws["divergences"][0]
            shared = len(os.path.commonprefix([d0["a"], d0["b"]]))
            check(0 < shared < max(len(d0["a"]), len(d0["b"])), f"a divergence shares a strict prefix {d0}")
        lines = rt.stdout.splitlines()
        w = [x for x in lines if x.startswith("INFO R1w seed 101 (wrong seed) replay vs stored seed 1 (reported, not "
                                                "gated):")]
        check(len(w) == 1 and re.search(r"t\d+ shared \d+ chars margin 0\.700", w[0] if w else ""),
              f"report has no R1w INFO line with shared prefixes: {lines}")
        check(not any(x.startswith("FAIL R1w") or x.startswith("PASS R1w") for x in lines), "R1w gated")
        check(sum(x.startswith("PASS R2") for x in lines) == 2 and "INFO R1 seed 101 vs seed 1" in rt.stdout,
              f"the R2 lines and the turn-1 INFO line must stay: {lines}")
        (rp, mg, rt), rep = three(base)
        tags = [c["tag"] for c in rep["comparisons"]] if rep else []
        check(rt.returncode == 0 and not any(t.startswith("R1w") for t in tags) and "R1w" not in rt.stdout,
              f"without --wrong-seed nothing changes: {tags}")
        check(len(tags) == 3, f"default comparisons {tags}")
        p = subprocess.run([sys.executable, "-B", "verify_dev_runs.py", "--root", base, "--model", "SEEDED",
                            "--render", "plain", "--no-prompt", "--engines", eng, "--rerun", "3"],
                           cwd=HERE, capture_output=True, text=True, timeout=300)
        check("R1w" not in p.stdout and p.stdout.count("PASS R") == 2, f"verify_dev_runs unchanged {p.stdout[-400:]}")
        d = tempfile.mkdtemp(prefix="rc12_wsrep_")                   # stage report on a failing R2: exit 1
        div = dict(id="x", turn=2, history=[], a="It is blue.", b="It is red.", stop_a="eos", stop_b="eos")
        json.dump(dict(engine="fake", hf={}, info=["i"], comparisons=[dict(tag="R2 greedy lockstep vs sequential",
                  conversations=1, identical=0, divergences=[div])]), open(os.path.join(d, "replay.json"), "w"))
        json.dump([0.9], open(os.path.join(d, "margins.json"), "w"))
        p = stage(base, d, "report")
        check(p.returncode == 1 and "FAIL R2 greedy lockstep vs sequential" in p.stdout, f"report FAIL {p.stdout}")
        shutil.rmtree(d, ignore_errors=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    print(f"test_rerun_wrong_seed: {len(FAILS)} failures")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
