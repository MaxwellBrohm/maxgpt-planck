"""RC-12 dev_batch.py tests without a model (notes STEP 9), in-process with fake:IDEAL_ALT (lockstep, PerConv).
  layout     <root>/<model>/<render>/<seed>/ and <seed>_owncf/ for every seed, each with DONE, meta.json,
             transcripts.jsonl and scores.jsonl; the main runs hold every record, the twins only OWN records
  rows       each run's transcripts.jsonl equals runner.run's own rows for the same records (sequential), the
             twins with own_cf
  skip       a rerun does nothing and rewrites nothing (mtimes unchanged); a run dir without DONE is refused
  partial    a stale <dir>.partial is deleted and the run redone
  gate       a vllm: or hfb: responder without parity.json, or with a failing verdict, exits before loading anything,
             unless engines.json names that engine for the model (then it goes past the gate); a model engines.json
             lists is refused on any other engine
Run: python3 -B test_dev_batch.py   (exit 1 on any failure)"""
import json
import os
import sys
import tempfile

import dev_batch as DB
import fakes_family as FF
import runner as R


def call(argv):
    old = sys.argv
    sys.argv = ["dev_batch.py"] + argv
    try:
        return DB.main()
    except SystemExit as e:
        return str(e.code)
    finally:
        sys.argv = old


def mtimes(root):
    return {os.path.join(d, f): os.path.getmtime(os.path.join(d, f)) for d, _, fs in os.walk(root) for f in fs}


def c_layout_rows(root, limit):
    fails = []
    recs, own = R.load(R.DEV, None, limit), R.load(R.DEV, ["OWN"], limit)
    for seed in (None, 1):
        for own_cf in (False, True):
            d = os.path.join(root, "IDEAL_ALT", "plain", DB.seed_name(seed) + ("_owncf" if own_cf else ""))
            if not all(os.path.exists(os.path.join(d, f)) for f in ("DONE", "meta.json", "transcripts.jsonl",
                                                                    "scores.jsonl")):
                fails.append(f"layout: {d} incomplete")
                continue
            want = R.run(own if own_cf else recs, FF.make("IDEAL_ALT"), "plain", [seed], None, None, "IDEAL_ALT", 0,
                         own_cf)
            got = open(os.path.join(d, "transcripts.jsonl")).read().splitlines()
            if got != [json.dumps(r) for r in want]:
                fails.append(f"rows: {d} differs from runner.run")
            meta = json.load(open(os.path.join(d, "meta.json")))
            if (meta["conversations"], meta["own_cf"], meta["seed"]) != (len(want), own_cf, seed):
                fails.append(f"layout: {d} meta {meta['conversations']} {meta['own_cf']} {meta['seed']}")
    return fails


def checks():
    try:
        return _checks()
    except Exception as e:  # noqa: BLE001  (a raising check is a failure line, not a crash)
        return [f"checks raised {type(e).__name__}: {str(e)[:150]}"]


def _checks():
    fails, limit = [], 40
    with tempfile.TemporaryDirectory() as root:
        base = ["--responder", "fake:IDEAL_ALT", "--render", "plain", "--root", root, "--limit", str(limit)]
        rc = call(base + ["--seeds", "greedy,1"])
        if rc != 0:
            fails.append(f"layout: first run returned {rc}")
        fails += c_layout_rows(root, limit)
        before = mtimes(root)
        if call(base + ["--seeds", "greedy,1"]) != 0 or mtimes(root) != before:
            fails.append("skip: a rerun changed files")
        stale = os.path.join(root, "IDEAL_ALT", "plain", "2.partial")
        os.makedirs(stale)
        open(os.path.join(stale, "junk"), "w").close()
        call(base + ["--seeds", "2"])
        done2 = os.path.join(root, "IDEAL_ALT", "plain", "2", "DONE")
        if os.path.exists(stale) or not os.path.exists(done2) or os.path.exists(os.path.join(stale[:-8], "junk")):
            fails.append("partial: a stale .partial was not replaced by a clean finished run")
        os.remove(done2)
        rc = call(base + ["--seeds", "2"])
        if "without DONE" not in str(rc):
            fails.append(f"skip: a run dir without DONE was not refused ({rc})")
        rc = call(["--responder", "vllm:org/NoModel", "--render", "template", "--root", root])
        if "parity gate" not in str(rc):
            fails.append(f"gate: vllm without parity.json not refused ({rc})")
        pdir = os.path.join(root, "NoModel", "parity")
        os.makedirs(pdir)
        json.dump({"verdict": "DIFFERS"}, open(os.path.join(pdir, "parity.json"), "w"))
        rc = call(["--responder", "vllm:org/NoModel", "--render", "template", "--root", root])
        if "parity gate" not in str(rc):
            fails.append(f"gate: vllm with verdict DIFFERS not refused ({rc})")
        eng = os.path.join(root, "engines.json")
        json.dump({"models": {"org/NoModel": {"engine": "vllm"}}}, open(eng, "w"))
        try:                                     # past the gate it goes on to load org/NoModel and fails there
            rc = call(["--responder", "vllm:org/NoModel", "--render", "template", "--root", root, "--engines", eng])
        except Exception as e:  # noqa: BLE001
            rc = f"went past the gate ({type(e).__name__})"
        if "went past the gate" not in str(rc):
            fails.append(f"gate: engines.json names vllm, a DIFFERS verdict still refused ({rc})")
        rc = call(["--responder", "hfb:org/NoModel", "--render", "template", "--root", root, "--hf-untested-ok",
                   "--engines", eng])
        if "engine gate" not in str(rc):
            fails.append(f"gate: engines.json names vllm, hfb not refused ({rc})")
        rc = call(["--responder", "hfb:org/NoModel", "--render", "template", "--root", root, "--hf-untested-ok"])
        if "parity gate" not in str(rc):
            fails.append(f"gate: hfb with verdict DIFFERS not refused ({rc})")
        rc = call(["--responder", "hfb:org/Other", "--render", "template", "--root", root, "--hf-untested-ok"])
        if "parity gate" not in str(rc):
            fails.append(f"gate: hfb without parity.json not refused ({rc})")
    return fails


def main():
    fails = checks()
    print("\n".join(f"FAIL {f}" for f in fails) or "ALL DEV_BATCH CHECKS PASS (fake:IDEAL_ALT, no model)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
