"""Mutation check for tests/test_preflight.py: each mutant is one deliberate bug in a scratch copy of preflight.py;
killed = the test file fails (pytest exit 1); a pattern not found exactly once is INVALID. PC only (torch).
  ~/planck/venv/bin/python experiments/E2_lr_transfer/tests/mutation_preflight.py [--only a,b]   (about 10 min:
  run it detached)
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
E2 = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(E2))
M = [
    ("no_tokenizer_hash", '(("tokenizer", code), ("tokenizer_manifest", code)', '(("tokenizer_manifest", code)'),
    ("no_shard_manifest_hash", ', ("shard_manifest", planck)):', '):'),
    ("no_eval_manifest_hash", 'if not os.path.isfile(evm) or sha256_file(evm) != pv.get("evalset_manifest_sha256"):',
     'if not os.path.isfile(evm):'),
    ("no_evalset_field", 'elif json.load(open(evm))["evalset_sha256"] != pv.get("evalset_sha256"):', 'elif False:'),
    ("source_order_ignored", "if want != have:", "if sorted(want) != sorted(have):"),
    ("ids_unchecked", '        if (d.get("pad_id"), chat.get("end_id")', '        if False and (d.get("pad_id"), chat.get("end_id")'),
    ("vocab_unchecked", 'if cfg["model"]["vocab_size"] != tk["vocab"]:', "if False:"),
    ("share_unchecked", 'if not float(s.get("share", 0)) > 0:', 'if float(s.get("share", 0)) < 0:'),
    ("glob_unchecked", "if n == 0:", "if n < 0:"),
    ("read_but_unset_allowed", "if k in reads and not has:", "if False:"),
    ("set_but_unread_allowed", "if has and k not in reads and", "if False and has and k not in reads and"),
    ("off_value_inverted", 'v != OFF.get(k, "<none>"):', 'v == OFF.get(k, "<none>"):'),
    ("auto_allowed", 'if strict and v == "auto":', "if False:"),
    ("engine_fixed_unchecked", 'if strict and not cfg.get("engine_fixed"):', "if False:"),
    ("data_seed_unchecked", 'if "data_seed" in cfg and "data_seed" not in reads:', "if False:"),
    ("scorer_unchecked", "if {k: ev.get(k) for k in SCORER} != SCORER:", "if False:"),
    ("trunk_last_point_off_by_one", "pts[-1] > int(total) - 1", "pts[-1] > int(total)"),
    ("warmup_unchecked", 'if sc.get("warmup_steps") is None:', "if False:"),
    ("branch_lr_unchecked", 'if float(oc.get(k, -1)) != float(po.get(k, -2)):', "if False:"),
    ("branch_total_unchecked", 'if int(ck["step"]) + int(sc.get("decay_steps", -1)) != int(total):', "if False:"),
    ("branch_seed_unchecked", 'if ck["config"].get("seed") != cfg.get("seed"):', "if False:"),
    ("branch_shape_unchecked", 'if ck["model_cfg"] != mcfg.to_dict():', "if False:"),
    ("prereg_never_strict", "pr = runio.check_prereg(cfg_path, strict)", "pr = runio.check_prereg(cfg_path, False)"),
    ("missing_init_allowed", "            if init_check:\n", "            if False:\n"),
]


def run_one(name, old, new):
    with tempfile.TemporaryDirectory() as d:
        root = os.path.join(d, "r")
        for sub in ("harness", "tokenizer", "data_prep", "experiments"):
            shutil.copytree(os.path.join(REPO, sub), os.path.join(root, sub),
                            ignore=shutil.ignore_patterns("__pycache__"))
        p = os.path.join(root, "experiments", "E2_lr_transfer", "preflight.py")
        src = open(p).read()
        if src.count(old) != 1:
            return f"INVALID (pattern found {src.count(old)} times)"
        open(p, "w").write(src.replace(old, new))
        r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", os.path.join(
            root, "experiments", "E2_lr_transfer", "tests", "test_preflight.py")], capture_output=True, text=True,
            timeout=600)
        last = (r.stdout.strip().splitlines() or ["?"])[-1]
        return {1: "killed", 0: "SURVIVED"}.get(r.returncode, f"INVALID rc {r.returncode}") + f"  [{last}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    todo = [m for m in M if not a.only or m[0] in a.only.split(",")]
    bad = 0
    for m in todo:
        res = run_one(*m)
        bad += not res.startswith("killed")
        print(f"{m[0]:32s} {res}", flush=True)
    print(f"{len(todo) - bad}/{len(todo)} killed", flush=True)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
