"""Mutation test of verify_dev_runs.py and dev_batch.py's run-config fields against test_verify_dev_runs.py (Max's
rule: every test claim is watched failing; notes STEP 9 VERIFY). Each mutant replaces one text in a scratch copy of
rc12 (the *.py files and dev/), then test_verify_dev_runs.py runs there; KILLED = it exits non-zero. The unmutated
copy must pass; malformed = the text is not found exactly once. No model, Mac only.
  python3 -B mutation_verify.py    (writes logs/mutation_verify.txt; exit 1 unless the baseline passes and all die)"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
V, D = "verify_dev_runs.py", "dev_batch.py"
E004 = os.path.join(HERE, "..", "experiments", "E004_general_updating", "code")   # quick_shortcuts' import, by path
ENV = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (E004, os.environ.get("PYTHONPATH")) if p))
MUTANTS = [
    (D, "dtype = chosen_dtype(model_id, args.engines) or args.dtype", "dtype = args.dtype",
     "engines.json's dtype ignored"),
    (D, "decode=HR.DECODE, audit=engine_audit(eng))", "audit=engine_audit(eng))", "decode not recorded"),
    (V, 'say("FAIL", f"C2 {name}: no finished _owncf twin (R would be None)")', "pass", "missing twin passes"),
    (V, 'if m.get("decode") != HR.DECODE:', "if False:", "decode not checked"),
    (V, 'if t["stop"] not in OK_STOPS[render] or t["reply"] != t["reply"].strip():', "if False:",
     "stop set and strip not checked"),
    (V, 'if text.strip() != t["reply"] or cut != (t["stop"] == "role"):', "if False:", "plain cut not checked"),
    (V, "if own != tw or flags", "if False and flags", "twin ids not compared"),
    (V, 'bad = [r["id"] for r in run["rows"] if r["seed"] != run["meta"]["seed"]]', "bad = []", "row seeds unchecked"),
    (V, 'kind = "FAIL" if common and same == len(common)', 'kind = "INFO" if common and same == len(common)',
     "seed-blind sampling passes"),
    (V, 'if len(run["rows"]) != m["conversations"] or', "if", "row count unchecked"),
    (V, 'cmp(f"R1 seed {seed} lockstep replay vs stored", again, stored)', "pass", "no sampled replay"),
    (V, 'cmp("R2 greedy lockstep replay vs stored greedy run", lock, greedy["rows"][:args.rerun])', "pass",
     "no greedy replay"),
    (V, 'if a.get("vllm_default_sampling") not in ({}, None) or', "if", "vLLM defaults unchecked"),
    (V, 'if a.get("gen_config_stops_not_in_hf"):', "if False:", "gen-config stop ids unchecked"),
    (V, 'if a.get("sampled") != want or', "if", "vLLM kwargs unchecked"),
    (V, "if leak:", "if False:", "HF generation_config fields unchecked"),
    (V, '(ta["reply"], ta["stop"]) != (tb["reply"], tb["stop"])]', "False]", "replay comparison blind"),
]


def run_in(copy):
    p = subprocess.run([sys.executable, "-B", "test_verify_dev_runs.py"], cwd=copy, capture_output=True, text=True,
                       timeout=600, env=ENV)
    return p.returncode, (p.stdout + p.stderr)[-300:]


def main():
    t0, lines, bad = time.time(), [], 0
    for k, (f, old, new, what) in enumerate([(None, None, None, "baseline")] + MUTANTS):
        tmp = tempfile.mkdtemp()
        copy = os.path.join(tmp, "rc12")
        shutil.copytree(HERE, copy, ignore=shutil.ignore_patterns("logs", "runs", "__pycache__"))
        if k:
            p = os.path.join(copy, f)
            src = open(p).read()
            if src.count(old) != 1:
                lines.append(f"MALFORMED #{k} {f}: {what}")
                bad += 1
                shutil.rmtree(tmp)
                continue
            open(p, "w").write(src.replace(old, new))
        rc, tail = run_in(copy)
        shutil.rmtree(tmp)
        if not k:
            lines.append(f"baseline: exit {rc}" + ("" if rc == 0 else f"  {tail!r}"))
            bad += rc != 0
            continue
        lines.append(f"{'KILLED  ' if rc else 'SURVIVED'} #{k} {f}: {what}")
        bad += rc == 0
    lines.append(f"{len(MUTANTS)} mutants, {bad} problems, {time.time() - t0:.0f} s")
    lines.append("ALL MUTANTS KILLED" if not bad else "NOT ALL KILLED")
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    open(os.path.join(HERE, "logs", "mutation_verify.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
