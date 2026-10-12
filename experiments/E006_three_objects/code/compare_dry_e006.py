"""E006 dry-run comparison (notes.txt WHAT STAYS IDENTICAL 1; queue step 1). No model: reads the dry runs' run.json.
  identity  dryC5 (E005's entry point e005_ft_test.py through numerics_e006.py) vs dryC6a (e006_ft_test.py --arm C):
            the 5 step losses, the loss self-check values (hf and answer-only, every batch) and the probe
            trajectory are bitwise equal
  repro     dryC6a vs dryC6b (the same entry point twice). If they are not bitwise equal on this GPU, identity
            becomes: C5 vs C6a differ by no more than C6a vs C6b do (the largest absolute difference); logged
  arms      dryP and dryG finished without a self-test failure, trained both renders, and dryG trained replay
            threads; dryG's first-batch self-check equals C's (within the repro difference)
  timing    seconds per step of each dry run (printed; the ceilings are fixed in notes.txt, not derived here)
usage: python -B compare_dry_e006.py"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")
SLUG = "HuggingFaceTB__SmolLM2-135M-Instruct"


def run(tag):
    p = os.path.join(OUT, f"{SLUG}__{tag}__run.json")
    return json.load(open(p)) if os.path.exists(p) else None


def vec(m):
    """every number the identity compares, in a fixed order."""
    out = [float(x) for x in m.get("losses", [])]
    for b in sorted(m.get("loss_selfcheck", {})):
        if b != "replay":
            out += [m["loss_selfcheck"][b]["hf"], m["loss_selfcheck"][b]["answer_only"]]
    for p in m.get("probe", []):
        out += [float(v) for k, v in sorted(p.items()) if k != "step" and v is not None]
    return out


def maxdiff(a, b):
    if len(a) != len(b):
        return float("inf")
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def main():
    fails = []
    R = {t: run(t) for t in ("dryC5", "dryC6a", "dryC6b", "dryP", "dryG")}
    miss = [t for t, m in R.items() if m is None]
    if miss:
        print(f"RESULT: FAILED: missing dry runs {miss}")
        return 1
    c5, c6a, c6b = (vec(R[t]) for t in ("dryC5", "dryC6a", "dryC6b"))
    rep = maxdiff(c6a, c6b)
    idn = maxdiff(c5, c6a)
    print(f"repro dryC6a vs dryC6b: max |diff| {rep} over {len(c6a)} numbers ({'bitwise equal' if rep == 0 else 'NOT'})")
    print(f"identity dryC5 vs dryC6a: max |diff| {idn} over {len(c5)} numbers")
    if rep == 0 and idn != 0:
        fails.append("E006's C path is not bitwise equal to E005's entry point on this GPU")
    if rep != 0 and idn > rep:
        fails.append(f"E005 vs E006 C path differ ({idn}) by more than two E006 runs do ({rep})")
    for t in ("dryC6a", "dryP", "dryG"):
        m = R[t]
        if m.get("selftest_fail"):
            fails.append(f"{t}: selftest failures {m['selftest_fail']}")
        tr = m.get("renders_trained") or {}
        if not all(tr.values()):
            fails.append(f"{t}: renders trained {tr}")
    g = R["dryG"]
    if not (g.get("replay_digest") or {}).get("n"):
        fails.append("dryG trained no replay thread")
    fc = [[m["loss_selfcheck"]["first"][k] for k in ("hf", "answer_only", "n_labelled")] for m in (R["dryC6a"], g)]
    if fc[0][2] != fc[1][2] or maxdiff(fc[0][:2], fc[1][:2]) > rep:
        fails.append(f"dryG's first-batch self-check {fc[1]} differs from C's {fc[0]} (its update part is not C's)")
    for t, m in R.items():
        print(f"timing {t}: s_per_step {m.get('s_per_step')} train_s {m.get('train_s')} total_s {m.get('total_s')}")
    for f in fails:
        print("FAIL ", f)
    print("RESULT: " + ("ALL PASS" if not fails else f"{len(fails)} FAILED"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
