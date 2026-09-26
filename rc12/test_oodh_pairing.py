"""Tests of the OOD-H pairing checks and the refusal matcher (oodh_wording.py; STEP 11 FIX ROUND, the reviewer's
oodh_adv cases). Synthetic per-thread scores only: no model, no OOD-H items. An exception inside a test is reported
as that test's FAIL.
  python3 -B test_oodh_pairing.py    (exit 1 on any failure; mutation_oodh_wording.py runs it and test_oodh_wording)"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import oodh_wording as OW

HERE = os.path.dirname(os.path.abspath(__file__))
PLANCK = "runs/planck-30m/final.pt"
IDS = [f"oodh-{i:03d}" for i in range(20)]
FAILS = []


def check(ok, msg):
    if not ok:
        FAILS.append(msg)
        print("FAIL", msg)


def rows(units, seed=1, render="template", responder=PLANCK, train_seed=0):
    return [dict(id=t, family="OODH", seed=seed, train_seed=train_seed, render=render, unit=u, own_cf=False,
                 responder=responder) for t, u in zip(IDS, units)]


def sampled(units, responder=PLANCK, render="template"):
    return sum((rows(units, s, render, responder) for s in (1, 2, 3)), [])


LOW, HIGH = [0.0] * 10 + [1.0] * 10, [1.0] * 20
Q = OW.COMPARATOR


def refused(fn, needle):
    try:
        fn()
    except ValueError as e:
        return needle in str(e)
    return False


def test_responders():
    check(refused(lambda: OW.paired_diff(sampled(LOW), sampled(HIGH, "HuggingFaceTB/SmolLM2-135M-Instruct"), n=20),
                  "is not Qwen/Qwen2.5-0.5B-Instruct"), "a comparator that is not Qwen2.5-0.5B-Instruct accepted")
    ci = OW.paired_diff(sampled(LOW), sampled(HIGH, "vllm:" + Q), n=20)
    check(ci["responder_b"] == "vllm:" + Q and ci["D"] == -50.0, f"a kind-prefixed comparator id refused {ci}")
    check(refused(lambda: OW.paired_diff(sampled(LOW, Q), sampled(HIGH, Q), n=20), "is the comparator"),
          "Planck rows from the comparator accepted")
    two = sampled(HIGH, Q)[:-1] + [dict(sampled(HIGH, Q)[-1], responder="Qwen/Qwen3-0.6B")]
    check(refused(lambda: OW.paired_diff(sampled(LOW), two, n=20), "2 responders"), "two comparator responders")
    none = [dict(r, responder=None) for r in sampled(HIGH)]
    check(refused(lambda: OW.paired_diff(sampled(LOW), none, n=20), "is not Qwen"), "comparator without responder")


def test_decoding():
    greedy = rows(HIGH, seed=None, responder=Q)
    check(refused(lambda: OW.paired_diff(sampled(LOW), greedy, n=20), "greedy and the other's sampled"),
          "Planck sampled vs comparator greedy accepted (reviewer case a)")
    check(refused(lambda: OW.paired_diff(rows(LOW, seed=None), sampled(HIGH, Q), n=20), "greedy and the other"),
          "Planck greedy vs comparator sampled accepted")
    ci = OW.paired_diff(rows(LOW, seed=None), greedy, n=20)
    check(ci["greedy"] and any("greedy" in x for x in ci["diagnostic"]), f"greedy pair not labeled {ci['diagnostic']}")
    check(refused(lambda: OW.decide(ci), "diagnostic"), "a claim worded from greedy rows")
    ci = OW.paired_diff(sampled(LOW), sampled(HIGH, Q), n=20)
    check(ci["diagnostic"] == [] and not ci["greedy"] and OW.decide(ci)["decision"] == "qualified",
          f"the claim path (template, sampled) labeled diagnostic {ci['diagnostic']}")


def test_render():
    ci = OW.paired_diff(sampled(LOW, render="plain"), sampled(HIGH, Q, render="plain"), n=20)
    check(any("plain render" in x for x in ci["diagnostic"]), f"plain pair not labeled {ci['diagnostic']}")
    check(refused(lambda: OW.decide(ci), "diagnostic"), "a claim worded from the plain render (reviewer case c)")


def test_part2():
    ci = OW.paired_diff(sampled(LOW), sampled(HIGH, Q), n=20)
    check(ci["part2_threads"] is None and refused(lambda: OW.decide(ci, part2_complete=True), "no Part 2 thread"),
          "Part 2 complete without knowing which threads are Part 1")
    ci = OW.paired_diff(sampled(LOW), sampled(HIGH, Q), n=20, part1=IDS)
    check(ci["part2_threads"] == 0 and refused(lambda: OW.decide(ci, part2_complete=True), "no Part 2 thread"),
          "Part 2 complete on Part 1 threads only (reviewer finding)")
    ci = OW.paired_diff(sampled(LOW), sampled(HIGH, Q), n=20, part1=IDS[:18])
    d = OW.decide(ci, part2_complete=True)
    check(ci["part2_threads"] == 2 and not d["provisional"] and OW.PROVISIONAL not in d["text"], "Part 2 scored")


def test_refusal_matcher():
    q = OW.decide(OW.paired_diff(sampled(LOW), sampled(HIGH, Q), n=20))
    tail = " " + q["oodh_sentence"] + " " + OW.PROVISIONAL
    check(q["decision"] == "qualified" and OW.refusals(q["text"], q) == [], "own qualified text refused")
    for head in ("Planck-30M is Non-inferior to Qwen2.5-0.5B-Instruct on RC-12.",
                 "Planck-30M is noninferior to Qwen2.5-0.5B-Instruct on RC-12.",
                 "Planck-30M is non-inferior to Qwen2.5-0.5B-Instruct on RC-12, which uses RC-12's format.",
                 "Planck-30M is NON INFERIOR to Qwen2.5-0.5B-Instruct on RC-12.",
                 "Planck-30M is non‑inferior to Qwen2.5-0.5B-Instruct on RC-12.",
                 "Planck-30M shows non-inferiority to Qwen2.5-0.5B-Instruct on RC-12."):
        check(any("unqualified" in x for x in OW.refusals(head + tail, q)), f"accepted while qualified: {head}")
    u = OW.decide(OW.paired_diff(sampled(HIGH), sampled(HIGH, Q), n=20))
    tail_u = " " + u["oodh_sentence"] + " " + OW.PROVISIONAL
    check(u["decision"] == "unqualified" and
          OW.refusals("Planck-30M is Non-inferior to Qwen2.5-0.5B-Instruct on RC-12." + tail_u, u) == [],
          "an unqualified wording refused while unqualified")


def test_cli():
    d = tempfile.mkdtemp(prefix="rc12_oodhp_")
    paths = {}
    for name, rr in (("a", sampled(LOW)), ("b", sampled(HIGH, Q)), ("pa", sampled(LOW, render="plain")),
                     ("pb", sampled(HIGH, Q, render="plain")), ("smol", sampled(HIGH, "HuggingFaceTB/SmolLM2-135M")),
                     ("greedy", rows(HIGH, seed=None, responder=Q))):
        paths[name] = os.path.join(d, name + ".jsonl")
        open(paths[name], "w").write("".join(json.dumps(r) + "\n" for r in rr))
    run = lambda *x: subprocess.run([sys.executable, "-B", os.path.join(HERE, "oodh_wording.py"), "--n", "50", *x],
                                    capture_output=True, text=True, timeout=100)
    p = run("--planck", paths["pa"], "--comparator", paths["pb"])
    out = json.loads(p.stdout) if p.returncode == 0 else {}
    check(p.returncode == 0 and out.get("decision") is None and out["ci"]["diagnostic"], f"cli plain {p.returncode}")
    for other, needle in (("smol", "is not Qwen"), ("greedy", "greedy and the other")):
        p = run("--planck", paths["a"], "--comparator", paths[other])
        check(p.returncode == 2 and needle in p.stderr, f"cli {other}: exit {p.returncode} {p.stderr[-160:]}")
    shutil.rmtree(d, ignore_errors=True)


def main():
    for t in (test_responders, test_decoding, test_render, test_part2, test_refusal_matcher, test_cli):
        try:
            t()
        except Exception as e:                      # reported as the test's FAIL, never a silent pass
            check(False, f"{t.__name__} raised {type(e).__name__}: {str(e)[:200]}")
    print(f"test_oodh_pairing: {len(FAILS)} failures")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
