"""Tests of the OOD-H wording rule (oodh_wording.py, imported by score_stats as ST.OW). Synthetic per-thread
scores only: no model, no OOD-H items. An exception inside a test is reported as that test's FAIL.
  python3 -B test_oodh_wording.py      (exit 1 on any failure; mutation_oodh_wording.py runs it against its mutants)"""
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

import oodh_wording as OW
import score_stats as ST

HERE = os.path.dirname(os.path.abspath(__file__))
PREREG = os.path.join(os.path.dirname(HERE), "prereg", "RC-12.draft.txt")
FAILS = []


def check(ok, msg):
    if not ok:
        FAILS.append(msg)
        print("FAIL", msg)


def rows(units, seed=0, train_seed=0, render="template", ids=None, **kw):
    ids = ids or [f"oodh-{i:03d}" for i in range(len(units))]
    return [dict(id=t, family="OODH", seed=seed, train_seed=train_seed, render=render, unit=u, own_cf=False, **kw)
            for t, u in zip(ids, units)]


PLANCK = "runs/planck-30m/final.pt"
P1 = [f"oodh-{i:03d}" for i in range(15)]       # a "Part 1" id list: 5 of the 20 synthetic threads are Part 2


def stamp(rr, name):
    return [r if r.get("responder") else dict(r, responder=name) for r in rr]


def pd(a, b, **kw):
    """OW.paired_diff with responders stamped where missing: a = a Planck checkpoint, b = the comparator."""
    return OW.paired_diff(stamp(a, PLANCK), stamp(b, OW.COMPARATOR), **kw)


def ones(k, n=20):
    return [1.0] * k + [0.0] * (n - k)


def refused(fn, needle):
    try:
        fn()
    except ValueError as e:
        return needle in str(e)
    return False


def test_boundary():
    below = pd(rows(ones(10)), rows(ones(11)), n=50)     # 10/20 vs 11/20: D = -5 with float noise
    above = pd(rows(ones(11)), rows(ones(12)), n=50)
    check(below["D"] < -5.0 and above["D"] > -5.0, f"fixtures lost their float noise {below['D']!r} {above['D']!r}")
    for ci in (below, above):
        d = OW.decide(ci)
        check(d["decision"] == "unqualified" and d["refused"] == [], f"D exactly -5 ({ci['D']!r}) is not worse")
        check(OW.UNQUALIFIED in d["text"] and OW.QUALIFIED not in d["text"], "exactly -5: unqualified text")
        check("Qwen2.5-0.5B-Instruct is -5.00 points" in d["oodh_sentence"], f"exactly -5 shown as {d['oodh_sentence']}")
    ci = pd(rows([1.0] * 8 + [0.999] + [0.0] * 11), rows(ones(10)), n=50)     # D = -5.005
    d = OW.decide(ci)
    check(d["decision"] == "qualified" and d["refused"] == [OW.UNQUALIFIED], f"D {ci['D']:.4f}: {d['decision']}")
    check(OW.QUALIFIED in d["text"] and OW.UNQUALIFIED not in d["text"], "qualified text keeps the unqualified phrase")
    check("Qwen2.5-0.5B-Instruct is -5.005 points" in d["oodh_sentence"], f"-5.005 shown as {d['oodh_sentence']}")
    for k_a, k_b, want in ((12, 20, "qualified"), (20, 20, "unqualified"), (16, 20, "qualified"),
                           (17, 20, "qualified"), (18, 20, "unqualified"), (20, 12, "unqualified"),
                           (4, 10, "qualified")):
        ci = pd(rows(ones(k_a, 50)), rows(ones(k_b, 50)), n=50)
        got = OW.decide(ci)["decision"]
        check(got == want, f"D {ci['D']:.1f} ({k_a} vs {k_b} of 50): {got}, want {want}")


def test_provisional_and_text():
    ci = pd(rows(ones(5)), rows(ones(9)), n=50, part1=P1)
    d = OW.decide(ci)
    check(d["provisional"] and d["text"].endswith(OW.PROVISIONAL) and d["note"] == OW.PROVISIONAL,
          "not provisional while Part 2 is incomplete")
    d2 = OW.decide(ci, part2_complete=True)
    check(not d2["provisional"] and OW.PROVISIONAL not in d2["text"] and d2["note"] is None, "Part 2 complete")
    check(d["oodh_sentence"] in d["text"] and d2["oodh_sentence"] in d2["text"], "OOD-H sentence not beside the claim")
    check(d["oodh_sentence"] == f"On OOD-H, 20 conversations with human-written user turns, the paired difference "
          f"against Qwen2.5-0.5B-Instruct is -20.00 points (95% CI {ci['lo']:.2f} to {ci['hi']:.2f}).",
          f"sentence {d['oodh_sentence']}")
    f = OW.decide(ci, fill=dict(N=30, B=18, k=3, s7lo="45.10"))
    check(f["text"].startswith("Planck-30M (30M total parameters, 18M non-embedding) clears the state bar on "
                               "RC-12's format, a pre-registered") and "is 45.10 of 100" in f["text"] and
          "scored {y} (3 training seeds, {c} conversations)" in f["text"], "fill")
    quote = []                                   # s1 Level R quote, placeholders as written there
    lines = open(PREREG).read().split("\n")
    i = next(j for j, ln in enumerate(lines) if ln.startswith('  Level R: "'))
    while not quote or not quote[-1].endswith('"'):
        quote.append(lines[i].strip())
        i += 1
    s1 = " ".join(quote)[len('Level R: "'):-1]
    check(OW.LEVEL_R.format_map(OW._Keep(oodh=OW.OODH_SENTENCE)) == s1, "LEVEL_R differs from the s1 wording")


def test_refusals():
    q = OW.decide(pd(rows(ones(5)), rows(ones(9)), n=50))
    u = OW.decide(pd(rows(ones(9)), rows(ones(9)), n=50, part1=P1), part2_complete=True)
    check(OW.refusals(q["text"], q) == [] and OW.refusals(u["text"], u) == [], "own texts refused")
    bad = q["text"].replace(OW.QUALIFIED, OW.UNQUALIFIED)
    check(any("unqualified" in x for x in OW.refusals(bad, q)), "unqualified wording accepted while qualified")
    para = "Planck-30M clears the state bar on RC-12. {} " + OW.PROVISIONAL
    check(any("unqualified" in x for x in OW.refusals(para.format(q["oodh_sentence"]), q)),
          "a paraphrase without 'RC-12's format' accepted")
    check(OW.refusals(para.format(u["oodh_sentence"]), u) == [], "the unqualified paraphrase refused while unqualified")
    check(OW.refusals(u["text"].replace(OW.UNQUALIFIED, OW.QUALIFIED), u) == [], "qualified wording refused (allowed)")
    check(any("OOD-H result" in x for x in OW.refusals(q["text"].replace(q["oodh_sentence"], ""), q)),
          "claim without the OOD-H result accepted")
    check(any("provisional" in x for x in OW.refusals(q["text"].replace(OW.PROVISIONAL, ""), q)),
          "provisional note missing accepted")


def test_bootstrap():
    rng = random.Random(7)
    b = [rng.choice([0, 1 / 3, 0.5, 2 / 3, 0.9]) for _ in range(150)]
    a = [x + 0.1 for x in b]
    ra = rows(a, seed=0, train_seed=1) + rows(a, seed=1, train_seed=1) + rows(a, seed=0, train_seed=2)
    rb = rows(b, seed=0) + rows(b, seed=1) + rows(b, seed=2)
    ci = pd(ra, rb, n=300)
    check(all(abs(ci[k] - 10) < 1e-9 for k in ("D", "lo", "hi")), f"paired shift of 10 points: {ci}")
    check((ci["runs_a"], ci["runs_b"], ci["train_seeds_a"], ci["h"]) == (3, 3, 2, 150), f"counts {ci}")
    x = [rng.choice([0, 0.5, 1]) for _ in range(150)]
    y = [rng.choice([0, 0.5, 1]) for _ in range(150)]
    ci = pd(rows(x), rows(y), n=4000, seed=3)
    dd = [100 * (p - q) for p, q in zip(x, y)]
    mu = sum(dd) / 150
    se = (sum((v - mu) ** 2 for v in dd) / 150) ** 0.5 / 150 ** 0.5
    check(abs(ci["D"] - mu) < 1e-9, f"D {ci['D']} vs {mu}")
    check(abs((mu - ci["lo"]) / (1.96 * se) - 1) < 0.15 and abs((ci["hi"] - mu) / (1.96 * se) - 1) < 0.15,
          f"percentile CI {ci['lo']:.2f}..{ci['hi']:.2f} vs normal {mu - 1.96 * se:.2f}..{mu + 1.96 * se:.2f}")
    check(ci == pd(rows(x), rows(y), n=4000, seed=3), "bootstrap not deterministic for a seed")


def test_nesting_and_seeds():
    ra = rows(ones(20), 0, 0) + rows(ones(0), 1, 0) + rows(ones(0), 2, 0) + rows(ones(20), 0, 1)
    ci = pd(ra, rows(ones(0)), n=20)
    check(abs(ci["D"] - 200 / 3) < 1e-9, f"nested mean (1/3 and 1 over training seeds) = {ci['D']}, want 66.67")
    rg = rows(ones(0), seed=None) + rows(ones(20), seed=0)
    ci = pd(rg, rows(ones(0), seed=None) + rows(ones(0), seed=4), n=20)
    check(ci["D"] == 100.0 and ci["runs_a"] == 1, f"greedy rows scored beside sampled ones: {ci}")
    ci = pd(rows(ones(20), seed=None), rows(ones(0), seed=None), n=20)
    check(ci["D"] == 100.0, "greedy-only models not scored on greedy")
    check(pd(rows(ones(6)), rows(ones(14)), n=20)["D"] == -40.0, "sign: Planck minus comparator")


def test_refused_inputs():
    a, b = rows(ones(10)), rows(ones(12))
    miss = a + rows(ones(10), seed=1)[:-1]
    check(refused(lambda: pd(miss, b, n=5), "lacks 1 of 20"), "a missing thread in one run")
    check(refused(lambda: pd(a, b[:-1], n=5), "lacks 1 of 20"), "comparator lacks a thread")
    extra = b + rows([1.0], ids=["oodh-999"])
    check(refused(lambda: pd(a, extra, n=5), "lacks 1 of 21"), "comparator has an extra thread")
    other = rows(ones(10), ids=[f"oodh-{i:03d}" for i in range(100, 120)])
    check(refused(lambda: pd(a, other, n=5), "lacks 20 of 40"), "disjoint thread sets")
    ids = [r["id"] for r in a]
    check(pd(a, b, n=5, expected=ids)["items_checked"], "expected ids refused")
    check(refused(lambda: pd(a, b, n=5, expected=ids + ["oodh-150"]), "differ from the item file"),
          "a thread of the item file missing from both models")
    check(refused(lambda: pd(a + a[:1], b, n=5), "twice"), "duplicate thread")
    check(refused(lambda: pd(a[:-1] + [dict(a[-1], family="RECALL")], b, n=5), "not OODH"), "family")
    check(refused(lambda: pd(a[:-1] + [dict(a[-1], own_cf=True)], b, n=5), "own-cf"), "own_cf row")
    for u in (None, float("nan"), 1.5, -0.1, True):
        check(refused(lambda: pd(a[:-1] + [dict(a[-1], unit=u)], b, n=5), "not a number"), f"unit {u}")
    check(refused(lambda: pd(a, rows(ones(12), render="plain"), n=5), "renders differ"), "render")
    check(refused(lambda: pd(a + rows(ones(10), seed=1, render="plain"), b, n=5), "two renders"),
          "two renders in one model")
    check(refused(lambda: pd([], b, n=5), "no OOD-H rows"), "no rows")


def test_cli_and_score_stats():
    check(ST.OW is OW and ST.OW.decide is OW.decide, "score_stats does not expose the wording rule as ST.OW")
    d = tempfile.mkdtemp(prefix="rc12_oodh_")
    paths = {}
    for name, rr in (("a", stamp(rows(ones(8)) + rows(ones(8), seed=1), PLANCK)),
                     ("b", stamp(rows(ones(12)), OW.COMPARATOR)), ("short", stamp(rows(ones(12))[:-1], OW.COMPARATOR)),
                     ("items", [dict(id=f"oodh-{i:03d}") for i in range(20)]),
                     ("part1", [dict(id=f"oodh-{i:03d}") for i in range(15)])):
        paths[name] = os.path.join(d, name + ".jsonl")
        open(paths[name], "w").write("".join(json.dumps(r) + "\n" for r in rr))
    run = lambda *x: subprocess.run([sys.executable, "-B", os.path.join(HERE, "oodh_wording.py"), "--n", "200", *x],
                                    capture_output=True, text=True, timeout=100)
    p = run("--planck", paths["a"], "--comparator", paths["b"], "--data", paths["items"], "--fill", "N=30,k=3")
    out = json.loads(p.stdout) if p.returncode == 0 else {}
    dec = out.get("decision", {})
    check(p.returncode == 0 and dec.get("decision") == "qualified" and dec.get("provisional") and
          out["ci"]["items_checked"] and "Planck-30M" in dec.get("text", ""), f"cli {p.returncode} {p.stderr[-200:]}")
    p = run("--planck", paths["a"], "--comparator", paths["b"], "--data", paths["items"], "--part1", paths["part1"],
            "--part2-complete")
    check(p.returncode == 0 and json.loads(p.stdout)["decision"]["provisional"] is False and
          json.loads(p.stdout)["ci"]["part2_threads"] == 5, f"cli --part2-complete {p.returncode} {p.stderr[-200:]}")
    p = run("--planck", paths["a"], "--comparator", paths["b"], "--part1", paths["part1"], "--part2-complete")
    check(p.returncode == 2 and "needs --data" in p.stderr, "cli --part2-complete without --data not refused")
    p = run("--planck", paths["a"], "--comparator", paths["b"], "--data", paths["items"], "--part1", paths["items"],
            "--part2-complete")
    check(p.returncode == 2 and "no Part 2 thread" in p.stderr, "cli --part2-complete with Part 1 threads only")
    p = run("--planck", paths["a"], "--comparator", paths["short"])
    check(p.returncode == 2 and "REFUSED" in p.stderr and "lacks" in p.stderr, f"cli refusal exit {p.returncode}")
    check("torch" not in sys.modules, "torch imported")
    shutil.rmtree(d, ignore_errors=True)


def main():
    for t in (test_boundary, test_provisional_and_text, test_refusals, test_bootstrap, test_nesting_and_seeds,
              test_refused_inputs, test_cli_and_score_stats):
        try:
            t()
        except Exception as e:                      # reported as the test's FAIL, never a silent pass
            check(False, f"{t.__name__} raised {type(e).__name__}: {str(e)[:200]}")
    print(f"test_oodh_wording: {len(FAILS)} failures")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
