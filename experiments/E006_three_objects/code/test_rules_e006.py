"""E006 rules tests (notes.txt CODE TO WRITE, test_rules_e006). No model. suite(RL) checks every label branch of
rules_e006 on synthetic inputs with known answers, the boundaries (>= vs >, < vs <=), the bootstrap's point value,
pairing and denominators; mutation_e006.py runs the same suite on mutated copies of rules_e006.py (every mutant must
be killed, and a crash is not a kill).
usage: python -B test_rules_e006.py   (exit 0 = all pass)"""
import sys
sys.dont_write_bytecode = True

import rules_e006 as RL0


def grid(fn, seeds=(1, 2, 3, 4, 5), n=100):
    return {s: {i: fn(s, i) for i in range(n)} for s in seeds}


def suite(RL):
    F = []

    def ok(c, m):
        if not c:
            F.append(m)
    # ---- paired bootstrap: point value, CI, pairing, sign
    one, zero = grid(lambda s, i: 1), grid(lambda s, i: 0)
    ok(RL.paired(one, zero, n_boot=500)[:3] == (1.0, 1.0, 1.0), "paired: all right minus all wrong = 1, CI 1..1")
    half = grid(lambda s, i: int(i < 50))
    d, lo, hi, k, n = RL.paired(half, zero, n_boot=2000)
    ok(d == 0.5 and lo < 0.5 < hi and 0.35 < lo and hi < 0.65 and (k, n) == (5, 100), f"paired: d .5 CI {lo, hi}")
    ok(RL.paired(zero, half, n_boot=500)[0] == -0.5, "paired: sign is X - Y")
    same = grid(lambda s, i: (s * 7 + i * 3) % 2)
    ok(RL.paired(same, same, n_boot=500)[:3] == (0.0, 0.0, 0.0), "paired: X == Y gives 0, CI 0..0 (paired by item)")
    shifted = {s: {i: (s * 7 + i * 3) % 2 for i in range(100)} for s in (1, 2, 3, 4, 5)}
    other = {s: {i: (s * 7 + (i + 1) * 3) % 2 for i in range(100)} for s in (1, 2, 3, 4, 5)}
    r = RL.paired(shifted, other, n_boot=500)
    ok(r[0] == 0.0 and r[1] < 0 < r[2], "paired: equal means but different items gives a CI around 0")
    ok(RL.paired(one, {1: one[1]}, n_boot=100)[3] == 1, "paired: only common seeds are used")
    ok(RL.paired(one, {}, n_boot=100) is None, "paired: no common seed -> None")
    # ---- chat bootstraps: turn-weighted mean and per-model checks
    Xs = {s: {"a": [{"TF": 1}] * 4, "b": [{"TF": 0}]} for s in (1, 2, 3, 4, 5)}
    Ys = {s: {"a": [{"TF": 0}] * 4, "b": [{"TF": 0}]} for s in (1, 2, 3, 4, 5)}
    ok(abs(RL.chat_boot(Xs, Ys, "TF", n_boot=500)[0] - 0.8) < 1e-12, "chat_boot: turn-weighted (4 of 5 turns)")
    Xc = {s: {"a": 3, "b": 1} for s in (1, 2, 3, 4, 5)}
    Yc = {s: {"a": 1, "b": 1} for s in (1, 2, 3, 4, 5)}
    ok(RL.checks_boot(Xc, Yc, n_boot=500)[0] == 2.0, "checks_boot: +2 checks per model")
    # ---- Q-H5
    g, p, w, z = (0.10, 0.04, 0.16), (0.10, 0.01, 0.20), (-0.10, -0.2, -0.01), (0.0, -0.05, 0.05)
    L4ok = {"LIK": (0.02, -0.01, 0.05), "GEN": (0.02, -0.01, 0.05)}
    L4lo = {"LIK": (-0.06, -0.10, -0.01), "GEN": (0.02, -0.01, 0.05)}
    nol = {"LIK": (0.0, -0.03, 0.03), "GEN": (0.0, -0.03, 0.03)}
    q = RL.q_h5(g, p, L4ok, 0.10, 0.0, nol)
    ok(q["label"] == "RESTORED" and not q["flags"] and q["suffix"] == "no loss where the gold is last", f"RESTORED {q}")
    ok(RL.q_h5(g, p, L4lo, 0.10, 0.0, nol)["label"] == "PARTLY RESTORED", "PARTLY RESTORED when P - L4 CI high <= 0")
    ok(RL.q_h5(g, p, None, 0.10, 0.0, nol)["label"] == "PARTLY RESTORED", "PARTLY RESTORED without L4")
    ok(RL.q_h5(g, p, L4ok, 0.10, 0.03, nol)["label"].startswith("GAIN"), "GAIN when the device shift is 0.03")
    ok(RL.q_h5(g, p, L4ok, 0.10, -0.029, nol)["label"] == "RESTORED", "device shift -0.029 keeps RESTORED")
    ok(RL.q_h5(g, p, L4ok, 0.10, None, nol)["label"].startswith("GAIN"), "GAIN when the device shift is unknown")
    ok(RL.q_h5(g, p, L4ok, 0.10, -0.05, nol)["label"].startswith("GAIN"), "GAIN when the device shift is -0.05")
    ok(RL.q_h5(g, p, L4ok, 0.049, 0.0, nol)["flags"] == ["NO DROP TO RESTORE ON CUDA"], "NO DROP flag below 0.05")
    ok(RL.q_h5(g, p, L4ok, 0.05, 0.0, nol)["flags"] == [], "no flag at exactly 0.05")
    ok(RL.q_h5(g, p, None, None, 0.0, nol)["flags"] == ["L4 NOT READ"], "L4 NOT READ flag when L4 - C is not read")
    ok(RL.q_h5(z, z, None, None, 0.0, nol)["flags"] == ["L4 NOT READ"], "L4 NOT READ flag on a non-gain label too")
    tr = {"LIK": (-0.05, -0.09, -0.01), "GEN": (0.0, -0.03, 0.03)}
    ok(RL.q_h5(g, p, L4ok, 0.1, 0.0, tr)["suffix"] == "recency traded (LIK)", "recency traded on LIK")
    tr2 = {"LIK": (-0.049, -0.09, -0.01), "GEN": (-0.06, -0.09, 0.001)}
    ok(RL.q_h5(g, p, L4ok, 0.1, 0.0, tr2)["suffix"] == "no loss where the gold is last", "trade needs <= -0.05, CI < 0")
    ok(RL.q_h5((0.06, 0.001, 0.1), (0.06, 0.001, 0.1), L4ok, 0.1, 0.0, nol)["label"] == "RESTORED", "gain at 0.06")
    ok(RL.q_h5((0.0599, 0.01, 0.1), p, L4ok, 0.1, 0.0, nol)["label"] == "INCONCLUSIVE", "0.0599 is no gain")
    ok(RL.q_h5((0.1, 0.0, 0.2), p, L4ok, 0.1, 0.0, nol)["label"] == "INCONCLUSIVE", "CI low 0 is not > 0")
    ok(RL.q_h5(w, w, L4ok, 0.1, 0.0, nol)["label"] == "WORSE", "WORSE on both")
    ok(RL.q_h5(z, z, L4ok, 0.1, 0.0, nol)["label"] == "NOT THE CAUSE", "NOT THE CAUSE (CI high < 0.06)")
    ok(RL.q_h5((0, -0.05, 0.06), z, L4ok, 0.1, 0.0, nol)["label"] == "INCONCLUSIVE", "CI high 0.06 is not < 0.06")
    ok(RL.q_h5(g, z, L4ok, 0.1, 0.0, nol)["label"] == "INCONCLUSIVE", "LIK gain, GEN null -> INCONCLUSIVE")
    ok(RL.q_h5(w, z, L4ok, 0.1, 0.0, nol)["label"] == "NOT THE CAUSE", "WORSE on one only -> NOT THE CAUSE")
    ok(RL.q_h5(None, z, L4ok, 0.1, 0.0, nol)["label"] == "NOT READ", "NOT READ without a measure")
    # ---- Q-chat
    neg, pos, nul = (-0.4, -0.5, -0.3), (0.2, 0.1, 0.3), (0.0, -0.1, 0.1)
    lp, ck = (-0.1, -0.2, 0.0), (1.0, -1.0, 3.0)
    ok(RL.q_chat(neg, neg, 0.55, 0.10, 0.56, 0.11, lp, ck, True)["label"] == "PROTECTS", "PROTECTS")
    ok(RL.q_chat(neg, neg, 0.55, 0.275, 0.56, 0.28, lp, ck, True)["label"] == "PROTECTS", "ratio exactly 0.5 holds")
    for bad, name, args in (((0.0, -0.1, 0.10), "(b) loops", "loop"), ((0.0, -3.0, 2.0), "(c) probe checks", "checks"),
                            (None, "(d) stopping", "stops")):
        kw = dict(loop=lp, checks=ck, stops=True)
        kw[args] = bad if bad is not None else False
        r = RL.q_chat(neg, neg, 0.55, 0.10, 0.56, 0.11, kw["loop"], kw["checks"], kw["stops"])
        ok(r["label"] == "PARTLY PROTECTS" and r["failed"] == [name], f"PARTLY PROTECTS naming {name}: {r}")
    ok(RL.q_chat(neg, neg, 0.55, 0.30, 0.56, 0.11, lp, ck, True)["label"] == "SMALL EFFECT", "SMALL EFFECT (TF ratio)")
    ok(RL.q_chat(neg, (-0.1, -0.2, 0.01), 0.55, 0.1, 0.56, 0.5, lp, ck, True)["label"] == "NO EFFECT",
       "TF2 CI reaching 0 -> no effect")
    ok(RL.q_chat(pos, nul, 0.2, 0.4, 0.2, 0.2, lp, ck, True)["label"] == "WORSE", "WORSE when TF rises")
    ok(RL.q_chat(nul, nul, 0.5, 0.5, 0.5, 0.5, lp, ck, True)["label"] == "NO EFFECT", "NO EFFECT")
    ok(RL.q_chat(None, nul, 0.5, 0.5, 0.5, 0.5, lp, ck, True)["label"] == "NOT READ", "chat NOT READ")
    # ---- knowledge and cost
    ok(RL.knowledge((0.03, 0.001, 0.06)) == "REDUCES THE COST", "knowledge REDUCES at 0.03")
    ok(RL.knowledge((0.029, 0.001, 0.06)) == "SMALL REDUCTION", "knowledge SMALL REDUCTION")
    ok(RL.knowledge((0.03, 0.0, 0.06)) == "NO EFFECT", "knowledge CI low 0 is no effect")
    ok(RL.knowledge((-0.03, -0.06, -0.001)) == "INCREASES THE COST", "knowledge INCREASES")
    ok(RL.knowledge((-0.03, -0.06, 0.0)) == "NO EFFECT", "knowledge CI high 0 is no effect")
    ok(RL.cost_mark((-0.05, -0.08, -0.01)) == "COST" and RL.cost_mark((-0.049, -0.08, -0.01)) == "LOWER" and
       RL.cost_mark((-0.06, -0.1, 0.0)) == "-" and RL.cost_mark(None) == "NOT READ", "cost marks and boundaries")
    return F


def main():
    F = suite(RL0)
    for f in F:
        print("FAIL ", f)
    print("RESULT: " + ("ALL PASS" if not F else f"{len(F)} FAILED"))
    return 1 if F else 0


if __name__ == "__main__":
    sys.exit(main())
