"""SCREENS2 SEQUENTIAL RULE (screens2_seq.py) and its Monte Carlo check (screens2_seqsim.py). Mac CPU, no model.
  spending   alpha(t) Pocock-type; the boundaries spend exactly 0.0155 then 0.0095 (integrator) and the shared-W joint
             law is right (an independent numpy Monte Carlo); the z case reproduces the published two-look values
  reading    look 1 at c1 (above C7's t(0.975)), every d_s; look 2 at c2 with 9 of 10 d_s; EQUAL at the look's c;
             a diverged seed; the stagewise p; only INCONCLUSIVE extends; diverged BASE seeds keep the spend at k1/10
  sim        the simulator's vectorised rule equals the stdlib rule replicate by replicate (B4 (3) included), its bare
             boundaries give 0.025, and the recorded screens2_seqsim.json keeps every error rate at or under 0.025
"""
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import analyze_lib as A  # noqa: E402
import screens2_seq as Q  # noqa: E402
import screens2_seqsim as S  # noqa: E402

DFS = (4, 7, 9)


def test_spend():
    assert abs(Q.spend(0.5) - 0.025 * math.log(1 + (math.e - 1) / 2)) < 1e-15 and abs(Q.spend(1.0) - 0.025) < 1e-15
    assert abs(Q.spend(0.5) - 0.0155034) < 1e-6 and Q.spend(0.4) < Q.spend(0.5)


@pytest.mark.parametrize("df", DFS)
def test_bounds_spend_exactly(df):
    b = Q.bounds(df)
    assert abs((1 - A.t_cdf(b["c1"], df)) - b["a1"]) < 1e-9
    assert abs(Q.cont_cross(b["c1"], b["c2"], b["rho"], df) - (0.025 - b["a1"])) < 1e-9
    assert b["c1"] > A.t_ppf(0.975, df) and abs(b["rho"] - math.sqrt(0.5)) < 1e-15


@pytest.mark.parametrize("df", DFS)
def test_bounds_against_independent_monte_carlo(df):
    rng, n = np.random.default_rng(7 + df), 2_000_000
    z1 = rng.standard_normal(n)
    z2 = (z1 + rng.standard_normal(n)) / math.sqrt(2)              # mean of 10 seeds vs mean of the first 5
    w = np.sqrt(rng.chisquare(df, n) / df)                          # ONE reference SD for both looks
    b = Q.bounds(df)
    p = float(((z1 / w > b["c1"]) | (z2 / w > b["c2"])).mean())
    assert abs(p - 0.025) < 4 * math.sqrt(0.025 * 0.975 / n), p
    p1 = float((z1 / w > b["c1"]).mean())
    assert abs(p1 - b["a1"]) < 4 * math.sqrt(b["a1"] / n), p1


def test_z_case_published_values():
    b = Q.bounds(None)                                              # Lan-DeMets Pocock-type, 2 equal looks, 0.025
    assert abs(b["c1"] - 2.157) < 0.001 and abs(b["c2"] - 2.201) < 0.001, b


def test_diverged_base_seeds_keep_spend_at_planned_fraction():
    b = Q.bounds(7, 5, 9)
    assert abs(b["a1"] - Q.spend(0.5)) < 1e-15 and abs(b["rho"] - math.sqrt(5 / 9)) < 1e-15
    b4 = Q.bounds(7, 4, 9)
    assert abs(b4["a1"] - Q.spend(0.4)) < 1e-15
    for x in (b, b4):
        assert abs(Q.cont_cross(x["c1"], x["c2"], x["rho"], 7) - (0.025 - x["a1"])) < 1e-9
    with pytest.raises(ValueError):
        Q.bounds(7, 5, 5)


def test_sign_min():
    assert [Q.sign_min(k) for k in (5, 8, 9, 10)] == [5, 8, 8, 9]


def _d(dbar_rel, signs, base=1.15):
    return [base * dbar_rel / 100 * s for s in signs]


def test_look1_threshold_above_c7():
    sd, df, bm = 0.0035 * 1.15, 7, 1.15
    t_c7, c1 = A.t_ppf(0.975, df), Q.bounds(df)["c1"]
    mid = (t_c7 + c1) / 2 * sd / math.sqrt(5)                       # between the two thresholds, every seed negative
    d = [-mid] * 5
    assert A.metric_reading(d, sd, df, bm)["label"] == "BETTER"
    r = Q.reading(d, sd, df, bm, 1)
    assert r["label"] == "EQUAL" and abs(r["thr"] - c1 * sd / math.sqrt(5)) < 1e-15 and r["need"] == 5
    assert abs(r["p_two_sided"] - A.metric_reading(d, sd, df, bm)["p_two_sided"]) < 1e-12


def test_look1_every_seed():
    d = _d(1.5, [-1, -1, -1, -1, 0.2])
    assert Q.reading(d, 0.003, 7, 1.15, 1)["label"] != "BETTER"


def test_look2_sign_clause_and_c2():
    sd, df, bm = 0.0035 * 1.15, 7, 1.15
    big = 3 * Q.bounds(df)["c2"] * sd / math.sqrt(10)
    nine = [-big * 1.2] * 9 + [big * 0.1]
    eight = [-big * 1.3] * 8 + [big * 0.1] * 2
    r9, r8 = Q.reading(nine, sd, df, bm, 2, 5), Q.reading(eight, sd, df, bm, 2, 5)
    assert r9["label"] == "BETTER" and r9["need"] == 9 and r8["label"] != "BETTER"
    assert abs(r9["thr"] - Q.bounds(df)["c2"] * sd / math.sqrt(10)) < 1e-15
    w9 = Q.reading([-x for x in nine], sd, df, bm, 2, 5)
    assert w9["label"] == "WORSE"


def test_look2_equal_uses_c2():
    sd, df, bm = 0.004 * 1.15, 4, 1.15
    c2 = Q.bounds(df)["c2"]
    edge = bm * 0.01 - c2 * sd / math.sqrt(10)
    def ds(target):                                                 # 2 of 10 negative: never WORSE (needs 9)
        return [(10 * target + 2 * 0.001) / 8] * 8 + [-0.001] * 2
    assert Q.reading(ds(edge * 0.99), sd, df, bm, 2, 5)["label"] == "EQUAL"
    assert Q.reading(ds(edge * 1.01), sd, df, bm, 2, 5)["label"] == "UNRESOLVED"


def test_diverged_seed_at_look2():
    r = Q.reading([A.INF] + [0.01] * 9, 0.003, 7, 1.15, 2, 5)
    assert r["label"] == "WORSE" and r["p_two_sided"] == 0.0
    r8 = Q.reading([A.INF] + [0.01] * 8 + [-0.001], 0.003, 7, 1.15, 2, 5)  # 9 on the WORSE side only with the +inf
    assert r8["label"] == "WORSE"
    assert Q.reading([A.INF] + [-0.01] * 9, 0.003, 7, 1.15, 2, 5)["label"] == "UNRESOLVED"


@pytest.mark.parametrize("df", DFS)
def test_stagewise_p(df):
    b = Q.bounds(df)
    p_c2 = Q.p_stagewise(b["c2"], 2, df)
    assert 0.0495 < p_c2 <= 0.05 and Q.p_stagewise(0.0, 2, df) > 0.99
    ps = [Q.p_stagewise(t, 2, df) for t in (b["c2"], b["c2"] + 1, b["c2"] + 3, 20.0)]
    assert all(x > y for x, y in zip(ps, ps[1:])) and ps[-1] >= 2 * b["a1"] - 1e-12


def test_extends_only_inconclusive():
    assert Q.extends("INCONCLUSIVE")
    assert not any(Q.extends(v) for v in ("NOMINATE", "REJECT", "TIE", "GUARD-FAIL"))


def _stdlib_rule(d, sd_ref, init, df_ref):
    """The registered rule, per replicate, from screens2_seq + analyze_lib (B4 (3) at each look)."""
    def one(k, look):
        x = list(d[:k])
        obs = float(np.std(x, ddof=1))
        sd, df = sd_ref, df_ref
        if obs > sd_ref * A.sd_bound(k - 1):
            sd, df = (obs, k - 1) if obs > init else (init, 4)
        return Q.reading(x, sd, df, 100.0, look, 5)["label"]
    first = one(5, 1)
    return first if first != "UNRESOLVED" else "X" + one(10, 2)


@pytest.mark.parametrize("cls,mu", [("new init", -0.5), ("SIA x1.5", -0.4), ("new init x1.5", 0.0), ("SIA", -0.25)])
def test_sim_rule_equals_stdlib_rule(cls, mu):
    rng = np.random.default_rng(11)
    x = S.draw(rng, cls, mu, 400)
    r = S.rule(x, "SEQ")
    b1, e1 = r["B1"], r["E1"]
    w1 = ~r["ext"] & ~b1 & ~e1
    seen = set()
    for i in range(400):
        want = _stdlib_rule(x["d"][i], float(x["sd"][i]), float(x["init"][i]), x["df"])
        if not r["ext"][i]:
            got = "BETTER" if b1[i] else "EQUAL" if e1[i] else "WORSE" if w1[i] else "?"
        else:
            got = "X" + ("BETTER" if r["B"][i] else "WORSE" if r["W"][i] else "EQUAL" if r["E"][i] else "UNRESOLVED")
        assert got == want, (i, got, want)
        seen.add(want)
    assert len(seen) >= 2, seen


def test_sim_reread_reference_per_class():
    """analyze2_lib.read_arm re-reads a new-init arm at its own class reference; other classes at an independent
    new-init draw (B4 (3))."""
    rng = np.random.default_rng(5)
    for cls in ("new init", "new init x1.5"):
        x = S.draw(rng, cls, 0.0, 50)
        assert np.array_equal(x["init"], x["sd"]), cls
    x = S.draw(rng, "SIA", 0.0, 50)
    assert not np.array_equal(x["init"], x["sd"])


def test_sim_bare_boundaries_spend_alpha():
    rng, n = np.random.default_rng(3), 400_000
    for cls in ("new init", "SIA at u"):
        r = S.rates(rng, cls, 0.0, "bare", n)
        e = S.rates(rng, cls, 1.0, "bare", n)
        for p in (r["B"], e["E"]):
            assert abs(p - 0.025) < 4 * math.sqrt(0.025 * 0.975 / n), (cls, p)
    n2 = 2_000_000                                                  # look 2's own share, where c2 != c1 (df 4)
    r = S.rates(rng, "new init", 0.0, "bare", n2)
    a2 = 0.025 - Q.spend(0.5)
    assert abs((r["B"] - r["B1"]) - a2) < 4 * math.sqrt(a2 / n2), r


def test_recorded_simulation_keeps_error_rates():
    rec = json.load(open(os.path.join(os.path.dirname(HERE), "screens2_seqsim.json")))
    n, tol = rec["reps"], 4 * math.sqrt(0.025 * 0.975 / rec["reps"])
    assert n >= 1_000_000 and rec["sign_min_10"] == 9
    for cls, v in rec["bare"].items():
        assert abs(v["mu 0"]["B"] - 0.025) < tol and abs(v["mu +delta"]["E"] - 0.025) < tol, cls
    for cls in ("same init", "SIA", "SIA at u", "new init"):
        g = rec["grid"][cls]
        assert g["0.0"]["SEQ"]["B"] <= 0.025 and g["0.0"]["SEQ"]["W"] <= 0.025, cls
        assert g["-1.0"]["SEQ"]["E"] <= 0.025 + tol and g["-1.25"]["SEQ"]["E"] <= 0.025, cls
