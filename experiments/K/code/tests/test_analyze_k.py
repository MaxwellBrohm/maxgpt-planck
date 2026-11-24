"""analyze_k.py: SPEC 10's KR on hand-computed fixtures (every verdict branch, both directions, DIVERGENCE), the Holm
reading, CE-DISAGREE, GENERAL-LANGUAGE COST, SPEC 9's time course and TRANSIENT, the summaries from scorer rows (B
groups, F by reference kind), and the append-only readings file."""
import json
import math
import statistics as st

import pytest

import analyze_k as A

S3 = (201, 202, 203)


def d3(*xs):
    return dict(zip(S3, xs))


def test_t_quantiles_match_spec_10():
    assert [round(A.t_ppf(0.975, df), 3) for df in (4, 6, 8)] == [2.776, 2.447, 2.306]


def test_kr_pools_residuals_and_reads_each_branch():
    c = {"up": d3(0.05, 0.06, 0.07), "down": d3(-0.06, -0.05, -0.07), "eq": d3(0.0, 0.004, -0.004),
         "mixed": d3(0.03, -0.002, 0.04), "small": d3(0.021, 0.019, 0.020)}
    r = A.kr(c, 0.03)
    res = [x - st.mean(v.values()) for v in c.values() for x in v.values()]
    sd = math.sqrt(sum(x * x for x in res) / 10)              # df = 5 contrasts x (3 - 1)
    thr = A.t_ppf(0.975, 10) * sd / math.sqrt(3)
    assert r["up"]["df"] == 10 and r["up"]["sd_ref"] == pytest.approx(sd) and r["up"]["thr"] == pytest.approx(thr)
    assert {n: v["verdict"] for n, v in r.items()} == {"up": "HIGHER", "down": "LOWER", "eq": "EQUAL",
                                                      "mixed": "UNRESOLVED", "small": "UNRESOLVED"}
    assert thr < 0.02 < 0.03                                   # "small" clears thr with every d_s > 0: delta decides
    p = 2 * (1 - A.t_cdf(0.06 / (sd / math.sqrt(3)), 10))
    assert r["up"]["p"] == pytest.approx(p) and r["up"]["p"] < 0.05


def test_kr_higher_needs_every_seed_positive():
    c = {"a": d3(0.10, -0.001, 0.11), "z1": d3(0.0, 0.0, 0.0), "z2": d3(0.0, 0.0, 0.0), "z3": d3(0.0, 0.0, 0.0),
         "z4": d3(0.0, 0.0, 0.0)}
    r = A.kr(c, 0.03)["a"]
    assert r["dbar"] > r["thr"] and r["dbar"] >= 0.03 and r["verdict"] == "UNRESOLVED"


def test_kr_equal_needs_the_interval_inside_delta():
    c = {"a": d3(0.0, 0.01, -0.01), "b": d3(0.02, 0.03, 0.025)}
    r = A.kr(c, 0.03)
    assert abs(r["b"]["dbar"]) < 0.03 and abs(r["b"]["dbar"]) + r["b"]["thr"] > 0.03
    assert r["b"]["verdict"] == "UNRESOLVED" and r["a"]["verdict"] == "EQUAL"


def test_divergence_scores_worst_and_a_diverged_reference_drops_its_seed():
    arm, ref = d3(0.5, None, 0.4), d3(0.6, 0.6, 0.55)
    d = A.paired(arm, ref)
    assert d[202] == -math.inf and len(d) == 3
    r = A.kr({"x": d, "y": d3(0.01, 0.0, -0.01)}, 0.03)
    assert r["x"]["verdict"] == "LOWER" and "UNSTABLE" in r["x"]["labels"] and r["x"]["p"] is None
    assert r["y"]["df"] == 2                                   # the diverged contrast adds no residual
    assert A.paired(d3(0.5, None, 0.4), ref, higher_better=False)[202] == math.inf
    dd = A.paired(d3(0.5, 0.5, 0.4), d3(0.6, None, 0.55))
    assert sorted(dd) == [201, 203]
    one = A.kr({"z": {201: 0.1}}, 0.03)["z"]
    assert one["verdict"] == "UNRESOLVED" and one["labels"] == ["fewer than 2 seeds"]


def test_holm_steps_down_and_stops():
    h = A.holm({"a": 0.01, "b": 0.03, "c": 0.04, "d": None})
    assert h == {"a": True, "b": False, "c": False, "d": None}          # 0.01 <= 0.05/3; 0.03 > 0.05/2
    h = A.holm({"a": 0.012, "b": 0.02, "c": 0.04})
    assert h == {"a": True, "b": True, "c": True}                       # 0.012 <= 0.0167, 0.02 <= 0.025, 0.04 <= 0.05
    assert A.holm({"a": 0.2, "b": 0.001}) == {"a": False, "b": True}


def test_ce_disagree():
    up = {"verdict": "HIGHER", "labels": []}
    assert A.skill_label(up, {"dbar": -0.1})["verdict"] == "HIGHER"
    r = A.skill_label(up, {"dbar": 0.05})
    assert r["verdict"] == "UNRESOLVED" and r["labels"] == ["CE-DISAGREE"] and r["raw_verdict"] == "HIGHER"
    assert A.skill_label({"verdict": "LOWER", "labels": []}, {"dbar": 0.2})["verdict"] == "LOWER"
    assert A.skill_label({"verdict": "LOWER", "labels": []}, {"dbar": -0.2})["verdict"] == "UNRESOLVED"
    assert A.skill_label({"verdict": "EQUAL", "labels": []}, {"dbar": 0.2})["verdict"] == "EQUAL"
    # K3 round 15 (REVIEW 16b T-6): a skill verdict whose CE_C contrast has no dbar (no seed read) does not stand
    # (mutant r16_label_ce_missing_ok)
    r = A.skill_label(up, {})
    assert (r["verdict"], r["labels"], r.get("raw_verdict")) == ("UNRESOLVED", ["CE-DISAGREE"], "HIGHER"), r
    # K3 round 16 (2026-11-22, REVIEW 17b; mutant r17c_label_ce_zero survived): CE_C must MOVE with the skill verdict;
    # a CE_C dbar of exactly 0 leaves either verdict CE-DISAGREE
    for v in ("HIGHER", "LOWER"):
        r = A.skill_label({"verdict": v, "labels": []}, {"dbar": 0.0})
        assert (r["verdict"], r["labels"], r.get("raw_verdict")) == ("UNRESOLVED", ["CE-DISAGREE"], v), r


def test_bpb_cost():
    assert A.bpb_cost(d3(0.02, 0.03, 0.025), 0.005, 4)["labels"] == ["GENERAL-LANGUAGE COST"]
    assert not A.bpb_cost(d3(0.02, -0.001, 0.03), 0.005, 4)["worse"]
    assert not A.bpb_cost(d3(0.004, 0.005, 0.006), 0.005, 4)["worse"]    # under thr = 2.776 x 0.005 / sqrt 3


def test_course_and_transient():
    curve = [(153 * i, v) for i, v in enumerate([0.1, 0.2, 0.55, 0.6, 0.6, 0.62, 0.6, 0.52, 0.5, 0.48, 0.5, 0.45])]
    c = A.course(curve)
    mm = max(sum(x for _, x in curve[i - 4:i + 1]) / 5 for i in range(4, 12))
    assert c["peak"] == pytest.approx(mm) and c["final"] == pytest.approx(0.49) and c["peak_step"] == 153 * 6
    flat = [(153 * i, 0.5) for i in range(12)]
    t = A.transient({201: curve, 202: curve, 203: flat})
    assert t["transient"] and t["seeds_dropping"] == 2
    assert not A.transient({201: curve, 202: flat, 203: flat})["transient"]
    with pytest.raises(ValueError):
        A.course(curve[:4])
    # K3 round 15 (REVIEW 16b T-6): a tie of the moving mean goes to the EARLIER window (mutant r16_peak_tie_late)
    assert A.course(flat)["peak_step"] == 153 * 4


def row(i, s, fam, lik, n_c=10, ce=1.0, gen=False):
    return {"id": i, "set": s, "fam": fam, "lik": lik, "n_c": n_c, "ce_c": ce, "gen": gen}


def test_summary_from_rows():
    items, rows = [], []
    for j, (kref, lik) in enumerate([("ell", 1.0), ("ell", 0.0), ("al", 1.0), ("al", 1.0)]):
        items.append({"id": f"F{j}", "set": "K-EVAL-ID", "meta": {"kref": kref, "fref": "al" if j else "ell"}})
        rows.append(row(f"F{j}", "K-EVAL-ID", "F", lik))
    liks = [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.0]     # group 1 all right; group 2 a tie: not right
    for j, lik in enumerate(liks):
        grp = None if j == 8 else (j % 4)
        items.append({"id": f"B{j}", "set": "K-EVAL-ID", "meta": {"struct": grp is not None, "grp": grp}})
        rows.append(row(f"B{j}", "K-EVAL-ID", "B", lik))
    rows.append(row("O1", "OOD-DFAR", "R", 1.0))
    items.append({"id": "O1", "set": "OOD-DFAR", "meta": {}})
    s = A.summary(rows, items)
    e = s["sets"]["K-EVAL-ID"]
    assert e["F_by_kref"]["ell"]["lik"] == 0.5 and e["F_by_kref"]["al"]["lik"] == 1.0
    assert e["B_groups"] == {"n": 2, "acc": 0.5} and e["B_small"]["n"] == 1 and e["B_cube"]["n"] == 8
    assert s["S_ID"] == pytest.approx(sum(r["lik"] for r in rows[:-1]) / 13) and s["S_ID_S"] is None
    assert s["OOD"] == {"OOD-DFAR": 1.0} and e["family"]["F"]["chance"] == pytest.approx(0.1)
    # K3 round 15 (REVIEW 16b T-6): a B group of other than 4 questions is refused, not scored (mutant
    # r16_b_group_size_unchecked): drop group 2's last member
    cut = [x for x in items if x["id"] != "B7"]
    with pytest.raises(ValueError, match="exactly 4"):
        A.summary([o for o in rows if o["id"] != "B7"], cut)


def test_readings_are_append_only(tmp_path):
    p = str(tmp_path / "results" / "readings.jsonl")
    A.append_reading(p, {"reading": "x", "seeds": [201]})
    A.append_reading(p, {"reading": "y", "seeds": [202]})
    rows = [json.loads(x) for x in open(p)]
    assert [r["reading"] for r in rows] == ["x", "y"] and all("time" in r for r in rows)
    with open(p, "a") as f:
        f.write('{"cut')
    with pytest.raises(ValueError, match="incomplete"):
        A.append_reading(p, {"reading": "z"})
    # K3 round 15 (REVIEW 16b T-5): a last line that parses but lacks its newline is refused too, by the newline
    # rule (before, only json.loads refused '{"cut'; mutant r16_readings_partial_line_ok glued the next row to it)
    q = str(tmp_path / "results" / "r2.jsonl")
    with open(q, "w") as f:
        f.write('{"reading": "w"}')
    with pytest.raises(ValueError, match="incomplete"):
        A.append_reading(q, {"reading": "z"})
    assert open(q).read() == '{"reading": "w"}'


def test_probe_curve_reads_the_hook_log_and_refuses_errors(tmp_path):
    p = str(tmp_path / "k_probe.jsonl")
    rows = [{"step": 153 * i, "skill_lik": {"K-PROBE": 0.1 * i}, "fact_acc": {"K-PROBE": 0.01 * i}} for i in (1, 2)]
    open(p, "w").write("".join(json.dumps(r) + "\n" for r in rows))
    assert A.probe_curve(p) == [(153, 0.1), (306, 0.2)]
    assert A.probe_curve(p, "fact_acc") == [(153, 0.01), (306, 0.02)]
    with open(p, "a") as f:
        f.write(json.dumps({"step": 459, "error": "boom"}) + "\n")
    with pytest.raises(ValueError, match="error"):
        A.probe_curve(p)


def test_kr_with_two_seeds_divides_by_root_two():
    """REVIEW 14c D5: a contrast left with 2 seeds (a diverged reference seed) takes thr = t(0.975, df) SD_ref /
    sqrt(2). Here dbar 0.024 lies between that thr (0.0260) and the sqrt(3) figure (0.0212), so the verdict turns on
    it: UNRESOLVED, not HIGHER, at delta 0.02 (mutant r14c_thr_sqrt3)."""
    c = {"x": {201: 0.014, 203: 0.034}, "y": d3(0.01, 0.0, -0.01)}
    r = A.kr(c, 0.02)
    sd = math.sqrt((2 * 0.01 ** 2 + 2 * 0.01 ** 2) / 3)          # df = (2 - 1) + (3 - 1)
    assert r["x"]["k"] == 2 and r["x"]["df"] == 3 and r["x"]["sd_ref"] == pytest.approx(sd)
    assert r["x"]["thr"] == pytest.approx(A.t_ppf(0.975, 3) * sd / math.sqrt(2))
    assert r["x"]["dbar"] == pytest.approx(0.024)
    assert A.t_ppf(0.975, 3) * sd / math.sqrt(3) < r["x"]["dbar"] < r["x"]["thr"]
    assert r["x"]["verdict"] == "UNRESOLVED"
    assert r["x"]["p"] == pytest.approx(2 * (1 - A.t_cdf(0.024 / (sd / math.sqrt(2)), 3)))


def test_transient_at_a_drop_of_exactly_three_points():
    """REVIEW 14c D5: K1 [K1-d] reads TRANSIENT at peak - final >= 3 points; a drop of exactly 0.03 (0.13 to 0.10,
    exact in floating point here) counts (mutant r14c_transient_gt)."""
    lo = 0.1
    drop = [(153 * i, v) for i, v in enumerate([lo + 0.03] * 5 + [lo] * 5)]
    c = A.course(drop)
    assert c["peak"] - c["final"] == 0.03
    flat = [(153 * i, 0.5) for i in range(10)]
    t = A.transient({201: drop, 202: drop, 203: flat})
    assert t["seeds_dropping"] == 2 and t["transient"]


def test_ood_cube_items_without_a_group_stay_in_the_cube_row():
    """REVIEW 14c D5: the B cube row is read by structure. OOD sets carry cube items with no group (gen.py's
    k_eval_ood: groups=False), which must stay in B_cube, with no group row (mutant r14c_bcube_by_grp)."""
    items = [{"id": "B0", "set": "OOD-VALH", "meta": {"struct": True, "k": 8}},
             {"id": "B1", "set": "OOD-VALH", "meta": {"struct": False, "k": 3}}]
    rows = [row("B0", "OOD-VALH", "B", 1.0), row("B1", "OOD-VALH", "B", 0.0)]
    e = A.summary(rows, items)["sets"]["OOD-VALH"]
    assert e["B_cube"]["n"] == 1 and e["B_cube"]["lik"] == 1.0 and e["B_small"]["n"] == 1 and "B_groups" not in e


def g2(*rows):
    """{g: {199: a, 200: b}} from (g, a, b) rows: K-DEV objectives at the two selection seeds (G-2)."""
    return {g: {199: a, 200: b} for g, a, b in rows}


def test_pick_g_takes_the_mean_over_both_seeds():
    """G-2: the pick is the g with the least mean K-DEV objective over seeds 199 and 200. Seed 199 alone picks 0.5 and
    seed 200 alone 2 (each an edge, so each alone would ask an extension); the mean picks 1, final, SEED-SPLIT."""
    r = A.pick_g(g2((0.5, 2.900, 3.200), (1.0, 2.950, 3.000), (2.0, 3.200, 2.900)))
    assert (r["g"], r["final"], r["extend"]) == (1.0, True, None)
    assert r["mean"] == pytest.approx({0.5: 3.05, 1.0: 2.975, 2.0: 3.05})
    assert r["per_seed"] == {199: 0.5, 200: 2.0} and r["labels"] == ["SEED-SPLIT"]


def test_pick_g_ties_go_to_the_lower_lr():
    """Ties at 4 decimals of the mean go to the lower g (the lower LR): 0.5 and 1 tie at 3.0000; 1 alone would win at
    full precision. Agreeing seeds carry no label."""
    r = A.pick_g(g2((0.5, 3.00004, 3.00004), (1.0, 3.00001, 3.00001), (2.0, 3.1, 3.1)))
    assert (r["g"], r["final"], r["extend"]) == (0.5, False, 0.25) and r["labels"] == []
    assert r["per_seed"] == {199: 0.5, 200: 0.5}        # K3 round 15 (REVIEW 16b T-6): each seed's own argmin ties
    #                                                    the same way (mutant r16_pick_seed_tie_high)
    r = A.pick_g(g2((0.5, 3.1, 3.1), (1.0, 3.00004, 3.00004), (2.0, 3.00001, 3.00001)))
    assert (r["g"], r["final"]) == (1.0, True)


def test_pick_g_extends_an_edge_once():
    """A pick at an edge asks one factor-2 point beyond it, run at both seeds; with it in the table the pick is final,
    even at the new edge. A second point beyond, or a point off the grid's factor-2 line, is refused."""
    base = ((0.5, 3.2, 3.2), (1.0, 3.1, 3.1), (2.0, 3.0, 3.0))
    r = A.pick_g(g2(*base))
    assert (r["g"], r["final"], r["extend"]) == (2.0, False, 4.0)
    r = A.pick_g(g2(*base, (4.0, 2.9, 2.95)))
    assert (r["g"], r["final"], r["extend"]) == (4.0, True, None)
    r = A.pick_g(g2(*base, (4.0, 3.05, 3.05)))
    assert (r["g"], r["final"], r["extend"]) == (2.0, True, None)
    for bad in (g2(*base, (4.0, 2.9, 2.9), (8.0, 2.8, 2.8)), g2(*base, (3.0, 2.9, 2.9))):
        with pytest.raises(ValueError):
            A.pick_g(bad)


def test_pick_g_never_picks_a_diverged_g_and_needs_both_seeds():
    """A g that diverged at either seed has the mean +inf (SPEC 10 DIVERGENCE: never dropped, never picked), though its
    other seed is the best value in the table; a g read at one seed only is refused; every g diverged: no pick."""
    r = A.pick_g(g2((0.5, 3.2, 3.2), (1.0, 2.0, None), (2.0, 3.0, 3.1)))
    assert r["g"] == 2.0 and r["mean"][1.0] == math.inf and r["per_seed"] == {199: 1.0, 200: 2.0}
    with pytest.raises(ValueError):
        A.pick_g({0.5: {199: 3.0, 200: 3.0}, 1.0: {200: 2.9}, 2.0: {199: 3.1, 200: 3.1}})
    r = A.pick_g(g2((0.5, None, 3.0), (1.0, 3.0, None), (2.0, None, None)))
    assert r["g"] is None and r["final"] and "every g diverged" in r["labels"]


def test_pick_g_takes_only_the_extension_the_grid_asks():
    """K3 round 14 (REVIEW 15b T-2): a table missing a grid point is refused (a lost g-check run would leave a pick
    from two points; mutant r15_pick_grid_point_missing_ok), and a point off the grid is taken only when it is the one
    extension the grid's own pick asks: an interior pick asks none, an edge pick only the point beyond its own edge
    (mutant r14_pick_ext_unasked_ok, the pre-round-14 check: any factor-2 point beyond either edge was taken, final)."""
    with pytest.raises(ValueError, match="the grid"):
        A.pick_g(g2((0.5, 3.2, 3.2), (2.0, 3.0, 3.0)))
    with pytest.raises(ValueError, match="asks no extension"):
        A.pick_g(g2((0.5, 3.1, 3.1), (1.0, 3.0, 3.0), (2.0, 3.2, 3.2), (0.25, 2.9, 2.9)))
    with pytest.raises(ValueError, match="asks 4.0"):
        A.pick_g(g2((0.5, 3.2, 3.2), (1.0, 3.1, 3.1), (2.0, 3.0, 3.0), (0.25, 2.9, 2.9)))
    with pytest.raises(ValueError, match="asks 0.25"):
        A.pick_g(g2((0.5, 3.0, 3.0), (1.0, 3.1, 3.1), (2.0, 3.2, 3.2), (4.0, 2.9, 2.9)))
    r = A.pick_g(g2((0.5, 3.0, 3.0), (1.0, 3.1, 3.1), (2.0, 3.2, 3.2), (0.25, 3.05, 3.05)))
    assert (r["g"], r["final"], r["extend"]) == (0.5, True, None)


def test_pick_g_needs_exactly_the_two_seeds_and_reads_nan_as_diverged():
    """K3 round 14 (REVIEW 15b T-6): a g read at a third seed is refused, not averaged over two of its three (mutant
    r15_pick_extra_seed_ok). A non-finite objective is a diverged seed: a NaN at seed 199 on g 0.5 is not seed 199's
    argmin, which is 1 (before round 14 it was reported 0.5 and SEED-SPLIT went unprinted; mutant r14_pick_nan_finite),
    and the g's mean is +inf, so it is never picked; an inf alike."""
    with pytest.raises(ValueError, match="seeds"):
        A.pick_g({0.5: {199: 3.0, 200: 3.0, 201: 2.0}, 1.0: {199: 3.1, 200: 3.1}, 2.0: {199: 3.2, 200: 3.2}})
    r = A.pick_g(g2((0.5, math.nan, 3.0), (1.0, 3.1, 3.1), (2.0, 3.2, 3.2)))
    assert r["per_seed"] == {199: 1.0, 200: 0.5} and r["labels"] == ["SEED-SPLIT"], r
    assert (r["g"], r["final"]) == (1.0, True) and r["mean"][0.5] == math.inf
    r = A.pick_g(g2((0.5, 3.1, 3.1), (1.0, 3.2, 3.2), (2.0, 3.0, math.inf)))
    assert r["per_seed"] == {199: 2.0, 200: 0.5} and r["mean"][2.0] == math.inf and (r["g"], r["extend"]) == (0.5, 0.25)


def test_deltas_are_spec_10s():
    """K3 round 14 (REVIEW 15b T-3): SPEC 10's deltas, which no other test read (mutants r15_delta_sid,
    r15_bits_delta_scale, r14_delta_gen, r14_delta_rate): S_ID 3 points, GEN 5, the CONFLICT and K5 rates 10, stored
    bits 10% of the reference mean over its finite seeds (a diverged reference seed, None, is left out)."""
    assert A.DELTA == {"S_ID": 0.03, "GEN": 0.05, "RATE": 0.10}
    assert A.bits_delta({201: 100.0, 202: 120.0, 203: None}) == pytest.approx(11.0)


def test_kr_lower_is_the_mirror_of_higher():
    """K3 round 16 (2026-11-21, REVIEW 17b): SPEC 10's LOWER is HIGHER's mirror image, so it needs every d_s < 0 and
    -dbar >= delta, as HIGHER needs every d_s > 0 and dbar >= delta. Before this test only a contrast far below on
    every seed read LOWER, so LOWER with one seed of the wrong sign (mutant r17_kr_lower_any) and LOWER below delta
    (r17_kr_lower_no_delta) survived; both turn a K1 reading (CROWD-OUT is FH LOWER than F0)."""
    c = {"a": d3(-0.10, 0.001, -0.11), "z1": d3(0.0, 0.0, 0.0), "z2": d3(0.0, 0.0, 0.0), "z3": d3(0.0, 0.0, 0.0),
         "z4": d3(0.0, 0.0, 0.0)}
    r = A.kr(c, 0.03)["a"]
    assert -r["dbar"] > r["thr"] and -r["dbar"] >= 0.03 and r["d"][202] > 0
    assert r["verdict"] == "UNRESOLVED", r
    # REVIEW 17b's hand case: K1's two contrasts, FH - F0 clears thr on every seed but sits under delta
    c = {"FH-F0": d3(-0.026, -0.024, -0.025), "FL-F0": d3(-0.001, 0.0, 0.001)}
    r = A.kr(c, 0.03)
    fh = r["FH-F0"]
    assert fh["dbar"] == pytest.approx(-0.025) and -fh["dbar"] > fh["thr"] and all(x < 0 for x in fh["d"].values())
    assert abs(fh["dbar"]) + fh["thr"] < 0.03 and fh["verdict"] == "EQUAL", fh
    # and the same contrast past delta on every seed is LOWER
    c["FH-F0"] = d3(-0.041, -0.039, -0.040)
    assert A.kr(c, 0.03)["FH-F0"]["verdict"] == "LOWER"
