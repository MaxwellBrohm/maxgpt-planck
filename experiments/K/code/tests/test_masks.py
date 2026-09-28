"""K2 M1 planted-set fixture (K2 notes DEPENDENCIES, K2-e and K2-f; code/notes.txt R-2 and K3 R-5). tests/plantfix.py
plants one unit per fact (F1-F4), one per skill (S1-S2) and one shared unit X (fact F5 and skill S3) in a scrambled
tiny model; F1, F3 and S1 are saturated (CE barely moves until the gate is under about 0.2), the case the build
defaults missed. At its defaults masks.fit must return exactly the planted set at the planted budget; no other
unit's P(closed) may exceed TAU_FP, and every planted unit's must reach TP_FLOOR (its gate moved well past the 0.17
it starts at). The full check (10 cases x 20 seeds x both floor conventions, 400 fits, 3-4 processes) is
    python tests/mask_sweep.py OUT.jsonl --seeds 12 13 ... 31 --floors k2 default
which reads the same defaults. RUN is what the suite's time allows: a 2-unit set on the default floors and the mixed
plant's 4 fact units on K2's floors (each missed by the build defaults, 300 steps, init 1.0), and MX-SS on seed 6
on K2's floors, REVIEW 2 R-5's miss at K2-e's defaults (1,000 steps, lam 1 per unit: the saturated skill unit S1
left out for the unplanted (0, 11)). MX-SF on seed 0 leaves a planted gate at P(closed) 0.09 at 300 steps.
K3 round 3 (REVIEW 4b M-1): MX-KF seed 98 on the default floors, where one run leaves the saturated fact unit F1
stuck open (P(closed) 0.001, the unplanted (0, 1) hardened instead); fit keeps its second restart, which finds the
set (test_a_stuck_run_is_outscored_by_its_restart).
K3 round 4 (REVIEW 5b T-b): MX-SF seed 300 on the default floors, whose first run hardens the shared unit X in
place of F2; the pick must weigh the objective (SF: g_fact - g_skill), test_the_restart_pick_weighs_the_objective."""
import numpy as np
import pytest
import torch

import ablate as A
import masks as M
import plantfix as PF

TAU_FP, TP_FLOOR = 0.5, 0.3
RUN = [("KF2", 1, "default"), ("MX-SF", 0, "k2"), ("MX-SS", 6, "k2"),       # MX-SS seed 6: REVIEW 2 R-5's miss
       ("MX-KF", 98, "default"),                                              # REVIEW 4b M-1: a stuck first run
       ("MX-SF", 300, "default")]           # REVIEW 5b T-b: a selective objective whose first run keeps X for F2


@pytest.fixture
def one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)                   # tiny model: dispatch-bound, one thread is the fastest
    yield
    torch.set_num_threads(n)


def _ces(m, names):
    """Per-prompt CE at the last position, one batched forward."""
    allp, gold = {**PF.FACTS, **PF.SKILLS}, PF.build()[3]
    idx = torch.tensor([allp[n] for n in names])                 # every fixture prompt has 3 tokens
    with torch.no_grad():
        lg = m(idx)[0][:, -1]
    return torch.nn.functional.cross_entropy(lg, torch.tensor([gold[n] for n in names]), reduction="none").tolist()


def _scaled(m, layer, j, z, names):
    def pre(mod, a):
        h = a[0].clone()
        h[..., j] *= z
        return (h,)
    hd = m.blocks[layer].mlp.down_proj.register_forward_pre_hook(pre)
    try:
        return _ces(m, names)
    finally:
        hd.remove()


def test_fixture_is_clean(one_thread):
    """Each planted unit is the only strong unit for its prompts and silent on the rest; the saturated plants are
    saturated and the graded ones graded (without saturation the build defaults pass every case)."""
    m, facts, skills, gold, units = PF.build()
    names = list(facts) + list(skills)
    with torch.no_grad():
        for n in names:
            p = {**facts, **skills}[n]
            assert int(m(torch.tensor([p]))[0][0, -1].argmax()) == gold[n], n
    eff = A.unit_effects(m, lambda mm: _ces(mm, names))
    hid = m.blocks[0].mlp.gate_proj.out_features
    planted_rows = {layer * hid + j: k for k, (layer, j) in units.items()}
    for row, k in planted_rows.items():
        for c, n in enumerate(names):
            if n in PF.SERVES[k]:
                assert eff[row, c] > 3.5, (k, n, eff[row, c])
            else:
                assert abs(eff[row, c]) < 1e-3, (k, n, eff[row, c])
    others = np.delete(eff, list(planted_rows), axis=0)
    assert others[:, :len(facts)].max() < 0.5                # no unplanted unit carries a fact
    for k in ("F1", "F3", "S1"):
        layer, j = units[k]
        n = PF.SERVES[k][0]
        base, z02 = _ces(m, [n])[0], _scaled(m, layer, j, 0.2, [n])[0]
        assert z02 - base < 0.25, (k, base, z02)            # saturated: near flat down to z = 0.2
    for k in ("F2", "S2"):
        layer, j = units[k]
        n = PF.SERVES[k][0]
        assert _scaled(m, layer, j, 0.5, [n])[0] - _ces(m, [n])[0] > 0.3, k    # graded


@pytest.mark.parametrize("case,seed,floors", RUN)
def test_planted_set_found_at_defaults(case, seed, floors, one_thread):
    obj, facts, b, want = PF.CASES[case]
    m, *_, units = PF.build()
    fc, sc, fl = PF.objectives(facts)
    r = M.fit(m, fc, sc, obj, budget=b, seed=seed, floors=fl if floors == "k2" else (None, None))
    want_u = {units[k] for k in want}
    assert PF.as_set(r["units"]) == want_u, (case, seed, r["units"])
    drop = {(layer, j): v for layer, ps in r["p_drop"].items() for j, v in enumerate(ps)}
    assert len(drop) == 3 * 32
    assert min(drop[u] for u in want_u) >= TP_FLOOR, (case, seed, [drop[u] for u in sorted(want_u)])
    fp = {u: round(v, 3) for u, v in drop.items() if u not in want_u and v > TAU_FP}
    assert not fp, (case, seed, fp)


def test_penalty_is_per_budget_unit_and_defaults_are_the_verified_ones():
    """K2-f: the budget term is priced per budget unit, and fit's defaults are the values the planted-set sweep
    verified (2,000 steps, lam 6, init 0.0, per budget, K2 notes K2-f; 2 restarts, K2-h). On the fixture's budgets
    (1-5) lam 6 per unit also finds the RUN cases, so only this test separates the two scales."""
    import inspect
    e = torch.tensor(12.0)
    assert float(M.penalty(e, 10, 6.0)) == pytest.approx(1.2) and float(M.penalty(e, 10, 6.0, False)) == 12.0
    assert float(M.penalty(torch.tensor(9.0), 10, 6.0)) == 0.0
    d = {k: v.default for k, v in inspect.signature(M.fit).parameters.items()}
    assert (d["steps"], d["lam"], d["init"], d["per_budget"], d["lr"], d["restarts"]) == (2000, 6.0, 0.0, True, 0.05, 2)


def test_sweep_reads_what_fit_returns(one_thread):
    """The sweep reads the set at S steps inside a longer run (tests/mask_sweep.py); at masks.fit's defaults that is
    the set a fit of S steps returns, E[dropped] equal to float noise (REVIEW 3b D-5: the check compared other
    settings and was off by 0.030)."""
    import mask_sweep as MS
    same, gap, top = MS.check_equivalence(seed=1, steps=120)
    assert same and gap < 1e-4, (same, gap, top)


def test_a_stuck_run_is_outscored_by_its_restart(one_thread):
    """K3 round 3 (REVIEW 4b M-1): a single run can leave a saturated unit stuck open (its gate sees no gain until it
    is nearly shut, and the budget penalty pushes it open). MX-KF seed 98 on fit's default floors is such a run: F1
    (2, 5) left at P(closed) 0.001, the unplanted (0, 1) hardened in its place (REVIEW 4b: 4,000 steps or lam 10 do
    not free it). Its exact objective (the hardened set ablated) is lower than the planted set's, so fit's second
    restart (seed 98 + 7919), which finds the set, is kept (RUN checks the default fit)."""
    obj, facts, b, want = PF.CASES["MX-KF"]
    m, *_, units = PF.build()
    fc, sc, _ = PF.objectives(facts)
    one = M.fit(m, fc, sc, obj, budget=b, seed=98, restarts=1)
    stuck = PF.as_set(one["units"])
    assert units["F1"] not in stuck and (0, 1) in stuck and one["p_drop"][2][5] < 0.05, one["units"]
    two = M.fit(m, fc, sc, obj, budget=b, seed=98 + 7919, restarts=1)
    assert PF.as_set(two["units"]) == {units[k] for k in want} and two["hard"] > one["hard"] + 0.5
    c0 = one["ce0"]
    assert M._hard(m, one["units"], fc, sc, c0, (c0[0] + 1.0, c0[1] + 1.0), (1.0, 0.0)) == one["hard"]


def test_the_restart_pick_weighs_the_objective(one_thread):
    """K3 round 4 (REVIEW 5b T-b): _hard is the fit's objective, weights included, so on a selective objective the
    restart pick prefers the selective set. MX-SF (SF: g_fact - g_skill), fit's default floors: {F1, F3, F4, X}
    (MX-SF seed 300's first run: X serves fact F5 and skill S3) scores 3.39 against the planted {F1, F2, F3, F4}'s
    4.02, while g_fact + g_skill ranks it first (4.48). And _hard is linear in the weights for every objective."""
    obj, facts, b, want = PF.CASES["MX-SF"]
    m, *_, units = PF.build()
    fc, sc, _ = PF.objectives(facts)
    with torch.no_grad():
        c0 = (float(fc(m)), float(sc(m)))
    fl = (c0[0] + 1.0, c0[1] + 1.0)

    def sets(keys):
        out = {}
        for k in keys:
            out.setdefault(units[k][0], []).append(units[k][1])
        return out
    planted, with_x = sets(sorted(want)), sets(["F1", "F3", "F4", "X"])
    h = {n: M._hard(m, u, fc, sc, c0, fl, M.OBJECTIVES["SF"]) for n, u in (("planted", planted), ("x", with_x))}
    assert h["x"] < h["planted"] - 0.5, h
    gf, gs = (M._hard(m, with_x, fc, sc, c0, fl, w) for w in ((1.0, 0.0), (0.0, 1.0)))
    assert gf + gs > h["planted"] + 0.3, (gf, gs, h)            # the unweighted sum would keep the set with X
    for w in M.OBJECTIVES.values():
        assert abs(M._hard(m, with_x, fc, sc, c0, fl, w) - (w[0] * gf + w[1] * gs)) < 1e-6, w
