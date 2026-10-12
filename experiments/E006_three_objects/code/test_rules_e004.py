"""E004 decision-rule tests on synthetic inputs, plus mutants that must be killed (notes.txt TESTS REQUIRED:
pass rule, majority rule, reading labels, lock-in rule, LR picks; mutants mean-instead-of-min, LIK only, GEN only,
and LR-pick mutants). No model, no files. Exit 0 iff the baseline passes every test and every mutant is killed by a
failed assertion (not by a crash). Writes ../logs/test_rules_e004.txt.
usage: python3 -B test_rules_e004.py"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrics_ft as MF
import rules_e004 as RU

P = RU.PASS


def cells(lik=0.9, gen=0.9, **over):
    """all 9 families at lik / gen; over: {"LIK_H1": 0.5, "GEN_H4": 0.7, ...}"""
    c = {"LIK": {f: lik for f in P}, "GEN": {f: gen for f in P}}
    for k, v in over.items():
        part, fam = k.split("_", 1)
        c[part][fam] = v
    return c


def t_seed_pass():
    assert RU.seed_pass(cells())
    assert RU.seed_pass(cells(0.8, 0.8)), "0.8 exactly passes (>=)"
    assert not RU.seed_pass(cells(LIK_H1=0.79)), "one LIK cell under 0.8 fails the seed"
    assert not RU.seed_pass(cells(GEN_C_twoslot=0.5)), "one GEN cell under 0.8 fails the seed"
    assert not RU.seed_pass(cells(1.0, 1.0, LIK_H2=0.7)), "min, not mean: 8 cells at 1.0 cannot carry one at 0.7"
    assert not RU.seed_pass(cells(LIK_H3=None)), "a missing cell fails"
    assert not RU.seed_pass(None)


def t_majority():
    good, bad = cells(), cells(LIK_H1=0.2)
    assert RU.model_pass([good] * 3 + [bad] * 2, 5)
    assert not RU.model_pass([good] * 2 + [bad] * 3, 5)
    assert RU.model_pass([good, good, bad], 3) and not RU.model_pass([good, bad, bad], 3)
    assert not RU.model_pass([good, good, None, None, None], 5), "missing seeds count against (planned n)"
    assert not RU.model_pass([good, good], 5), "majority over the PLANNED seeds, not the results"
    assert not RU.model_pass([good, good, bad, bad], 4), "strict majority: 2 of 4 is not a majority"


def t_reading_labels():
    good = cells()
    r = RU.reading([good, good, good, cells(LIK_H1=0.1), None], 5)
    assert r["label"] == "PASS", r
    r = RU.reading([cells(GEN_H4=0.5), cells(GEN_H4=0.6, GEN_H7=0.1), cells(GEN_C_noupd=0.2)], 3)
    assert r["label"] == "PARTIAL-G" and r["gen_fail"] == ["H4", "H7", "C_noupd"], r
    # both PARTIAL-G and PARTIAL-X conditions hold: PARTIAL-G is checked first
    r = RU.reading([cells(GEN_H4=0.5), cells(GEN_H4=0.5), cells(LIK_H1=0.1)], 3)
    assert r["label"] == "PARTIAL-G", r
    r = RU.reading([cells(LIK_H4=0.5), cells(LIK_H4=0.3, GEN_H5=0.1), cells(LIK_H1=0.1)], 3)
    assert r["label"] == "PARTIAL-X" and r["limit"] == ["H4", "H5"], r
    r = RU.reading([cells(LIK_H4=0.5), cells(LIK_H4=0.3), cells(LIK_H1=0.1)], 3, model_id="roneneldan/TinyStories-8M")
    assert r["label"] == "PARTIAL-X" and r["limit"] == ["H4"] and "positional confound" in r["detail"], r
    r = RU.reading([cells(LIK_H4=0.5, LIK_H5=0.5, LIK_H6=0.5)] * 3, 3)
    assert r["label"] == "FAIL", "three held-out axes are not PARTIAL-X"
    r = RU.reading([cells(LIK_H1=0.5)] * 3, 3, id_accs=[0.9, 0.9, 0.9])
    assert r["label"] == "FAIL", "an H1 LIK failure is never partial"
    assert any("shortcut signature" in s for s in r["sub"]) and any("does not transfer" in s for s in r["sub"]), r
    r = RU.reading([cells(LIK_C_noupd=0.1)] * 3, 3, id_accs=[0.5, 0.5, 0.9])
    assert r["label"] == "FAIL" and any("recency" in s for s in r["sub"]) and any("in distribution" in s for s in r["sub"]), r
    r = RU.reading([cells(LIK_H4=0.5), cells(LIK_H5=0.5), cells(LIK_H6=0.5)], 3)
    assert r["label"] == "PARTIAL-X" and r["limit"] == ["H4", "H5"], "X = {H4, H5} covers 2 of 3 seeds"
    r = RU.reading([cells(LIK_H3=0.5, GEN_H4=0.1), cells(LIK_H5=0.5, LIK_H6=0.5), cells(GEN_H7=0.5, LIK_H3=0.1)], 3)
    assert r["label"] == "FAIL", "no single X of <= 2 axes covers a majority"
    assert RU.untouched_pass(cells()) and not RU.untouched_pass(cells(GEN_H1=0.0))


def t_below_chance():
    assert RU.below_chance(0.1, 0.2) and not RU.below_chance(0.2, 0.2) and not RU.below_chance(None, 0.2)


def t_lock_in():
    tr = lambda xs: [dict(step=50 * i, **{f: x for f in P}) for i, x in enumerate(xs)]
    assert MF.lock_in_step(tr([0.1, 0.9, 0.9, 0.9]), keys=tuple(P)) == 50
    assert MF.lock_in_step(tr([0.9, 0.5, 0.9, 0.9]), keys=tuple(P)) == 100, "must STAY >= 0.8 to the end"
    assert MF.lock_in_step(tr([0.9, 0.9, 0.9, 0.7]), keys=tuple(P)) is None
    t = tr([0.9, 0.9])
    t[1]["C_twoslot"] = 0.5
    assert MF.lock_in_step(t, keys=tuple(P)) is None, "min over the 9 families"


def t_pick_135m():
    row = lambda mn, me: {"min": mn, "mean": me}
    assert RU.pick_135m({"5e-05": row(0.5, 0.7), "1.5e-04": row(0.6, 0.7)}) == ("CHOSEN", "1.5e-04"), "+0.10 exactly"
    assert RU.pick_135m({"5e-05": row(0.5, 0.7), "1.5e-04": row(0.59, 0.9)}) == ("CHOSEN", "5e-05")
    assert RU.pick_135m({"5e-05": row(0.5, 0.7), "1.5e-04": row(0.8, 0.69)}) == ("CHOSEN", "5e-05"), "mean lower"
    assert RU.pick_135m({"5e-05": row(0.5, 0.7), "1.5e-04": None}) == ("CHOSEN", "5e-05")
    assert RU.pick_135m({"5e-05": None, "1.5e-04": row(0.9, 0.9)})[0] == "NONE"


def t_pick_grid():
    row = lambda mn, me: {"min": mn, "mean": me}
    g = ["5e-05", "3e-04", "1e-03"]
    assert RU.pick_grid({"5e-05": row(0.9, 0.9), "3e-04": row(0.85, 0.95), "1e-03": None}, g) == ("CHOSEN", "5e-05")
    assert RU.pick_grid({"5e-05": row(0.9, 0.9), "3e-04": row(0.9, 0.95), "1e-03": row(0.9, 0.95)}, g) \
        == ("CHOSEN", "3e-04"), "ties on min go to the higher mean, then the SMALLER LR"
    assert RU.pick_grid({"5e-05": row(0.1, 0.2), "3e-04": row(0.2, 0.3), "1e-03": row(0.5, 0.6)}, g) == ("EXTEND", "3e-03")
    assert RU.pick_grid({"5e-05": row(0.1, 0.2), "3e-04": row(0.2, 0.3), "1e-03": row(0.85, 0.9)}, g) == ("CHOSEN", "1e-03")
    g4 = g + ["3e-03"]
    assert RU.pick_grid({lr: row(0.1 * i, 0.5) for i, lr in enumerate(g4, 1)}, g4) == ("EXTEND", "1e-02")
    rows = {"5e-05": row(0.1, 0.2), "3e-04": row(0.2, 0.3), "1e-03": row(0.5, 0.6), "3e-03": row(0.6, 0.6)}
    assert RU.pick_grid(rows, g, final=True) == ("CHOSEN", "3e-03"), "the final pick never extends"
    rows["3e-03"] = row(0.3, 0.4)
    assert RU.pick_grid(rows, g, final=True) == ("CHOSEN", "1e-03"), "final: the top grid LR at min < 0.8 is kept"
    assert RU.pick_grid({"5e-05": None, "3e-04": None}, g)[0] == "NONE"
    assert RU.dev_score({f: 0.9 for f in P} | {"H5": 0.4}) == {"min": 0.4, "mean": round((8 * 0.9 + 0.4) / 9, 4)}
    assert RU.dev_score({f: 0.9 for f in P if f != "H7"})["min"] == 0.0, "a missing family scores 0"


TESTS = [t_seed_pass, t_majority, t_reading_labels, t_below_chance, t_lock_in, t_pick_135m, t_pick_grid]


def run_all():
    fails, crashes = [], []
    for t in TESTS:
        try:
            t()
        except AssertionError as e:
            fails.append(f"{t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            crashes.append(f"{t.__name__}: {type(e).__name__}: {e}")
    return fails, crashes


# ---------------- mutants (each replaces one module attribute, then the whole suite runs) ----------------
def _mean_seed_pass(c):
    xs = [x for part in ("LIK", "GEN") for x in (c or {}).get(part, {}).values() if isinstance(x, (int, float))]
    return bool(c) and len(xs) == 18 and sum(xs) / len(xs) >= RU.THRESH


def _partial_x_any(seed_cells, n):
    for size in (1, 2, 3):
        for X in __import__("itertools").combinations(RU.X_AXES, size):
            k = sum(1 for c in seed_cells if c is not None and set(RU.failing(c)) <= set(X))
            if k * 2 > n:
                return list(X), k
    return None


def _lock_first_reach(traj, keys=("LW", "TS", "NU")):
    for t in traj:
        if all(t.get(k) is not None and t[k] >= MF.THRESH for k in keys):
            return t["step"]
    return None


_orig_reading = RU.reading


def _reading_x_first(seed_cells, n, id_accs=(), model_id=""):
    px = RU.partial_x(list(seed_cells), n)
    if px and not RU.model_pass(list(seed_cells), n):
        return {"label": "PARTIAL-X", "limit": px[0], "detail": "", "sub": []}
    return _orig_reading(seed_cells, n, id_accs, model_id)


MUTANTS = [
    ("seed pass: mean instead of min", "seed_pass", _mean_seed_pass),
    ("seed pass: LIK only", "seed_pass", lambda c: c is not None and all(RU.ok(c["LIK"].get(f)) for f in P)),
    ("seed pass: GEN only", "seed_pass", lambda c: c is not None and all(RU.ok(c["GEN"].get(f)) for f in P)),
    ("threshold strict >", "ok", lambda x: isinstance(x, (int, float)) and x > RU.THRESH),
    ("majority over results", "majority", lambda flags, n: sum(bool(x) for x in flags) * 2 > len(flags)),
    ("majority: half is enough", "majority", lambda flags, n: sum(bool(x) for x in flags) * 2 >= n),
    ("PARTIAL-X: three axes allowed", "partial_x", _partial_x_any),
    ("PARTIAL-X before PARTIAL-G", "reading", _reading_x_first),
    ("PARTIAL-X may hold H1", "X_AXES", ["H1", "H3", "H4", "H5", "H6", "H7"]),
    ("below chance: <=", "below_chance", lambda a, c: a is not None and c is not None and a <= c),
    ("lock-in: first reach", "MF.lock_in_step", _lock_first_reach),
    ("135M: gain 0.05", "LR_135M_MIN_GAIN", 0.05),
    ("135M: mean condition dropped", "pick_135m",
     lambda rows: ("CHOSEN", "1.5e-04") if rows.get("1.5e-04") and rows.get("5e-05") and
     rows["1.5e-04"]["min"] >= rows["5e-05"]["min"] + 0.1 - 1e-9 else ("CHOSEN", "5e-05")),
    ("135M: 5e-05 ineligible -> 1.5e-04", "pick_135m",
     lambda rows: ("CHOSEN", "1.5e-04") if rows.get("5e-05") is None else _orig_135m(rows)),
    ("grid: ties to the larger LR", "pick_grid",
     lambda rows, grid, final=False: _orig_grid({k: (dict(v, mean=v["mean"] + 1e-6 * float(k)) if v else v)
                                                 for k, v in rows.items()}, grid, final)),
    ("grid: never extend", "pick_grid", lambda rows, grid, final=False: _orig_grid(rows, grid, True)),
    ("grid: extend when final", "pick_grid", lambda rows, grid, final=False: _orig_grid(rows, grid, False)),
    ("grid: extend at min >= 0.8", "THRESH_grid", None),
    ("dev score: mean", "dev_score", lambda fa: {"min": round(sum(fa.get(f) or 0 for f in P) / 9, 4),
                                                 "mean": round(sum(fa.get(f) or 0 for f in P) / 9, 4)}),
]
_orig_135m, _orig_grid = RU.pick_135m, RU.pick_grid


def apply(name, attr, val):
    """-> undo function."""
    if attr == "MF.lock_in_step":
        old = MF.lock_in_step
        MF.lock_in_step = val
        return lambda: setattr(MF, "lock_in_step", old)
    if attr == "THRESH_grid":  # the grid extension reads THRESH; lower it so min 0.85 still extends
        old = RU.pick_grid
        RU.pick_grid = lambda rows, grid, final=False: _grid_thresh(rows, grid, final, 0.9)
        return lambda: setattr(RU, "pick_grid", old)
    old = getattr(RU, attr)
    setattr(RU, attr, val)
    return lambda: setattr(RU, attr, old)


def _grid_thresh(rows, grid, final, th):
    saved = RU.THRESH
    RU.THRESH = th
    try:
        return _orig_grid(rows, grid, final)
    finally:
        RU.THRESH = saved


def main():
    lines = []
    fails, crashes = run_all()
    lines.append(f"baseline: {len(TESTS)} tests, {len(fails)} failed, {len(crashes)} crashed")
    lines += [f"  BASELINE FAIL {x}" for x in fails + crashes]
    ok = not fails and not crashes
    killed = 0
    for name, attr, val in MUTANTS:
        undo = apply(name, attr, val)
        try:
            f, c = run_all()
        finally:
            undo()
        by_assert = bool(f)
        killed += by_assert
        ok &= by_assert
        lines.append(f"  {'KILLED' if by_assert else 'SURVIVED'} {name}: {len(f)} assertion failures"
                     + (f", {len(c)} crashes" if c else "") + (f" (first: {f[0][:110]})" if f else ""))
    lines.append(f"mutants killed by an assertion: {killed}/{len(MUTANTS)}")
    lines.append("ALL PASS" if ok else "FAILED")
    txt = "\n".join(lines)
    print(txt)
    os.makedirs(os.path.join(os.path.dirname(HERE), "logs"), exist_ok=True)
    open(os.path.join(os.path.dirname(HERE), "logs", "test_rules_e004.txt"), "w").write(txt + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
