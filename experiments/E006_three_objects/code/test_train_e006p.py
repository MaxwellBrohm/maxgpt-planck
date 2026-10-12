"""E006 arm P generator tests (notes.txt CODE TO WRITE, test_train_e006p). No model, no tokenizer, stdlib only.
  equal      with Q_SPLIT = F_YFIRST = 0, stream_p(s) equals train_e005.stream(s) on the first 20,000 draws, s = 1-5
  one change the counterfactual pairing, per seed, every drawn example of the first N_PAIR: (1) the block and render
             sequence equals C's by drawn index; (2) the E004 block equals C's at the same drawn positions; (3) at
             each ALIAS/IND example, E005's planner and P's planner run from the SAME R_new state consume R_new
             identically, and P's plan equals C's plan when no change is drawn, else C's plan transformed exactly as
             notes (a)/(b) say (written out here independently of train_e006p); an unchanged example equals C's
             example rendered from the same state; (4) every example before the first changed one equals C's
             stream byte for byte; (5) the change is drawn exactly where it is eligible (Q_SPLIT = 1: always)
  structure  every P example passes checks_e006p.all_problems_p (E005's table, with the P structure rule)
  rates      the IND y-first share is F_YFIRST within 0.03 (pooled); ALIAS eligible share split = Q_SPLIT
  position   the drawn position statistics reproduce the numbers in notes.txt (4 decimals)
  source     e006_sets.cand_ids_notorch is lik.cand_ids's source text
usage: python3 -B test_train_e006p.py [--quick]   (exit 0 = every check passed)"""
import ast
import itertools
import os
import random
import sys
sys.dont_write_bytecode = True

import train_e004 as T4
import train_e005 as T5
import train_e006p as TP
import position_e006 as PO
import checks_e006p as CP

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (1, 2, 3, 4, 5)
# drawn position statistics, first 6,404 draws per seed, pooled s1-5 (GL2, GL1, GLX2), fixed in notes.txt
WANT = {"E004": (0.5586, 0.0384, 0.2189), "C": (0.6658, 0.0690, 0.2665), "P": (0.4369, 0.0832, 0.2080)}


def _r(state):
    r = random.Random()
    r.setstate(state)
    return r


def expect_split(events):
    b = [e for e in events if e["o"] == "B"]
    return [b[0]] + [e for e in events if e["o"] == "A"] + b[1:]


def expect_yfirst(events, rp_state, f):
    rp = _r(rp_state)
    changed = rp.random() < f
    if not changed:
        return events, False
    x = [e for e in events if e["o"] == "X"]
    y = [e for e in events if e["o"] == "Y"]
    post = [dict(e) for e in y[1:]]
    post[0]["gap"] = y[0]["gap"]
    if post[0]["ref"] in ("pron", "ell"):
        post[0]["ref"] = T4._pick(rp, T4.REF_FAR)
    return [dict(y[0], gap=None)] + x + post, True


def pair_seed(seed, n, q=None, f=None):
    """-> (failures, counts) of the counterfactual pairing on one seed."""
    q = TP.Q_SPLIT if q is None else q
    f = TP.F_YFIRST if f is None else f
    fails, cnt = [], {"alias": 0, "ind": 0, "split": 0, "yfirst": 0, "eligible": 0, "same_as_C": 0}
    r_mix, r_new = random.Random(1000 * seed + 46), random.Random(1000 * seed + 45)
    r_pos = random.Random(1000 * seed + TP.R_POS)
    e4, cs, ps = T4.stream(seed), T5.stream(seed), TP.stream_p(seed, q, f)
    diverged = False
    for i in range(n):
        exP, exC = next(ps), next(cs)
        u = r_mix.random()
        blk = "e004" if u < T5.BLOCKS["e004"] else ("alias" if u < T5.BLOCKS["e004"] + T5.BLOCKS["alias"] else "ind")
        rend = "chat" if r_mix.random() < T5.P_CHAT else "plain"
        if (exP["block"], exP["render"]) != (blk, rend) or (exC["block"], exC["render"]) != (blk, rend):
            fails.append(f"s{seed}#{i}: block/render sequence differs from C's")
            break
        if blk == "e004":
            if exP != dict(next(e4), block="e004", render=rend) or exP != exC:
                fails.append(f"s{seed}#{i}: E004-block example differs")
            continue
        cnt[blk] += 1
        st, pst = r_new.getstate(), r_pos.getstate()
        rc, rn = _r(st), _r(st)
        if blk == "alias":
            planC = T5.plan_alias(rc)
            planP = TP.plan_alias_p(rn, _r(pst), q)
            elig = planC[0] in ("latest", "earlier") and planC[1] in ("adjacent", "filler") and \
                planC[3][0]["o"] == "B" and sum(e["o"] == "B" for e in planC[3]) >= 2
            cnt["eligible"] += elig
            want = expect_split(planC[3]) if planP[5] else planC[3]
            if planP[5] and not elig:
                fails.append(f"s{seed}#{i}: split drawn on an ineligible plan")
            if q >= 1 and elig and not planP[5]:
                fails.append(f"s{seed}#{i}: eligible plan not split at Q_SPLIT = 1")
            ok = planP[:3] == planC[:3] and planP[4] == planC[4] and planP[3] == want
            changed = planP[5]
        else:
            planC = T5.plan_ind(rc)
            planP = TP.plan_ind_p(rn, _r(pst), f)
            want, ch = expect_yfirst(planC[1], pst, f)
            ok = planP[0] == planC[0] and planP[1] == want and planP[2] == ch
            changed = planP[2]
        if rn.getstate() != rc.getstate():
            fails.append(f"s{seed}#{i}: P's planner consumes R_new differently from E005's")
        if not ok:
            fails.append(f"s{seed}#{i}: P's {blk} plan is not C's plan {'transformed' if changed else 'unchanged'}")
        gen_c = T5.gen_alias if blk == "alias" else T5.gen_ind
        exCf = dict(gen_c(_r(st)), render=rend)
        gen_p = (lambda: TP.gen_alias_p(r_new, r_pos, q)) if blk == "alias" else (lambda: TP.gen_ind_p(r_new, r_pos, f))
        exPr = dict(gen_p(), render=rend)
        if exPr != exP:
            fails.append(f"s{seed}#{i}: the paired replay of stream_p differs from stream_p")
        if changed:
            cnt["split" if blk == "alias" else "yfirst"] += 1
            if exP.get("p_change") != ("split" if blk == "alias" else "yfirst"):
                fails.append(f"s{seed}#{i}: changed example without its p_change mark")
            for k in ("kind", "vtype", "case", "placement", "both_aliased", "objects") + (("asked",) if blk == "alias" else ()):
                if exP.get(k) != exCf.get(k):
                    fails.append(f"s{seed}#{i}: changed example's {k} differs from C's from the same state")
            diverged = True
        else:
            if exP != exCf:
                fails.append(f"s{seed}#{i}: unchanged example differs from C's from the same state")
        if not diverged:
            cnt["same_as_C"] += exP == exC
            if exP != exC:
                fails.append(f"s{seed}#{i}: example before the first change differs from C's stream")
        if len(fails) > 5:
            break
    return fails, cnt


def run(n_eq=20000, n_pair=6404, n_struct=6500, seeds=SEEDS, numbers=("E004", "C", "P")):
    fails, lines = [], []
    for s in seeds:
        a = list(itertools.islice(TP.stream_p(s, 0, 0), n_eq))
        b = list(itertools.islice(T5.stream(s), n_eq))
        if a != b:
            fails.append(f"s{s}: Q = F = 0 does not equal train_e005.stream on {n_eq} draws")
    if TP.take_p(1, 300) != TP.take_p(1, 300) or TP.take_p(1, 300) == TP.take_p(2, 300):
        fails.append("stream_p is not deterministic per seed")
    tot = {}
    for s in seeds:
        fl, cnt = pair_seed(s, n_pair)
        fails += fl
        for k, v in cnt.items():
            tot[k] = tot.get(k, 0) + v
        lines.append(f"pairing s{s}: {cnt}")
    if tot["ind"] and abs(tot["yfirst"] / tot["ind"] - TP.F_YFIRST) > 0.03:
        fails.append(f"IND y-first share {tot['yfirst'] / tot['ind']:.3f} vs F_YFIRST {TP.F_YFIRST}")
    if tot["eligible"] and TP.Q_SPLIT >= 1 and tot["split"] != tot["eligible"]:
        fails.append(f"split {tot['split']} of {tot['eligible']} eligible at Q_SPLIT = 1")
    lines.append(f"pooled: {tot}")
    bad = 0
    for s in seeds:
        for ex in TP.take_p(s, n_struct):
            pr = CP.all_problems_p(ex)
            if pr:
                bad += 1
                if bad <= 3:
                    fails.append(f"s{s} P example fails the per-example table: {pr}")
    lines.append(f"per-example table: {bad} failing of {n_struct * len(seeds)}")
    for name, mk in (("E004", T4.stream), ("C", T5.stream), ("P", TP.stream_p)):
        if name in numbers:
            pl = PO.pooled([PO.stats(list(itertools.islice(mk(s), 6404))) for s in SEEDS])
            got = tuple(None if pl[k] is None else round(pl[k], 4) for k in ("GL2", "GL1", "GLX2"))
            lines.append(f"position {name}: GL2 {got[0]} GL1 {got[1]} GLX2 {got[2]}")
            if got != WANT[name]:
                fails.append(f"position statistics of {name} {got} != notes {WANT[name]}")
    lim = {"GL2": 0.5, "Delta": 0.4, "GLX2": 0.2}
    at = {"all": {"GL2": 0.5, "Delta": 0.4, "GLX2": 0.2}, "alias": {"GL2": 0.5}, "ind": {"GL2": 0.5}}
    if PO.gates(at, lim):
        fails.append(f"position gate fails a stream exactly at the limits: {PO.gates(at, lim)}")
    for blk, key in (("all", "GL2"), ("alias", "GL2"), ("ind", "GL2"), ("all", "Delta"), ("all", "GLX2")):
        above = {b: dict(v) for b, v in at.items()}
        above[blk][key] += 1e-6
        if len(PO.gates(above, lim)) != 1:
            fails.append(f"position gate does not fail (once) a stream 1e-6 above the limit on {blk} {key}")
    if "C" in numbers:
        c = PO.stats(list(itertools.islice(T5.stream(1), 6404)))
        if not PO.gates(c, {"GL2": WANT["E004"][0], "Delta": WANT["E004"][0] - WANT["E004"][1],
                            "GLX2": WANT["E004"][2]}):
            fails.append("the position gate passes E005's own stream (Q = F = 0), which it must fail")
    src = open(os.path.join(HERE, "lik.py")).read()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "cand_ids")
    mine = open(os.path.join(HERE, "e006_sets.py")).read()
    fm = next(n for n in ast.parse(mine).body if isinstance(n, ast.FunctionDef) and n.name == "cand_ids_notorch")
    code = lambda f: [n for n in f.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    body = lambda s, f: "\n".join(s.splitlines()[code(f)[0].lineno - 1:f.end_lineno])
    if body(src, fn) != body(mine, fm):
        fails.append("e006_sets.cand_ids_notorch is not lik.cand_ids's code")
    return fails, lines


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    quick = "--quick" in argv
    fails, lines = run(n_eq=2000 if quick else 20000, n_pair=1500 if quick else 6404, n_struct=800 if quick else 6500,
                       numbers=() if quick else ("E004", "C", "P"))
    print("\n".join(lines))
    for f in fails:
        print("FAIL ", f)
    print("RESULT: " + ("ALL PASS" if not fails else f"{len(fails)} FAILED"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
