"""E005 step 2: mutation test of the new checks (checks_e005*.py, purity_e005.py, shares_e005.py, the E004-block
identity test and the oracle gate of oracles_e005.py). Each mutant breaks a pool, the generator, the stream or an
example on purpose; EVERY checker named for it must flag it (a crash is NOT a kill). The unmutated baselines
must be clean first: seed 0's first 800 draws for the per-example, pool, identity and oracle checkers, and
seeds 1-5 x 6,500 draws pooled for the share checks. Generator-level mutants here, example-level ones in
mutation_e005_b.py.
usage: python3 -B mutation_e005.py > ../logs/mutation_e005.txt   (exit code 0 = every mutant killed)"""
import sys
sys.dont_write_bytecode = True
import copy
import types

import train_e004 as T4
import train_e005 as T5
import pools_alias_train as PA
import pools_train as PT
import heldout_e004 as H
import checks_train as C
import checks_train_b as CB
import checks_e005 as CE
import checks_e005_b as CB5
import purity_e005 as P5
import shares_e005 as SH
import oracles_e005 as O5
from mutation_e005_b import EX_MUTANTS, EXTRA_GEN, ngram_flags

N, SEEDS, NS = 800, (1, 2, 3, 4, 5), 6500
EG = P5.eval_alias_grams()
PER_EX = {"structure": CB5.check_structure_e5, "references": C.check_references, "alias refs": CB5.check_alias_refs,
          "values": C.check_values, "echo": CB.check_echo, "heldout": CE.check_heldout_e5}
SNAP = copy.deepcopy((PA.SURNAMES, PA.JOINS, PA.ALIAS_CORR, PA.CASES, PA.N_B_CORR, PT.POOLS))
ATTRS = {k: getattr(T5, k) for k in ("plan_alias", "plan_ind", "BLOCKS", "P_CHAT", "P_ASK_X", "P_BOTH_ALIASED",
                                     "P_HONORIFIC", "T")}
T4_ATTRS = {k: getattr(T4, k) for k in ("D_RANGE", "K_RANGE")}


def restore():
    sur, joins, corr, cases, nb, pools = copy.deepcopy(SNAP)
    PA.SURNAMES[:] = sur
    PA.N_B_CORR[:] = nb
    for d, s in ((PA.JOINS, joins), (PA.ALIAS_CORR, corr)):
        for k in d:
            d[k][:] = s[k]
    PA.CASES.clear()
    PA.CASES.update(cases)
    for vt, P in PT.POOLS.items():
        for key in P:
            P[key][:] = pools[vt][key]
    for k, v in ATTRS.items():
        setattr(T5, k, v)
    for k, v in T4_ATTRS.items():
        setattr(T4, k, v)


def base():
    return T5.take(0, N)


def pooled():
    return [x for s in SEEDS for x in T5.take(s, NS)]


def n_flagged(check, exs):
    """how many units the named checker flags: examples, pool problems, failed share lines, gate failures."""
    kind, _, sub = check.partition(":")
    if kind == "purity":
        return sum(1 for ex in exs if sub in P5.violations(ex, EG))
    if kind == "pool":
        return sum(1 for m in CE.check_alias_pools(PA) if sub in m)
    if kind == "identity":
        return int(not SH.e004_identity(exs, 0)[0])
    if kind == "shares":
        return sum(1 for ok, m in SH.share_checks(exs) if not ok and sub in m)
    if kind == "ngram":
        return ngram_flags(sub, exs)
    if kind == "oracle":
        return sum(1 for m in O5.gate(O5.score(exs)[0]) if sub in m)
    return sum(1 for ex in exs if any(sub in m for m in PER_EX[kind](ex)))


# ---------------- generator- and stream-level mutants ----------------
def gen_with(fn, sample=base):
    def run(_):
        fn()
        return sample()
    return run


def patch_plan_alias(edit):
    orig = ATTRS["plan_alias"]

    def plan(rng):
        case, place, both, events, lab = orig(rng)
        return case, place, both, edit(case, place, events), lab
    return lambda: setattr(T5, "plan_alias", plan)


def patch_plan_ind(edit):
    orig = ATTRS["plan_ind"]

    def plan(rng):
        asked, events = orig(rng)
        return asked, edit(events)
    return lambda: setattr(T5, "plan_ind", plan)


def k4(case, place, ev):
    if case != "latest":
        return ev
    ev = [e for e in ev if not (e["o"] == "B" and e["role"] == "corr")]
    i = next(j for j, e in enumerate(ev) if e["o"] == "A")
    return ev[:i + 1] + [T5.ev("A")] + ev[i + 1:]


def adj_gap(case, place, ev):
    for e in ev:
        if e["ref"] == "alias" and place == "adjacent":
            e["gap"] = 1
    return ev


def other_before(case, place, ev):
    """OTHER: A's latest statement moved to the end, so B's alias correction precedes it."""
    if case != "other":
        return ev
    la = ev[max(j for j, e in enumerate(ev) if e["o"] == "A")]
    return [e for e in ev if e is not la] + [la]


def b_split(case, place, ev):
    bc = [e for e in ev if e["o"] == "B" and e["role"] == "corr"]
    if case == "other" or place == "other_obj" or not bc:
        return ev
    bo = next(e for e in ev if e["o"] == "B" and e["role"] == "orig")
    return [bo] + [e for e in ev if e["o"] == "A"] + bc


def three_obj(ev):
    """a third object Z stated before Y; Y keeps only its last correction, so 7 values always suffice."""
    y_corr = [e for e in ev if e["o"] == "Y" and e["role"] == "corr"]
    ev = [e for e in ev if not (e["o"] == "Y" and e["role"] == "corr")] + y_corr[-1:]
    i = next(j for j, e in enumerate(ev) if e["o"] == "Y")
    return ev[:i] + [T5.ev("Z", "orig", "full")] + ev[i:]


def ind_last(**kw):
    def f(ev):
        last = max(j for j, e in enumerate(ev) if e["o"] == "X")
        ev[last].update(kw)
        return ev
    return f


def ind_y_first(ev):
    last = max(j for j, e in enumerate(ev) if e["o"] == "X")
    yo = next(e for e in ev if e["o"] == "Y" and e["role"] == "orig")
    rest = [e for e in ev if e is not yo]
    return rest[:last] + [yo] + rest[last:]


def e004_proxy(edit):
    """T5 sees a train_e004 whose stream is edited; train_e004 itself (and the identity test's reference) is not."""
    def apply():
        ns = types.SimpleNamespace(**{k: getattr(T4, k) for k in dir(T4) if not k.startswith("__")})

        def stream(seed):
            for i, ex in enumerate(T4.stream(seed)):
                out = edit(i, ex)
                if out is not None:
                    yield out
        ns.stream = stream
        T5.T = ns
    return apply


def setv(obj, **kw):
    return lambda: [setattr(obj, k, v) for k, v in kw.items()]


GEN_MUTANTS = [
    ("eval surnames in the training surname pool", gen_with(lambda: PA.SURNAMES.__setitem__(
        slice(0, 4), H.ALIAS_SURNAMES[:4])), ["pool:eval surname", "heldout:eval surname", "purity:eval surname"]),
    ("eval join 'with' as a training join", gen_with(lambda: PA.JOINS["weekday"].__setitem__(slice(None), ["with {a}"] * 3)),
     ["pool:eval join word", "heldout:eval join", "purity:eval join"]),
    ("eval join 'run by' as a training join", gen_with(lambda: PA.JOINS["city"].__setitem__(
        slice(None), ["run by {a}"] * 3)), ["pool:eval join word", "heldout:eval join", "purity:eval join"]),
    ("alias template sharing a 3-gram with an eval template", gen_with(lambda: PA.ALIAS_CORR["weekday"].__setitem__(
        slice(None), ["{a} can only do {v} now."] * 6)), ["pool:3-gram shared", "purity:eval alias 3-gram"]),
    ("alias template with a pronoun", gen_with(lambda: PA.ALIAS_CORR["colour"].__setitem__(
        slice(None), ["{a} says it is {v} now."] * 6)), ["pool:pronoun", "alias refs:pronoun"]),
    ("three objects in IND (H5)", gen_with(patch_plan_ind(three_obj)),
     ["structure:not exactly 2 objects", "purity:H5 obj3"]),
    ("k = 4 on the asked object (H3)", gen_with(patch_plan_alias(k4)),
     ["structure:more than 3 corrections", "purity:H3 k45"]),
    ("d = 20 (H4)", gen_with(setv(T4, D_RANGE=(20, 20))), ["structure:out of range", "purity:H4 d20"]),
    ("eval objects in the training pool (H6)", gen_with(lambda: PT.POOLS["weekday"]["objects"].__setitem__(
        slice(None), [("tax consultation", "consultation"), ("guitar recital", "recital"), ("massage", "massage")])),
     ["heldout:held-out object", "purity:H6 object"]),
    ("held-out value in a training pool (H6)", gen_with(lambda: PT.POOLS["colour"]["values"].append("tennis")),
     ["values:held-out type", "purity:H6 value"]),
    ("adjacent alias correction after a filler", gen_with(patch_plan_alias(adj_gap)), ["structure:adjacent placement"]),
    ("OTHER alias before A's latest", gen_with(patch_plan_alias(other_before)), ["structure:OTHER"]),
    ("B split around A's chain (adjacent/filler)", gen_with(patch_plan_alias(b_split)), ["structure:B not all before"]),
    ("IND X's last correction direct", gen_with(patch_plan_ind(ind_last(ref="full", gap=None))),
     ["structure:not ellipsis/pronoun"]),
    ("IND Y stated before X's last", gen_with(patch_plan_ind(ind_y_first)),
     ["structure:Y stated before", "references:not adjacent"]),
    ("IND indirect correction after a filler", gen_with(patch_plan_ind(ind_last(gap=1))), ["references:not adjacent"]),
    ("E004 block skips one example", gen_with(e004_proxy(lambda i, ex: None if i == 9 else ex)), ["identity"]),
    ("E004 block example edited", gen_with(e004_proxy(lambda i, ex: dict(ex, d=ex["d"] + 1) if i == 3 else ex)),
     ["identity"]),
    ("block shares 65/25/10", gen_with(setv(T5, BLOCKS={"e004": .65, "alias": .25, "ind": .10}), pooled),
     ["shares:block"]),
    ("chat render 45%", gen_with(setv(T5, P_CHAT=0.45), pooled), ["shares:chat render"]),
    ("alias cases 50/25/25", gen_with(lambda: PA.CASES.update(latest=0.5, earlier=0.25), pooled),
     ["shares:alias case"]),
    ("both aliased 25%", gen_with(setv(T5, P_BOTH_ALIASED=0.25), pooled), ["shares:both aliased"]),
    ("honorific titles 50%", gen_with(setv(T5, P_HONORIFIC=0.5), pooled), ["shares:honorific"]),
    ("B's corrections 0/1/2 uniform", gen_with(lambda: PA.N_B_CORR.__setitem__(slice(None), [0, 1, 2]), pooled),
     ["shares:B's corrections"]),
    ("IND asks X 60%", gen_with(setv(T5, P_ASK_X=0.6), pooled), ["shares:IND asks X"]),
    ("all-LATEST alias stream (O7 shortcut)", gen_with(lambda: (setv(T5, BLOCKS={"e004": 0, "alias": 1.0, "ind": 0})(),
                                                                PA.CASES.clear(), PA.CASES.update(latest=1.0))),
     ["oracle:O7"]),
]


def main():
    B, PL = base(), pooled()
    clean = (not any(CB5.all_problems(ex) for ex in B) and not any(P5.violations(ex, EG) for ex in B)
             and not CE.check_alias_pools(PA) and SH.e004_identity(B, 0)[0] and not O5.gate(O5.score(B)[0])
             and all(ok for ok, _ in SH.share_checks(PL)) and not ngram_flags("template", B)
             and not ngram_flags("text", B))
    print(f"baseline (seed 0 x {N}; seeds 1-5 x {NS} for shares) clean on every checker: {clean}\n")
    survived = 0
    for name, run, checks in GEN_MUTANTS + EXTRA_GEN + EX_MUTANTS:
        restore()
        try:
            exs = run(B)
            got = {c: n_flagged(c, exs) for c in checks}
        except Exception as e:                                   # a crash is not a kill
            got = {"CRASH " + type(e).__name__ + ": " + str(e)[:80]: 0}
        finally:
            restore()
        killed = all(v > 0 for v in got.values())
        survived += not killed
        print(f"{'KILLED  ' if killed else 'SURVIVED'} {name:52s} " + ", ".join(f"{c}={v}" for c, v in got.items()))
    n = len(GEN_MUTANTS) + len(EXTRA_GEN) + len(EX_MUTANTS)
    print(f"\nRESULT: {n - survived}/{n} mutants killed" + ("" if clean else "; BASELINE NOT CLEAN"))
    return 0 if clean and not survived else 1


if __name__ == "__main__":
    sys.exit(main())
