"""E004 step 2: mutation test of the training-generator checks (checks_train.py, checks_train_b.py and the
determinism test in test_train_gen.py). Each mutant breaks the generator, a pool or an example on purpose; the
named check must report it (a crash is NOT a kill). The unmutated baseline must be clean on the same draw.
usage: python3 mutation_train_gen.py > ../logs/mutation_train_gen.txt   (exit code 0 = every mutant killed)"""
import sys
sys.dont_write_bytecode = True
import copy
import random

import train_e004 as T
import pools_train as PT
import fillers_train as FT
import checks_train as C
import checks_train_b as CB

N = 400
PER_EX = {"structure": C.check_structure, "references": C.check_references, "values": C.check_values,
          "answer": CB.check_answer, "echo": CB.check_echo, "heldout": CB.check_heldout}
SNAP = copy.deepcopy((PT.POOLS, FT.FILLERS_TRAIN, PT.MARKERS, PT.NONCORR_MARKERS, PT.ACKS, PT.MARKER_FMT))
ATTRS = {k: getattr(T, k) for k in ("K_RANGE", "D_RANGE", "REF_FAR", "REF_ELIG", "stream", "VTYPES")}


def restore():
    pools, fills, markers, noncorr, acks, fmt = copy.deepcopy(SNAP)
    for vt, P in PT.POOLS.items():
        for key in P:
            P[key][:] = pools[vt][key]
    FT.FILLERS_TRAIN[:] = fills
    PT.MARKERS[:] = markers
    PT.NONCORR_MARKERS[:] = noncorr
    for role in PT.ACKS:
        PT.ACKS[role][:] = acks[role]
    PT.MARKER_FMT.clear()
    PT.MARKER_FMT.update(fmt)
    for k, v in ATTRS.items():
        setattr(T, k, v)


def n_bad(check, exs):
    return sum(1 for ex in exs if check(ex))


def pool_level():
    return {"pools": len(CB.check_pools()), "fillers": len(CB.check_fillers()),
            "e001": len(CB.e001_filler_overlap())}


def determinism_broken():
    return T.take(0, 50) != T.take(0, 50)


# ---- mutants: (name, apply, how to detect) where detect(exs) -> dict of check -> problem count ----
def m_pool(vt, role, tpl):
    return lambda: PT.POOLS[vt][role].append(tpl)


def m_only(vt, **roles):
    """replace whole pools so the mutant occurs in (nearly) every dialogue of that value type."""
    def f():
        for role, tpl in roles.items():
            PT.POOLS[vt][role][:] = [tpl]
    return f


def m_k_leak():
    T.K_RANGE = (1, 4)
    T.VTYPES = ["city"]           # 16 values: enough for k = 4 in every kind, so the leak cannot crash


def m_ack_echo():
    PT.ACKS["orig"][:] = ["It is set for {v}."]
    PT.POOLS["weekday"]["ans"][:] = ["Your {o} is set for {v}."]


def m_filler(q, a):
    return lambda: FT.FILLERS_TRAIN.append((q, a))


def m_eval_marker():
    PT.MARKERS.append("oops,")
    PT.MARKER_FMT["oops,"] = "Oops, {s}"


def m_setattr(name, value):
    return lambda: setattr(T, name, value)


def m_unseeded():
    T.stream = lambda seed: (T.gen(random.Random()) for _ in iter(int, 1))


EX_EDITS = {   # mutants applied to sampled examples rather than to the generator
    "gold = first mention": lambda ex: dict(ex, gold=ex["stmts"][0]["value"]),
    "question copies a statement frame": lambda ex: dict(ex, question=ex["turns"][ex["stmts"][0]["turn"]][0]
                                                          .replace(ex["stmts"][0]["value"], "what")),
    "undeclared kind (binding family)": lambda ex: dict(ex, kind="bind"),
    "held-out value type": lambda ex: dict(ex, vtype="sport"),
    "three objects": lambda ex: dict(ex, objects=ex["objects"] + [("tent", "tent")], n_obj=ex["n_obj"] + 1),
    "k recorded wrong": lambda ex: dict(ex, k=ex["k"] + 1),
    "answer without newline": lambda ex: dict(ex, answer=ex["answer"].rstrip("\n")),
}

MUTANTS = [
    ("echo in training (question)", m_only("weekday", ask="So the {o} got moved to which day?",
                                           corr="The {o} got moved to {v}."), ["pools", "echo"]),
    ("echo in training (ack vs answer)", m_ack_echo, ["pools", "echo"]),
    ("alias defined in training", m_pool("weekday", "orig", "My {o} with Mr. Adeyemi is on {v}."), ["heldout"]),
    ("eval marker in training", m_eval_marker, ["heldout", "references"]),
    ("held-out object in a filler", m_filler("How do I track a package delivery?", "Check the tracking page."),
     ["fillers", "heldout"]),
    ("held-out value in a template", m_only("colour", orig="I played tennis near the {v} {o}."),
     ["pools", "values"]),
    ("training value in a filler", m_filler("Why is Friday so tiring?", "The whole week catches up with you."),
     ["fillers", "values"]),
    ("digit in a filler", m_filler("How long is a nap?", "About 20 minutes is plenty."), ["fillers", "values"]),
    ("E001 filler 5-gram", m_filler("How do I keep herbs fresh longer?", "Trim the stems and chill them."),
     ["e001"]),
    ("k range leak (1-4)", m_k_leak, ["structure"]),
    ("d range leak (0-12)", m_setattr("D_RANGE", (0, 12)), ["structure"]),
    ("indirect reference not adjacent", m_setattr("REF_FAR", {"full": .3, "head": .3, "pron": .2, "ell": .2}),
     ["references"]),
    ("retracting marker on a non-correction", lambda: PT.NONCORR_MARKERS.append("sorry, I meant"),
     ["references"]),
    ("bare answer", m_pool("city", "ans", "{v}."), ["answer"]),
    ("negation in an answer", m_pool("month", "ans_upd", "{v}, not the old month."), ["answer", "pools"]),
    ("hedge in an answer", m_pool("colour", "ans", "Probably {v}."), ["answer", "pools"]),
    ("user's voice in an answer", m_pool("weekday", "ans", "My {o} is on {v}."), ["answer"]),
    ("unseeded stream", m_unseeded, ["determinism"]),
]


def run():
    fails = []
    restore()
    base = T.take(0, N)
    base_counts = {k: n_bad(f, base) for k, f in PER_EX.items()}
    base_counts.update(pool_level())
    base_counts["determinism"] = int(determinism_broken())
    base_counts["coverage"] = len(C.coverage_missing(T.take(0, 6400)))
    clean = not any(base_counts.values())
    print(f"baseline (unmutated, n = {N}): {base_counts} -> {'clean' if clean else 'NOT CLEAN'}")
    if not clean:
        fails.append("baseline not clean")
    for name, apply, expect in MUTANTS:
        restore()
        try:
            apply()
            exs = T.take(0, N) if name != "unseeded stream" else []
            got = {k: n_bad(PER_EX[k], exs) for k in expect if k in PER_EX}
            got.update({k: v for k, v in pool_level().items() if k in expect})
            if "determinism" in expect:
                got["determinism"] = int(determinism_broken())
            killed = all(got[k] > 0 for k in expect)
            print(f"{'KILLED' if killed else 'SURVIVED'}  {name}: {got}")
        except Exception as e:           # a crash is not a kill
            killed = False
            print(f"CRASH   {name}: {type(e).__name__}: {e}")
        if not killed:
            fails.append(name)
    restore()
    for name, edit in EX_EDITS.items():
        exs = [edit(ex) for ex in base]
        got, crashed = {}, []
        for k, f in PER_EX.items():      # a check that crashes does not count; another must report it
            try:
                got[k] = n_bad(f, exs)
            except Exception as e:
                crashed.append(f"{k}:{type(e).__name__}")
        killed = any(got.values())
        print(f"{'KILLED' if killed else 'SURVIVED'}  {name}: { {k: v for k, v in got.items() if v} }"
              + (f" (crashed, not counted: {crashed})" if crashed else ""))
        if not killed:
            fails.append(name)
    PT.POOLS["weekday"]["corr"].append("The {o} now falls on {v}.")      # entry added after the draw
    miss = C.coverage_missing(base)
    killed = ("weekday", "corr", len(PT.POOLS["weekday"]["corr"]) - 1) in miss
    print(f"{'KILLED' if killed else 'SURVIVED'}  unused pool entry: coverage reports {len(miss)} missing entries")
    if not killed:
        fails.append("unused pool entry")
    restore()
    print("\nRESULT:", "ALL MUTANTS KILLED" if not fails else f"{len(fails)} NOT KILLED: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(run())
