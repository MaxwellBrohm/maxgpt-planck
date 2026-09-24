"""E004 step-3 mutation test, part 2: purity mutants (a training example gets one held-out axis value; the axis
must appear in purity_e004.violations) and n-gram mutants (a training template leaks into an eval pool; the
template-level 5-gram check must see it). Imported by mutation_eval.py, which runs everything."""
import ngram_overlap as NG


# ---- purity mutants: mutate(ex) -> ex; the named axis must appear in purity_e004.violations(ex)
def _h7(ex):
    from pools_eval import H7_FILLERS
    ex["turns"].insert(0, H7_FILLERS[0])
    for s in ex["stmts"]:
        s["turn"] += 1
    return ex


def _obj3(ex):
    """add statements (and object entries) until the example has three objects."""
    for o in range(len(ex["objects"]), 3):
        ex["objects"] = ex["objects"] + [("spare object %d" % o, "spare")]
        ex["stmts"] = ex["stmts"] + [dict(ex["stmts"][0], obj=o)]
    return ex


def _k4(ex):
    base = [s for s in ex["stmts"] if s["obj"] == 0][-1]
    ex["stmts"] += [dict(base, role="corr") for _ in range(4)]
    return ex


PURITY_MUTANTS = [
    ("eval object in a training turn", lambda ex: _set_turn(ex, 0, " The desk lamp is lovely."), "H6 object"),
    ("d 20 (text only, annotation unchanged)", lambda ex: dict(ex, turns=ex["turns"] + [ex["turns"][0]] * 11),
     "H4 d20"),
    ("third object statement", lambda ex: _obj3(ex), "H5 obj3"),
    ("an object corrected 4+ times", _k4, "H3 k45"),
    ("question echoes a statement", lambda ex: dict(ex, question=ex["turns"][ex["stmts"][0]["turn"]][0]),
     "H1/H2 echo"),
    ("alias surname in a turn", lambda ex: _set_turn(ex, 0, " Dr. Okafor agreed."), "H1/H2 alias"),
    ("H7 filler in training", _h7, "H7 filler"),
    ("eval marker on a statement", lambda ex: _set_user(ex, ex["stmts"][-1]["turn"], "Oops, "), "eval marker"),
    ("digit in the answer", lambda ex: dict(ex, answer=" It is 5 now.\n"), "H6 value"),
]


def _set_turn(ex, i, add):
    u, a = ex["turns"][i]
    ex["turns"][i] = (u, a + add)
    return ex


def _set_user(ex, i, pre):
    u, a = ex["turns"][i]
    ex["turns"][i] = (pre + u, a)
    return ex


def ngram_leak(pool_key, role, tpl):
    from pools_eval import ALL_EVAL_POOLS, ACKS_E
    target = ACKS_E["orig"] if role == "ack" else ALL_EVAL_POOLS[pool_key][role]
    target.append(tpl)
    try:
        return {g for g, _ in NG.template_grams("eval")} & {g for g, _ in NG.template_grams("train")}
    finally:
        target.pop()
