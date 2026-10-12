"""E005 step 2 mutation test, part 2: example-level mutants (a copy of the baseline draw with one or a few examples
broken on purpose). Each entry: (name, fn(exs) -> mutated list, [checkers that must ALL flag it]). The harness
(mutation_e005.py) defines the checker names: "<check>:<substring of a problem message>", "purity:<axis>", ...."""
import copy

import pools_alias_train as PA
from pools_alias_train import JOINS
from pools_eval import H7_FILLERS


def _first(exs, pred):
    for i, ex in enumerate(exs):
        if pred(ex):
            return i
    raise LookupError("no example matches the mutant's precondition")


def _edit(exs, pred, fn):
    """deep-copied list with fn applied to the first example matching pred."""
    out = copy.deepcopy(exs)
    fn(out[_first(out, pred)])
    return out


def _replace(ex, old, new):
    ex["turns"] = [(u.replace(old, new), a.replace(old, new)) for u, a in ex["turns"]]
    ex["question"], ex["answer"] = ex["question"].replace(old, new), ex["answer"].replace(old, new)
    ex["aliases"] = {o: (new if a == old else a) for o, a in ex.get("aliases", {}).items()}


def _alias_stmt(ex):
    return next(s for s in ex["stmts"] if s["ref"] == "alias")


def _set_user(ex, s, text):
    u, a = ex["turns"][s["turn"]]
    ex["turns"][s["turn"]] = (text, a)


is_alias = lambda ex: ex["block"] == "alias"
is_both = lambda ex: ex["block"] == "alias" and ex["both_aliased"]
is_single = lambda ex: ex["block"] == "alias" and not ex["both_aliased"]
corr_alias = lambda ex: ex["aliases"][ex["alias_obj"]]


def m_pair(ex):
    _replace(ex, corr_alias(ex), "Ms. Petrov")


def m_title_e004(ex):
    s = ex["stmts"][0]
    _set_user(ex, s, "Dr. Iwasaki says: " + ex["turns"][s["turn"]][0])


def m_ack_title(ex):
    s = _alias_stmt(ex)
    u, a = ex["turns"][s["turn"]]
    ex["turns"][s["turn"]] = (u, f"Thanks, {corr_alias(ex)} it is.")


def m_alias_in_ind(ex):
    ph = ex["objects"][0][0]
    s = ex["stmts"][0]
    _set_user(ex, s, ex["turns"][s["turn"]][0].replace(ph, ph + " that Dr. Iwasaki set up"))
    ex["aliases"] = {0: "Dr. Iwasaki"}


def m_undeclared(ex):
    ex["aliases"] = {o: a for o, a in ex["aliases"].items() if o == ex["alias_obj"]}
    ex["both_aliased"] = False


def m_foreign(ex):
    _replace(ex, corr_alias(ex), "Mr. Smith")


def m_names_obj(ex):
    s = _alias_stmt(ex)
    _set_user(ex, s, f"{corr_alias(ex)} moved the {ex['objects'][s['obj']][1]} to {s['value']}.")


def m_h7(ex):
    ex["turns"][-1] = H7_FILLERS[0]


def m_eval_marker(ex):
    s = _alias_stmt(ex)
    _set_user(ex, s, "Hold on, " + ex["turns"][s["turn"]][0])


def m_echo(ex):
    ex["question"] = ex["turns"][[s for s in ex["stmts"] if s["obj"] == ex["asked"]][-1]["turn"]][0]


def m_second_alias(ex):
    other = next(a for o, a in ex["aliases"].items() if o != ex["alias_obj"])
    s = _alias_stmt(ex)
    _set_user(ex, s, ex["turns"][s["turn"]][0].replace(corr_alias(ex), other))


def m_same_surname(ex):
    a1 = corr_alias(ex)
    o2, a2 = next((o, a) for o, a in ex["aliases"].items() if o != ex["alias_obj"])
    _replace(ex, a2, a2.split(" ")[0] + " " + a1.split(" ", 1)[1])


def m_no_definition(ex):
    o = ex["alias_obj"]
    s = next(st for st in ex["stmts"] if st["obj"] == o and st["role"] == "orig")
    u = ex["turns"][s["turn"]][0]
    for j in JOINS[ex["vtype"]]:
        u = u.replace(" " + j.format(a=ex["aliases"][o]), "")
    _set_user(ex, s, u)


def m_two_alias(ex):
    s = next(st for st in ex["stmts"] if st["obj"] == ex["alias_obj"] and st["role"] == "corr" and st["ref"] != "alias")
    s["ref"] = "alias"
    _set_user(ex, s, f"{corr_alias(ex)} has me down for {s['value']}.")


def m_wrong_gold(exs):
    out = copy.deepcopy(exs)
    for ex in out[::10]:
        ex["gold"] = next((s["value"] for s in ex["stmts"] if s["value"] != ex["gold"]), ex["gold"])
    return out


def relabel(**kw):
    def f(ex):
        ex.update(kw)
    return f


EX_MUTANTS = [
    ("eval honorific+surname pair as an alias", lambda xs: _edit(xs, is_alias, m_pair),
     ["heldout:eval alias pair", "purity:eval alias pair"]),
    ("honorific in an E004-block statement", lambda xs: _edit(xs, lambda e: e["block"] == "e004", m_title_e004),
     ["heldout:honorific or role outside", "purity:undeclared name"]),
    ("alias in an acknowledgement", lambda xs: _edit(xs, is_alias, m_ack_title), ["alias refs:a title outside"]),
    ("alias defined in an IND example", lambda xs: _edit(xs, lambda e: e["block"] == "ind", m_alias_in_ind),
     ["heldout:aliases outside the alias block", "structure:undeclared IND"]),
    ("second alias left undeclared", lambda xs: _edit(xs, is_both, m_undeclared),
     ["heldout:honorific or role outside", "purity:undeclared name"]),
    ("alias from outside the training pools", lambda xs: _edit(xs, is_alias, m_foreign),
     ["heldout:alias not from the training pools"]),
    ("alias correction names its object", lambda xs: _edit(xs, is_alias, m_names_obj),
     ["alias refs:names an object"]),
    ("H7 filler in a training dialogue", lambda xs: _edit(xs, lambda e: e["block"] == "alias" and e["d"] > 0, m_h7),
     ["values:neither a statement nor a training filler", "purity:H7 filler"]),
    ("eval marker on an alias correction", lambda xs: _edit(xs, is_alias, m_eval_marker),
     ["heldout:eval marker", "purity:eval marker"]),
    ("question echoes the gold statement", lambda xs: _edit(xs, is_alias, m_echo), ["echo:echo", "purity:H1/H2 echo"]),
    ("other alias used in the alias correction", lambda xs: _edit(xs, is_both, m_second_alias),
     ["alias refs:without its object's alias"]),
    ("two aliases share a surname", lambda xs: _edit(xs, is_both, m_same_surname), ["structure:share a surname"]),
    ("alias never defined in the original", lambda xs: _edit(xs, is_alias, m_no_definition),
     ["alias refs:not defined right after"]),
    ("two alias corrections of one object", lambda xs: _edit(
        xs, lambda e: e.get("case") == "latest" and e["k"] >= 2, m_two_alias), ["structure:2 alias corrections"]),
    ("other_obj example labelled filler", lambda xs: _edit(
        xs, lambda e: e.get("placement") == "other_obj" and e["case"] == "latest", relabel(placement="filler")),
     ["structure:filler placement"]),
    ("adjacent example labelled other_obj", lambda xs: _edit(
        xs, lambda e: e.get("placement") == "adjacent", relabel(placement="other_obj")),
     ["structure:other-object placement"]),
    ("EARLIER example labelled LATEST", lambda xs: _edit(
        xs, lambda e: e.get("case") == "earlier", relabel(case="latest", kind="alias_latest")), ["structure:LATEST"]),
    ("single-alias example labelled both_aliased", lambda xs: _edit(xs, is_single, relabel(both_aliased=True)),
     ["structure:both_aliased"]),
    ("wrong gold on every 10th example", m_wrong_gold, ["oracle:IDEAL"]),
]


# ---------------- n-gram overlap (ngram_overlap_e005.py's template and text 5-gram gates) ----------------
_EV = {}


def ngram_flags(sub, exs):
    """"template": template-level 5-grams shared with the eval templates; "text": word 5-grams of the training texts
    of exs shared with any eval/dev/probe text (both gated at 0 in ngram_overlap_e005.py)."""
    import items_e004 as I
    import ngram_overlap as N4
    import ngram_overlap_e005 as NG5
    from text_e004 import words
    if not _EV:
        src, _ = N4.eval_texts({name: I.draw(name) for name in I.DRAWS})
        _EV["text5"] = set().union(*(N4.grams(words(t), 5) for ts in src.values() for t in ts))
        _EV["tpl"] = {g for g, _ in N4.template_grams("eval")}
    if sub == "template":
        return len(_EV["tpl"] & {g for g, _ in NG5.template_train_grams()})
    texts, _ = NG5.train_texts(exs)
    return len(_EV["text5"] & set().union(*(N4.grams(words(t), 5) for t in texts)))


def _regen(patch):
    def run(_):
        import train_e005 as T5
        patch()
        return T5.take(0, 800)
    return run


def _before_original():
    """OTHER, adjacent/filler: B's alias correction moved to the very start (before B's original)."""
    import train_e005 as T5
    orig = T5.plan_alias

    def plan(rng):
        case, place, both, ev, lab = orig(rng)
        if case == "other" and place != "other_obj":
            al = next(e for e in ev if e["ref"] == "alias")
            ev = [al] + [e for e in ev if e is not al]
        return case, place, both, ev, lab
    T5.plan_alias = plan


EXTRA_GEN = [
    ("eval alias template copied into training", _regen(lambda: PA.ALIAS_CORR["weekday"].__setitem__(
        slice(None), ["{a} can only fit me in on {v} now."] * 6)),
     ["ngram:template", "ngram:text", "pool:3-gram shared"]),
    ("alias correction before its object's original", _regen(_before_original),
     ["structure:before its object's original"]),
]
