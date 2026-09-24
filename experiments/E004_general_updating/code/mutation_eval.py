"""E004 step-3 mutation test: every check this step relies on must be seen failing. Each mutant breaks one
thing on purpose; it must be CAUGHT by the named check, and not by a crash. The unbroken baseline must be clean.
  design mutants (shortcut gates, shortcuts.family_gate): the builder or a scorer is broken, the gate must fail
  item mutants (checks_eval.ITEM_CHECKS): one rendered item is corrupted, the named check must flag it
  purity and n-gram mutants: mutation_eval_b.py
Usage: python3 -B mutation_eval.py > ../logs/mutation_eval.txt"""
import copy
import sys

import items_e004 as I
import items_plan as IP
import items_plan_b as IPB
import oracles_e004 as O
import purity_e004 as PU
import ngram_overlap as NG
from mutation_eval_b import PURITY_MUTANTS, ngram_leak
from checks_eval import ITEM_CHECKS
from shortcuts import family_gate
from shortcuts_x import FAKE, score

CHECK = dict(ITEM_CHECKS)


def gate_fails(fam, items, ideal=O.ideal):
    scores = {name: score(items, fn) for name, fn in FAKE}
    scores["IDEAL"] = score(items, ideal)
    return family_gate(fam, scores)[0]


def patched(module, name, new, fn):
    old = getattr(module, name)
    setattr(module, name, new)
    try:
        return fn()
    finally:
        setattr(module, name, old)


# ---- design mutants: return the gate failures (non-empty = caught)
def d_noupd_a_first():
    orig = IPB._noupd_events
    return patched(IPB, "_noupd_events", lambda r, f, c, n: orig(r, dict(f, a_first=True), c, n),
                   lambda: gate_fails("C_noupd", I.build(4004, "C_noupd", 64)))


def d_b_stated_once():
    orig = IP.b_block
    return patched(IP, "b_block", lambda r, m, n, obj=1, last_indirect=None: orig(r, m, 0, obj),
                   lambda: gate_fails("H3", I.build(4004, "H3", 64)))


def d_h1_latest_named():
    orig = IP.a_chain
    return patched(IP, "a_chain", lambda r, k, latest, *a, **kw: orig(r, k, "full", *a, **kw),
                   lambda: gate_fails("H1", I.build(4004, "H1", 64)))


def d_h5_asked_last():
    """H5 with nothing after the asked object's latest statement: recency (last mention) passes."""
    orig = IPB._h5_one

    def one(rng, f):
        s = orig(rng, f)
        last = max(i for i, e in enumerate(s["events"]) if e["obj"] == s["asked"])
        s["events"] = s["events"][:last + 1]
        return s
    return patched(IPB, "_h5_one", one, lambda: gate_fails("H5", I.build(4004, "H5", 64)))


def d_ideal_broken():
    return gate_fails("H2", I.build(4004, "H2", 64), ideal=O.o1_first)


def d_gold_reader():
    """a fake scorer that reads the gold: the gate must fail it."""
    FAKE.append(("M gold reader", lambda V: V.it["gold"]))
    try:
        return gate_fails("H4", I.build(4004, "H4", 64))
    finally:
        FAKE.pop()


# ---- item mutants: (family, mutate(item) -> item or None if not applicable, check name)
def _stmt(it, pred):
    return next((s for s in it["stmts"] if pred(s)), None)


def m_gap_before_pron(it):
    s = _stmt(it, lambda s: s["ref"] in ("pron", "ell"))
    if s is None:
        return None
    it["turns"].insert(s["turn"], ("Why is the sky dark at night?", "Because the sun is below the horizon."))
    for x in it["stmts"]:
        x["turn"] += x["turn"] >= s["turn"]
    return it


def m_value_in_filler(it):
    st = {s["turn"] for s in it["stmts"]}
    i = next(i for i in range(len(it["turns"])) if i not in st)
    u, a = it["turns"][i]
    it["turns"][i] = (u, a + " " + it["values"][0] + " works too.")
    return it


def m_question_echo(it):
    s = _stmt(it, lambda s: s["ref"] == "full" and s["obj"] == it["asked"])
    if s is None or it["frame"] is not None:
        return None
    it["question"] = it["turns"][s["turn"]][0].replace(s["value"], "which one").rstrip(".") + "?"
    return it


def m_h1_latest_shares(it):
    s = [x for x in it["stmts"] if x["obj"] == it["asked"]][-1]
    u, a = it["turns"][s["turn"]]
    it["turns"][s["turn"]] = (u.rstrip(".") + " for the " + it["objects"][it["asked"]][1] + ".", a)
    return it


def m_drop_last_filler(it):
    it["turns"].pop()
    return it


def m_gold_reused(it):
    s = next(x for x in it["stmts"] if x["value"] != it["gold"])
    u, a = it["turns"][s["turn"]]
    it["turns"][s["turn"]] = (u.replace(s["value"], it["gold"]), a.replace(s["value"], it["gold"]))
    s["value"] = it["gold"]
    return it


def m_alias_dropped(it):
    if not it["alias"]:
        return None
    s = _stmt(it, lambda s: s["role"] == "orig" and s["obj"] == it["alias_obj"])
    u, a = it["turns"][s["turn"]]
    it["turns"][s["turn"]] = (u.replace(" " + it["alias"], ""), a)
    return it


def m_training_filler(it):
    from pools_train import FILLERS_TRAIN
    it["turns"][-1] = FILLERS_TRAIN[0]
    return it


def m_k_out_of_range(it):
    it["k"] = 7
    return it


ITEM_MUTANTS = [("pron/ell not adjacent", "H3", m_gap_before_pron, "refs"),
                ("value inside a filler", "H4", m_value_in_filler, "text"),
                ("question echoes a statement", "H6", m_question_echo, "echo"),
                ("H1 latest correction shares a content word", "H1", m_h1_latest_shares, "content"),
                ("d one short", "H7", m_drop_last_filler, "fillers"),
                ("gold value reused by another statement", "H3", m_gold_reused, "text"),
                ("alias missing from the original", "H2", m_alias_dropped, "refs"),
                ("training filler in an eval item", "H4", m_training_filler, "fillers"),
                ("k outside the family range", "H4", m_k_out_of_range, "structure")]


def run(name, fn):
    try:
        return name, bool(fn()), None
    except Exception as e:                     # a crash is not a catch
        return name, False, f"{type(e).__name__}: {e}"


def main():
    rows, base = [], []
    for fam in ("H1", "H2", "H3", "H4", "H5", "C_noupd"):
        base += gate_fails(fam, I.build(4004, fam, 64))
    items = {f: I.build(4004, f, 64) for f in ("H1", "H2", "H3", "H4", "H6", "H7")}
    base += [f"{f} {c}" for f, its in items.items() for it in its for c, fn in ITEM_CHECKS if fn(it)]
    exs = PU.T.take(1, 300)
    base += [f"purity {a}" for ex in exs for a in PU.violations(ex)]
    base += [f"ngram {g}" for g in ({g for g, _ in NG.template_grams("eval")} & {g for g, _ in NG.template_grams("train")})]
    print(f"baseline (gates on H1-H5 and C_noupd, item checks on 384 items, purity on 300 training examples, "
          f"template 5-grams): {'CLEAN' if not base else base[:5]}\n")
    for name, fn in [("C_noupd with A always first", d_noupd_a_first), ("B stated once (no B correction)",
                     d_b_stated_once), ("H1 latest correction named", d_h1_latest_named),
                     ("H5 asked object's latest moved last", d_h5_asked_last), ("IDEAL answers first mention",
                     d_ideal_broken), ("fake scorer reads the gold", d_gold_reader)]:
        rows.append(("design / gate", *run(name, fn)))
    for name, fam, mut, check in ITEM_MUTANTS:
        def fn(fam=fam, mut=mut, check=check):
            for it in items[fam]:
                m = mut(copy.deepcopy(it))
                if m is not None:
                    return CHECK[check](m)
            raise RuntimeError("no applicable item")
        rows.append((f"item / {check}", *run(name, fn)))
    for name, mut, axis in PURITY_MUTANTS:
        rows.append((f"purity / {axis}", *run(name, lambda mut=mut, axis=axis: axis in PU.violations(
            mut(copy.deepcopy(exs[0]))) and (axis != "H7 filler" or PU.h7_overlap([mut(copy.deepcopy(exs[0]))])))))
    rows.append(("ngram / template", *run("training template in an eval pool",
                                          lambda: ngram_leak("city", "orig", "I'm heading to {v} for the {o}."))))
    rows.append(("ngram / template", *run("training acknowledgement in the eval acks",
                                          lambda: ngram_leak(None, "ack", "Thanks for letting me know."))))
    bad = 0
    for kind, name, caught, err in rows:
        bad += not caught
        print(f"{'KILLED' if caught else 'SURVIVED'}  {kind:26s} {name}" + (f"  [crash: {err}]" if err else ""))
    print(f"\n{len(rows) - bad} of {len(rows)} mutants killed (none by a crash: "
          f"{'yes' if not any(r[3] for r in rows) else 'no'}); baseline {'clean' if not base else 'NOT clean'}")
    ok = not bad and not base
    print("RESULT: " + ("ALL PASS" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
