"""Mutation test of test_gens.py / test_gens_fam.py (Max's rule: a test never watched failing is not evidence).
Each mutant breaks ONE property of the real dev records (or of the filler pool) in memory and must turn the
matching check red WITHOUT a crash. The clean records must pass. Run: python3 -B mutation_gens.py"""
import copy
import os
import sys
import tempfile

import build_dev
import common as C
import test_gens as T
import test_gens_fam as TF


def first(recs, fam, cell=None):
    return next(r for r in recs if r["family"] == fam and (cell is None or r["cell"] == cell))


def main_p(r):
    return TF.main_p(r)


def m_gold_in_question(recs):
    r = first(recs, "RECALL", "d4-7")
    p = main_p(r)
    p["question"] = r["turns"][p["turn"] - 1]["text"] = p["question"][:-1] + f", is it {p['gold']}?"


def m_gold_after_source(recs):
    r = first(recs, "CORR", "U-diff")
    p = main_p(r)
    t = r["turns"][p["src"][0]]           # the turn right after the source
    t["text"] += f" I love {p['gold']} anyway."


def m_candidate_in_question(recs):
    r = first(recs, "TWOHOP", "door")
    p = main_p(r)
    other = [c for c in p["candidates"] if c != p["gold"]][0]
    p["question"] = r["turns"][p["turn"] - 1]["text"] = p["question"].replace("color", f"{other} or other color")


def m_drop_record(recs):
    recs.remove(first(recs, "LOOKUP"))


def m_repeat_ideal(recs):
    r = first(recs, "TOPIC")
    r["turns"][3]["ideal"] = r["turns"][2]["ideal"]


def m_long_turn(recs):
    r = first(recs, "T0")
    r["turns"][0]["text"] += " and then" * 20


def m_d_mismatch(recs):
    main_p(first(recs, "RECALL", "d1-3"))["d"] += 1


def m_recall_bin(recs):
    r = first(recs, "RECALL", "d8-11")
    main_p(r)["d"] = 2
    r["cell"] = "d8-11"


def m_recall_pos(recs):
    r = first(recs, "RECALL", "d4-7")
    r["meta"]["pos"] = {"first": "last", "middle": "first", "last": "middle"}[r["meta"]["pos"]]


def m_bind_same_order(recs):
    a = first(recs, "BIND", "owner")
    b = next(r for r in recs if r["meta"].get("pair_id") == a["meta"]["pair_id"] and r is not a)
    b["turns"] = copy.deepcopy(a["turns"])
    b["meta"]["order"] = a["meta"]["order"]


def m_corr_dc(recs):
    main_p(first(recs, "CORR", "U-same"))["dc"] = 3


def m_udiff_shares_word(recs):
    r = first(recs, "CORR", "U-diff")
    p = main_p(r)
    t = r["turns"][p["src"][0] - 1]
    t["text"] = t["text"][:-1] + f" for the {r['meta']['objA']}."


def m_indirect_not_adjacent(recs):
    for r in recs:
        if r["family"] == "CORR":
            for t in r["turns"]:
                if t["kind"] == "C" and t["facts"][0]["form"] in ("pronoun", "ellipsis", "demonstrative"):
                    prev = r["turns"][t["i"] - 2]
                    prev["kind"], prev["facts"] = "D", []
                    return


def m_echo_on_gold(recs):     # a second echoing statement: the gold's, in an item where a lure already echoes
    r = [x for x in recs if x["family"] == "TWOHOP" and x["cell"] == "day"
         and main_p(x)["echo_value"] != main_p(x)["gold"]][0]
    p = main_p(r)
    t = r["turns"][p["src"][0] - 1]
    t["text"] = f"The day for the {t['facts'][0]['object']} is {p['gold']}."


def m_persist_breaks_rule(recs):
    r = first(recs, "PERSIST", "hold")
    p = r["probes"][-1]
    p["ideal"] = r["turns"][p["turn"] - 1]["ideal"] = "plain reply. Two sentences here"


def m_persist_old_rule_kept(recs):
    r = next(r for r in recs if r["family"] == "PERSIST" and r["meta"]["rules"][0][0] == "closing"
             and r["cell"] == "override")
    p = r["probes"][-1]
    p["ideal"] = r["turns"][p["turn"] - 1]["ideal"] = p["ideal"][:-1] + ", " + p["fmt"]["absent"]["arg"] + "."


def m_lookup_absent_key(recs):
    r = first(recs, "LOOKUP", "rota")
    x = [q for q in r["probes"] if q["kind"] == "X"][0]
    x["key"] = main_p(r)["key"]


def m_k_ref_in_question(recs):
    r = first(recs, "K", "followup")
    p = main_p(r)
    p["question"] = r["turns"][p["turn"] - 1]["text"] = p["question"] + " In " + r["meta"]["referent"] + "."


def m_knowledge_flag(recs):
    first(recs, "TOPIC")["knowledge"] = True


def m_placement(recs):
    r = next(r for r in recs if r["family"] == "OWN" and main_p(r)["turn"] == 12)
    p = main_p(r)
    t = r["turns"][11]
    r["turns"][10], r["turns"][11] = dict(t, i=11), dict(r["turns"][10], i=12)
    p["turn"] = 11


def m_topic_names_gold(recs):
    r = first(recs, "TOPIC")
    p = main_p(r)
    p["question"] = r["turns"][p["turn"] - 1]["text"] = p["question"] + f" The {p['gold']}?"


def m_loop_repeat_request(recs):
    first(recs, "LOOP")["turns"][5]["text"] = "Repeat the story again."


# mutant -> a substring the INTENDED check's message must contain (so a kill by an unrelated check is not
# counted as evidence for this one)
MUTANTS = [(m_gold_in_question, "question contains"), (m_gold_after_source, "repeats the gold"),
           (m_candidate_in_question, "question contains"), (m_drop_record, "family counts"),
           (m_repeat_ideal, "IDEAL replies repeat"), (m_long_turn, "words >"), (m_d_mismatch, "d mismatch"),
           (m_recall_bin, "outside d8-11"), (m_recall_pos, "gold position"), (m_bind_same_order, "orders"),
           (m_corr_dc, "dc 3 < 4"), (m_udiff_shares_word, "U-diff question shares"),
           (m_indirect_not_adjacent, "indirect correction not right after"), (m_echo_on_gold, "echo=False"),
           (m_persist_breaks_rule, "IDEAL breaks"), (m_persist_old_rule_kept, "old rule kept"),
           (m_lookup_absent_key, "X key placement"), (m_k_ref_in_question, "K ref"),
           (m_knowledge_flag, "knowledge flag"), (m_placement, "main probes at u12"),
           (m_topic_names_gold, "question contains"), (m_loop_repeat_request, "repeat request")]


def run_checks(recs):
    T.FAILS.clear()
    for fn in (T.test_counts, T.test_structure, T.test_l2, T.test_g8):
        fn(recs)
    return list(T.FAILS) + TF.run(recs)


def filler_mutants():
    """the filler pool checks: a pool value, a digit, a name, a 2-sentence IDEAL, an E001 5-gram."""
    out = []
    bad = [(("Why is the ocean teal on sunny days?", "Light scatters in the water, which is what you see."),
            "contains 'teal'"),
           (("How do I fix a squeak in the stairs?", "Drive a screw near the nail, about 2 inches away."), "digit"),
           (("Is it true Marcus likes rain?", "Some people love rain for the calm it brings."), "name-like"),
           (("How do I clean a mug?", "Soak it first. Then scrub it gently."), "not one sentence"),
           (("How can I stop my glasses from fogging up?", "A tiny bit of soap on the lenses helps a lot."),
            "5-gram")]
    for item, want in bad:
        C.FILLERS.append(item)
        T.FAILS.clear()
        T.test_fillers([])
        out.append((f"filler {want}", sum(want in f for f in T.FAILS)))
        C.FILLERS.pop()
    return out


def main():
    recs = [r for r in T.load()]
    clean = run_checks(copy.deepcopy(recs))
    ok = not clean
    print(f"clean records: {len(clean)} failures")
    for m, want in MUTANTS:
        mr = copy.deepcopy(recs)
        try:
            m(mr)
            fails = run_checks(mr)
            n = len(fails)
            status = "killed" if any(want in f for f in fails) else f"SURVIVED (wanted {want!r})"
        except Exception as e:                     # a crash is not a kill
            status, n = f"CRASH {type(e).__name__}: {e}", 0
        ok &= status == "killed"
        print(f"  {m.__name__:28s} {status} ({n} failures)")
    for name, n in filler_mutants():
        ok &= n > 0
        print(f"  {name:28s} {'killed' if n else 'SURVIVED'} ({n} failures)")
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(build_dev.dumps(recs[:-1]))
    real, build_dev.OUT = build_dev.OUT, f.name
    T.FAILS.clear()
    T.test_determinism(recs)
    build_dev.OUT = real
    os.unlink(f.name)
    ok &= bool(T.FAILS)
    print(f"  {'m_file_differs':28s} {'killed' if T.FAILS else 'SURVIVED'} ({len(T.FAILS)} failures)")
    print("ALL MUTANTS KILLED" if ok else "MUTATION TEST FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
