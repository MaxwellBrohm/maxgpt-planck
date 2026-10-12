"""Mutation test for the grader self-test: break graders on purpose and confirm
battery.selftest() goes red for each mutant. Exits non-zero if any mutant survives."""
import importlib, battery as B

def run_mutant(name, patch):
    importlib.reload(B)
    patch(B)
    try:
        B.selftest()
        return False  # mutant survived
    except AssertionError:
        return True

def always(v):
    def p(B):
        for t in B.TESTS:
            for c in t["checks"]:
                c["fn"] = (lambda a, r, v=v: v)
    return p

def one_family(fam, v):
    def p(B):
        for t in B.TESTS:
            for c in t["checks"]:
                if c["name"].startswith(fam):
                    c["fn"] = (lambda a, r, v=v: v)
    return p

def drop_guard(B):
    for t in B.TESTS:
        for c in t["checks"]:
            if c.get("guard") and "gold" in c:
                c["fn"] = (lambda a, r, g=c["gold"]: B.mentions(g, a))

def leak_gold_into_question(B):
    t = [t for t in B.TESTS if t["id"] == "R_d2_name"][0]
    t["turns"][-1] = "Is my name Priya?"

def break_n_sentences(B):
    B.n_sentences = lambda t: 1

def break_parse_list(B):
    B.parse_list = lambda t: ["a", "b", "c"]

def drop_capture(B):
    B.captures = lambda p, a: False

def drop_fresh(B):
    B.fresh = lambda a, r, t: True

def drop_vocative(B):
    B.strip_vocative = lambda p, a: a

mutants = {
    "greeting-echo filter removed": drop_vocative,
    "format freshness guard removed": drop_fresh,
    "role-capture guard removed": drop_capture,
    "all graders return True": always(True),
    "all graders return False": always(False),
    "recall graders return True": one_family("recall", True),
    "one-sentence graders return True": one_family("one_sentence", True),
    "caps graders return True": one_family("caps", True),
    "end-question graders return True": one_family("end_q", True),
    "corrected graders return True": one_family("corrected", True),
    "deflection guard removed": drop_guard,
    "gold answer leaked into its question": leak_gold_into_question,
    "sentence counter always 1": break_n_sentences,
    "list parser returns junk": break_parse_list,
}
survivors = []
for k, p in mutants.items():
    killed = run_mutant(k, p)
    print(("KILLED  " if killed else "SURVIVED"), k)
    if not killed:
        survivors.append(k)
importlib.reload(B)
B.selftest()
print("unmutated selftest passes;", "all mutants killed" if not survivors else f"SURVIVORS: {survivors}")
raise SystemExit(1 if survivors else 0)
