"""Mutation test for the hardened chat-probe graders: break graders on purpose and confirm
battery.selftest() goes red for each mutant. Exits non-zero if any mutant survives.

Part 1: the original 14 mutants (patch module attributes after a reload).
Part 2 (E001): one mutant per new guard, made by editing the SOURCE text of battery.py and
executing it as a fresh module, so inline rules can be broken too. Each edit must match the
source exactly once, or the mutant counts as broken (not silently skipped).
"""
import importlib, os, sys, types

import battery as B

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "battery.py")).read()


# ------------------------------------------------------------------ part 1 (original)
def run_mutant(patch):
    importlib.reload(B)
    patch(B)
    try:
        B.selftest()
        return False
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


ORIGINAL = {
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

# ------------------------------------------------------------------ part 2 (E001 source mutants)
NEVER = 'rx(r"(?!x)x")'
SOURCE_MUTANTS = {
    "negation guard removed": [
        ('NEGATION = rx(r"(?:\\bnot|', 'NEGATION = rx(r"(?!x)x(?:\\bnot|'),
        ('NO_BEFORE = rx(r"\\bno\\W+$")', f"NO_BEFORE = {NEVER}")],
    "question-sentence rule removed": [("    if is_question(sent):\n        return False\n", "")],
    "'or' guess-list rule removed": [("def or_alternative(text, m, s0, s1):", "def or_alternative(text, m, s0, s1):\n    return False")],
    "'or' rule back to any nearby 'or' (false-fail fix removed)": [
        ("            if w[0].isupper() or w[0].isdigit():", "            if True:")],
    "hedge rule removed": [('HEDGE = rx(r"\\b(?:maybe', 'HEDGE = rx(r"(?!x)x\\b(?:maybe')],
    "example rule removed": [('HYPO = rx(r"\\b(?:for example', 'HYPO = rx(r"(?!x)x\\b(?:for example')],
    "conditional rule removed": [('COND = rx(r"\\b(?:if|', 'COND = rx(r"(?!x)x\\b(?:if|')],
    "conditional scope runs to sentence end (false-fail fix removed)": [
        ('    return not re.search(r",|\\bthen\\b", before[last.end():], re.I)', '    return True')],
    "list-line guess rule removed": [("    if single and bare_list_value(text, m):\n        return False\n", "")],
    "list-line rule counts labelled lines (false-fail fix removed)": [
        ('    return len(re.findall(r"[\\w\']+", body)) <= 4 and ":" not in head', '    return True')],
    "unsure-reply rule removed": [('UNSURE = rx(r"\\b(?:i\'m not sure', 'UNSURE = rx(r"(?!x)x\\b(?:i\'m not sure')],
    "nearest-person capture removed (original sentence rule only)": [
        ("        if marks:\n            if FIRST_BIND.fullmatch(marks[-1].group(0)):\n                return True\n        else:\n",
         "        if True:\n")],
    "'I <verb>' binding removed (possessives and I'm only)": [
        ('FIRST_BIND_RX = r"(?:my|mine|i\'m|i am|i was|i\'ve|i have|me|i\\s+(?!(?:" + NEUTRAL_I + r")\\b)[a-z]+)"',
         'FIRST_BIND_RX = r"(?:my|mine|i\'m|i am|i was|i\'ve|i have|me)"')],
    "cognition verbs count as binding (false-fail fix removed)": [('NEUTRAL_I = r"think|', 'NEUTRAL_I = r"(?!x)x"\nNEUTRAL_OLD = r"think|')],
    "capture fallback back to any first person (false-fail fix removed)": [
        ("            if FIRST_BIND.search(sent) and not SECOND_P.search(sent):", "            if FIRST_P.search(sent) and not SECOND_P.search(sent):")],
    "competing-number rule removed": [
        ("        if number is not None and competing_numbers(ans, number, allowed):\n            return False\n", "")],
    "incidental-unit rule removed": [
        ('            spans = [(a, b) for a, b in spans if not re.match(r"\\s*(?:" + units + r")\\b", t[b:], re.I)]\n', "")],
    "K_time back to any standalone 4 / 3": [
        ('T4 = r"(?<![\\d:.])4(?::00)?', 'T4 = r"(?<![\\d:])4(?![\\d])|(?<![\\d:.])4(?::00)?'),
        ('T3 = r"(?<![\\d:.])3(?::00)?', 'T3 = r"(?<![\\d:])3(?![\\d])\\s*(pm|p\\.m|o\'clock)|\\bthree\\b|(?<![\\d:.])3(?::00)?')],
    "change-phrase requirement removed (last mention wins)": [
        ("        return max(pos_new) > max(pos_old) and changed", "        return max(pos_new) > max(pos_old)")],
    "stale-after context removed": [('STALE_POST = rx(r"^\\W*', 'STALE_POST = rx(r"(?!x)x^\\W*')],
    "stale-before context removed": [('STALE_CTX = rx(r"(?:\\bfrom|', 'STALE_CTX = rx(r"(?!x)x(?:\\bfrom|')],
    "truncation rule removed": [('        if check["name"].startswith(TRUNC_SENSITIVE):\n            return None, True\n', "")],
    "truncation rule applied to every check": [('        if check["name"].startswith(TRUNC_SENSITIVE):', '        if True:')],
    "'correct day is' change phrase removed": [("|\\binstead|\\bcorrect(?:ed|ion)?)", "|\\binstead)")],
    "unsure-reply rule widened to 'here are some ideas' (false-fail fix removed)": [
        ("i don't know|i do not know)\\b\")", "i don't know|i do not know|some options|here are some (?:options|ideas|suggestions))\\b\")")],
    "new deflection phrases removed": [('NO_MEMORY_2 = rx(r"(?:don\'t|do not) have', 'NO_MEMORY_2 = rx(r"(?!x)x(?:don\'t|do not) have')],
    "known wrong binding (parrot) removed": [('PARROT = r"\\b(?:you', 'PARROT = r"(?!x)x\\b(?:you')],
    "RI_binding hedged-alternative pattern removed": [('            r"|\\b(?:it(?:\'s| is)|maybe|or)\\s+lena\\b")', '            r"")')],
    "knowledge checks graded as strictly as user facts": [
        ("    if not strict:\n        return True\n", "")],
}


def run_source_mutant(edits):
    src = SRC
    for a, b in edits:
        c = src.count(a)
        if c != 1:
            return f"BROKEN MUTANT: edit target found {c} times: {a[:60]!r}"
        src = src.replace(a, b)
    mod = types.ModuleType("battery_mutant")
    try:
        exec(compile(src, "battery_mutant", "exec"), mod.__dict__)
        mod.selftest()
        return False
    except AssertionError:
        return True
    except Exception as e:  # a crash also counts as red, but report it
        return f"KILLED BY CRASH: {type(e).__name__}: {str(e)[:80]}"


if __name__ == "__main__":
    survivors, broken = [], []
    for k, p in ORIGINAL.items():
        killed = run_mutant(p)
        print(("KILLED  " if killed else "SURVIVED"), k)
        if not killed:
            survivors.append(k)
    importlib.reload(B)
    for k, edits in SOURCE_MUTANTS.items():
        r = run_source_mutant(edits)
        if isinstance(r, str) and r.startswith("BROKEN"):
            print("BROKEN  ", k, r)
            broken.append(k)
        elif r:
            print("KILLED  ", k, "" if r is True else r)
        else:
            print("SURVIVED", k)
            survivors.append(k)
    B.selftest()
    print(f"unmutated selftest passes; {len(ORIGINAL) + len(SOURCE_MUTANTS)} mutants;",
          "all killed" if not survivors and not broken else f"SURVIVORS: {survivors} BROKEN: {broken}")
    sys.exit(1 if survivors or broken else 0)
