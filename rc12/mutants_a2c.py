"""RC-12 grader mutants for lock step A2c (2026-11-19, rc12/LOCK_STEPS.md M7; graders.g_abs's anycase line,
fixtures_a2c.py, rc12/notes.txt STEP 16A2C). SOURCE mutants as in mutants_graders.py (module, old text, new text; old
occurs exactly once), which appends them to its SOURCE; its OFF list holds the A2c switch "lookcase" (the whole rule).
One mutant per clause, each named after the fixture kind that kills it; each survives the fixtures without
fixtures_a2c: the scope (LOOKUP records only: dropped, or widened to OOD-H's family OODH alone, which also survives
every other A2c fixture; its pin a2c_scope_relabelled_oodh was added when A2c was finished, 2026-11-21), the any-case
flag, every pool value (not only the lure, not only the conversation's values), capitals and mixed case (not only lower
case), the possessive and quotes (any non-letter after or before the value), the reply's start, a reply written all in
capitals, whole-word boundaries. Not mutated: the lenient twin's holder check (it reads values as written, unchanged),
the X probe alone (LOOKUP's one ABS probe is X, which fixtures_a2b asserts on every build)."""

SOURCE_A2C = [
    ("graders", '"lookcase" not in T.OFF and rec["family"] == "LOOKUP"', '"lookcase" not in T.OFF'),
    # a2c_scope_lower_as_written
    ("graders", '"lookcase" not in T.OFF and rec["family"] == "LOOKUP"',
     '"lookcase" not in T.OFF and rec["family"] in ("LOOKUP", "OODH")'),
    # a2c_scope_relabelled_oodh
    ("graders", "(anycase and V.value_re(v, False).search(text))", "(anycase and V.value_re(v).search(text))"),
    # a2c_lure_lower
    ("graders", "(anycase and V.value_re(v, False).search(text))",
     '(anycase and re.search(r"(?<![A-Za-z0-9])" + re.escape(v.lower()) + r"(?![A-Za-z0-9])", text))'),
    # a2c_lure_upper
    ("graders", "(anycase and V.value_re(v, False).search(text))",
     '(anycase and re.search(r"(?<![A-Za-z0-9])(?:" + re.escape(v.lower()) + "|" + re.escape(v.upper())'
     ' + r")(?![A-Za-z0-9])", text))'),
    # a2c_lure_mixed
    ("graders", "(anycase and V.value_re(v, False).search(text))",
     '(anycase and re.search(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9\'])", text, re.I))'),
    # a2c_lure_poss
    ("graders", "(anycase and V.value_re(v, False).search(text))",
     '(anycase and re.search(r"(?<=\\s)" + re.escape(v) + r"(?![A-Za-z0-9])", text, re.I))'),
    # a2c_lure_quoted
    ("graders", "(anycase and V.value_re(v, False).search(text))", "(anycase and V.value_re(v, False).search(text, 1))"),
    # a2c_lure_start
    ("graders", "(anycase and V.value_re(v, False).search(text))",
     "(anycase and not text.isupper() and V.value_re(v, False).search(text))"),
    # a2c_caps_reply_lure
    ("graders", "(anycase and V.value_re(v, False).search(text))",
     '(anycase and v == probe.get("other_value") and V.value_re(v, False).search(text))'),
    # a2c_other_lower
    ("graders", "(anycase and V.value_re(v, False).search(text))",
     "(anycase and v in in_context(rec, probe, pool) and V.value_re(v, False).search(text))"),
    # a2c_guess_lower
    ("graders", "(anycase and V.value_re(v, False).search(text))", "(anycase and v.lower() in text.lower())"),
    # a2c_substring_ok
]
