"""RC-12 grader mutants (used by mutation_graders.py). Four kinds, each must be KILLED (some fixture or
conversation test gets the wrong verdict) and none may crash:
  CLAUSE   a named clause removed from every result (E004 style: "remove it, watch a fixture go red")
  OFF      a sub-rule switched off inside the text helpers (grade_text.OFF)
  PARAM    a threshold moved
  SOURCE   a one-line source edit of the grader code (module, old text, new text); old must occur exactly once
Equivalent mutants are not listed; notes.txt names the clauses dropped because no fixture can isolate them."""

CLAUSES = ["v1_degen", "v2_gold", "v2_unsure", "v3_other", "v3_guess", "v3_shotgun", "v4_voice", "v5_echo",
           "a1_degen", "a2_cue", "a3_value", "a4_voice",
           "f1_degen", "f2_echo", "f3_short", "f4_rule", "f5_old",
           "d0_source", "d1_degen", "d2_gold", "d3_other", "d4_assert", "d5_unsure", "d6_voice",
           "r1_degen", "r2_echo", "r3_unsure", "r4_capture", "r5_name", "r6_own", "r7_self",
           "loop_LOOP", "loop_RUNAWAY", "loop_EMPTY", "loop_LEAK", "loop_echo"]

OFF = ["neg", "postneg", "question", "hedge", "hypo", "cond", "alt", "list", "than", "voc", "capture", "myobj",
       "assistrev", "unsure", "echo", "leak", "tri", "equal", "selfcopy", "runaway", "empty", "stale", "guess",
       "shotgun", "echo_contains", "cue", "absvalue", "absvoice", "short", "rule", "old", "source", "other", "assert",
       "listvoice", "claim", "ownname", "self", "stuff"]

PARAM = [("grade_text", "P", "neg_window", 1), ("grade_text", "P", "echo", 0.95), ("grade_text", "P", "alt_chars", 0),
         ("grade_loop", "LP", "tri_rep", 5), ("grade_loop", "LP", "selfcopy", 0.9),
         ("grade_loop", "LP", "selfcopy_min", 40), ("grade_fmt_dyn", "FP", "min_words", 1),
         ("grade_fmt_dyn", "FP", "caps", 0.4), ("grade_fmt_dyn", "FP", "stuff", 5)]

SOURCE = [
    ("grade_text", r"|n't\b|\binstead of", r"|\binstead of"),
    ("grade_text", 'before = NO_INTERJ.sub(" ", text[s0:pos])', "before = text[s0:pos]"),
    ("grade_text", r'HEDGE = rx(r"\b(?:maybe|perhaps', r'HEDGE = rx(r"\b(?:perhaps'),
    ("grade_text", '|guessing|i think"', '|guessing"'),
    ("grade_text", r'not re.search(r",|\bthen\b", before[last.end():], re.I)', "True"),
    ("grade_text", "FIRST_BIND.fullmatch(marks[-1].group(0))", "FIRST_BIND.fullmatch(marks[0].group(0))"),
    ("grade_text", "set(ws[i + 1:i + 4]) & objwords", "set(ws[i + 1:i + 2]) & objwords"),
    ("grade_text", 'return s.endswith("?")', 'return s.endswith("??")'),
    ("grade_text", "return r == q", "return False"),
    ("grade_text", 'if text[max(0, m.start() - 1):m.start()] == "/" or text[m.end():m.end() + 1] == "/":',
     "if False:"),
    ("grade_text", 'return len(toks(body)) <= 4 and ":" not in head', "return len(toks(body)) <= 4"),
    ("grade_text", "if a >= m.end() and NEG_CUE.search(text[m.end():a]):", "if False:"),
    ("grade_text", r'if b <= m.start() and re.search(r"\bbut\b", text[b:m.start()], re.I):', "if False:"),
    ("grade_text", r"\b(?:hi|hello|hey|dear", r"\b(?:hello|hey|dear"),
    ("grade_text", "(?:picked|chose|chosen|went with", "(?:chose|chosen|went with"),
    ("grade_text", r'rx(r"\b(?:your name is|', r'rx(r"\b(?:your nickname is|'),
    ("grade_text", "|i (?:can't|cannot|don't|do not) (?:remember|recall)", ""),
    ("grade_text", "(?:yet )?(?:told|mentioned|said", "(?:yet )?(?:mentioned|said"),
    ("grade_text", r"|\b(?:isn't|is not|aren't|are not|not) (?:listed )?(?:on|in|part of|included in) "
                   r"(?:the|that|this|your)\b", ""),
    # step 5 audit cue additions (grade_cues.py spliced into grade_text): each must be killed by its fixture
    ("grade_text", '|likely|" + GC.HEDGE_MORE + r")', '|likely)'),
    ("grade_text", 'GC.HEDGE_MORE + r")', 'GC.HEDGE_MORE.replace("guess|", "") + r")'),
    ("grade_text", 'GC.HEDGE_MORE + r")', 'GC.HEDGE_MORE.replace("|as far as i (?:know|remember|recall|can tell)", "") + r")'),
    ("grade_text", "(HEDGE.search(sent) or GC.near_hedge(text, s0, s1, HEDGE))", "HEDGE.search(sent)"),
    ("grade_text", 'r"|" + GC.ABS_MORE)', 'r"")'),
    ("grade_text", "GC.ABS_MORE)", 'GC.ABS_MORE.replace("(?:come up|been mentioned)", "(?:been mentioned)"))'),
    ("grade_text", "GC.ABS_MORE)", r'GC.ABS_MORE.replace(r"|\bskips\b", ""))'),
    ("grade_loop", "re.I | re.M)", "re.I)"),
    ("grade_loop", "any(ws == p for p in prior_ws)", "any(ws is p for p in prior_ws)"),
    ("grade_loop", 'stop == "cap"', 'stop == "length"'),
    # OD6 (iii), 2026-09-25: equality charges only replies to asking turns (P X Q D); statement-turn repeats are
    # ack-repeats (killed by fixtures_more.lfix loop_request_* and the od6_* conversation tests)
    ("grade_loop", "equals_earlier(ws, equality_set(prior_ws, prior_kinds, kind))", "equals_earlier(ws, prior_ws)"),
    ("grade_loop", "equals_earlier(ws, equality_set(prior_ws, prior_kinds, kind))",
     "not asks(kind) and equals_earlier(ws, prior_ws)"),
    ("grade_loop", 'ASKS = frozenset("PXQD")', 'ASKS = frozenset("PXQ")'),
    ("grade_loop", "classify(r, s, replies[:i], k, kinds[:i])[0]",
     "classify(r, s, [x for x, y in zip(replies[:i], kinds) if asks(y)], k, [y for y in kinds[:i] if asks(y)])[0]"),
    ("grade_loop", "return equals_earlier(T.lwords(T.norm(reply)), [T.lwords(T.norm(p)) for p in prior])",
     "return False"),
    # verifier 2026-09-25: equality is charged on Q and X replies (conversation tests q_copy, x_copy), and the
    # answer-repeat report (ack_of_answer: a statement-turn reply equal to an earlier reply to an asking turn)
    ("grade_loop", '"equal" not in T.OFF and equals_earlier(',
     '"equal" not in T.OFF and kind != "Q" and equals_earlier('),
    ("grade_loop", 'ASKS = frozenset("PXQD")', 'ASKS = frozenset("PXD")'),
    ("grade_loop", 'ASKS = frozenset("PXQD")', 'ASKS = frozenset("PQD")'),
    ("grade_loop", "answers = [p for p, k in zip(prior, prior_kinds) if asks(k)]", "answers = list(prior)"),
    ("grade_loop", "return equals_earlier(T.lwords(T.norm(reply)), [T.lwords(T.norm(a)) for a in answers])",
     "return False"),
    ("graders", "ack_of_answer=L.conversation_answer_repeats(replies, kinds)",
     "ack_of_answer=L.conversation_acks(replies, kinds)"),
    # F1, decided 2026-09-25 (prereg draft s17 OD6 follow-up): on a statement turn the equality set is the earlier
    # replies to ASKING turns (grade_loop.equality_set). Killed by the od6_* conversation tests (stated_after,
    # ack_twice, ack_fixed) and lfix loop_request_repeat.
    ("grade_loop", "return [p for p, k in zip(prior_ws, prior_kinds) if asks(k)]", "return []"),
    ("grade_loop", "return [p for p, k in zip(prior_ws, prior_kinds) if asks(k)]", "return prior_ws"),
    ("grade_loop", "zip(prior_ws, prior_kinds) if asks(k)]", "zip(prior_ws, prior_kinds) if not asks(k)]"),
    ("grade_loop", "classify(r, s, replies[:i], k, kinds[:i])[0]", 'classify(r, s, replies[:i], k, ["S"] * i)[0]'),
    ("grade_loop", "classify(r, s, replies[:i], k, kinds[:i])[0]", "classify(r, s, replies[:i], k, kinds[1:i + 1])[0]"),
    ("graders", "kinds = L.turn_kinds(rec)", 'kinds = ["P"] * len(replies)'),
    ("grade_loop", 'return [loop_kind(t) for t in sorted(rec["turns"], key=lambda t: t["i"])]',
     'return ["P"] * len(rec["turns"])'),
    # F2, decided 2026-09-25 (prereg draft s17 OD6 follow-up (2)): a D turn annotated asks: false (a small-talk
    # statement filler) has loop kind "d", which does not ask (grade_loop.loop_kind). Killed by the od6_* tests
    # smalltalk_ack, smalltalk_parrot (small talk asks again), same_answer, question_after, request_twice (every D
    # exempt; asks read off the question mark instead of the annotation).
    ("grade_loop", 'return "D" if t["asks"] else "d"', 'return "D"'),
    ("grade_loop", 'return "D" if t["asks"] else "d"', 'return "d"'),
    ("grade_loop", 'return "D" if t["asks"] else "d"', 'return "D" if t["text"].endswith("?") else "d"'),
    ("grade_loop", 'ASKS = frozenset("PXQD")', 'ASKS = frozenset("PXQDd")'),
    ("graders", 'L.classify(reply, stop, prior, probe["kind"], L.turn_kinds(rec)[:len(prior)])',
     'L.classify(reply, stop, prior, "S", ["S"] * len(prior))'),
    # verifier 2026-09-25 (round 2): a ONE-word answer parroted on a statement turn is a LOOP (od6_one_word_stated);
    # a D turn without asks is refused at grading, never read as asking (asks_missing_refused); the F3 measurement
    # graders.eq_only (eq_only_restate, eq_only_wrong, eq_only_stutter)
    ("grade_loop", "return [p for p, k in zip(prior_ws, prior_kinds) if asks(k)]",
     "return [p for p, k in zip(prior_ws, prior_kinds) if asks(k) and len(p) > 1]"),
    ("grade_loop", 'assert type(t.get("asks")) is bool, f"D turn u{t[\'i\']} has no asks annotation (F2)"',
     'if type(t.get("asks")) is not bool: return "D"'),
    ("graders", 'return flags == ["LOOP"] and why == ["equal"]', 'return flags == ["LOOP"]'),
    ("graders", 'if len(res["fails"]) != 1 or res["fails"][0] not in DEGEN_CLAUSES:',
     'if not res["fails"] or res["fails"][0] not in DEGEN_CLAUSES:'),
    ("graders", "eq_only=eq_only(res, *args))", "eq_only=False)"),
    ("graders", "v in stale and gold_ok", "v in stale and not gold_ok"),
    ("graders", "all(T.mentioned(text, v) for v in cands)", "any(T.mentioned(text, v) for v in cands)"),
    ("graders", "for v in outside)", "for v in outside[1:])"),
    ("graders", "ws |= set(T.lwords(h))", "ws |= set()"),
    ("graders", 'float(all(r["ok"] for r in results))', 'float(any(r["ok"] for r in results))'),
    ("graders", "T.FIRST_BIND.search(s) and not T.SECOND_P.search(s)", "T.FIRST_BIND.search(s)"),
    ("graders", 'ABS_NOT_BIND.sub(" ", text)', "text"),
    ("grade_fmt_dyn", "not any(c.isupper() for c in letters)", "sum(c.isupper() for c in letters) < 3"),
    ("grade_fmt_dyn", "ws[0] == arg.lower()", "arg.lower() in ws[:3]"),
    ("grade_fmt_dyn", "T.lwords(tail(text))[-len(want):] == want", "set(want) <= set(T.lwords(text))"),
    ("grade_fmt_dyn", 'return tail(text).endswith("?")', 'return "?" in text'),
    ("grade_fmt_dyn", "return n_sentences(text) == 1", "return n_sentences(text) <= 2"),
    ("grade_fmt_dyn", "return arg.lower() in T.lwords(text)", "return arg.lower() in text.lower()"),
    ("grade_fmt_dyn", 'return s.startswith("[") and s.endswith("]")', 'return s.startswith("[")'),
    ("grade_fmt_dyn", "len(picked) != 1", "len(picked) < 1"),
    ("grade_fmt_dyn", "shape_ok = len(items) == 3", "shape_ok = len(items) >= 3"),
    ("grade_fmt_dyn", "all(s in have for s in gold_st)", "any(s in have for s in gold_st)"),
    ("grade_fmt_dyn", "stems(items[1])[:4] if", "stems(items[0])[:4] if"),
    ("grade_fmt_dyn", 'other_st |= {s for it in probe.get("lure_items") or [] for s in stems(it)}', "pass"),
    ("grade_fmt_dyn", 'and rule_ok(f["absent"]["rule"]', 'and not rule_ok(f["absent"]["rule"]'),
    ("grade_role", "if T.FIRST_ANY.search(s) and not T.SECOND_P.search(s) and any(",
     "if T.FIRST_ANY.search(s) and any("),
    ("grade_role", " or T.assist_reversed(text, own)", ""),
    ("grade_role", "if i <= turn and T.captures(v, text)", "if T.captures(v, text)"),
    ("grade_role", r're.search(r"\bmy\s+" + re.escape(h)', r're.search(r"\byour\s+" + re.escape(h)'),
]
