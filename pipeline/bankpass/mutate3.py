"""Mutation test of the W3 code (Max's rule: a test never watched failing is not evidence). mutate.py's staging and
runner on a SCRATCH copy, with the W3 tests and one mutant per rule a W3 test claims. The repo is never edited.

    python pipeline/bankpass/mutate3.py SCRATCH_DIR [--only N,M]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mutate as MU  # noqa: E402

TESTS = ["tests/test_bankpass_w3.py", "tests/test_bankpass_w3b.py", "tests/test_bankpass_w3c.py"]
M = [
    ("judge3.py", "    return norm_value(answer) == norm_value(expected)", "    return norm_value(answer) in norm_value(expected)"),
    ("judge3.py", "        return norm_items(answer) == norm_items(expected)",
     "        return sorted(norm_items(answer)) == sorted(norm_items(expected))"),
    ("judge3.py", "        return norm_value(answer) in NOT_SAID_ALIASES", "        return True"),
    ("judge3.py", '    return _LEAD.sub("", a).strip()', "    return a"),
    ("judge3.py", "            break\n    return f", "            pass\n    return f"),
    ("judge3.py", 'for p in re.split(r",|\\band\\b", str(x))]', "for p in [str(x)]]"),
    ("w3state.py", '            it.update(status="dropped", drop="DUP_EXACT_PREJUDGE"', '            it.update(status="kept", drop=None'),
    ("w3state.py", '        if all(b["model"] != blk["model"] for b in lst):', "        if True:"),
    ("w3state.py", "    return ok | {c for c, n in errs.items() if n >= max_errors}", "    return ok"),
    ("w3judge.py", 'if it["status"] != "kept" or it["author"]["model"] == teacher or judged_by(it, teacher):',
     'if it["status"] != "kept" or judged_by(it, teacher):'),
    ("w3judge.py", "NEXT = {ORDER[i]: ORDER[(i + 1) % 3] for i in range(3)}", "NEXT = {o: o for o in ORDER}"),
    ("w3deps.py", "if any(f and g and len(f & g) / len(f | g) >= TOPIC_J for g in fams):\n                continue",
     "if False:\n                continue"),
    ("w3deps.py", 'if it["features"].get("likelihood") == "common" and common >= TOPIC_QUOTA[t] // 2:', "if False:"),
    ("w3deps.py", "TOPIC_J, INTENTS, TOPEN_K, REL_BATCH, VERB_BATCH = 0.5, 8,",
     "TOPIC_J, INTENTS, TOPEN_K, REL_BATCH, VERB_BATCH = 0.5, 7,"),
    ("w3deps.py", "for it in (it for it, a in chosen if a == teacher):", "for it in (it for it, a in chosen):"),
    ("w3deps.py", "        if not cands or not topic_calls_done(recs, t, plan_calls):", "        if not cands:"),
    ("w3items.py", "0 if cap else re.I))", "re.I))"),
    ("w3items.py", '            (drops.append([w, bad[0][0]]) if bad else words.append(w))', "            words.append(w)"),
    ("w3items.py", "            if not w or w in seen:", "            if not w:"),
    ("w3items.py", '    if spec.get("role") in ("query", "bait"):', "    if False:"),
    ("itemize.py", 'lo, hi = specs.pool_specs()["topic"]["min_w"], specs.pool_specs()["topic"]["max_w"]', "lo, hi = 1, 4"),
    ("specs.py", '            req -= {"old", "p_obj"}', "            pass"),
    ("w3plan.py", "LINE_OVERGEN = 3", "LINE_OVERGEN = 1"),
    ("w3render.py", 'GREEDY_KINDS = {"judge", "vote",', 'GREEDY_KINDS = {"vote",'),
    ("w3fill.py", 'or (bank.startswith("key.") and not bank.endswith(".plant")):', "):"),
    ("hold.py", '    todo = iter([c for c in calls if c["call_id"] not in done])', "    todo = iter(list(calls))"),
    ("hold.py", "SMOKE_N, SMOKE_BAD = 8, 6", "SMOKE_N, SMOKE_BAD = 8, 9"),
    ("hold.py", "                    gone = True\n", "                    gone = False\n"),
    ("client.py", '        if d.get("regex_sha256") != D.sha(spec["regex"]):', "        if False:"),
    ("client.py", "        if self.phrases and not d.get(\"phrases\"):", "        if False:"),
    ("client.py", 'if not preset and (d.get("sampling") or {}).get("temperature") != 0.0:', "if False:"),
    ("client.py", 'if not base.startswith(("http://127.0.0.1:", "http://localhost:")):', "if False:"),
    ("bankserve.py", 'if rx is not None and rest.get("constraint") is not None:', "if False:"),
    ("bankserve.py", '        if not info.get("structured_backend"):', "        if False:"),
    ("prompts3.py", '        ask = ask.replace("{lab}", "thing given for that line")', '        ask = ask.replace("{lab}", "{lab}")'),
    ("w3amend.py", 'SUPERSEDED = ("pool.", "pos.", "notewas.", "topen.", "verbs.pos.", "topic.")',
     'SUPERSEDED = ("pos.", "notewas.", "topen.", "verbs.pos.", "topic.")'),
    ("w3amend.py", 'SUPERSEDED = ("pool.", "pos.", "notewas.", "topen.", "verbs.pos.", "topic.")',
     'SUPERSEDED = ("pool.", "pos.", "notewas.", "topen.", "verbs.pos.")'),
    ("w3amend.py", '\n                and r.get("finish") == "length" and', "\n                and"),
    ("w3amend.py", 'not r["call_id"].endswith("r1") and ', ""),
    ("w3amend.py", "seed_words=None", 'seed_words=c.get("seed_words")'),
    ("w3state.py", 'and (w3amend.superseded_item(it)', "and (False"),
    ("w3state.py", 'or (src in its and its[src].get("drop") == "SUPERSEDED_V1")', "or False"),
    ("w3items.py", '    if spec.get("role") == "corr":', "    if False:"),
    ("w3items.py", 'hits = [("TOPIC_NAME", x) for x in w[1:] if x[:1].isupper()][:1]', "hits = []"),
    ("w3items.py", 'hits += [("TOPIC_PERSONAL", m.group(0)) for m in [PERSONAL.search(it["text"])] if m]', "pass"),
    ("w3items.py", '(_PART.sub("", store.straight(x)).strip() for x in rec["lines"])',
     '(store.straight(x) for x in rec["lines"])'),
    ("itemize.py", 'if role in ("plant", "corr", "twin") and f.get("article"):', 'if role == "plant" and f.get("article"):'),
    ("gates.py", "nouns = [x for x in nouns if BK.pronouns(key, x)[0]] or nouns", "pass"),
    ("w3render.py", 'c["n"], c.get("voice")) if c.get("v") == 2', 'c["n"], c.get("voice")) if False'),
    ("prompts.py", '_X = "\\\\n0-9*#`\\"_"', '_X = "\\\\n"'),
    ("w3amend.py", "return superseded(cid) or (amend4_bank(it[\"bank\"]) and not current6(cid))", "return superseded(cid)"),
    ("w3amend.py", '    return call_id.endswith(("v6", "v6r1"))', '    return call_id.endswith("v6")'),
    ("w3amend.py", '            continue                         # a superseded call is never retried', "            pass"),
    ("w3render.py", 'question = dec4 and c["class"] == "K"', 'question = c["class"] == "K"'),
    ("w3render.py", 'lits = [LIT4[bank](f) for f in c["fills"]]', "pass"),
    ("specs.py", '                allowed |= {"o"}', "                pass"),
    ("gates.py", 'if spec["side"] == "user" and bank != "swap.q" and', 'if spec["side"] == "user" and'),
    ("w3state.py", "    prejudge_dedup(its)\n    jb = judge_blocks(recs)", "    jb = judge_blocks(recs)"),
]


def main(argv=None):
    MU.TESTS[:] = TESTS
    MU.M[:] = [m for m in M if m[1] != m[2]]
    return MU.main(argv)


if __name__ == "__main__":
    sys.exit(main())
