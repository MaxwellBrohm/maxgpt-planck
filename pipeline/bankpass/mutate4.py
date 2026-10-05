"""Mutation test of the W4 / W5 code (Max's rule: a test never watched failing is not evidence): one mutant per rule a
W4 or W5 test claims, each applied to a SCRATCH copy (mutate.py's staging and runner), never the repo. Paths are
relative to pipeline/, so the skeleton-side wiring (banks, events, assemble, pools, topic_words) is covered too.

    python pipeline/bankpass/mutate4.py SCRATCH_DIR [--only N,M]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mutate as MU  # noqa: E402

TESTS = ["tests/test_bankpass_w4.py", "tests/test_bankpass_w4b.py", "tests/test_bankpass_w4c.py",
         "tests/test_bankpass_w5.py", "tests/test_bankpass_load.py"]
W, G_, BK, F_, FZ, V, D, ME = ("bankpass/w4words.py", "bankpass/w4gates.py", "bankpass/w4banks.py",
                               "bankpass/w4feats.py", "bankpass/w4freeze.py", "bankpass/w4vote.py",
                               "bankpass/w5dry.py", "bankpass/w5measure.py")
M = [
    (W, "NEED_VOTES, NEED_AGREE = 3, 2", "NEED_VOTES, NEED_AGREE = 2, 2"),
    (W, 'if proxy in CONTENT or proxy == "func":\n        c[proxy] += 1', "if False:\n        c[proxy] += 1"),
    (W, '        return proxy, "pending"', '        return proxy, "amb"'),
    (W, '    return not w3amend.superseded(r["call_id"])', "    return True"),
    (W, 'if r.get("kind") != kind or r.get("problem") or r.get("error"):', 'if r.get("kind") != kind:'),
    (W, 'if sum(v == "same" for v in vs.values()) < NEED_AGREE:', 'if sum(v == "same" for v in vs.values()) < 1:'),
    (W, 'elif adj not in fams or fams[adj]["pos"] != "adj" or fams[adj]["pos_status"] != "confirmed":',
     "elif adj not in fams:"),
    (W, "if n < NEED_AGREE or form == regular_ed(verb) or form == verb:", "if n < NEED_AGREE or form == verb:"),
    (W, "            if form not in eligible:", "            if False:"),
    (W, 'elif other and other["rank"] and (other["pos_status"], other["pos"]) != ("confirmed", "verb"):', "elif False:"),
    (W, '            by[form].update(rank="", lists="",', '            by[form].update('),
    (W, 'and r["pos"] in CONTENT\n                                   and r["pos_status"] == "confirmed"))',
     'and r["pos"] in CONTENT))'),
    ("bankpass/wordload.py", '"pos": r.get("pos") or r["pos_proxy"]', '"pos": r["pos_proxy"]'),
    (G_, "N_ECHO, N_MINED, N_DEC13, N_FILLS = 8, 6, 13, 5", "N_ECHO, N_MINED, N_DEC13, N_FILLS = 9, 6, 13, 5"),
    (G_, "N_ECHO, N_MINED, N_DEC13, N_FILLS = 8, 6, 13, 5", "N_ECHO, N_MINED, N_DEC13, N_FILLS = 8, 7, 13, 5"),
    (G_, "        for _ in range(N_FILLS):", "        for _ in range(0):"),
    (G_, "        for run in _HOLE.split(template):", '        for run in [_HOLE.sub(" ", template)]:'),
    (G_, '    g["w4"] = [list(h) for h in hits]', '    g["w4"] = []'),
    (G_, "            if tuple(ws[i:i + self.n]) in self.set:\n                return \" \".join(ws[i:i + self.n])",
     "            if tuple(ws[i:i + self.n]) in self.set:\n                return \" \".join(sorted(self.set & "
     "grams(ws, self.n), key=hash)[0])"),
    (G_, '            if oodh.in_oodh_reserve(t["message_tree_id"]):', "            if True:"),
    (G_, 'out += [p.get("question") or "", (p.get("prefix")', 'out += [(p.get("prefix")'),
    (BK, '("JUDGE_MISSING" if it.get("w3") == "pending"', '("W3_DROPPED" if it.get("w3") == "pending"'),
    (BK, 'elif vt == "assistant_name" and', 'elif vt in ("assistant_name", "pet_name") and'),
    (BK, 'if r["text"].lower() in program_values():', "if False:"),
    (BK, '            if num != "one":', '            if num == "more":'),
    (BK, '{"woman": "F", "man": "M"}.get(sex, "U")', '{"woman": "F"}.get(sex, "U")'),
    (BK, "            if len(a) < MIN_ATTRS:", "            if len(a) < 1:"),
    (BK, "    trim.dedup(recs)\n    trim.author_thirds(recs, seed)\n    return recs, {}",
     "    trim.dedup(recs)\n    return recs, {}"),
    (BK, "if n < REQ_MIN_AUTHORS or not d", "if n < 1 or not d"),
    (BK, ' or d["pos"] != part \\\n', " \\\n"),
    (BK, "or not d[\"required_ok\"] or h in generic \\\n", "or not d[\"required_ok\"] \\\n"),
    (BK, "                    or h in TW.NOT_REQUIRED or not heldout.pool_ok(h):", "                    or not heldout.pool_ok(h):"),
    (BK, 'or not d or not d["lists"] & {"RS", "RM"} or', "or not d or"),
    (BK, "AVOID_N, REQ_MIN_AUTHORS, GENERIC_SHARE, MIN_ATTRS = 80, 2, 0.05, 2",
     "AVOID_N, REQ_MIN_AUTHORS, GENERIC_SHARE, MIN_ATTRS = 80, 2, 0.04, 2"),
    (BK, 'key=lambda r: r["id"]):\n        k = store.norm_key(r["text"])',
     'key=lambda r: r["id"], reverse=True):\n        k = store.norm_key(r["text"])'),
    (BK, 'if d["pos"] not in ("adj", "adv") or', 'if d["pos"] not in ("adj", "adv", "noun") or'),
    (BK, ' or h in L.FUNCTION_WORDS \\\n                or h in L.STOPWORDS or d["flags"]:', ' or d["flags"]:'),
    (BK, '        if r["status"] == "kept" and r["id"] not in selected:', "        if False:"),
    (BK, 'r["text"] = f["topic"] + ": " + "; ".join(', 'r["text"] = "; ".join('),
    (F_, "        if c and c.most_common(1)[0][1] >= 2:", "        if c and c.most_common(1)[0][1] >= 1:"),
    (F_, 'if r["kind"] == kind and not r.get("problem")', 'if r["kind"] in ("vote", "vote2") and not r.get("problem")'),
    (F_, ' and ans.get("added") == "no"', ""),
    (F_, "        return c[0][0] if c and c[0][1] >= 2 else None", "        return c[0][0] if c else None"),
    (F_, "        elif typed.setdefault(lab, vt) != vt:", "        elif False:"),
    (F_, '        elif any(a[0] == lab for a in out[i["features"]["kind_id"]]):', "        elif False:"),
    (F_, 'if tid in topic_text and f.get("topic") != topic_text[tid] and i.get("w3") == "kept":', "if False:"),
    (F_, "def wordset_topics(items, min_authors=2):", "def wordset_topics(items, min_authors=1):"),
    (F_, ' and r["plan"]["topic"] == text.get(r["plan"]["topic_id"])}', "}"),
    (F_, '        if r["status"] == "kept" and r["id"] not in with_sets:', "        if False:"),
    (V, 'return "(?:" + "|".join(str(i) for i in range(n, 0, -1)) + ")"',
     'return "(?:" + "|".join(str(i) for i in range(n + 1, 0, -1)) + ")"'),
    (V, "if not rows[0].isdigit() or not 1 <= int(rows[0]) <= n:", "if not rows[0].isdigit():"),
    (V, '        if c["call_id"] in done:', "        if False:"),
    ("bankpass/load.py", '        self.refs.update({f"label.{k}": f"label.{k}@fake:FAKE" for k in BK.KEYS})', "        pass"),
    ("bankpass/load.py", '            bs.list_names[bank.split(".", 1)[1]] = texts[0]', "            pass"),
    ("bankpass/load.py", "        undo += [_swap_set(BK.FEMALE, bs.sex[0]), _swap_set(BK.MALE, bs.sex[1])]", "        pass"),
    ("bankpass/load.py", "            q[part].extend(w for w in ws if w not in q[part])", "            pass"),
    ("events_base.py", "p = B.p_exact(options[0][0], self.p_exact) if options else self.p_exact", "p = self.p_exact"),
    ("banks.py", '    return line_id.rsplit(".", 1)[0]', '    return line_id.split(".", 1)[0]'),
    ("banks.py", "    return LOOKUP_PRED_ATTR.get(attr) or LOOKUP_PRED[vtype]", "    return LOOKUP_PRED[vtype]"),
    ("events_c.py", "        pred = B.lookup_pred(attr, vt).format(v=val)", "        pred = B.LOOKUP_PRED[vt].format(v=val)"),
    ("assemble.py", "cands = [w for w in TW.req_words(topic, part)", "cands = [w for w in TW.words(topic, part)"),
    ("assemble.py", '    prov["fake"] = bool(fake_refs(prov))', '    prov["fake"] = True'),
    ("assemble.py", '"entity_kind",\n                                                            "avoid_word"}', '"entity_kind"}'),
    ("assemble.py", '"lists": sorted(B.LIST_REFS.values())}', '"lists": []}'),
    ("assemble.py", ' + list(prov.get("lists", [])) \\\n', " \\\n"),
    ("pools.py", '    if counted and "article" in counted and not cap:', "    if False:"),
    ("topic_words.py", '    return tuple(w for w in row.get(part, ()) if not heldout.vocab_hits(w))',
     "    return tuple(row.get(part, ()))"),
    (D, 'fake[r.split("@")[0] if "@" in r else r] += 1', "fake[r] += 1"),
    (D, '"skeletons_fake": sum(bool(A.fake_refs(s["provenance"])) for s in sks)', '"skeletons_fake": 0'),
    (ME, 'if t["role"] == "assistant" and it.split(";")[0] in CB.FILLER_ASSIST:', 'if t["role"] == "assistant":'),
    (ME, '(prev_rule == "fam2" and len(shared) >= 2)', '(prev_rule == "fam2" and len(shared) >= 1)'),
    (ME, "                and len(w) > 2} - self.generic", "                and len(w) > 2}"),
    (ME, 'donors = [x for x in ls if x[0] != a and not tp & set(rows[x[0]][0]["topic_text"].values())]',
     "donors = [x for x in ls if x[0] != a]"),
    (ME, '            forms = {w} | {f for fs in d["forms"].values() for f in fs}', "            forms = {w}"),
    (ME, 'if w.isalpha() and w not in vals]', "if w.isalpha()]"),
    (ME, "    ranked = sorted(((len(v) / n, h) for h, v in where.items() if len(v) / n >= share), reverse=True)",
     "    ranked = sorted(((len(v) / n, h) for h, v in where.items()), reverse=True)"),
    (ME, '                if (prev_rule == "cur" and (len(shared) >= 2 or len(shared) >= 0.5 * len(own) > 0)) or \\',
     '                if (prev_rule == "cur" and len(shared) >= 2) or \\'),
]


def main(argv=None):
    import subprocess  # noqa: F401  (mutate.py's runner)
    argv = argv if argv is not None else sys.argv[1:]
    scratch = argv[0]
    only = {int(x) for x in argv[2].split(",")} if len(argv) > 2 and argv[1] == "--only" else None
    MU.TESTS[:] = TESTS
    MU.stage(scratch)
    rc, tail = MU.run_tests(scratch)
    print(f"baseline rc={rc} {tail}", flush=True)
    if rc != 0:
        return 1
    survived = []
    for i, (f, old, new) in enumerate(M):
        if only and i not in only:
            continue
        path = os.path.join(scratch, "pipeline", f)
        src = open(path, encoding="utf-8").read()
        if src.count(old) != 1:
            print(f"#{i} {f}: pattern found {src.count(old)} times, NOT APPLIED", flush=True)
            survived.append(i)
            continue
        open(path, "w", encoding="utf-8").write(src.replace(old, new))
        try:
            rc, tail = MU.run_tests(scratch)
        finally:
            open(path, "w", encoding="utf-8").write(src)
        print(f"#{i} {f}: {'KILLED' if rc else 'SURVIVED'} ({tail})", flush=True)
        if rc == 0:
            survived.append(i)
    print(f"mutants {len(M) if not only else len(only)}, survived {survived}")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
