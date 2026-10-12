"""Mutation test of the bank pass (Max's rule: a test that has never been watched failing is not evidence). Copies
pipeline/, corpus/ and the E004 code to a SCRATCH directory, applies one mutant at a time there, runs the bank pass
tests, and reports killed / survived. The repo is never edited.

    python pipeline/bankpass/mutate.py SCRATCH_DIR [--only N,M]"""
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTS = ["tests/test_bankpass_core.py", "tests/test_bankpass_gen.py", "tests/test_bankpass_load.py",
         "tests/test_bankpass_words.py", "tests/test_bankpass_wordload.py", "tests/test_bankpass_human.py",
         "tests/test_bankpass_humanbuild.py"]
M = [  # (file under pipeline/bankpass, old, new): each guards one rule a test claims
    ("store.py", 'if a.get("revision") != t["revision"]:', "if False:"),
    ("store.py", 'if src.get("license") not in HUMAN_LICENSES:', "if False:"),
    ("store.py", "if last and tolerate_torn_tail:", "if True:"),
    ("store.py", 'if "fake" in shares:', "if False:"),
    ("templatize.py", "if len(hits) > 1:", "if len(hits) > 2:"),
    ("templatize.py", "flags = 0 if v[:1].isupper() else re.I", "flags = re.I"),
    ("templatize.py", 'ok = "the {o}" in low and "my {o}" not in low', 'ok = "the {o}" in low or "my {o}" in low'),
    ("templatize.py", "return [plain] if len(items) == 2 else [oxford, plain]", "return [plain]"),
    ("templatize.py", 'OPTIONAL = {"old"}', "OPTIONAL = set()"),
    ("templatize.py", 'return "{M}" + t[:1].lower() + t[1:]', 'return "{M}" + t'),
    ("gates.py", 'for code, rx in (("DASH", G.DASH_RE), ("DIGIT"', 'for code, rx in (("DIGIT"'),
    ("gates.py", 'if role in ("plant", "corr", "twin") and "?" in s:', "if False:"),
    ("gates.py", 'if spec["side"] == "user" and re.search(r"(?<![a-z])your', 'if False and re.search(r"(?<![a-z])your'),
    ("gates.py", 'if not spec["min_w"] <= n_w <= spec["max_w"]:', "if False:"),
    ("gates.py", "PROMPT_N, MINED_N, N_FILLS = 5, 6, 20", "PROMPT_N, MINED_N, N_FILLS = 9, 6, 20"),
    ("gates.py", "    if v:\n        out.append((\"HELDOUT_VOCAB\", v[0]))", "    if False:\n        pass"),
    ("gates.py", "        if v or e:", "        if False:"),
    ("gates.py", "sum((q, s) in bad for s in statements) * 2 > len(statements)",
     "sum((q, s) in bad for s in statements) * 2 >= len(statements)"),
    ("gates.py", 'out = [] if heldout.pool_ok(value) else [("HELDOUT_POOL", value)]', "out = []"),
    ("judge.py", "or author in models:", ":"),
    ("judge.py", "if answers.get(qid) is None or answers[qid] != _norm(exp):", "if answers.get(qid) is None:"),
    ("judge.py", "if len(got.get(qid, [])) == 1 else None", "if got.get(qid) else None"),
    ("trim.py", "NEAR_J, FRAME_MAX, AUTHOR_SLACK, P_EXACT = 0.6,", "NEAR_J, FRAME_MAX, AUTHOR_SLACK, P_EXACT = 0.95,"),
    ("trim.py", "AUTHOR_SLACK, P_EXACT = 0.6, 0.05, 1.1, 0.7", "AUTHOR_SLACK, P_EXACT = 0.6, 0.05, 3.0, 0.7"),
    ("trim.py", "min(P_EXACT, P_EXACT * kept / target)", "P_EXACT * kept / target"),
    ("trim.py", "        if k in seen:\n", "        if False:\n"),
    ("admit.py", 'if store.sha256_file(path) != meta.get("sha256"):', "if False:"),
    ("admit.py", 'if g.get("gate_hash") != gh:', "if False:"),
    ("admit.py", 'if g.get("hits"):', "if False:"),
    ("admit.py", 'and r.get("class") in JUDGED_CLASSES and judge.vote(r):', "and False:"),
    ("admit.py", 'if a.get("kind") not in (expected_kind(bank), "fake"):', "if False:"),
    ("admit.py", "        if k in keys:\n", "        if False:\n"),
    ("admit.py", 'if man.get("fixture") and not allow_fixture:', "if False:"),
    ("admit.py", "SHARE_MIN_N, SHARE_TOL = 30, 0.05", "SHARE_MIN_N, SHARE_TOL = 30, 0.5"),
    ("admit.py", "            if hits:\n", "            if False:\n"),
    ("admit.py", "    if not kept:\n", "    if False:\n"),
    ("admit.py", 'if a.get("kind") == "fake" and not fixture:', "if False:"),
    ("admit.py", 'if man.get("status") != "frozen" and not', 'if False and not'),
    ("load.py", "_swap_dict(BK.KEYS, bs.keys)", "_swap_attr(BK, 'KEYS', bs.keys)"),
    ("load.py", "        B.PROVENANCE, TW.PROVENANCE = old_prov, old_tw\n", "        pass\n"),
    ("load.py", 'return "FAKE" if self.fake_banks else "BANKSET"', 'return "BANKSET"'),
    ("gen.py", 'if "sealed" in parts:', "if False:"),
    ("gen.py", 'if any(parts[i] == "planck" and', 'if False and any(parts[i] == "planck" and'),
    ("gen.py", 'if not (isinstance(max_ok, dict) and max_ok.get("by") == "Max" and max_ok.get("date")):', "if False:"),
    ("gen.py", 'if call["call_id"] in done:', "if False:"),
    ("gen.py", "if clock() - start >= hold_s:", "if False:"),
    ("gen.py", 'if out.get("model") != call["teacher"]:', "if False:"),
    ("gen.py", 'if not call.get("ready", True):', "if False:"),
    ("prompts.py", 'if not rows or rows[-1] != "END":', "if not rows:"),
    ("prompts.py", 'body = D.noend() if literal is None else', 'body = D.noend() if True else'),
    ("wordlist.py", "% 20\n", "% 10\n"),
    ("wordlist.py", 'WORD = re.compile(r"(?<![^\\W\\d_])[a-z]+', 'WORD = re.compile(r"[a-z]+'),
    ("wordlist.py", '    acc["adj"].update(ADJ_BE.findall(low))\n', ""),
    ("wordlist.py", "        if w[0].isupper():", "        if True:"),
    ("wordfam.py", "ELIG_GROUPS, LISTS = 3,", "ELIG_GROUPS, LISTS = 2,"),
    ("wordfam.py", 'bp in ("noun", "verb") and fp != "adj"', 'True'),
    ("wordfam.py", "max(r[f\"df_{g}\"], 0.5)", "max(r[f\"df_{g}\"], 0)"),
    ("wordfam.py", "rate = {p: r[p] / tot[p] for p in POS}", "rate = {p: r[p] for p in POS}"),
    ("wordfam.py", "PROPER_MAX, PROPER_MIN_USES, POS_MIN, POS_SHARE_MIN, BASE_TF_RATIO = 0.5,",
     "PROPER_MAX, PROPER_MIN_USES, POS_MIN, POS_SHARE_MIN, BASE_TF_RATIO = 0.99,"),
    ("wordfam.py", "    if heldout.vocab_hits(w):", "    if False:"),
    ("wordfam.py", "if set(flags.get(b, ())) & set(DROP_FLAGS) or (tf", "if (tf"),
    ("wordfam.py", '    elif not heldout.pool_ok(w):\n        out.append("heldout_head")',
     '    elif not heldout.pool_ok(w):\n        out.append("heldout_vocab")'),
    ("wordfam.py", 'and not res["flags"][h]', ""),
    ("wordfam.py", 'DROP_FLAGS = ("proper", "heldout_vocab",', 'DROP_FLAGS = ("heldout_vocab",'),
    ("wordfam.py", "        if w in func:\n            out[w] = (\"func\", 1.0)", "        if False:\n            pass"),
    ("wordfam.py", "b in elig and len(b) >= 3]", "b in elig]"),
    ("wordfam.py", "        if any(b in func for b, _ in cands):\n            continue\n", ""),
    ("wordfam.py", "(tf and tf[b] * BASE_TF_RATIO < tf[w])", "False"),
    ("wordfam.py", "max(POS_MIN, POS_SHARE_MIN * r[\"tf\"])", "POS_MIN"),
    ("wordstats.py", '["jaccard_top5000"]) < 0.95', '["jaccard_top5000"]) < 0.3'),
    ("wordfam.py", 'plural = res["rule_of"].get(ws[-1]) == "s"', "plural = False"),
    ("wordfam.py", 'last["mass"] > last["a"] + last["an"]', 'last["mass"] > 99'),
    ("wordlist.py", "r\"(?<![A-Za-z'])[a-z][a-z']*[,;:]? (?=([A-Za-z][a-z']*)(?![^\\W\\d_]))\"",
     "r\"(?<=[a-z,;:] )([A-Za-z][a-z']*)(?![^\\W\\d_])\""),
    ("wordlist.py", "({w: n for w, n in c.items() if w in keep} if keep else dict(c))", "dict(c)"),
    ("mine.py", "    if heldout.vocab_hits(s):", "    if False:"),
    ("mine.py", 'chosen = pick(rows, a.per_class, ok=lambda r: not res.has(r["text"]), notes=notes)',
     "chosen = pick(rows, a.per_class, notes=notes)"),
    ("mine.py", "        return any(k in self.turns[i] for i in cands)", "        return False"),
    ("mine.py", "if in_oodh_reserve(meta[\"tree_id\"]):", "if False:"),
    ("mine.py", "if any(w[:1].isupper() and not PRONOUN_I.fullmatch(w)", "if False and any(w[:1].isupper()"),
    ("mine.py", "and not PRONOUN_I.fullmatch(w) for", "for"),
    ("specs.py", "return vt in P.POOLS and P.POOLS[vt].provenance", "return True or P.POOLS[vt].provenance"),
    ("specs.py", "and not vt.startswith(\"req_\") and vt != \"topic\"", "and vt != \"topic\""),
    ("specs.py", "return bank.split(\".\", 1)[1] in KEYS", "return True"),
    ("specs.py", "P.POOLS[vt].provenance != \"PROGRAM\"", "True"),
    ("specs.py", "    return False\n\n\ndef hole_problems", "    return True\n\n\ndef hole_problems"),
    ("admit.py", "        if not specs.loadable(bank):", "        if False:"),
    ("load.py", "            raise LoadRefused({bank: [(\"UNLOADABLE\", bank)]})", "            continue"),
    ("load.py", "elif bank.startswith(\"pool.\") or bank == \"topic\":", "elif bank.startswith(\"pool.\"):"),
    ("wordload.py", "if expect_sha256 and sha != expect_sha256:", "if False:"),
    ("wordload.py", "                if meta is None:\n", "                if False:\n"),
    ("wordload.py", "if any(c not in cols for c in NEED):", "if False:"),
    ("wordload.py", "and self.families[h][\"pos_status\"] in USABLE_POS", ""),
    ("wordload.py", "return self.form_of.get(w, w)", "return w"),
    ("wordload.py", "if d and d[\"a\"] + d[\"an\"] >= 3:", "if False:"),
    ("wordload.py", "plural = self.rule_of.get(ws[-1]) == \"s\"", "plural = False"),
    ("wordload.py", "mass = bool(last) and last[\"mass\"] >= 3", "mass = False and last[\"mass\"] >= 3"),
    ("wordload.py", "added = wl.english() - P._ENGLISHISH", "added = wl.english()"),
    ("wordload.py", "            if name == rel:", "            if True:"),
    ("wordload.py", "    return read(p, expect_sha256) if p else fake()", "    return fake()"),
    ("plan.py", "list(seeds or P.pool(\"req_noun\").values)", "list(P.pool(\"req_noun\").values)"),
    # W2 (2026-10-04): sources, human pools, persona seeds, safety list, build, aux admit, loader features
    ("sources.py", "                if not m:\n", "                if False:\n"),
    ("sources.py", "if pin and sha != pin:", "if False:"),
    ("sources.py", "if pin or not os.path.exists(path):", "if pin:"),
    ("sources.py", 'if rec.get("license") != SOURCES[name]["license"] or', "if"),
    ("sources.py", "bad = verify_one(root, name, rec)", "bad = []"),
    ("sources.py", 'if not rec or rec.get("status") != "ok":', "if not rec:"),
    ("human.py", "if not FORM_RE.fullmatch(value):", "if False:"),
    ("human.py", "if value.lower() in PROGRAM:", "if False:"),
    ("human.py", 'return bool(fam) and "proper" not in fam["flags"]', "return bool(fam)"),
    ("human.py", "if ctx.unsafe and ctx.unsafe.search(value):", "if False:"),
    ("human.py", "SEX_PURE = 0.9", "SEX_PURE = 0.5"),
    ("human.py", "key=lambda e: (rate[e], e))", "key=lambda e: (-rate[e], e))"),
    ("human.py", 'if name.startswith("MC"):', "if False:"),
    ("human.py", "name[len(p):] in listed for p in PREFIXES)", "True for p in PREFIXES)"),
    ("human.py", '"DUP_NAME" if v.lower() in seen else ', ""),
    ("human.py", "if r[7] in CITY_CODES:\n            seen.add", "if True:\n            seen.add"),
    ("human.py", "        if kept >= target:\n", "        if False:\n"),
    ("human.py", "for nm in cands:\n                if got[band] >= want:", "for nm in cands:\n                if False:"),
    ("human.py", "for r in cands:\n            if got[band] >= want:", "for r in cands:\n            if False:"),
    ("human.py", '("common", common, quota - quota // 2)', '("common", common, quota)'),
    ("humanseed.py", "if any(_bare(w) in own for w in rest):", "if False:"),
    ("humanseed.py", 'if not drop and int(row["age"]) < 18:', "if False:"),
    ("humanseed.py", "while name and _tok(name[-1]).lower() in PARTICLES:", "while False:"),
    ("humanseed.py", "return name if name and shared >= len(name) else []", "return name"),
    ("humanseed.py", 'if t.endswith((",", "\'s")):\n            break', "if False:\n            break"),
    ("humanseed.py", "or name[-1].endswith(\"'s\"):", ":"),
    ("humanseed.py", '(body if name[-1].endswith(",") else "someone who " + body)', '("someone who " + body)'),
    ("humanseed.py", '"DUP" if store.norm_key(text) in seen else None', "None"),
    ("humanseed.py", '_LONG.sub(", ", t)', "t"),
    ("humanseed.py", 'drop = "DUP" if t.lower() in seen else None', "drop = None"),
    ("gates.py", 'out = [("HELDOUT_VOCAB", h) for h in heldout.vocab_hits(text)[:1]]', "out = []"),
    ("admit.py", 'for bank, meta in sorted((man.get("aux") or {}).items()):', "for bank, meta in []:"),
    ("admit.py", 'if bank.startswith(("seed.", "rubric.")):', "if False:"),
    ("admit.py", 'hits = gates.pool_checks(r["text"], bank[5:]) if', 'hits = [] if False and'),
    ("store.py", 'part = "aux" if bank in (aux or {}) else "banks"', 'part = "banks"'),
    ("load.py", 'bs.features[vt] = {r["text"]: r.get("features") or {} for r in recs}', "pass"),
    ("humanbuild.py", "if ctx.unsafe is None:", "if False:"),
    ("load.py", 'meta["ref"].split(":", 1)[1]', 'meta["ref"].rsplit(":", 1)[1]'),
    ("humanbuild.py", "{b: m for b, m in banks.items() if b in LOADED}", "{b: m for b, m in banks.items()}"),
]
SWAP_ATTR = "\n\ndef _swap_attr(mod, name, new):\n    old = getattr(mod, name)\n    setattr(mod, name, new)\n" \
            "    return lambda: setattr(mod, name, old)\n"


def stage(scratch):
    if os.path.exists(scratch):
        shutil.rmtree(scratch)
    for rel in ("pipeline", "corpus", os.path.join("experiments", "E004_general_updating", "code")):
        shutil.copytree(os.path.join(REPO, rel), os.path.join(scratch, rel),
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))


def run_tests(scratch):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PLANCK_SKEL_N="200")
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"] + TESTS,
                       cwd=os.path.join(scratch, "pipeline"), env=env, capture_output=True, text=True, timeout=900)
    tail = (p.stdout.strip().splitlines() or [""])[-1]
    return p.returncode, tail


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    scratch = argv[0]
    only = {int(x) for x in argv[2].split(",")} if len(argv) > 2 and argv[1] == "--only" else None
    stage(scratch)
    rc, tail = run_tests(scratch)
    print(f"baseline rc={rc} {tail}", flush=True)
    if rc != 0:
        return 1
    survived = []
    for i, (f, old, new) in enumerate(M):
        if only and i not in only:
            continue
        path = os.path.join(scratch, "pipeline", "bankpass", f)
        src = open(path, encoding="utf-8").read()
        if src.count(old) != 1:
            print(f"#{i} {f}: pattern found {src.count(old)} times, NOT APPLIED", flush=True)
            survived.append(i)
            continue
        mutated = src.replace(old, new) + (SWAP_ATTR if "_swap_attr" in new else "")
        open(path, "w", encoding="utf-8").write(mutated)
        try:
            rc, tail = run_tests(scratch)
        finally:
            open(path, "w", encoding="utf-8").write(src)
        print(f"#{i} {f}: {'KILLED' if rc else 'SURVIVED'} ({tail})", flush=True)
        if rc == 0:
            survived.append(i)
    print(f"mutants {len(M) if not only else len(only)}, survived {survived}")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
