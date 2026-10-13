"""Mutation test of the lock machinery (rc12/notes.txt STEP 15) against test_lock.py, test_lock_b.py, test_lock_c.py
and test_lock_d.py (Max's rule: every test claim is watched failing). Each mutant replaces one or more texts in a
scratch copy of rc12 (with the E004 code it imports and the prereg draft), then the tests run there; KILLED = one
exits non-zero. The unmutated copy must pass; MALFORMED = a text not found exactly once. At most 4 mutants at a
time.
No model.
  python3 -B mutation_lock.py   (writes logs/mutation_lock.txt; exit 1 unless the baseline passes and all die)"""
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
E004 = os.path.join("experiments", "E004_general_updating", "code")
TESTS = ["test_lock.py", "test_lock_b.py", "test_lock_c.py", "test_lock_d.py"]
SPC, CM, PV, PS, BD, TG, FL = ("split_content.py", "common.py", "pools_vals.py", "pools_slots.py", "build_dev.py",
                               "test_gens.py", "fam_role_lookup.py")
SO, SK, CR, SS, LC, LF = ("sealed_overlap.py", "sealed_kit.py", "sealed_crypto.py", "sealed_search.py",
                          "lock_check.py", "lock_finalize.py")
PKF, RN, PUB, MGF = "sealed_pack.py", "runner.py", "sealed_public.py", "mutation_graders.py"
APPLY_V = "SC.apply(globals(), CONTENT)      # dev: no-op; sealed: its own values (prereg s13 S3, split_content.py)\n"
APPLY_W = "SC.apply(globals(), CONTENT)      # dev: no-op; sealed: its own wording (prereg s13 S3, split_content.py)\n"
MUTANTS = [
    (SPC, [("    path = content_path(mod)\n    if path is None:",
            "    return\n    path = content_path(mod)\n    if path is None:")], "hook never applies"),
    (SPC, [("""                                       ".rc12_workcopy")) or content_path(mod):""",
            """                                       ".rc12_workcopy")) and False:""")],
     "dev runs beside sealed content"),
    (SPC, [("        if not (isinstance(node, ast.Assign) and len(node.targets) == 1",
            "        if False and not (isinstance(node, ast.Assign) and len(node.targets) == 1")],
     "code accepted in a content file"),
    (SPC, [("        d = shape_diff(g[key], over[key], f\"{mod}.{key}\", free_keys=n.endswith(\"*\")) or \\\n"
            "            pinned(mod, key, g[key], over[key])",
            "        d = shape_diff(g[key], over[key], f\"{mod}.{key}\", free_keys=n.endswith(\"*\"))")],
     "identifier pins not checked"),
    (SPC, [("        if d:\n            raise SystemExit", "        if False:\n            raise SystemExit")],
     "shape not enforced"),
    (SPC, [("    if set(over) != want:", "    if False:")], "extra or missing names allowed"),
    (SPC, [("return None if fields(dev) == fields(new) else", "return None if True else")], "placeholders unchecked"),
    (SPC, [('            return f"{where}: length', '            return None and f"{where}: length')],
     "list lengths unchecked"),
    (SPC, [("            if set(dev) != set(new):", "            if False:")], "dict keys unchecked"),
    (SPC, [("return None if dev == new else", "return None if True else")], "numbers unchecked"),
    (SPC, [('    if not re.fullmatch(r"[0-9a-f]{32,}", SEED):', "    if False:")], "short seed accepted"),
    (SPC, [('        raise SystemExit(f"split {SPLIT}: content_{mod}.py is missing (prereg s13 S3)")',
            "        return")],
     "missing content tolerated"),
    (CM, [('random.Random(f"RC12:{SPLIT}:{family}', 'random.Random(f"RC12:dev:{family}')], "streams ignore the split"),
    (CM, [('return f"rc12-{SPLIT}-{family.lower()}', 'return f"rc12-dev-{family.lower()}')], "ids ignore the split"),
    (CM, [("split=SPLIT, family", 'split="dev", family')], "records say dev"),
    (CM, [("        seed = SEED if seed is None else seed", "        seed = 1212 if seed is None else seed")],
     "records carry the dev seed"),
    (PV, [(APPLY_V, ""), ("_WRE = {}", "SC.apply(globals(), CONTENT)\n_WRE = {}")], "pools applied after derivation"),
    (PS, [(APPLY_W + "SLOT", "SLOT"),
           ("SLOT = {s[\"key\"]: s for s in RECALL_SLOTS}      # derived, after the content\n",
            "SLOT = {s[\"key\"]: s for s in RECALL_SLOTS}\n" + APPLY_W)],
     "SLOT derived before the content"),
    (BD, [('    if C.SPLIT != "dev" and not os.path.isfile(MARKER):', "    if False:")], "no work-copy guard"),
    (TG, [('    check(any(v.endswith("s") for v in V.INSTRUMENT + V.FOOD + V.PROJECT),', "    check(True,")],
     "no s-ending value passes"),
    (TG, [('        check(want is not None and [n[True], n[False]] == want["asks"]', "        check(True")],
     "sealed asks counts unchecked"),
    (FL, [("    return ROW_SEP.join(ROW.format(k=k, v=v) for k, v in zip(keys, vals))",
           '    return ", ".join(f"{k}: {v}" for k, v in zip(keys, vals))')], "LOOKUP row format hardcoded"),
    (SO, [('"FOOD", "PROJECT", "BOAT"', '"FOOD", "WEEKDAY", "PROJECT", "BOAT"')], "a closed class checked"),
    (SO, [("def grams(texts, n=5):", "def grams(texts, n=6):")], "6-grams"),
    (SO, [("            sorted(short_values(new, free) & short_values(dev, free)))", "            [])")],
     "(d) dropped"),
    (SO, [("            if \"{\" not in x and 1 <= len(words(x)) <= 4 and x.lower() not in closed:",
           "            if \"{\" not in x and 1 <= len(words(x)) <= 4:")], "(d) flags closed classes"),
    (SO, [("                if not (isinstance(x, tuple) and i in pins):", "                if True:")],
     "(d) flags pinned identifiers"),
    (SO, [("sorted(shared), sorted(long_strings(new) & long_strings(dev))", "sorted(shared), []")], "(c) dropped"),
    (SO, [('        pools += [f"{m}.{n}: {v} (a dev value of another pool)" for v in mine & across]\n', "")],
     "cross-pool values allowed"),
    (SK, [("        elif hashlib.sha256(", "        elif False and hashlib.sha256(")], "changed files pass"),
    (SK, [('                bad.append(f"extra {p}")', "                pass")], "extra files pass"),
    (SK, [("    if inside(work, repo) and not inside(", "    if False and not inside(")], "work copy anywhere"),
    (SK, [("    if os.path.exists(work) and os.listdir(work):", "    if False:")], "prepare into a used folder"),
    (SK, [("        if stop and not ok:\n            break", "        if False:\n            break")],
     "gates never stop early"),
    (CR, [("    if in_git_tree(path):", "    if False:")], "key inside a repo"),
    (CR, [("    if os.path.exists(path):\n        die(", "    if False:\n        die(")], "private file overwritten"),
    (CR, [("O_EXCL, 0o600)", "O_EXCL, 0o644)")], "world-readable key"),
    (CR, [("                   if not m.isfile() or want.get(m.name) != sha(t.extractfile(m).read())]",
            "                   if False]")], "per-file hashes unchecked"),
    (CR, [('        bad += [n for n in want if n not in names and n not in (TAR, "generator", ENC_NAME)]\n', "")],
     "missing archived files pass"),
    (CR, [("    with open(hashes, \"a\") as f:\n        f.write(f\"sha256 {sha(open(out, 'rb').read())}",
           "    with open(os.devnull, \"a\") as f:\n        f.write(f\"sha256 {sha(open(out, 'rb').read())}")],
     "ciphertext hash not appended to S5"),
    (CR, [("    if not re.search(rf\"^sha256 {sha(open(tar, 'rb').read())}  {TAR} \", open(hashes).read(), re.M):",
           "    if False:")], "an unnamed tar encrypted"),
    (CR, [("""        hashes.update(sha(" ".join(w[i:i + n]).encode()) for i in range(len(w) - n + 1))""",
           """        hashes.update(sha(" ".join(t.split()[i:i + n]).encode()) for i in range(len(w) - n + 1))""")],
     "13-grams not normalized"),
    (CR, [("def ngrams(work, oodh, out, n=13):", "def ngrams(work, oodh, out, n=12):")], "12-grams"),
    (CR, [('            texts += [t["text"], t.get("ideal") or ""]', '            texts += [t["text"]]')],
     "OOD-H assistant replies left out"),
    (SS, [("            os.remove(trial)\n        log.write", "            pass\n        log.write")],
     "failed seeds kept"),
    (SS, [("            except (SystemExit, subprocess.TimeoutExpired):", "            except ZeroDivisionError:")],
     "a failed build stops the draws"),
    (SS, [("    if supersede and not re.fullmatch(REASON, supersede):", "    if False:")], "free-text restart reasons"),
    (SS, [("    if K.inside(log_path, work):", "    if False:")], "draw log inside the work copy"),
    (SS, [("        for i in range(done + 1, done + max_draws + 1):",
            "        for i in range(done + 1, done + max_draws + 2):")],
     "draw cap off by one"),
    (LC, [('    bad = [f"prereg still carries {m}" for m in MARKERS if re.search(m, text)]', "    bad = []")],
     "lock markers ignored"),
    (LC, [('    bad += [f"prereg lacks the S5 hash of {n}" for h, n in hashes if h not in flat]\n', "")],
     "S5 hashes not required in the prereg"),
    (LC, [('    if not K.gates_clean(open(os.path.join(root, GATES)).read()):', "    if False:")],
     "failing gates accepted"),
    (LC, [("        if diff.returncode or changed:", "        if False:")], "code changes after the build pass"),
    (LC, [("    if have != cur:", "    if False:")], "stale manifest passes"),
    (LC, [('            bad.append(f"{path} is not the {name} the archive holds")', "            pass")],
     "public copies unlike the archived files pass"),
    (LC, [("    if want not in re.sub(", "    if False and want not in re.sub(")], "OOD-H counts unchecked"),
    (LC, [('            bad.append(f"tracked file that must never be in git: {p}")', "            pass")],
     "tracked sealed files pass"),
    (LF, [("        if text.count(old) != 1:", "        if text.count(old) > 1:")], "drifted draft edited"),
    (LF, [('    live = draws.rsplit("\\nsuperseded:", 1)[-1]', "    live = draws")], "restarts not honored"),
    (PUB, [('    if name == "build_stdout":\n        return', '    if False:\n        return')],
     "build output published"),
    (PUB, [("""    plain = re.fullmatch(r"[\\w .,:;()=<>/%+-]*", last) and redact(last, words) == last""",
            """    plain = redact(last, words) == last""")], "unplain gate output published"),
    (PUB, [("""    plain = re.fullmatch(r"[\\w .,:;()=<>/%+-]*", last) and redact(last, words) == last""",
            """    plain = re.fullmatch(r"[\\w .,:;()=<>/%+-]*", last)""")], "sealed words in gate output published"),
    (PUB, [("""    return re.sub(WORD, lambda m: "(withheld)" if m.group(0).lower() in words else m.group(0), line, flags=re.I)""",
            """    return line""")], "sealed words in the draw log"),
    (SS, [("            except subprocess.TimeoutExpired:\n                log.write(f\"draw {i}: FAIL; a gate timed out\\n\")",
           "            except ZeroDivisionError:\n                log.write(f\"draw {i}: FAIL; a gate timed out\\n\")")],
     "a gate timeout stops the draws unlogged"),
    (PKF, [("    K.check_place(out, repo)", "    pass")], "plaintext tar anywhere"),
    (TG, [("    check(sorted(M.PROJECTS) == sorted(V.PROJECT), \"PROJECTS keys are not the PROJECT pool\")", "")],
     "PROJECTS keys unchecked"),
    (SPC, [("    items = [(k, dev[k], new[k]) for k in dev] if isinstance(dev, dict) else list(zip(range(len(dev)), dev, new))",
            "    items = list(zip(range(len(dev)), dev.values(), new.values())) if isinstance(dev, dict) else "
            "list(zip(range(len(dev)), dev, new))")], "pins compared by position"),
    (SK, [("            if os.path.islink(full) or n == \"__pycache__\" or n.endswith((\".pyc\", \".pyo\")):",
           "            if False:")], "bytecode or links in a work copy"),
    (SK, [("                if os.path.isfile(path)))", "                if False))")], "gate inputs not recorded"),
    (SS, [("    if done and not supersede:", "    if False:")], "second acceptance allowed"),
    (SS, [("            os.remove(os.path.join(os.path.dirname(seed_file), x))", "            pass")],
     "cut trial seeds kept"),
    (SS, [('    done = len(re.findall(r"^draw \\d+:", old, re.M))', "    done = 0")], "draw numbers restart"),
    (PUB, [("""    line = re.sub(r"'[^']*'|\\"[^\\"]*\\"", "'(withheld)'", line)""", "    pass")],
     "quoted text in the draw log"),
    (PKF, [("    if not K.gates_clean(gates):", "    if False:")], "a failing gate table packed"),
    (PKF, [("""    lines.append(f"decon13 hashes {count} (the S6 list's count; """,
            """    0 and lines.append(f"decon13 hashes {count} (the S6 list's count; """)], "S6 count not published"),
    (PKF, [("        if recorded.get(name) != CR.sha(data[name]):", "        if False:")], "ungated inputs packed"),
    (PKF, [("    if rebuild_items(work, seed_file, split) != data[\"rc12_sealed.jsonl\"]:", "    if False:")],
     "items the seed does not rebuild packed"),
    (PKF, [("    if not last.startswith(f\"commit {commit} split {split}:\") or len(accepted) != 1 or \\\n"
            "            accepted[0] != CR.sha(open(seed_file, \"rb\").read().strip()):",
            "    if False:")], "an unaccepted seed packed"),
    (PKF, [("    if f\"items sha256 {CR.sha(data['rc12_sealed.jsonl'])}\" not in s1:", "    if False:")],
     "an S1 file of other items packed"),
    (PKF, [("    elif \"S1 overall: PASS\" not in s1 and split == \"sealed\":", "    elif False:")],
     "a failed S1 packed"),
    (PKF, [("    if f\"commit {commit} split {split}\" not in gates.split(\"\\n\", 1)[0]:", "    if False:")],
     "a gate table of another commit packed"),
    (PKF, [("    if K.check(work, repo=repo):", "    if False:")], "a changed work copy packed"),
    (LC, [('    paths = list(K.ARCHIVE_PATHS) + ["oodh", CORPUS]', "    paths = list(K.ARCHIVE_PATHS) + [CORPUS]")],
     "OOD-H code changes pass"),
    (LC, [('    paths = list(K.ARCHIVE_PATHS) + ["oodh", CORPUS]', '    paths = list(K.ARCHIVE_PATHS) + ["oodh"]'),
          ('                   or p.startswith("oodh/") and p.endswith(".py") or p == CORPUS]',
           '                   or p.startswith("oodh/") and p.endswith(".py")]')], "corpus/oodh.py changes pass"),
    (SK, [("    return \"failing steps: 0\" in text and all(name in rows for name, _ in GATES)",
           "    return \"failing steps: 0\" in text")], "a quick-gates-only table passes"),
    (LF, [("    if not ceil:\n        sys.exit(", "    if False:\n        sys.exit(")], "no ceiling required"),
    (LC, [('    if not re.search(r"^decon13 hashes \\d+ ", s5, re.M):', "    if False:")], "S6 count not required"),
    (LC, [("    if not enc or enc.group(1) != sha_file(os.path.join(root, ARCHIVE)):", "    if False:")],
     "a ciphertext S5 does not name passes"),
    (LC, [("    if not split or (split.groups() != (\"sealed\", \"PASS\") and not dryrun):", "    if False:")],
     "a dry-run split passes"),
    (LC, [("    if None in commits or len({m.group(1) for m in commits}) != 1:", "    if False:")],
     "different commits pass"),
    (LC, [("    if not seed or acc != [seed.group(1)]:", "    if False:")], "another seed passes"),
    (LC, [("""                re.search(r"(^|/)(content_[a-z_]+\\.py|rc12_sealed[^/]*|decon13[^/]*)$", p):""",
           """                False:""")], "tracked content files pass"),
    (RN, [("    if bad:                                      # graders read the split's own pools",
           "    if False:                                    # graders read the split's own pools")],
     "runner loads another split's records"),
    (LF, [('        ("(proposal thresholds):", "(thresholds fixed at the lock):"),\n', "")], "an edit dropped"),
    (LC, [('p.endswith(SUFFIX) and not p.startswith(("rc12/logs/", LOCK + "/"))\n',
           'p.endswith(SUFFIX) and not p.startswith(("rc12/logs/", LOCK))\n')], "code named rc12/lock* exempt"),
    (SO, [(', "fam_bind.BIND_NAME": ("objw",)}', "}")], "(d) flags BIND's object words"),
    (SO, [('D_SKIP = {"pools_fill_a.FILLERS_A": (2,), ', "D_SKIP = {")], "(d) flags filler labels"),
    (SO, [('"fam_own_topic.OWN_AVOID", ', "")], "(d) flags an avoid list"),
    (PUB, [("    new = set()\n", "    return set()\n")], "no content word withheld"),
    (TG, [('    check(low(FB.BIND_AVOID) <= objs,', "    check(True,")], "stale BIND_AVOID passes"),
    (SS, [('\n         "mutation_graders_G6"}', "}")], "mutation_graders not a draw gate"),
    (SS, [("""                log.write(f"draw {i}: STOPPED; the kit refused (its message is on the screen)\\n")\n""", "")],
     "a stopped draw unlogged"),
    (TG, [("        check(low(avoid) <= set().union(", "        check(True or low(avoid) <= set().union(")],
     "a stale avoid word passes"),
    (TG, [("        check(all(low(avoid) & r for r in runs),", "        check(True,")],
     "an item without its avoid word passes"),
    (SS, [("    g10 = g10.group(1) + quiet_text(g10.group(2), words) if g10", "    g10 = g10.group(1) + g10.group(2) if g10")],
     "G10 detail unredacted"),
    (SS, [("    g10 = g10.group(1) + quiet_text(g10.group(2), words) if g10",
           "    g10 = quiet_text(g10.group(1) + g10.group(2), words) if g10")], "G10 verdict redacted"),
    (SS, [("    fails = [quiet_text(f, words) for f in", "    fails = [f for f in")], "e2e FAIL lines unredacted"),
    (TG, [('re.sub(r"\\{[^}]*\\}", " ", " ".join(ts))', '" ".join(ts)')], "a {field} name counts as an object word"),
    (TG, [("    check(low(FB.BIND_AVOID) <= objs,", "    check(set(FB.BIND_AVOID) <= objs,")],
     "avoid words compared case-sensitively"),
    (TG, [("        check(low(avoid) <= set().union(", "        check(set(avoid) <= set().union(")],
     "OWN/LOOKUP stale check case-sensitive"),
    (TG, [("        check(all(low(avoid) & r for r in runs),", "        check(all(set(avoid) & r for r in runs),")],
     "OWN/LOOKUP item check case-sensitive"),
    (TG, [("<= 1 for w in low(avoid)),", "<= 1 for w in set(avoid)),")], "OWN/LOOKUP own-word check case-sensitive"),
    (MGF, [('        if any((f["label"], f["grader"]) not in kinds for f in new):', "        if False:")],
     "G6 fixture kinds hang on record order"),
    (MGF, [('        if any((f["label"], f["grader"]) not in kinds for f in new):', "        if True:")],
     "G6 takes every record"),
    ("fixtures_more.py", [('patch={src: f"{cap(o0)} is great. {cap(o1)} is great too."}',
                           'patch={src: f"Maybe {o0} or {o1}?"}')], "no two-picks source fixture"),
    ("fixtures_more.py", [('sub_word(p["ideal"], gold, o0), False,', 'p["ideal"], False,'),
                          ('patch={src: f"{cap(o0)} is great. {cap(o1)} is great too."}',
                           'patch={src: f"{cap(gold)} is great. {cap(alt)} is great too."}')],
     "two-picks fixture kills only when the gold is the first option"),
    ("fixtures_more.py", [('sub_word(p["ideal"], gold, o0), False,', 'sub_word(p["ideal"], gold, alt), False,')],
     "two-picks fixture replies with the other option"),
    ("fixtures_more.py", [('patch={src: f"{cap(o0)} is great. {cap(o1)} is great too."}',
                           'patch={src: f"{cap(o0)} is great. It is not {o1}."}')],
     "two-picks fixture with one asserted pick (real graders pass it)"),
    (TG, [("        check(all(sum(w in r for r in runs) <= 1 for w in low(avoid)),", "        check(True,")],
     "a common word in a spare slot passes"),
    (SS, [("    return (f\"test_gens", "    return (lambda s: quiet_text(s, words))(f\"test_gens")],
     "fixed summary words redacted"),
]


def scratch():
    d = tempfile.mkdtemp(prefix="rc12mutlock_")
    shutil.copytree(HERE, os.path.join(d, "rc12"), ignore=shutil.ignore_patterns("runs", "logs", "__pycache__"))
    shutil.copytree(os.path.join(ROOT, E004), os.path.join(d, E004), ignore=shutil.ignore_patterns("__pycache__"))
    os.makedirs(os.path.join(d, "prereg"))
    for name in ("RC-12.draft.txt", "RC-12.md"):          # the draft before the lock, the locked file after it
        if os.path.exists(os.path.join(ROOT, "prereg", name)):
            shutil.copy(os.path.join(ROOT, "prereg", name), os.path.join(d, "prereg"))
            break
    return d


LOCKED = not os.path.exists(os.path.join(ROOT, "prereg", "RC-12.draft.txt"))   # finalize's edits are history then


def run(m=None):
    if m is not None and LOCKED and MUTANTS[m][0] == LF:
        return m, "n/a", "after the lock: test_lock_c checks the locked file, lock_finalize does not run"
    d = scratch()
    try:
        if m is not None:
            f, pairs, _ = MUTANTS[m]
            path = os.path.join(d, "rc12", f)
            text = open(path).read()
            for old, new in pairs:
                if text.count(old) != 1:
                    return m, "malformed", f"{old[:50]!r} found {text.count(old)} times"
                text = text.replace(old, new)
            open(path, "w").write(text)
        env = {k: v for k, v in os.environ.items() if not k.startswith("RC12_")}
        for t in TESTS:
            r = subprocess.run([sys.executable, "-B", t], cwd=os.path.join(d, "rc12"), capture_output=True, text=True,
                               timeout=900, env=env)
            if r.returncode != 0:
                lines = [x for x in r.stdout.splitlines() if x.startswith("FAIL")] or r.stderr.splitlines()[-1:]
                return m, "killed", f"{t}: {lines[0][:110] if lines else r.returncode}"
        return m, "pass", ""
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    _, base, why = run()
    out = [f"baseline: {base} {why}"]
    print(out[0], flush=True)
    with ProcessPoolExecutor(max_workers=4) as ex:
        results = sorted(ex.map(run, range(len(MUTANTS))))
    killed = 0
    na = sum(v == "n/a" for _, v, _ in results)
    for m, v, first in results:
        killed += v == "killed"
        out.append(f"{'KILLED' if v == 'killed' else v.upper():9s} {MUTANTS[m][0]}: {MUTANTS[m][2]} | {first}")
        print(out[-1], flush=True)
    out.append(f"MUTANTS killed {killed}/{len(MUTANTS) - na}" + (f" ({na} n/a after the lock)" if na else ""))
    print(out[-1])
    with open(os.path.join(HERE, "logs", "mutation_lock.txt"), "w") as f:
        f.write("\n".join(out) + "\n")
    return 0 if base == "pass" and killed == len(MUTANTS) - na else 1


if __name__ == "__main__":
    sys.exit(main())
