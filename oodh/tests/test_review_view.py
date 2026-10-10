"""oodh/review_view.py on the synthetic fixture (tests/fixture.py: made-up threads, no sealed file). Loads no model.
  python3 -B oodh/tests/test_review_view.py            the tests
  python3 -B oodh/tests/test_review_view.py --mutants  every mutant in M must turn the tests red (else exit 1)"""
import os, sys  # noqa: E401
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
OODH, REPO = os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != OODH]
import select  # noqa: E402,F401  the stdlib module first: oodh/ holds a select.py of its own
sys.path[:0] = [OODH, HERE]
import glob, hashlib, html, json, re, shutil, subprocess, tempfile  # noqa: E401,E402
from html.parser import HTMLParser  # noqa: E402
from pathlib import Path  # noqa: E402
import build as B, fixture as FX, hashes as H, review_view as RV  # noqa: E401,E402

SALT, TESTS, ABS = "planck-oodh-maxreview-v1:", [], "neither from the person nor from the assistant"  # not from RV
KIND = {"H-FACT": "fact the person gave", "H-ASK": "person asked for", "H-CORR": "correction", "H-ABS": "not know"}
SHA, NOT = "ab" * 32, "Not changed in the third review."
DEC = ("Made-up third-review decisions, shaped like c3_decisions.txt.\nChanges by kind:\n  kind a -> b: 001 t3 (first "
       "note), 002 t4;\n    003 t4 and 002 t5 <b>note</b> & more (x, y; z)\n  dropped: 003 t3 (too vague)\n  moved: "
       "002 t9 (no question there now)\nNot changed (checked): 001 t4 (kept),\n  001 t5 (kept too)\nRe-check: 001.\n")
test = lambda f: TESTS.append(f) or f  # noqa: E731
page_of = lambda recs: RV.render(recs, "oodh_part1.jsonl", SHA)[0]  # noqa: E731

def records(n=9):
    FX.make_world(root := os.path.join(tempfile.mkdtemp(prefix="oodh_rv_"), "sealed", "oodh"), n)
    return B.build(B.load_items(root), n)[0]

class Page(HTMLParser):
    """tags, attribute names, metas, sections (id: data-sample), checks; per section the text of say, q, gold, ideal"""
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.tags, self.attrs, self.metas, self.sections, self.checks = [], [], [], {}, []
        self.feed(text)
        self.close()
        self.by = {s[:s.index('"')]: {k: [html.unescape(t) for t in re.findall(f'class="{k}">([^<]*)<', s)] for k in (
            "say", "q", "gold", "ideal")} for s in text.split('<section class="thread" id="')[1:]}   # text is escaped

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self.tags.append(tag)
        self.attrs += [k for k, _ in attrs]
        self.metas += [a] if tag == "meta" else []
        self.sections.update({a["id"]: a.get("data-sample")} if tag == "section" else {})
        self.checks += [(a["data-t"], a["data-p"])] if "check" in (a.get("class") or "").split() else []

@test
def every_thread_turn_and_probe_appears():
    recs = records()
    assert {len([t for t in r["turns"] if "history_reply" in t]) for r in recs} == {2, 3}    # both fixture types
    pg = Page(page := page_of(recs))
    assert list(pg.sections) == [r["id"] for r in recs], pg.sections
    for r, sec in zip(recs, page.split("<section ")[1:]):
        got = pg.by[r["id"]]
        thread = [t for t in r["turns"] if "history_reply" in t]
        assert got["say"] == [x for t in thread for x in (t["text"], t["history_reply"])], (r["id"], got["say"])
        assert got["q"][1:] == [p["question"] for p in r["probes"]], (r["id"], got["q"])
        assert got["gold"] == [p["gold"] for p in r["probes"] if p["grader"] == "VAL"], (r["id"], got["gold"])
        assert got["ideal"] == [p["ideal"] for p in r["probes"]], (r["id"], got["ideal"])
        words = [w for p in r["probes"] for w in (     # per question, in page order: export ids, kind, stale, source
            [f'data-tree="{r["tree_id"]}" data-p="q{p["turn"]}" data-k="{p["probe_kind"]}"', KIND[p["probe_kind"]]]
            + [f'changed: {", ".join(p["stale"])}<'] * bool(p["stale"])
            + [ABS if p["grader"] == "ABS" else f'said it in message {" and ".join(map(str, p["src"]))}.<'])]
        assert re.search(".*?".join(map(re.escape, words)), sec, re.S), (r["id"], [w for w in words if w not in sec])
    want = [(r["id"], pid) for r in recs for pid in ["conversation"] + [f"q{p['turn']}" for p in r["probes"]]]
    assert pg.checks == want, pg.checks
    assert {p["probe_kind"] for r in recs for p in r["probes"]} == set(KIND)       # every kind is exercised, ABS too

@test
def sample_is_lowest_30_by_hash():
    recs = records(45)
    pg = Page(page := page_of(recs))
    key = {r["id"]: hashlib.sha256((SALT + r["tree_id"]).encode()).hexdigest() for r in recs}
    want = set(sorted(key, key=key.get)[:30])
    got = {rid for rid, s in pg.sections.items() if s == "1"}
    assert got == want, (len(got), sorted(got ^ want))
    assert {rid for rid, s in pg.sections.items() if s == "0"} == set(key) - want
    assert want != {r["id"] for r in recs[:30]}            # the fixture can tell the rule from file order
    assert re.findall(r"<h2>Sample (\d+) of 30 \(", page) == [str(i) for i in range(1, 31)]    # numbered in file order
    head = page[:page.index("<section")]
    assert SALT in head and "the 30 lowest are the sample" in head, "rule not stated at the top"
    assert set(Page(page_of(records(9))).sections.values()) == {"1"}       # fewer than 30 threads: all of them

@test
def turn_with_script_and_ampersand_renders_as_text():
    recs = records()
    nasty = '<script>alert("x")</script> & <b>bold</b> &amp; \'q\' "dq"'
    r = recs[0]
    r["turns"][0].update(text=nasty, history_reply=nasty + " reply")
    r["probes"][0].update(question=nasty + "?", gold="A & B <i>", ideal="</span></p></div><script>x()</script>",
                          accepted=["A & B <i>", "alt & <u>"])        # fixture accepted lists hold only the gold
    pg = Page(page := page_of(recs))
    assert "Also counts as right: alt &amp; &lt;u&gt;</p>" in page, "the other accepted answers are not shown"
    assert pg.tags.count("script") == 1 and "b" not in pg.tags and "i" not in pg.tags, pg.tags.count("script")
    got = pg.by[r["id"]]
    assert got["say"][:2] == [nasty, nasty + " reply"], got["say"][:2]
    assert got["q"][1] == nasty + "?" and got["gold"][0] == "A & B <i>", (got["q"], got["gold"])
    assert got["ideal"][0] == "</span></p></div><script>x()</script>", got["ideal"]
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; &lt;b&gt;" in page and "&amp;amp;" in page

@test
def recheck_page_keeps_those_threads_under_its_own_key_and_export_with_change_lines():
    recs, keep = records(45), ["oodh-001", "oodh-002", "oodh-003"]          # 003 is in the sample, 001 and 002 not
    full, (page, st) = page_of(recs), RV.render(recs, "oodh_part1.jsonl", SHA, set(keep), RV.parse_changes(DEC))
    h2 = lambda p: [h for h in re.findall(r"<h2>(.*?)</h2>", p) if h[-9:-1] in keep]  # noqa: E731  "Sample k of 30"
    js = lambda p: [re.search(k, p).group(1) for k in ('KEY="(.*?)"', 'download="(.*?)"', 'id="only" (\\w+)')]  # noqa
    assert list(Page(page).sections.items()) == [("oodh-001", "0"), ("oodh-002", "0"), ("oodh-003", "1")] and st[
        "threads"] == 3 and h2(page) == h2(full) != [], (Page(page).sections, h2(page), h2(full))
    assert (js(full), js(page)) == (["oodh-review-" + SHA[:16], "oodh_review_marks.json", "checked"], ["oodh-recheck-"
            + SHA[:16], "oodh_recheck_marks.json", "disabled"]) and "!only.disabled&&" in page, (js(full), js(page))
    assert all(s in page[:page.index("<section")] for s in ("from your sample that changed after your review",
               "the <strong>new</strong> ones", "Is the expected answer right", "downloads, oodh_recheck_marks.json"))
    got = [(s[:8], re.search(r'data-p="(\w+)"', c).group(1) if j else None, html.unescape(t))   # (thread, probe, line)
           for s in page.split('<section class="thread" id="')[1:] for j, c in enumerate(s.split('<div class="check"'))
           for t in re.findall(r'<p class="(?:chg|chg gone|exp fine)">(.*?)</p>', c)]
    C, S, G = "Changed in the third review: kind a -> b", ": <b>note</b> & more (x, y; z).", " Not on this page."
    assert got == [("oodh-001", "q3", C + ": first note."), ("oodh-001", "q4", NOT), ("oodh-001", "q5", NOT),
                   ("oodh-002", None, "Changed in the third review: moved: no question there now." + G),
                   ("oodh-002", "q4", C + "."), ("oodh-002", "q5", C + S), ("oodh-003", None, "Changed in the third "
                   "review: dropped: too vague." + G), ("oodh-003", "q3", NOT), ("oodh-003", "q4", C + S),
                   ("oodh-003", "q5", NOT)], got
    assert (st["changed"], st["gone"]) == (4, 2) and "<b>" not in page and "&lt;b&gt;note&lt;/b&gt; &amp;" in page
    assert not re.search('class="(chg|exp fine)', RV.render(recs, "x", SHA, set(keep))[0])    # no --changes, no lines

@test
def sha256_check_refuses_a_changed_file():
    os.makedirs(root := os.path.join(tempfile.mkdtemp(prefix="oodh_rv_"), "sealed", "oodh"))
    path, hp, out, rc, dec = (os.path.join(root, f) for f in ("oodh_part1.jsonl", "HASHES.txt", "REVIEW.html",
                                                              "REVIEW_RECHECK.html", "c3_decisions.txt"))
    recs, _ = records(), Path(dec).write_text(DEC)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    H.write_hashes(recs, False, len(recs), path, hp, [])
    args, only = ["--sealed", root, "--hashes", hp], ["--only", "oodh-001, oodh-003", "--changes", dec]
    assert RV.main(args) == 0 and os.path.getsize(out) > 0 and not os.remove(out)
    assert RV.main(args + ["--only", "oodh-001,oodh-x"]) == 2 and not os.path.exists(rc), "an unknown id not refused"
    data, hashes_text = open(path, encoding="utf-8").read(), open(hp).read()
    Path(path).write_text(data.replace("oodh-001", "oodh-00l", 1), encoding="utf-8")   # file sha256 moves, ids do not
    assert RV.main(args) == 3 and not os.path.exists(out), "a changed file was not refused"
    assert RV.main(args + only) == 3 and not os.path.exists(rc), "the re-check page did not check the file"
    Path(path).write_text(data, encoding="utf-8")
    ids = re.search(r"sorted tree ids .* sha256 ([0-9a-f]{64})", hashes_text).group(1)
    Path(hp).write_text(hashes_text.replace(ids, hashlib.sha256(b"other").hexdigest()))   # only the id hash wrong
    assert RV.main(args) == 3 and not os.path.exists(out), "a tree-id mismatch was not refused"
    Path(hp).write_text("\n".join(ln for ln in hashes_text.splitlines() if not ln.startswith("sealed/")) + "\n")
    assert RV.main(args) == 3 and not os.path.exists(out), "HASHES.txt naming no file was not refused"
    Path(hp).write_text(hashes_text)
    assert RV.main(args + only) == 0 and not os.path.exists(out) and 'class="chg' in (text := open(rc).read())
    assert list(Page(text).sections) == ["oodh-001", "oodh-003"] and RV.main(args) == 0 and os.path.exists(out)
    elsewhere = os.path.join(tempfile.mkdtemp(prefix="oodh_rv_open_"), "REVIEW.html")
    assert RV.main(args + ["--out", elsewhere]) == 2 and not os.path.exists(elsewhere), "wrote outside sealed/"

@test
def page_has_no_external_urls():
    pg = Page(page := page_of(records()))
    assert not set(pg.attrs) & {"src", "href", "srcset", "action", "formaction", "poster", "background", "data"}
    assert not set(pg.tags) & {"link", "img", "iframe", "object", "embed", "audio", "video", "source", "base"}
    hits = re.findall(r"://|//[a-z0-9.-]+\.[a-z]{2,}|@import|url\(|fetch\(|XMLHttpRequest|WebSocket|sendBeacon|"
                      r"EventSource", page)
    assert not hits, hits
    csp = [m["content"] for m in pg.metas if (m.get("http-equiv") or "").lower() == "content-security-policy"]
    assert len(csp) == 1 and "default-src 'none'" in csp[0], csp

@test
def time_estimate_counts_the_words_on_the_page():
    pg, (page, st) = Page((out := RV.render(records(45), "oodh_part1.jsonl", SHA))[0]), out
    on = [pg.by[rid] for rid, s in pg.sections.items() if s == "1"]
    words = sum(len(x.split()) for g in on for x in g["say"] + g["q"][1:] + g["ideal"])
    checks = sum(len(g["q"]) for g in on)
    assert (st["words"], st["sample"], st["sample_probes"]) == (words, 30, checks - 30), (st, words, checks)
    want = round(words / 200 + checks * 30 / 60)
    assert f"the sample has {words:,} words to read and {checks} checks, about {want} minutes" in page, want

M = [('if "history_reply" in t]', 'if "history_reply" in t][:-1]', "last thread turn dropped"),
     ('<p class="say">{E(t["history_reply"])}</p>', '<p class="say"></p>', "assistant turns not shown"),
     ('enumerate(r["probes"], 1)', 'enumerate(r["probes"][1:], 1)', "first probe dropped"),
     ('"planck-oodh-maxreview-v1:", 30', '"planck-oodh-maxreview-v2:", 30', "sample salt changed"),
     ('key=key)[:n]', 'key=key)[-n:]', "highest 30 instead of lowest"), ('key=key)[:n]', ')[:n]', "first 30 by id"),
     ('"planck-oodh-maxreview-v1:", 30', '"planck-oodh-maxreview-v1:", 29', "sample of 29"),
     ('html.escape("" if s is None else str(s), quote=True)', '"" if s is None else str(s)', "no escaping"),
     ('!= want:', '!= want and 0:', "file sha256 not checked"), ("default-src \\'none\\'; ", "", "no CSP default-src"),
     ('!= want_ids:', '!= want_ids and False:', "tree-id hash not checked"),
     ('    if not data or not ids:', '    if False:', "missing HASHES line not refused"),
     ('if "sealed" not in os.path', 'if False and "sealed" not in os.path', "writes outside sealed/"),
     ('<title>OOD-H {', '<link rel="stylesheet" href="https://fonts.example/a.css"><title>OOD-H {', "external font"),
     ('(p["question"], p.get("ideal"))', '(p["question"],)', "estimate skips IDEAL words"),
     ('round(w / WPM + c', 'round(w / WPM + 0 * c', "estimate skips check time"),
     ('neither from the person nor from the assistant', 'from the person', "ABS rule skips the assistant turns"),
     ('if alts else []', 'if 0 else []', "no other accepted answers"), ('if p.get("stale")', 'if 0', "no stale line"),
     ('if p.get("src")', 'if 0', "no source message"), ('data-k="{E(kind)}"', 'data-k=""', "export loses the kind"),
     ('data-tree="{E(r["tree_id"])}"', 'data-tree=""', "export loses the tree id"),
     ('E(KIND_WORDS.get(p["probe_kind"], p["probe_kind"]))', 'E(KIND_WORDS["H-FACT"])', "every question a fact"),
     ('r["tree_id"] in sample], 1)}', 'r["tree_id"] in sample][::-1], 1)}', "sample numbered backwards"),
     ('if not rc or r["id"] in only]', ']', "--only ignored"), ('"oodh-recheck-", "', '"oodh-review-", "', "same key"),
     ('"oodh_recheck_marks.json")}', '"oodh_review_marks.json")}', "re-check export under the full page's name"),
     ('"@KEY@", key + sha[:16]', '"@KEY@", "oodh-review-" + sha[:16]', "the script ignores the re-check key"),
     ('"@EXPORT@", export', '"@EXPORT@", "oodh_review_marks.json"', "the script ignores the re-check export name"),
     ('{"disabled" if rc else', '{"checked" if rc else', "re-check page hides threads outside the sample"),
     ('MODES[only is not None][0]', 'MODES[False][0]', "--only overwrites REVIEW.html"),
     ('if only is not None and (not only', 'if False and (not only', "an unknown --only id not refused"),
     ('are the <strong>new</strong> ones', 'are shown', "header does not say the answers are the new ones"),
     ('if tid == r["id"] for n in ns]', 'for n in ns]', "every thread gets every change line"),
     ('t == p["turn"] and not g]', 't == p["turn"]]', "a removed question's note lands on its successor"),
     ('bool(GONE.search(n[0])) or ', '', "a dropped question's note lands on its successor"),
     ('Not changed in the third review.', 'Changed in the third review.', "unchanged questions read as changed"),
     ('said = E(', 'said = (', "change notes not escaped"), (r'"\n {3,}"', r'"\n {9,}"', "continuations not joined"),
     (r'^\S.*:\n(', r'^\S.*\n(', "a paragraph outside the change list read"),
     ('[refs[-1].end():]', '[refs[0].end():]', "'a and b' split"), ('None if a.changes is None else parse',
                                                                       'None if True else parse', "--changes ignored"),
     (r'(?:\((?:[^()]|\([^()]*\))*\)|[^,;()])+', r'[^,;]+', "a note cut at a comma inside brackets"),
     (r're.sub(r"^\(([^()]*)\)$", r"\1"', r're.sub(r"^\(([^()]*)\)$x", r"\1"', "outer brackets kept")]

def mutants():
    """each row of M on a scratch copy of oodh/ (rc12/ and corpus/oodh.py beside it) with one exact text of
    review_view.py replaced: KILLED = the tests exit non-zero, MALFORMED = not 1 match; row 0 (no change) must pass"""
    survived = bad = 0
    for k, (old, new, why) in enumerate([("SAMPLE_SALT, SAMPLE_N", "SAMPLE_SALT, SAMPLE_N", "baseline")] + M):
        dst = tempfile.mkdtemp(prefix="mut_rv_")
        shutil.copytree(OODH, os.path.join(dst, "oodh"), ignore=shutil.ignore_patterns("__pycache__"))
        for sub, pats in (("rc12", ["*.py", "*.json"]), ("corpus", ["oodh.py"])):
            os.makedirs(os.path.join(dst, sub))
            for f in [f for pat in pats for f in glob.glob(os.path.join(REPO, sub, pat))]:
                shutil.copy(f, os.path.join(dst, sub))
        text = open(rv := os.path.join(dst, "oodh", "review_view.py")).read()
        Path(rv).write_text(text.replace(old, new))
        r = subprocess.run([sys.executable, "-B", os.path.join(dst, "oodh", "tests", "test_review_view.py")],
                           capture_output=True, text=True, timeout=300, cwd=dst)
        shutil.rmtree(dst, ignore_errors=True)
        code, out = (r.returncode if text.count(old) == 1 else None), r.stdout + r.stderr
        if k == 0 and code != 0:
            return print(f"baseline (unmutated copy): exit {code}\n{out}") or 1
        bad, survived = bad + (code is None), survived + (code == 0 and k > 0)
        print(f"#{k} {'MALFORMED' if code is None else 'KILLED' if code else 'SURVIVED' if k else 'PASSES'}: {why} ",
              [ln.split(":")[0][5:] for ln in out.splitlines() if ln.startswith("FAIL")])
    print(f"{len(M) - survived - bad} of {len(M)} killed, {survived} survived, {bad} malformed")
    return 0 if survived == 0 and bad == 0 else 1

def main():
    if "--mutants" in sys.argv:
        return mutants()
    failed = 0
    for f in TESTS:
        try:
            print(f() or f"ok   {f.__name__}")
        except Exception as e:                      # noqa: BLE001
            failed += print(f"FAIL {f.__name__}: {type(e).__name__}: {str(e)[:300]}") or 1
    return print(f"{len(TESTS) - failed} of {len(TESTS)} passed") or (1 if failed else 0)

if __name__ == "__main__":
    sys.exit(main())
