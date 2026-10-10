"""oodh/review_view.py on the synthetic fixture (tests/fixture.py: made-up threads, no sealed file). Loads no model.
  python3 -B oodh/tests/test_review_view.py            the tests
  python3 -B oodh/tests/test_review_view.py --mutants  every mutant in M must turn the tests red (else exit 1)"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
OODH, REPO = os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != OODH]
import select  # noqa: E402,F401  the stdlib module first: oodh/ holds a select.py of its own
sys.path[:0] = [OODH, HERE]

import glob, hashlib, json, re, shutil, subprocess, tempfile  # noqa: E401,E402
from html.parser import HTMLParser  # noqa: E402
from pathlib import Path  # noqa: E402

import build as B, fixture as FX, hashes as H, review_view as RV  # noqa: E401,E402

SALT = "planck-oodh-maxreview-v1:"     # written out here, not read from review_view
TESTS = []


def test(f):
    TESTS.append(f)
    return f


def records(n=9):
    root = os.path.join(tempfile.mkdtemp(prefix="oodh_rv_"), "sealed", "oodh")
    FX.make_world(root, n)
    return B.build(B.load_items(root), n)[0]


class Page(HTMLParser):
    """tags, attribute names, metas, sections (id: data-sample), checks; per section the text of say, q, gold, ideal"""
    TRACK = ("say", "q", "gold", "ideal")

    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.tags, self.attrs, self.metas, self.sections, self.checks, self.by = [], [], [], {}, [], {}
        self.sec = self.cur = self.cur_tag = None
        self.feed(text)
        self.close()

    def handle_starttag(self, tag, attrs):
        a, cls = dict(attrs), set((dict(attrs).get("class") or "").split())
        self.tags.append(tag)
        self.attrs += [k for k, _ in attrs]
        self.metas += [a] if tag == "meta" else []
        if tag == "section":
            self.sec, self.sections[a["id"]] = a["id"], a.get("data-sample")
            self.by[self.sec] = {k: [] for k in self.TRACK}
        if "check" in cls:
            self.checks.append((a["data-t"], a["data-p"]))
        hit = [k for k in self.TRACK if k in cls]
        if hit and self.sec:
            self.cur, self.cur_tag = hit[0], tag
            self.by[self.sec][self.cur].append("")

    def handle_endtag(self, tag):
        if tag == self.cur_tag:
            self.cur = self.cur_tag = None

    def handle_data(self, d):
        if self.cur:
            self.by[self.sec][self.cur][-1] += d


def page_of(recs):
    return RV.render(recs, "oodh_part1.jsonl", "ab" * 32)[0]


@test
def every_thread_turn_and_probe_appears():
    recs = records()
    assert {len([t for t in r["turns"] if "history_reply" in t]) for r in recs} == {2, 3}    # both fixture types
    pg = Page(page_of(recs))
    assert list(pg.sections) == [r["id"] for r in recs], pg.sections
    for r in recs:
        got = pg.by[r["id"]]
        thread = [t for t in r["turns"] if "history_reply" in t]
        assert got["say"] == [x for t in thread for x in (t["text"], t["history_reply"])], (r["id"], got["say"])
        assert got["q"][1:] == [p["question"] for p in r["probes"]], (r["id"], got["q"])
        assert got["gold"] == [p["gold"] for p in r["probes"] if p["grader"] == "VAL"], (r["id"], got["gold"])
        assert got["ideal"] == [p["ideal"] for p in r["probes"]], (r["id"], got["ideal"])
    want = [(r["id"], pid) for r in recs for pid in ["conversation"] + [f"q{p['turn']}" for p in r["probes"]]]
    assert pg.checks == want, pg.checks
    assert sum(1 for r in recs for p in r["probes"] if p["grader"] == "ABS") > 0      # the ABS branch is exercised


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
    head = page[:page.index("<section")]
    assert SALT in head and "the 30 lowest are the sample" in head, "rule not stated at the top"
    small = Page(page_of(records(9)))                        # fewer than 30 threads: all of them
    assert set(small.sections.values()) == {"1"}


@test
def turn_with_script_and_ampersand_renders_as_text():
    recs = records()
    nasty = '<script>alert("x")</script> & <b>bold</b> &amp; \'q\' "dq"'
    r = recs[0]
    r["turns"][0].update(text=nasty, history_reply=nasty + " reply")
    r["probes"][0].update(question=nasty + "?", gold="A & B <i>", ideal="</span></p></div><script>x()</script>")
    pg = Page(page := page_of(recs))
    assert pg.tags.count("script") == 1 and "b" not in pg.tags and "i" not in pg.tags, pg.tags.count("script")
    got = pg.by[r["id"]]
    assert got["say"][:2] == [nasty, nasty + " reply"], got["say"][:2]
    assert got["q"][1] == nasty + "?" and got["gold"][0] == "A & B <i>", (got["q"], got["gold"])
    assert got["ideal"][0] == "</span></p></div><script>x()</script>", got["ideal"]
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; &lt;b&gt;" in page and "&amp;amp;" in page


def sealed_world():
    recs = records()
    os.makedirs(root := os.path.join(tempfile.mkdtemp(prefix="oodh_rv_"), "sealed", "oodh"))
    path, hp = os.path.join(root, "oodh_part1.jsonl"), os.path.join(root, "HASHES.txt")
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    H.write_hashes(recs, False, len(recs), path, hp, [])
    return root, path, hp, os.path.join(root, "REVIEW.html")


@test
def sha256_check_refuses_a_changed_file():
    root, path, hp, out = sealed_world()
    args = ["--sealed", root, "--hashes", hp]
    assert RV.main(args) == 0 and os.path.getsize(out) > 0
    os.remove(out)
    data, hashes_text = open(path, encoding="utf-8").read(), open(hp).read()
    Path(path).write_text(data.replace("oodh-001", "oodh-00l", 1), encoding="utf-8")   # file sha256 moves, ids do not
    assert RV.main(args) == 3 and not os.path.exists(out), "a changed file was not refused"
    Path(path).write_text(data, encoding="utf-8")
    ids = re.search(r"sorted tree ids .* sha256 ([0-9a-f]{64})", hashes_text).group(1)
    Path(hp).write_text(hashes_text.replace(ids, hashlib.sha256(b"other").hexdigest()))   # only the id hash wrong
    assert RV.main(args) == 3 and not os.path.exists(out), "a tree-id mismatch was not refused"
    Path(hp).write_text("\n".join(ln for ln in hashes_text.splitlines() if not ln.startswith("sealed/")) + "\n")
    assert RV.main(args) == 3 and not os.path.exists(out), "HASHES.txt naming no file was not refused"
    Path(hp).write_text(hashes_text)
    assert RV.main(args) == 0 and os.path.exists(out)
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
    recs = records(45)
    pg = Page((out := RV.render(recs, "oodh_part1.jsonl", "ab" * 32))[0])
    page, st = out
    on = [pg.by[rid] for rid, s in pg.sections.items() if s == "1"]
    words = sum(len(x.split()) for g in on for x in g["say"] + g["q"][1:] + g["ideal"])
    checks = sum(len(g["q"]) for g in on)
    assert (st["words"], st["sample"], st["sample_probes"]) == (words, 30, checks - 30), (st, words, checks)
    want = round(words / 200 + checks * 30 / 60)
    assert f"the sample has {words:,} words to read and {checks} checks, about {want} minutes" in page, want


# Each mutant replaces one exact text in the scratch copy of oodh/review_view.py; KILLED = the tests exit non-zero.
M = [('if "history_reply" in t]', 'if "history_reply" in t][:-1]', "last thread turn dropped"),
     ('<p class="say">{E(t["history_reply"])}</p>', '<p class="say"></p>', "assistant turns not shown"),
     ('enumerate(r["probes"], 1)', 'enumerate(r["probes"][1:], 1)', "first probe dropped"),
     ('"planck-oodh-maxreview-v1:", 30', '"planck-oodh-maxreview-v2:", 30', "sample salt changed"),
     ('key=sample_key)[:n]', 'key=sample_key)[-n:]', "highest 30 instead of lowest"),
     ('key=sample_key)[:n]', ')[:n]', "first 30 by tree id"),
     ('"planck-oodh-maxreview-v1:", 30', '"planck-oodh-maxreview-v1:", 29', "sample of 29"),
     ('html.escape("" if s is None else str(s), quote=True)', '"" if s is None else str(s)', "no escaping"),
     ('if got != want:', 'if False:', "file sha256 not checked"),
     ('if got_ids != want_ids:', 'if False:', "tree-id hash not checked"),
     ('    if not data or not ids:', '    if False:', "missing HASHES line not refused"),
     ('if "sealed" not in os.path', 'if False and "sealed" not in os.path', "writes outside sealed/"),
     ('<title>OOD-H review</title>', '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter">'
      '<title>OOD-H review</title>', "external font"),
     ("default-src \\'none\\'; ", "", "no CSP default-src"),
     ('(p["question"], p.get("ideal"))', '(p["question"],)', "estimate skips IDEAL words"),
     ('round(words / WPM + checks', 'round(words / WPM + 0 * checks', "estimate skips check time")]


def run_copy(old, new):
    """(exit code, output) of the tests in a scratch copy with one text of review_view.py replaced; None: not 1 match"""
    dst = tempfile.mkdtemp(prefix="mut_rv_")
    shutil.copytree(OODH, os.path.join(dst, "oodh"), ignore=shutil.ignore_patterns("__pycache__"))
    for sub, pats in (("rc12", ["*.py", "*.json"]), ("corpus", ["oodh.py"])):
        os.makedirs(os.path.join(dst, sub))
        for f in [f for pat in pats for f in glob.glob(os.path.join(REPO, sub, pat))]:
            shutil.copy(f, os.path.join(dst, sub))
    rv = os.path.join(dst, "oodh", "review_view.py")
    text = open(rv).read()
    if text.count(old) != 1:
        return None, ""
    Path(rv).write_text(text.replace(old, new))
    r = subprocess.run([sys.executable, "-B", os.path.join(dst, "oodh", "tests", "test_review_view.py")],
                       capture_output=True, text=True, timeout=300, cwd=dst)
    shutil.rmtree(dst, ignore_errors=True)
    return r.returncode, r.stdout + r.stderr


def mutants():
    code, out = run_copy("SAMPLE_SALT, SAMPLE_N", "SAMPLE_SALT, SAMPLE_N")
    print(f"baseline (unmutated copy): exit {code}")
    if code != 0:
        return print(out) or 1
    survived = bad = 0
    for k, (old, new, why) in enumerate(M, 1):
        code, out = run_copy(old, new)
        fails = [ln.split(":")[0][5:] for ln in out.splitlines() if ln.startswith("FAIL")]
        bad, survived = bad + (code is None), survived + (code == 0)
        print(f"#{k} {'MALFORMED' if code is None else 'KILLED' if code else 'SURVIVED'}: {why}  {fails}")
    print(f"{len(M) - survived - bad} of {len(M)} killed, {survived} survived, {bad} malformed")
    return 0 if survived == 0 and bad == 0 else 1


def main():
    if "--mutants" in sys.argv:
        return mutants()
    failed = 0
    for f in TESTS:
        try:
            f()
            print(f"ok   {f.__name__}")
        except Exception as e:                      # noqa: BLE001
            failed += 1
            print(f"FAIL {f.__name__}: {type(e).__name__}: {str(e)[:300]}")
    print(f"{len(TESTS) - failed} of {len(TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
