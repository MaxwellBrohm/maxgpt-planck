"""SCREENS.txt AMENDMENT CAP-31 (2026-10-11): ORDER's cap is 31 GPU-hours on Max's answer. The entry's place and
dating, the question, its options and Max's answer verbatim, the code's cap (a flag past 31 h, never a cut), and the
entry's hours generated from the hours code and the logged hold times; with SCREENS_PC_COPY_1011 naming a copy of the
PC's queue_screens.txt, status.jsonl and c6diag.log of 2026-10-11, also from that copy. No model, nothing on the PC.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_cap31.py
"""
import collections, contextlib, io, math, os, re, subprocess, sys  # noqa: E401

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import analyze_lib as AL  # noqa: E402
import screens_hours as H  # noqa: E402
import screens_lib as L  # noqa: E402
from test_screens_integration import flat, lines  # noqa: E402

HEAD = "AMENDMENT CAP-31 ("
QUESTION = ("The screens budget rule now flags overruns to you instead of cutting anything. The GPU total is at 26.0 of "
            "the 28-hour budget. Two checks the audit asked for come next: re-running the base model to explain its "
            "slightly worse scores (~0.9 h), and scoring whether each model actually uses the conversation history "
            "(~1-2 h). Together they'd land at about 28 to 29 hours. OK to raise the screens budget to 31 hours so "
            "nothing gets cut or stalls?")
OPTIONS = [("Raise it to 31 h (Recommended)", "Both checks run and the screens finish; the change is recorded with your "
            "words before any of their results exist."),
           ("Raise it only as needed", "Run them anyway and record whatever the final total is; ask me again only if it "
            "goes past 31 h.")]
ANSWER = "do whatever you recommend, however i do have a question, whats the screens budget rule???? why is it in place??"
REPLAY = [("base_s101_rep1", "2026-10-10 21:32:08", "2026-10-10 21:50:31"),       # parse_holds on the 2026-10-11 copy
          ("base_s101_hookoff", "2026-10-10 22:28:09", "2026-10-10 22:46:30"),
          ("base_s101_rep2", "2026-10-10 23:32:59", "2026-10-10 23:51:19")]
C6 = [("2026-10-10 21:23:53", "2026-10-10 21:32:08"), ("2026-10-10 22:25:10", "2026-10-10 22:28:09")]  # c6diag.log
RUNS_N, RUNS_H, DRY = 99, 24.78472222222222, 0.02     # the copy's 99 runs and their hours; the dry run, counted
SHA = {"queue_screens.txt": "6c19954e97b3b35f", "status.jsonl": "dd76ba22c2533d4a", "c6diag.log": "474ebcc1ba931bdd"}
N_TESTS, BY_FILE = 9, {"analyze_lib.py": 2, "screens_hours.py": 1, "SCREENS.txt": 12}


def f3(x):
    return f"{x:.3f}"


def hours():
    other = sum(h for _, h, _ in H.OTHER)
    rep = [H.hours_between(a, b) for _, a, b in REPLAY]
    c6s = sum(H.hours_between(a, b) * 3600 for a, b in C6)
    rule = RUNS_H + other
    total = rule + c6s / 3600 + DRY
    return {"other": other, "rep": rep, "c6s": c6s, "rule": rule, "total": total}


def test_place_and_dating():
    ls = lines()
    heads = [i for i, ln in enumerate(ls) if ln.startswith(HEAD)]
    prev = next(i for i, ln in enumerate(ls) if ln.startswith("CORRECTIONS AFTER THE W1 VERIFY ("))
    assert len(heads) == 1 and heads[0] > prev and ls[heads[0] - 1] == ""
    f = flat(HEAD)
    assert f.startswith("AMENDMENT CAP-31 (2026-10-11 09:13 EDT, Claude, W1's commit step, on Max's answer below;")
    assert "nothing committed when written. Written before any C6 diagnostic, BASE replay or hook-off result is read" in f
    assert "none of the 133 c6/*.json files and none of the three runs' bpb.jsonl, diag.jsonl or log.jsonl had been " \
           "read since written" in f
    assert "\u2014" not in f and "\u2013" not in f


def test_question_options_and_answer_are_verbatim():
    f = flat(HEAD)
    assert f'ASKED (Claude to Max, 2026-10-11 08:21 EDT, verbatim): "{QUESTION}"' in f
    assert "Options, verbatim: " + "; ".join(f'"{a}", "{b}"' for a, b in OPTIONS) + "." in f
    assert f'MAX (the PI), 2026-10-11 09:03 EDT, verbatim: "{ANSWER}"' in f


def test_the_decision_and_what_did_not_change():
    f = flat(HEAD)
    for phrase in ("Claude recommended 31 h, so ORDER's cap is 31 GPU-hours from this entry on, read as AMENDMENT "
                   "CAP-FLAG reads it", "past the cap, nothing is cut or queued and the step brings Max a FLAG",
                   "Nothing else in CAP-FLAG changes.", "Max also asked what the rule is and why it is in place; that is "
                   "answered to him in chat, not here.", "analyze_lib.CAP = 31.0 (was 28.0)",
                   "screens.py hours (screens_hours report) now reads \"against 31 h\""):
        assert phrase in f, phrase


def test_the_code_cap_is_31_and_a_flag_never_a_cut():
    """This entry's 31 h, passed explicitly since AMENDMENT CAP-REMOVED (2026-10-11) made the code's cap infinite."""
    assert AL.CAP == math.inf and AL.cap_cut({})["cap"] == math.inf == AL.cap_check({})["cap"]   # CAP-REMOVED
    rem = {"BASE": 1.0, "S006": 2.0, "S003": 1.0}
    r = AL.cap_check(rem, measured=27.0, cap=31.0)                                        # exactly 31 h fits
    assert r["fits"] and not r["flag_to_max"] and r["proposal"] is None and r["cut"] == []
    r = AL.cap_check(rem, measured=27.0001, cap=31.0)
    assert r["flag_to_max"] and r["cut"] == [] and r["proposal"]["cut"] == ["S006"] and r["over_by"] > 0
    assert AL.cap_check(rem, measured=26.0, cap=28.0)["flag_to_max"]                     # over the old cap: a flag


def test_the_hours_report_reads_31_by_default():
    """This entry's default report read 31 h; since AMENDMENT CAP-REMOVED the default prints hours only (no cap line)
    and 31 h is passed explicitly to reproduce this entry's line."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        r = H.report(L.params(), cap=31.0)
    line = next(ln for ln in out.getvalue().splitlines() if ln.startswith("ORDER cap"))
    assert r["cap"]["cap"] == 31.0 == r["flag"]["cap"] and " h against 31 h: " in line
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        d = H.report(L.params())
    assert d["cap"] is None is d["flag"] and not any(ln.startswith("ORDER cap") for ln in out.getvalue().splitlines())


def test_what_the_question_had_wrong_is_generated():
    f, h = flat(HEAD), hours()
    c6e = float(re.search(r"about 22 min, (\d\.\d\d) GPU h", flat("AMENDMENT C6-DIAG ("))[1])
    assert f"base_s101_rep1 21:32:08 to 21:50:31, base_s101_hookoff 22:28:09 to 22:46:30, base_s101_rep2 23:32:59 to " \
           f"23:51:19, {f3(h['rep'][0])} h each), and the C6 queue {h['c6s']:.0f} s, {f3(h['c6s'] / 3600)} h " \
           "(c6diag.log: 21:23:53 to 21:32:08 and 22:25:10 to 22:28:09)" in f
    assert len({f3(x) for x in h["rep"]}) == 1 and f"held gpu.lock {f3(sum(h['rep']))} h (screens_hours.py" in f
    assert f"not \"~1-2 h\" (C6-DIAG's own estimate was {c6e:.2f} h)" in f
    assert "Its \"26.0\" was CAP-FLAG's POSITION NOW, 25.987 h, before them." in f and \
        "+ 2.120 h OTHER = 25.987 h" in flat("AMENDMENT CAP-FLAG (")
    assert 'it said the two checks "come next"; both had run the night before, from 2aa3787.' in f


def test_position_now_arithmetic():
    f, h = flat(HEAD), hours()
    assert f"{RUNS_N} runs measured, {f3(RUNS_H)} h, + {f3(h['other'])} h OTHER = {f3(h['rule'])} h " \
           "(screens_hours.py's rule input)" in f
    assert f"+ the C6 queue's {f3(h['c6s'] / 3600)} h as OTHER" in f and f"counted as {DRY:.2f} h = {f3(h['total'])} h; " \
           "queued 0 of 90 registered runs." in f
    old, new = AL.cap_check({}, h["total"], cap=28.0), AL.cap_check({}, h["total"], cap=31.0)
    assert old["fits"] and new["fits"] and not old["flag_to_max"]
    assert f"under 28 h too, with {f3(28.0 - h['total'])} h of headroom" in f
    assert f"with {f3(31.0 - h['total'])} h of headroom under 31 h" in f      # 31 h: this entry's cap (CAP-REMOVED)
    assert (f3(h["rule"]), f3(h["total"]), f3(28.0 - h["total"]), f3(31.0 - h["total"])) == \
        ("26.905", "27.112", "0.888", "3.888")
    for k, v in SHA.items():
        assert f"{k}, sha256 {v}..." in f or f"{k}, {v}..." in f, k


PC = os.environ.get("SCREENS_PC_COPY_1011", "")


@pytest.mark.skipif(not os.path.isfile(os.path.join(PC, "c6diag.log")), reason="needs the 2026-10-11 copy of the PC logs")
def test_position_now_is_the_1011_copy(tmp_path):
    for k, v in SHA.items():
        assert L.sha256(os.path.join(PC, k)).startswith(v), k
    q, st, c6 = (open(os.path.join(PC, k)).read() for k in SHA)
    t = tmp_path / "now.tsv"
    t.write_text(H.tsv_text(q, st))
    with contextlib.redirect_stdout(io.StringIO()):
        r = H.report(L.params(), str(t), cap=31.0)        # this entry's cap (none since AMENDMENT CAP-REMOVED)
    runs, other = H.load(str(t))
    assert (len(runs), sum(runs.values()), sum(r["tally"]["n_queued"].values())) == (RUNS_N, pytest.approx(RUNS_H), 0)
    assert f3(r["flag"]["total"]) == f3(hours()["rule"]) and not r["flag"]["flag_to_max"]
    assert [x[:3] for x in H.parse_holds(q, st)[-3:]] == REPLAY
    pairs, cur = [], None
    for ln in c6.splitlines():
        if re.match(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d c6diag: gpu\.lock held", ln):
            cur = ln[:19]
        elif re.match(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d c6diag: gpu\.lock released", ln):
            pairs.append((cur, ln[:19]))
    assert pairs == C6 and c6.rstrip().endswith("2026-10-10 22:28:09 c6diag: DONE")
    assert q.rstrip().splitlines()[-2:] == ["2026-10-10 23:51:19 MARK SCREENS BASE REPLAY DONE",
                                            "2026-10-10 23:51:19 plan finished: stage2_replay.txt"]
    assert sum(1 for ln in c6.splitlines() if ln.startswith("done ")) == 133


def test_counts():
    import mutation_screens as M
    from mutants_cap31 import MUTANTS_CAP31
    f = flat(HEAD)
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "--collect-only", "-p", "no:cacheprovider", __file__],
                         capture_output=True, text=True, cwd=HERE).stdout
    n = sum(1 for ln in out.splitlines() if "::" in ln)
    assert n == N_TESTS and f"tests/test_screens_cap31.py ({N_TESTS}:" in f
    by = collections.Counter(os.path.basename(m[1]) for m in MUTANTS_CAP31)
    assert by == BY_FILE and f"tests/mutants_cap31.py ({len(MUTANTS_CAP31)} mutants: " + \
        ", ".join(f"{k} {v}" for k, v in BY_FILE.items()) + ")" in f
    assert all(m in M.MUTANTS for m in MUTANTS_CAP31)
