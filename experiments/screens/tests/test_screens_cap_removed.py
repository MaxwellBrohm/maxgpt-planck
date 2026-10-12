"""SCREENS.txt AMENDMENT CAP-REMOVED (2026-10-11): ORDER's cap and cut order no longer exist for any SCREENS run. The
entry's place, dating and Max's words (equal to PLAN.md's), the code (CAP infinite; cap_check never flags and cap_cut
never cuts at the default; an explicit earlier cap keeps its arithmetic), and screens.py hours printing hours only.
No model, nothing on the PC.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_cap_removed.py
"""
import contextlib, io, math, os, sys  # noqa: E401

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import analyze_lib as AL  # noqa: E402
import privacy  # noqa: E402
import screens  # noqa: E402
import screens_hours as H  # noqa: E402
import screens_lib as L  # noqa: E402
from test_screens_integration import flat, lines  # noqa: E402

HEAD = "AMENDMENT CAP-REMOVED ("
MAX = ("so the thing is tho all the other chats that needed the pc arent running anymore, so why dont we just drop that "
       "gpu hour limit?? you can use it all the time no matter what unless i say otherwise and remember we prefer quality "
       "over speed 100 percent of the time ok?")
REM = {"BASE": 5.0, "S001": 5.0, "S002": 9.0, "S003": 9.0, "S004": 5.0, "S005": 9.0, "S006": 9.0, "S007": 9.0}


def test_place_dating_and_max_verbatim():
    ls = lines()
    heads = [i for i, ln in enumerate(ls) if ln.startswith(HEAD)]
    closed = next(i for i, ln in enumerate(ls) if ln.startswith("BATCH 1 CLOSED ("))
    seeds = next(i for i, ln in enumerate(ls) if ln.startswith("AMENDMENT SEEDS-103-105 ("))
    assert len(heads) == 1 and closed < heads[0] < seeds and ls[heads[0] - 1] == ""
    f = flat(HEAD)
    assert f.startswith("AMENDMENT CAP-REMOVED (2026-10-11 22:50 EDT, Claude, the SCREENS robustness workflow, on Max's "
                        "rule below;")
    assert "Written before any run it could affect: no SCREENS queue has run since stage2_replay's mark (2026-10-10 " \
           "23:51:19), plans/CURRENT named batch1_closed, and no robustness run exists." in f
    assert f'Max (the PI), 2026-10-11, verbatim (PLAN.md, GPU and quality policy, b790d85): "{MAX}"' in f
    plan = " ".join(open(os.path.join(L.ROOT, "PLAN.md")).read().split())
    assert f'Max: "{MAX}"' in plan
    q = ("Each pre-registration that carries a cap gets its own dated amendment removing it, written before any result "
         "the cap could have affected.")
    assert q in plan and f'"{q}" This is that amendment for SCREENS.' in f
    assert "\u2014" not in f and "\u2013" not in f and not privacy.hits("\n".join(ls[heads[0]:seeds]))


def test_the_decision_and_what_stays():
    f = flat(HEAD)
    for phrase in ("ORDER's CAP AND CUT RULE no longer exists for any SCREENS run, of batch 1 or of any later stage of it",
                   "no cap, no cut order, no never-cut list, no flag for hours", "never a reason to cut, defer or not "
                   "queue a run", "Unchanged: gpu.lock per run", "the E4 GATE (a scientific gate, not a budget rule)",
                   "No cut stands from them (S006's, the only one, was reversed by AMENDMENT S006-REINSTATE), so no run "
                   "is restored by this entry.", "analyze_lib.CAP = INF (was 31.0)", "screens.py robust has no cut."):
        assert phrase in f, phrase
    assert "AMENDMENT S006-REINSTATE (" in "\n".join(lines())


def test_the_code_has_no_cap():
    assert AL.CAP == math.inf == AL.INF
    for total in (0.0, 31.0001, 1e6, 1e300):
        r = AL.cap_check(REM, measured=total)
        assert r["fits"] and not r["flag_to_max"] and r["cut"] == [] and r["proposal"] is None and r["over_by"] == 0.0
        c = AL.cap_cut(REM, measured=total)
        assert c["cut"] == [] and c["fits"] and c["total_after"] == c["total_before"]
    r = AL.cap_check(REM, measured=0.0, cap=28.0)          # an earlier entry's cap keeps its arithmetic (the record)
    assert r["flag_to_max"] and r["proposal"]["cut"] == ["S006", "S003", "S007", "S002"] and r["cut"] == []
    assert AL.cap_cut(REM, cap=25.0)["cut"] == ["S006", "S003", "S007", "S002"]


def test_screens_py_hours_reports_hours_only(capsys):
    assert screens.main(["hours"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert not [ln for ln in out if ln.startswith(("ORDER cap", "runs only"))]
    assert not [ln for ln in out if "FLAG TO MAX" in ln or "cut order would take" in ln or "headroom" in ln]
    last = out[-1]
    assert last.startswith("GPU hours, reported only (AMENDMENT CAP-REMOVED: no cap, cut order or flag): measured ")
    with contextlib.redirect_stdout(io.StringIO()):
        r = H.report(L.params())
    assert (r["cap"], r["flag"], r["narrow"]) == (None, None, None)
    assert r["total"] == pytest.approx(r["measured_runs"] + r["other"] + sum(r["tally"]["queued"].values()))
    assert f"= {r['total']:.3f} h" in last and f"(runs {r['measured_runs']:.3f} h + other {r['other']:.3f} h)" in last
    out2 = io.StringIO()
    with contextlib.redirect_stdout(out2):
        r31 = H.report(L.params(), cap=31.0)               # an explicit earlier cap still prints its line (the record)
    assert any(ln.startswith("ORDER cap") and " h against 31 h: " in ln for ln in out2.getvalue().splitlines())
    assert r31["flag"]["cap"] == 31.0 and r31["cap"]["cap"] == 31.0


def test_screens_py_robust_takes_no_cut(capsys):
    with pytest.raises(SystemExit):
        screens.main(["robust", "--cut", "S006"])
    assert "unrecognized arguments: --cut S006" in capsys.readouterr().err
