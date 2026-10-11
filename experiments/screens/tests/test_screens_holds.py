"""SCREENS DEVIATION 1 (SCREENS.txt, 2026-10-10): screens_hours counts each gpu.lock hold once. A crash that
queue_lib.sh resumes inside the same hold is one hold (before, it was listed twice and load() stopped on "measured
twice"); a run resumed after a power cut, or after a heat stop, has one line per hold, each counted once, and the
time the queue was down or cooling is not GPU time. On the real queue log (96 runs, none resumed) the new parse
equals the old one: test_the_real_queue_log_parses_as_before (a copy of the PC files, when present). No model.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_holds.py
"""
import json, os, sys  # noqa: E401

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import screens_hours as H  # noqa: E402
import screens_lib as L  # noqa: E402

D = "2026-10-11 "
START = D + "05:00:00 queue start: exp SCREENS, plan stage2_x.txt (sha256 0123456789abcdef), code commit " + "a" * 40
CRASH = [START, D + "05:00:01 waiting for gpu.lock", D + "05:00:02 gpu.lock held; GPU memory in use 401 MiB",
         D + "05:00:04 train s007_smear_g1_s103.yaml --require-committed",
         D + "05:06:00 crash s007_smear_g1_s103 rc 1: resuming once from its last checkpoint",
         D + "05:06:00 train s007_smear_g1_s103.yaml --require-committed", D + "05:20:00 done s007_smear_g1_s103 rc 0",
         D + "05:20:01 score s007_smear_g1_s103/final_00007630", D + "05:22:00 waiting for gpu.lock"]
CUT = [START, D + "05:00:01 waiting for gpu.lock", D + "05:00:02 gpu.lock held; GPU memory in use 401 MiB",
       D + "05:00:04 train base_s103.yaml --require-committed", "\0" * 300,
       D + "06:30:00 boot resume: queue_screens.txt's last start line had no end (a power cut or a kill)",
       "\0\0\0" + D + "06:30:01 queue start: exp SCREENS, plan stage2_x.txt (sha256 0123456789abcdef), code commit " + "a" * 40,
       D + "06:30:01 waiting for gpu.lock", D + "06:30:05 gpu.lock held; GPU memory in use 300 MiB",
       D + "06:30:06 base_s103: the queue was cut while it trained: counted as its one crash; resuming once",
       D + "06:30:07 train base_s103.yaml --require-committed", D + "06:40:00 done base_s103 rc 0",
       D + "06:41:30 waiting for gpu.lock", D + "06:41:31 gpu.lock held; GPU memory in use 300 MiB",
       D + "06:41:32 train base_s104.yaml --require-committed"]                     # base_s104: running, no status
HEAT = [START, D + "05:00:01 waiting for gpu.lock", D + "05:00:02 gpu.lock held; GPU memory in use 401 MiB",
        D + "05:00:04 train s004_canonac_g1_s103.yaml --require-committed",
        D + "05:10:00 GPU at 88 C for 3 samples: stopping the run", D + "05:10:30 GPU temperature [82] not <= 75: waiting",
        D + "05:25:00 waiting for gpu.lock", D + "05:25:01 gpu.lock held; GPU memory in use 300 MiB",
        D + "05:25:02 train s004_canonac_g1_s103.yaml --require-committed", D + "05:40:00 done s004_canonac_g1_s103 rc 0"]


def status(*runs):
    return "".join(json.dumps({"time": t.replace(" ", "T"), "exp": "SCREENS", "run": r, "outcome": o}) + "\n"
                   for r, t, o in runs)


def h(a, b):
    return H.hours_between(D + a, D + b)


def test_a_crash_inside_one_hold_is_one_hold():
    st = status(("s007_smear_g1_s103", D + "05:21:59", "done and scored"))
    assert H.parse_queue("\n".join(CRASH), st) == [("s007_smear_g1_s103", D + "05:00:02", D + "05:21:59")]


def test_a_power_cut_gives_two_holds_without_the_downtime():
    st = status(("base_s103", D + "06:41:29", "done and scored"))
    got = H.parse_holds("\n".join(CUT), st)
    assert got == [("base_s103", D + "05:00:02", D + "05:00:04", H.SRC_CUT),            # ends at the cut queue's last line
                   ("base_s103", D + "06:30:05", D + "06:41:29", "queue_screens.txt")]   # base_s104 running: left out


def test_a_relaunch_after_a_nul_tail_is_a_restart():
    """A person's relaunch (no boot resume line): the new queue's start line follows the cut's NUL bytes on the same
    line. It still ends the cut hold at the cut queue's last line, not at the next lock request (no downtime counted)."""
    log = CUT[:5] + CUT[6:]
    st = status(("base_s103", D + "06:41:29", "done and scored"))
    assert H.parse_holds("\n".join(log), st)[0] == ("base_s103", D + "05:00:02", D + "05:00:04", H.SRC_CUT)


def test_a_heat_stop_gives_two_holds_without_the_cool_down():
    st = status(("s004_canonac_g1_s103", D + "05:41:00", "done and scored"))
    assert H.parse_holds("\n".join(HEAT), st) == [
        ("s004_canonac_g1_s103", D + "05:00:02", D + "05:10:30", H.SRC_EARLIER),
        ("s004_canonac_g1_s103", D + "05:25:01", D + "05:41:00", "queue_screens.txt")]


def test_a_gap_counted_at_a_restart_keeps_its_hold():
    """A second cut makes cut_count write GAP and its status before any lock: the cut hold still counts."""
    log = CUT[:7] + [D + "06:30:02 GAP base_s103: second crash (a cut, counted as a crash)"]
    st = status(("base_s103", D + "06:30:02", "gap"))
    assert H.parse_holds("\n".join(log), st) == [("base_s103", D + "05:00:02", D + "05:00:04", H.SRC_CUT)]


def test_holds_add_up_once_each_and_a_repeat_or_overlap_is_refused(tmp_path):
    st = status(("base_s103", D + "06:41:29", "done and scored"), ("s007_smear_g1_s103", D + "05:21:59", "done"))
    p = tmp_path / "m.tsv"
    p.write_text(H.tsv_text("\n".join(CUT), st) + "".join(
        ln + "\n" for ln in H.tsv_text("\n".join(CRASH), st).splitlines() if ln.startswith("run\t")))
    runs, other = H.load(str(p))
    assert runs == pytest.approx({"base_s103": h("05:00:02", "05:00:04") + h("06:30:05", "06:41:29"),
                                  "s007_smear_g1_s103": h("05:00:02", "05:21:59")}) and set(other) == {"c1a_gate", "eager_smokes"}
    assert sum(1 for ln in p.read_text().splitlines() if ln.startswith("run\tbase_s103\t")) == 2
    assert H.SRC_CUT in p.read_text()
    t = H.tally(L.params(), {**H.load()[0], **runs})
    assert H.slot("base_s103")[1] not in t["registered"] and t["extra"]["base_s103"] == pytest.approx(runs["base_s103"])
    lines = p.read_text().splitlines()
    one = next(ln for ln in lines if ln.startswith("run\tbase_s103\t"))
    p.write_text("\n".join(lines + [one]) + "\n")
    with pytest.raises(AssertionError, match="measured twice"):
        H.load(str(p))
    a, b = one.split("\t")[3:5]
    p.write_text("\n".join(lines + [f"run\tbase_s103\t0.1\t{a}\t{D}06:00:00\tqueue_screens.txt"]) + "\n")
    with pytest.raises(AssertionError, match="two holds overlap"):
        H.load(str(p))


PC = os.environ.get("SCREENS_PC_COPY", "")


@pytest.mark.skipif(not os.path.isfile(os.path.join(PC, "queue_screens.txt")), reason="needs a copy of the PC queue files")
def test_the_real_queue_log_parses_as_before():
    """Set SCREENS_PC_COPY to a directory holding the PC's queue_screens.txt and status.jsonl."""
    q, st = (open(os.path.join(PC, f)).read() for f in ("queue_screens.txt", "status.jsonl"))
    old = []
    end = {r["run"]: r["time"].replace("T", " ") for r in map(json.loads, filter(str.strip, st.splitlines()))}
    held = None
    for ln in q.splitlines():                       # the parse as committed before SCREENS DEVIATION 1
        t, msg = ln[:19], ln[20:]
        held = t if msg.startswith("gpu.lock held") else held
        if msg.startswith("train ") and msg.split()[1][:-5] in end:
            old.append((msg.split()[1][:-5], held, end[msg.split()[1][:-5]]))
    assert H.parse_queue(q, st) == old and len(old) >= 81
    committed = [ln for ln in open(H.TSV) if ln.startswith("run\t")]
    assert [ln + "\n" for ln in H.tsv_text(q, st).splitlines() if ln.startswith("run\t")][:len(committed)] == committed


QT = os.path.join(HERE, "qtest_pc")


def test_the_pc_stub_queue_parses_to_its_holds(tmp_path):
    """tests/qtest_pc: queue_screens.sh with DEVIATION 1 run end to end on the PC (bash 5, flock; stub training and
    scoring, a scratch PLANCK_HOME, no GPU; 2026-10-10 17:33): base_s902 cut while training then resumed; base_s903 cut,
    then a crash: GAP; base_s905 cut twice: GAP; base_s906's NUL-filled score file scored again (no train line: its
    final existed). Each hold counts once, a cut hold ends at the cut queue's last line, the GAP that cut_count wrote
    before any lock keeps its holds."""
    q, st = (open(os.path.join(QT, f)).read() for f in ("queue_screens.txt", "status.jsonl"))
    got = [(r, a[11:], b[11:], {H.SRC_CUT: "cut", H.SRC_EARLIER: "earlier"}.get(s, "last")) for r, a, b, s in H.parse_holds(q, st)]
    assert got == [("base_s901", "17:33:53", "17:33:54", "last"), ("base_s902", "17:33:54", "17:33:54", "cut"),
                   ("base_s902", "17:33:56", "17:33:57", "last"), ("base_s903", "17:33:57", "17:33:57", "cut"),
                   ("base_s903", "17:33:59", "17:34:00", "last"), ("base_s904", "17:34:00", "17:34:01", "last"),
                   ("base_s905", "17:34:01", "17:34:01", "cut"), ("base_s905", "17:34:03", "17:34:03", "cut"),
                   ("base_s906", "17:34:05", "17:34:05", "last")]
    assert q.count("counted as its one crash; resuming once") == 3 and "GAP base_s903: second crash (rc 1)" in q
    assert "GAP base_s905: second crash (a cut, counted as a crash)" in q and q.count("queue start: ") == 5
    assert "base_s906/bpb/final_00007630.json is not whole JSON (a cut write): moved aside, scoring again" in q
    assert [json.loads(x)["outcome"] for x in st.splitlines()] == ["done and scored"] * 2 + ["gap", "done and scored", "gap",
                                                                                              "done and scored"]
    t = tmp_path / "m.tsv"
    t.write_text(H.tsv_text(q, st))
    runs, _ = H.load(str(t))
    assert sorted(runs) == [f"base_s90{i}" for i in range(1, 7)] and runs["base_s902"] == pytest.approx(1 / 3600)
