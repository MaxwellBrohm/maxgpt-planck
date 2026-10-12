"""SCREENS DEVIATION 1 (SCREENS.txt, 2026-10-10): the queue's score files are durable before SCORED, a bpb/*.json a
power cut left unparsable is scored again, and a queue cut while a run trains counts as that run's one crash. The
bash functions (qlib_screens.sh, bash 3.2 here) run with a stub bpb.py; score_sync.py's publish order is recorded.
No model, no GPU.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_queue_durable.py
"""
import glob, json, os, re, subprocess, sys  # noqa: E401

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(SX))
sys.path.insert(0, SX)
import score_sync  # noqa: E402

QLIB, QUEUE = os.path.join(SX, "qlib_screens.sh"), os.path.join(SX, "queue_screens.sh")
E2LIB = os.path.join(ROOT, "experiments", "E2_lr_transfer", "queue_lib.sh")
STUB = '''import json, os, sys
a = sys.argv[1:]; out = a[a.index("--out") + 1]; ck = os.path.basename(a[0])[:-3]
open(os.environ["STUB_LOG"], "a").write(ck + "\\n")
if os.environ.get("STUB_FAIL") == ck:
    sys.exit(3)
sets = {s: {"bits": 100.0 * (i + 1), "bytes": 80 * (i + 1), "tokens": 30, "windows": 2, "bpb": 1.25, "truncated": 0}
        for i, s in enumerate(("oasst2", "cccc", "gutenberg", "wikimedia"))}
sets["oasst2"]["by_role"] = {"assistant": {"bits": 60.0, "bytes": 50, "tokens": 20, "windows": 2, "bpb": 1.2}}
json.dump({"checkpoint": ck + ".pt", "step": int(ck.split("_")[1]), "tokens_trained": 1, "evalset_sha256": "e" * 64,
           "tokenizer": {"sha256": "t" * 64}, "precision": "fp32", "max_windows": None, "sets": sets}, open(out, "w"), indent=1)
'''


def bash(tmp, body, env=None):
    """Run body after sourcing E2's queue_lib.sh and qlib_screens.sh, as queue_screens.sh does (CODE = a stub tree)."""
    code = tmp / "code"
    (code / "data_prep").mkdir(parents=True, exist_ok=True)
    (code / "data_prep" / "bpb.py").write_text(STUB)
    script = f'CODE="{code}"; . "{E2LIB}"; . "{QLIB}"; SX="{SX}"\n{body}\n'
    e = {**os.environ, "PLANCK_HOME": str(tmp / "ph"), "PLANCK_PY": sys.executable, "STUB_LOG": str(tmp / "calls.txt"),
         **(env or {})}
    return subprocess.run(["/bin/bash", "-c", script], capture_output=True, text=True, env=e)


def run_dir(tmp, ckpts=("final_00007630", "ckpt_00007497", "ckpt_00007344", "ckpt_00007191")):
    out = tmp / "runs" / "s007_smear_g1_s101"
    (out / "bpb").mkdir(parents=True)
    for c in ckpts:
        (out / f"{c}.pt").write_bytes(b"weights")
    return out


def calls(tmp):
    p = tmp / "calls.txt"
    return sorted(p.read_text().split()) if p.exists() else []


def lines_by_e2(out):
    return subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "E2_lr_transfer", "bpb_lines.py"), str(out)],
                          capture_output=True, text=True, check=True).stdout


def test_scores_final_and_two_rolling_then_publishes(tmp_path):
    out = run_dir(tmp_path)
    r = bash(tmp_path, f'score_screens "{out}"; echo rc $?')
    assert r.stdout.strip().endswith("rc 0"), r.stdout + r.stderr
    assert calls(tmp_path) == ["ckpt_00007344", "ckpt_00007497", "final_00007630"]       # never ckpt_00007191
    assert (out / "SCORED").exists() and (out / "bpb.jsonl").read_text() == lines_by_e2(out) != ""
    assert sorted(os.listdir(out / "bpb")) == ["ckpt_00007344.json", "ckpt_00007497.json", "final_00007630.json"]


def test_a_whole_score_file_is_kept(tmp_path):
    out = run_dir(tmp_path)
    assert bash(tmp_path, f'score_screens "{out}"').returncode == 0
    (tmp_path / "calls.txt").unlink()
    (out / "SCORED").unlink()
    assert bash(tmp_path, f'score_screens "{out}"').returncode == 0 and calls(tmp_path) == [] and (out / "SCORED").exists()


@pytest.mark.parametrize("damage", ["nul", "lost_tail", "array"])
def test_an_unparsable_score_file_is_moved_aside_and_scored_again(tmp_path, damage):
    out = run_dir(tmp_path)
    assert bash(tmp_path, f'score_screens "{out}"').returncode == 0
    f = out / "bpb" / "final_00007630.json"
    good = f.read_text()
    f.write_bytes({"nul": b"\0" * 4096, "lost_tail": good[: len(good) // 2].encode(), "array": b"[1, 2]"}[damage])
    (out / "SCORED").unlink()
    (out / "bpb.jsonl").unlink()
    (tmp_path / "calls.txt").unlink()
    r = bash(tmp_path, f'score_screens "{out}"')
    assert r.returncode == 0 and calls(tmp_path) == ["final_00007630"], r.stdout + r.stderr
    assert f.read_text() == good and (out / "bpb.jsonl").read_text() == lines_by_e2(out) and (out / "SCORED").exists()
    aside = glob.glob(str(out / "bpb" / "final_00007630.json.unparsable_*"))
    assert len(aside) == 1 and re.search(r"unparsable_\d{8}T\d{6}$", aside[0])
    assert "is not whole JSON (a cut write): moved aside, scoring again" in r.stdout


def test_a_failed_scoring_writes_no_scored(tmp_path):
    out = run_dir(tmp_path)
    r = bash(tmp_path, f'score_screens "{out}"; echo rc $?', {"STUB_FAIL": "ckpt_00007497"})
    assert r.stdout.strip().endswith("rc 1") and "scoring failed for ckpt_00007497" in r.stdout
    assert not (out / "SCORED").exists() and not (out / "bpb.jsonl").exists()


def test_publish_refuses_a_score_file_that_is_not_whole(tmp_path):
    out = run_dir(tmp_path)
    assert bash(tmp_path, f'score_screens "{out}"').returncode == 0
    for x in ("SCORED", "bpb.jsonl"):
        (out / x).unlink()
    (out / "bpb" / "ckpt_00007344.json").write_bytes(b'{"checkpoint": ')
    assert score_sync.main(["publish", str(out)]) == 1 and not (out / "SCORED").exists() and not (out / "bpb.jsonl").exists()
    assert score_sync.main(["publish", str(tmp_path)]) == 1                        # no bpb/*.json at all
    assert score_sync.main(["whole"]) == 2 and score_sync.main(["nonsense", "x"]) == 2


def test_publish_fsyncs_the_score_files_then_bpb_jsonl_then_scored(tmp_path, monkeypatch):
    out = run_dir(tmp_path)
    assert bash(tmp_path, f'score_screens "{out}"').returncode == 0
    for x in ("SCORED", "bpb.jsonl"):
        (out / x).unlink()
    seen, real_replace = [], score_sync.durable.replace
    monkeypatch.setattr(score_sync.durable, "fsync_file", lambda p: seen.append(("fsync", os.path.basename(p))))
    monkeypatch.setattr(score_sync.durable, "fsync_dir", lambda p: seen.append(("fsync_dir", os.path.basename(p))))
    monkeypatch.setattr(score_sync.durable, "replace", lambda t, p: (seen.append(("rename", os.path.basename(p))),
                                                                     real_replace(t, p)))
    assert score_sync.main(["publish", str(out)]) == 0
    names = [x for x in seen if x[0] != "fsync_dir" or x[1] == "bpb"]
    assert names == [("fsync", "ckpt_00007344.json"), ("fsync", "ckpt_00007497.json"), ("fsync", "final_00007630.json"),
                     ("fsync_dir", "bpb"), ("rename", "bpb.jsonl"), ("rename", "SCORED")]
    assert (out / "bpb.jsonl").read_text() == lines_by_e2(out) and (out / "SCORED").read_text() == ""


def test_cut_count(tmp_path):
    out = run_dir(tmp_path, ("ckpt_00003060",))
    rc = lambda: bash(tmp_path, f'cut_count "{out}"; echo rc $?').stdout.strip().splitlines()[-1]  # noqa: E731
    assert rc() == "rc 0" and not (out / "crashes.txt").exists()                     # no QRUN: nothing
    assert bash(tmp_path, f'qrun_set "{out}"').returncode == 0
    assert re.match(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d queue pid \d+ inside train_run\n$", (out / "QRUN").read_text())
    assert rc() == "rc 1" and not (out / "QRUN").exists() and not (out / "GAP").exists()   # the cut: crash 1, resume
    c = (out / "crashes.txt").read_text().splitlines()
    assert len(c) == 1 and "cut (power cut, reboot or kill while training; QRUN: " in c[0] and c[0].endswith("as a crash")
    assert bash(tmp_path, f'qrun_set "{out}"; qrun_clear "{out}"').returncode == 0 and not (out / "QRUN").exists()
    bash(tmp_path, f'qrun_set "{out}"')                                               # a second cut: GAP
    r = bash(tmp_path, f'cut_count "{out}"; echo rc $?')
    assert r.stdout.strip().endswith("rc 2") and "GAP s007_smear_g1_s101: second crash" in r.stdout
    assert (out / "GAP").read_text().startswith("second crash") and len((out / "crashes.txt").read_text().splitlines()) == 2


def test_a_crash_the_queue_saw_then_a_cut_is_a_gap(tmp_path):
    out = run_dir(tmp_path, ("ckpt_00003060",))
    (out / "crashes.txt").write_text("2026-10-10 06:01:00 rc 1\n")                     # train_run's own line
    bash(tmp_path, f'qrun_set "{out}"')
    assert bash(tmp_path, f'cut_count "{out}"; echo rc $?').stdout.strip().endswith("rc 2") and (out / "GAP").exists()


def test_a_cut_after_the_final_checkpoint_is_not_a_crash(tmp_path):
    out = run_dir(tmp_path)
    bash(tmp_path, f'qrun_set "{out}"')
    r = bash(tmp_path, f'cut_count "{out}"; echo rc $?')
    assert r.stdout.strip().endswith("rc 0") and "final checkpoint saved: not a crash" in r.stdout
    assert not (out / "QRUN").exists() and not (out / "crashes.txt").exists() and not (out / "GAP").exists()


def test_queue_screens_wiring():
    """do_train: the cut is counted after the skip check and before the screens check; QRUN brackets train_run
    exactly; scoring is score_screens, which writes SCORED itself (no bare touch of SCORED anywhere)."""
    q = open(QUEUE).read()
    assert subprocess.run(["/bin/bash", "-n", QUEUE]).returncode == 0
    body = q[q.index("do_train() {"):q.index("\nmapfile")]
    order = [body.index(x) for x in ("qlog \"skip $name", "cut_count \"$out\"\n    [ $? = 2 ] && { status \"$name\" gap; return 0; }",
                                     "screens.py check \"$name\"", "qrun_set \"$out\" ||",
                                     "train_run \"$cfg\" \"$out\" --require-committed; rc=$?\n        qrun_clear \"$out\"\n",
                                     "elif score_screens \"$out\"; then status \"$name\" \"done and scored\"")]
    assert order == sorted(order) and body.count("train_run ") == 1 and body.count("qrun_clear") == 1
    assert "touch \"$out/SCORED\"" not in q and "score_run " not in body and ". \"$CODE/experiments/screens/qlib_screens.sh\"" in q
