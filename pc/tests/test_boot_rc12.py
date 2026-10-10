"""boot.d/30_rc12.sh's code dir (rc12 notes STEP 9g VERIFY): a cut-off RC-12 queue resumes from the code dir its own
start line names (the queue has run from rc12_dg, the session queue rc12_dg3 and, for the Doge pass, rc12_doge2), not
from a fixed dir; RC12_CODE, when set, still overrides it (test_boot_resume.py's cases set it); a named dir without the
queue script is not resumed. Uses test_boot_resume.py's fakes (box fixture: fake queues under tmp_path)."""
import os

import pytest

from test_boot_resume import box, fake, put, script, started  # noqa: F401  (box is a pytest fixture)


def rc12_log(b, *codes):   # one start line per code dir (the last one cut off); the older runs ended on a STOP file
    lines = []
    for i, c in enumerate(codes):
        lines.append(f"2026-09-28T0{i}:00:00 queue: start, code {c}, root /r, pid {100 + i}, "
                     "gpu.lock sessions of 1800 s (tail 300 s)")
        lines.append(f"2026-09-28T0{i}:30:00 queue: " + ("STOP file, exiting" if i < len(codes) - 1 else "waiting"))
    put(str(b["home"] / "logs" / "rc12_dev_queue.log"), "\n".join(lines) + "\n")
    b["env"].pop("RC12_CODE")


def test_rc12_resumes_from_the_cut_off_runs_code(box):
    old, new = box["t"] / "rc12_dg", box["t"] / "rc12_dg3"
    fake(str(old / "queue_dev_baselines.sh"), "old")
    fake(str(new / "queue_dev_baselines.sh"), "new")
    rc12_log(box, old, new)
    out = script(box, "30_rc12.sh")
    assert "running after" in out and started(box).startswith(f"new  Q_CODE={new}"), out + started(box)


def test_rc12_cut_off_runs_code_missing_is_not_resumed(box):
    fake(str(box["t"] / "rc12_dg" / "queue_dev_baselines.sh"), "old")
    rc12_log(box, box["t"] / "rc12_dg", box["t"] / "gone")
    out = script(box, "30_rc12.sh")
    assert f"code dir [{box['t']}/gone]: cannot resume" in out and started(box) == "", out


# rc12/notes.txt STEP 14X: the run file. A queue since STEP 14X ends its start line with ", run ID" and writes
# logs/rc12_dev_queue.run (run=ID and its Q_ settings) first; the restart carries those settings.
KEYS = ["Q_MODELS", "Q_RENDERS", "Q_SEEDS", "Q_ROOT", "Q_ENGINES", "Q_TIMEOUT", "Q_SESSION_S", "Q_SESSION_TAIL",
        "Q_LIMIT"]
RUNFILE = {"Q_MODELS": "sz14/cRia-LM-75M-Instruct", "Q_RENDERS": "template plain", "Q_SEEDS": "greedy 1 2 3",
           "Q_ROOT": "/r/rc12_dev", "Q_ENGINES": "/c/rc12_cuts/engines.json", "Q_TIMEOUT": "1020",
           "Q_SESSION_S": "1500", "Q_SESSION_TAIL": "600", "Q_LIMIT": ""}


def env_fake(path):   # a fake queue that records the Q_ settings it was started with
    put(path, '#!/bin/bash\necho "rc12 Q_CODE=$Q_CODE' + "".join(f"|{k}=${{{k}-UNSET}}" for k in KEYS)
        + '" >> "$PLANCK_HOME/started"\nsleep 4\n', 0o755)


def run_case(b, rid="20261010T180000.4242", run=None, lines=None):
    code = b["t"] / "rc12_cuts"
    env_fake(str(code / "queue_dev_baselines.sh"))
    start = (f"2026-10-10T18:00:00 queue: start, code {code}, root /r/rc12_dev, pid 4242, gpu.lock sessions of 1500 s "
             f"(tail 600 s)" + (f", run {rid}" if rid else ""))
    put(str(b["home"] / "logs" / "rc12_dev_queue.log"), start + "\n2026-10-10T18:05:00 queue: waiting\n")
    run = RUNFILE if run is None else run
    text = lines if lines is not None else ["run=20261010T180000.4242", "pid=4242", f"code={code}"] + \
        [f"{k}={v}" for k, v in run.items()]
    put(str(b["home"] / "logs" / "rc12_dev_queue.run"), "\n".join(text) + "\n")
    b["env"].pop("RC12_CODE")
    return code


def test_rc12_restart_carries_the_run_files_settings(box):
    code = run_case(box)
    out = script(box, "30_rc12.sh")
    want = f"rc12 Q_CODE={code}" + "".join(f"|{k}={v}" for k, v in RUNFILE.items())
    assert "running after" in out and started(box).strip() == want, out + started(box)


def test_rc12_run_file_with_a_nul_tail_still_resumes(box):
    # a power cut can leave NUL bytes after the last line; GNU grep (the PC) then calls the file binary unless -a
    run_case(box, lines=["run=20261010T180000.4242", "pid=4242", "code=x"]
                    + [f"{k}={v}" for k, v in RUNFILE.items()] + ["\0" * 64])
    out = script(box, "30_rc12.sh")
    assert started(box).strip().endswith("|Q_SESSION_TAIL=600|Q_LIMIT="), out + started(box)


def test_rc12_older_start_line_without_a_run_id_uses_the_defaults(box):
    run_case(box, rid=None)
    out = script(box, "30_rc12.sh")
    assert "names no run" in out and started(box).count("=UNSET") == len(KEYS), out + started(box)


@pytest.mark.parametrize("why,kw,msg", [
    ("other run", dict(rid="20261010T190000.99"), "no run file of that run"),
    ("missing key", dict(run={k: v for k, v in RUNFILE.items() if k != "Q_TIMEOUT"}), "no Q_TIMEOUT line"),
    ("empty models", dict(run=dict(RUNFILE, Q_MODELS="")), "Q_MODELS is empty"),
    ("odd text", dict(run=dict(RUNFILE, Q_MODELS="a;touch x")), "unexpected text"),
    ("no run file", dict(lines=[]), "no run file of that run")])
def test_rc12_bad_run_file_is_not_resumed(box, why, kw, msg):
    run_case(box, **kw)
    if why == "no run file":
        os.remove(box["home"] / "logs" / "rc12_dev_queue.run")
    out = script(box, "30_rc12.sh")
    assert msg in out and "cannot resume" in out and started(box) == "", why + ": " + out
