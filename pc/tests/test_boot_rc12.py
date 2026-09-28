"""boot.d/30_rc12.sh's code dir (rc12 notes STEP 9g VERIFY): a cut-off RC-12 queue resumes from the code dir its own
start line names (the queue has run from rc12_dg, the session queue rc12_dg3 and, for the Doge pass, rc12_doge2), not
from a fixed dir; RC12_CODE, when set, still overrides it (test_boot_resume.py's cases set it); a named dir without the
queue script is not resumed. Uses test_boot_resume.py's fakes (box fixture: fake queues under tmp_path)."""
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
