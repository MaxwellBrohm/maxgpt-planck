"""boot.d/40_screens.sh: a cut-off SCREENS queue resumes on the plan and code clone its own last start line names,
only when no stop file applies, nothing of SCREENS runs, no queue exit follows that line and no run dir holds power-cut
damage the queue cannot repair. Uses test_boot_resume.py's box fixture (fakes under tmp_path, pgrep limited to them);
the clone is a real git repo whose queue_screens.sh is a fake that records its arguments."""
import os
import re
import shutil
import subprocess
import time
import zipfile

import pytest

from test_boot_resume import box, dispatch, fake, put, started  # noqa: F401  (box is a pytest fixture)

PLAN = "stage2_select.txt"
TRAIN = "2026-10-06 06:27:41 train s005_forget_g1_s1.yaml --require-committed\n"
DEV = "/Library/Developer/CommandLineTools"     # the Mac's git needs it (Xcode license); absent on the PC


def git(env, d, *a):
    return subprocess.run(["git", "-C", str(d), *a], env=env, check=True, capture_output=True, text=True).stdout.strip()


def clone(env, qcode, tag, plan_text):   # a git repo with a fake queue_screens.sh, renamed to qcode/<commit12>
    d = qcode / f"new_{tag}"
    fake(str(d / "experiments" / "screens" / "queue_screens.sh"), tag)
    put(str(d / "experiments" / "screens" / "plans" / PLAN), plan_text)
    put(str(d / "experiments" / "screens" / "plans" / "CURRENT"), "other_plan\n")
    put(str(d / "experiments" / "screens" / "plans" / "other_plan.txt"), "train x\n")
    git(env, d, "init", "-q")
    git(env, d, "add", "-A")
    git(env, d, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", tag)
    ref = git(env, d, "rev-parse", "HEAD")
    return ref, d


def whole_pt(path):   # a zip like torch.save's: local header first, end record last
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("archive/data.pkl", b"x" * 2000)


@pytest.fixture
def scr(box):
    if os.path.isdir(DEV):
        box["env"]["DEVELOPER_DIR"] = DEV
    env, home = box["env"], box["home"]
    ref, d = clone(env, home / "qcode", "screens", "wait_mark SCREENS X\ntrain s005_forget_g1_s1\n")
    c = home / "qcode" / ref[:12]
    os.rename(d, c)
    _, decoy = clone(env, home / "qcode", "WRONG", "train y\n")   # another commit's clone, sorted last: never used
    os.rename(decoy, home / "qcode" / "ffffffffffff")
    psha = subprocess.run(["sha256sum", str(c / "experiments" / "screens" / "plans" / PLAN)], capture_output=True,
                          text=True, check=True).stdout[:16]
    out = home / "runs" / "SCREENS"
    done, cut, trunk = out / "s005_forget_g0.5_s1", out / "s005_forget_g1_s1", out / "s003_adamw_e3_r1_trunk"
    for f in ("final_00007630.pt", "ckpt_00007344.pt", "ckpt_00007497.pt"):   # a finished, scored run
        whole_pt(str(done / f))
    put(str(done / "bpb" / "final_00007630.json"), '{"step": 7630}\n')
    put(str(done / "bpb.jsonl"), '{"run": "s005_forget_g0.5_s1", "ckpt": "final_00007630"}\n')
    put(str(done / "SCORED"), "")
    put(str(done / "crashed_0929" / "final_00000001.pt"), "")       # a person's moved-aside files: not read
    whole_pt(str(trunk / "final_00012208.pt"))                      # a finished trunk
    put(str(trunk / "TRUNK_DONE"), "")
    whole_pt(str(cut / "ckpt_00001000.pt"))                         # the run the power cut hit
    put(str(cut / "latest.json"), '{"path": "ckpt_00001000.pt", "step": 1000}')
    put(str(cut / "ckpt_00001500.pt.tmp"), "")                      # a save the cut stopped: never published
    put(str(cut / "bpb" / "ckpt_00001000.json"), "")                # 0 bytes, unscored: the queue re-scores it
    os.makedirs(home / "logs", exist_ok=True)
    start = (f"2026-10-06 05:18:04 queue start: exp SCREENS, plan {PLAN} (sha256 {psha}), code commit {ref}\n")
    return {**box, "ref": ref, "clone": c, "start": start, "out": out, "q": out / "queue_screens.txt",
            "done": done, "cut": cut, "trunk": trunk}


def log(s, text):
    put(str(s["q"]), text)
    return s["q"]


def run40(s, *args):
    r = subprocess.run(["bash", str(s["home"] / "boot.d" / "40_screens.sh"), *args], env=s["env"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode in (0, 1), r.stdout + r.stderr
    return r


def resumed(s):
    x = s["clone"] / "experiments" / "screens"
    return started(s).startswith(f"screens --code {s['clone']} --plan {x}/plans/{PLAN} ")


def test_cut_off_queue_resumes_its_own_plan_and_clone(scr):
    q = log(scr, scr["start"] + TRAIN)
    before = q.read_text()
    r = run40(scr, "--dry-run")
    assert "DRY RUN, would start" in r.stdout and started(scr) == "" and q.read_text() == before, r.stdout
    r = run40(scr)
    assert r.returncode == 0 and "running after" in r.stdout and resumed(scr), r.stdout + started(scr)
    lines = q.read_text().splitlines()
    assert lines[:-1] == before.splitlines() and " boot resume: " in lines[-1], lines


ENDS = ["plan finished: stage2_select.txt", "plan says stop", "queue stops at: train s005_forget_g1_s1",
        "STOP while waiting for SCREENS: SCREENS STAGE 2 SMOKES RECORDED", "bad plan line: frob x",
        "refusing: uncommitted changes in the code checkout:", "refusing: /x is not a git checkout"]


@pytest.mark.parametrize("end", ENDS)
def test_every_queue_exit_after_the_start_is_left_alone(scr, end):
    log(scr, scr["start"] + TRAIN + f"2026-10-06 06:30:00 {end}\n")
    r = run40(scr)
    assert "ended by itself" in r.stdout and started(scr) == "", r.stdout


@pytest.mark.parametrize("line", ["2026-10-06 06:00:00 MARK SCREENS plan finished: half way",
                                  "[preflight] refusing: data path missing"])
def test_end_words_not_at_a_queue_time_stamp_do_not_end_the_run(scr, line):
    log(scr, scr["start"] + line + "\n" + TRAIN)
    assert "cut off" in run40(scr).stdout and resumed(scr)


def test_only_the_last_start_counts(scr):
    fin = "2026-10-06 05:00:00 plan finished: stage1_seeds.txt\n"
    log(scr, scr["start"] + fin + scr["start"] + TRAIN)
    assert "cut off" in run40(scr).stdout and resumed(scr)


def test_a_run_that_ended_after_an_earlier_cut_is_left_alone(scr):
    log(scr, scr["start"] + TRAIN + scr["start"] + "2026-10-06 09:00:00 plan finished: stage2_select.txt\n")
    assert "ended by itself" in run40(scr).stdout and started(scr) == ""


def test_nul_tailed_log_resumes_and_the_note_starts_a_clean_line(scr):
    # 2026-09-29: a power cut left NUL bytes in a queue log, and the next start line was appended after them. grep
    # calls such a file binary unless -a. NULs this near the start of the file bite BSD grep (the Mac) as well as GNU
    # grep (the PC); test_boot_resume.py's NULs sit only at the tail, which BSD grep never inspects.
    q = scr["q"]
    os.makedirs(q.parent, exist_ok=True)
    with open(q, "wb") as f:
        f.write(b"\0" * 200 + scr["start"].encode() + TRAIN.encode() + b"\0" * 300)
    r = run40(scr)
    assert "cut off" in r.stdout and resumed(scr), r.stdout + r.stderr
    last = q.read_bytes().split(b"\n")[-2]
    assert re.match(rb"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d boot resume: ", last), last[:40]


@pytest.mark.parametrize("stop", ["boot.d/STOP", "boot.d/screens.STOP", "STOP", "PAUSE", "runs/SCREENS/NO_BOOT_RESUME"])
def test_stop_files_hold_it(scr, stop):
    log(scr, scr["start"] + TRAIN)
    put(str(scr["home"] / stop), "")
    assert "present: not resuming" in run40(scr).stdout and started(scr) == ""


@pytest.mark.parametrize("argv", [
    ["fq/queue_screens.sh", "--code", "C", "--plan", "P"],
    ["fq/train.py", "QC/experiments/S005_forget_gate/configs/s005_forget_g1_s1.yaml", "--device", "cuda"],
    ["fq/bpb.py", "HOME/runs/SCREENS/s005_forget_g1_s1/final_00007630.pt", "/ev"],
    ["fq/preflight.py", "QC/experiments/screens/configs/base_s101.yaml", "--code", "C"],
    ["fq/screens.py", "check", "s005_forget_g1_s1"],
    ["fq/launch_screens.sh"]])
def test_nothing_starts_while_any_part_of_screens_runs(scr, argv):
    log(scr, scr["start"] + TRAIN)
    argv = [a.replace("QC", str(scr["clone"])).replace("HOME", str(scr["home"])) for a in argv]
    put(str(scr["t"] / argv[0]), "#!/bin/bash\nsleep 5\n", 0o755)
    p = subprocess.Popen(["bash", str(scr["t"] / argv[0]), *argv[1:]], env=scr["env"])
    time.sleep(0.5)
    try:
        assert "running (pids" in run40(scr).stdout and started(scr) == ""
    finally:
        p.kill()


@pytest.mark.parametrize("case,why", [("no_clone", "no clone of commit"), ("other_commit", "the cut-off run used"),
                                      ("no_plan", "no plan stage2_select.txt"), ("plan_changed", "plan had")])
def test_missing_or_changed_clone_or_plan_is_refused(scr, case, why):
    log(scr, scr["start"] + TRAIN)
    x = scr["clone"] / "experiments" / "screens"
    if case == "no_clone":
        shutil.rmtree(scr["clone"])
    elif case == "other_commit":
        put(str(x / "plans" / "other_plan.txt"), "train z\n")
        git(scr["env"], scr["clone"], "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "moved")
    elif case == "no_plan":
        os.remove(x / "plans" / PLAN)
    else:
        with open(x / "plans" / PLAN, "a") as f:
            f.write("train extra\n")
    r = run40(scr)
    assert r.returncode == 1 and "cannot resume" in r.stdout and why in r.stdout and started(scr) == "", r.stdout


def test_no_log_or_no_start_line_resumes_nothing(scr):
    assert "SCREENS never ran here" in run40(scr).stdout and started(scr) == ""
    log(scr, TRAIN)
    assert "no queue start line" in run40(scr).stdout and started(scr) == ""


def truncate(p, keep):
    b = open(p, "rb").read()
    open(p, "wb").write(b[:keep])


DAMAGE = {   # name: (what the power cut left, the file the refusal must name)
    "latest_0_bytes": (lambda s: truncate(s["cut"] / "ckpt_00001000.pt", 0), "s005_forget_g1_s1/ckpt_00001000.pt"),
    "final_nul": (lambda s: open(s["done"] / "final_00007630.pt", "wb").write(b"\0" * 900),
                  "s005_forget_g0.5_s1/final_00007630.pt"),
    "ckpt_tail_lost": (lambda s: truncate(s["done"] / "ckpt_00007497.pt", 1000), "s005_forget_g0.5_s1/ckpt_00007497.pt"),
    "latest_names_missing": (lambda s: put(str(s["cut"] / "latest.json"), '{"path": "ckpt_00002000.pt"}'),
                             "s005_forget_g1_s1/latest.json"),
    "latest_nul": (lambda s: open(s["cut"] / "latest.json", "wb").write(b"\0" * 40), "s005_forget_g1_s1/latest.json"),
    "scored_empty": (lambda s: (truncate(s["done"] / "bpb.jsonl", 0), truncate(s["done"] / "bpb" / "final_00007630.json", 0)),
                     "s005_forget_g0.5_s1 is SCORED"),
    "scored_json_0_bytes": (lambda s: truncate(s["done"] / "bpb" / "final_00007630.json", 0),
                            "s005_forget_g0.5_s1/bpb/final_00007630.json"),
    "bpb_nul": (lambda s: open(s["cut"] / "bpb" / "ckpt_00001000.json", "wb").write(b"\0" * 300),
                "s005_forget_g1_s1/bpb/ckpt_00001000.json"),
    "trunk_no_final": (lambda s: os.remove(s["trunk"] / "final_00012208.pt"), "s003_adamw_e3_r1_trunk is TRUNK_DONE"),
}


@pytest.mark.parametrize("name", sorted(DAMAGE))
def test_power_cut_damage_is_left_for_a_person(scr, name):
    log(scr, scr["start"] + TRAIN)
    hurt, where = DAMAGE[name]
    hurt(scr)
    r = run40(scr)
    assert r.returncode == 1 and "power-cut damage" in r.stdout and where in r.stdout and started(scr) == "", r.stdout


def test_a_damage_check_that_cannot_run_resumes_nothing(scr):
    log(scr, scr["start"] + TRAIN)
    scr["env"]["BOOT_PY"] = str(scr["t"] / "no_python")
    r = run40(scr)
    assert r.returncode == 1 and "damage check did not run" in r.stdout and started(scr) == "", r.stdout


def test_dispatcher_runs_it_with_the_others(scr):
    log(scr, scr["start"] + TRAIN)
    r = dispatch(scr)
    assert r.returncode == 0 and "screens: running after" in r.stdout, r.stdout
    assert any(x.startswith("screens --code") for x in started(scr).splitlines()), started(scr)
