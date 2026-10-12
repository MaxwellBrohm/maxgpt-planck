"""boot.d/40_screens.sh, edge cases from the adversarial review of 2026-10-06 that test_boot_screens.py's fixture never
makes: lines a power cut glues together (NUL bytes and no newline, then the next writer's line), processes of the
other boot.d queues, and checkpoint and score files the main fixture keeps healthy or small. Uses its scr fixture."""
import json
import os
import subprocess
import time

import pytest

from test_boot_resume import box, put, started  # noqa: F401  (box is a pytest fixture)
from test_boot_screens import TRAIN, log, resumed, run40, scr  # noqa: F401  (scr is a pytest fixture)

OLD = ("2026-10-05 05:00:00 queue start: exp SCREENS, plan stage1_seeds.txt (sha256 0123456789abcdef), code commit "
       "93bc10b0000000000000000000000000000000aa\n")


def raw(s, data):
    os.makedirs(s["q"].parent, exist_ok=True)
    s["q"].write_bytes(data)


@pytest.mark.parametrize("end", ["queue stops at: train s005_forget_g0.5", "plan finished: stage1_seeds.txt"])
def test_an_older_end_glued_in_front_of_the_start_line_does_not_end_the_new_run(scr, end):
    # run A logged its end; a power cut kept part of that line, then NULs and no newline; a person started the queue
    # again by hand (its start line lands right after the NULs); that run was cut mid-training
    raw(scr, (OLD + f"2026-10-05 06:10:00 {end}").encode() + b"\0" * 200 + (scr["start"] + TRAIN).encode())
    r = run40(scr)
    assert "cut off" in r.stdout and resumed(scr), r.stdout


def test_a_refusal_glued_after_nuls_still_ends_the_run(scr):
    # the cut run's log, a NUL tail, then a relaunch by hand that refused before its own start line
    raw(scr, (scr["start"] + TRAIN + "2026-10-06 06:31:00 score s005_fo").encode() + b"\0" * 300
        + b"2026-10-06 07:00:00 refusing: uncommitted changes in the code checkout:\n M harness/x.py\n")
    r = run40(scr)
    assert "ended by itself" in r.stdout and started(scr) == "", r.stdout


def test_a_partial_older_start_glued_in_front_reads_the_newer_one(scr):
    # an older start line cut inside its commit, NULs, then the newer start: plan, sha256 and commit are the newer's
    raw(scr, OLD[:-30].encode() + b"\0" * 100 + (scr["start"] + TRAIN).encode())
    r = run40(scr)
    assert "cut off" in r.stdout and resumed(scr), r.stdout


@pytest.mark.parametrize("argv", [
    ["fq/python", "train.py", "QC/experiments/E2_lr_transfer/configs/5m_e12_r1_b125M.yaml", "--device", "cuda"],
    ["fq/python", "data_prep/bpb.py", "HOME/runs/E2/5m_e12_r1_b125M/final_00012208.pt", "/ev"],
    ["fq/queue_e2.sh", "--exp", "E2", "--code", "QC", "--plan", "QC/experiments/E2_lr_transfer/plans/a.txt"]])
def test_other_queues_running_do_not_hold_it_back(scr, argv):
    # boot_resume.sh runs 10_e2.sh (and 20, 30) first: SCREENS must still resume beside them
    log(scr, scr["start"] + TRAIN)
    argv = [a.replace("QC", str(scr["clone"])).replace("HOME", str(scr["home"])) for a in argv]
    put(str(scr["t"] / argv[0]), "#!/bin/bash\nsleep 8\n", 0o755)
    p = subprocess.Popen(["bash", str(scr["t"] / argv[0]), *argv[1:]], env=scr["env"])
    time.sleep(0.5)
    try:
        r = run40(scr, "--dry-run")
        assert "DRY RUN, would start" in r.stdout and "running (pids" not in r.stdout, r.stdout
    finally:
        p.kill()


def test_a_checkpoint_with_a_nul_head_and_a_whole_end_is_damage(scr):
    # zipfile.is_zipfile reads only the end record: lost first pages pass it, torch.load does not
    p = scr["done"] / "ckpt_00007344.pt"
    b = p.read_bytes()
    p.write_bytes(b"\0" * 64 + b[64:])
    log(scr, scr["start"] + TRAIN)
    r = run40(scr)
    assert r.returncode == 1 and "s005_forget_g0.5_s1/ckpt_00007344.pt" in r.stdout and started(scr) == "", r.stdout


def test_a_healthy_bpb_jsonl_of_many_rows_is_not_damage(scr):
    # bpb_lines.py writes one row per checkpoint x set x split (the main fixture has one row)
    rows = [{"ckpt": c, "step": 7630, "set": s, "split": "all", "bpb": 1.2}
            for c in ("final_00007630", "ckpt_00007497", "ckpt_00007344") for s in ("oasst2", "CHAT", "PROSE")]
    put(str(scr["done"] / "bpb.jsonl"), "".join(json.dumps(x) + "\n" for x in rows))
    log(scr, scr["start"] + TRAIN)
    r = run40(scr)
    assert "cut off" in r.stdout and resumed(scr), r.stdout


@pytest.mark.parametrize("end", ["", "2026-10-06 06:30:00 plan finished: stage2_select.txt\n"])
def test_a_start_line_the_cut_lost_is_left_for_a_person(scr, end):
    # the cut came 5-30 s after a later queue start: its code_sha256_<time>.txt (named just before its start line)
    # reached the disk, its start line did not (NULs). The start before it shows an end or not: neither is resumed.
    raw(scr, (scr["start"] + TRAIN + end).encode() + b"\0" * 400)
    for t in ("20261006_051804", "20261006_064000"):   # the visible start's own list, then the lost start's
        put(str(scr["out"] / f"code_sha256_{t}.txt"), "")
    r = run40(scr)
    assert r.returncode == 1 and "code_sha256_20261006_064000.txt is newer" in r.stdout and started(scr) == "", r.stdout


def test_each_start_lines_own_code_list_is_not_a_lost_start(scr):
    # the queue names its list in the same second as (or just before) its start line
    for t in ("20261005_210542", "20261006_051804"):
        put(str(scr["out"] / f"code_sha256_{t}.txt"), "")
    log(scr, OLD + "2026-10-05 22:00:00 plan finished: stage1_seeds.txt\n" + scr["start"] + TRAIN)
    r = run40(scr)
    assert "cut off" in r.stdout and resumed(scr), r.stdout
