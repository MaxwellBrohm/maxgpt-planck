"""queue_e2.sh / queue_lib.sh behaviour with a fake python (fake_py.py) and a fake nvidia-smi: no model, no GPU.
Needs bash >= 4 and flock (the PC's WSL; the Mac has neither).
  ~/planck/venv/bin/python -m pytest -q experiments/E2_lr_transfer/tests/test_queue.py
"""
import fcntl
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
E2 = os.path.dirname(HERE)
sys.path.insert(0, E2)
import e2plan  # noqa: E402

SMI = """#!/bin/bash
case "$*" in
  *temperature*) f=$FAKE_TEMPS
    if [ -s "$f" ]; then head -1 "$f"; [ "$(wc -l < "$f")" -gt 1 ] && sed -i 1d "$f"; else echo 50; fi ;;
  *memory.used*) echo "${FAKE_MEM:-1000}" ;;
esac
"""


@pytest.fixture
def ctx(tmp_path):
    home, code = tmp_path / "home", tmp_path / "code"
    xd = code / "experiments" / "E2_lr_transfer"
    for d in (home / "locks", home / "runs", xd / "configs", xd / "plans", code / "harness", code / "data_prep",
              code / "tokenizer" / "v0"):
        d.mkdir(parents=True)
    for f in ("queue_e2.sh", "queue_lib.sh", "bpb_lines.py", "preflight.py", "e2plan.py"):
        shutil.copy(os.path.join(E2, f), xd / f)
    (code / "harness" / "train.py").write_text("# placeholder\n")
    (code / "data_prep" / "bpb.py").write_text("# placeholder\n")
    (code / "tokenizer" / "v0" / "tok_v0_8k.json").write_text("{}\n")
    (xd / "configs" / "prereg.yaml").write_text("id: T\n")
    names = []
    for name, txt in e2plan.runs_for("5m", 3e-3, 1.0):
        (xd / "configs" / f"{name}.yaml").write_text(txt)
        names.append(name)
    (xd / "plans" / "test.txt").write_text("".join(f"train {n}\n" for n in names))
    g = ["git", "-C", str(code)]
    subprocess.run(g + ["init", "-q"], check=True)
    subprocess.run(g + ["add", "-A"], check=True)
    subprocess.run(g + ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "t"], check=True)
    smi = tmp_path / "smi.sh"
    smi.write_text(SMI)
    smi.chmod(0o755)
    fake = tmp_path / "fake_py.py"
    shutil.copy(os.path.join(HERE, "fake_py.py"), fake)
    fake.chmod(0o755)
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "PLANCK_HOME": str(home), "PLANCK_PY": str(fake),
           "PLANCK_NVIDIA_SMI": str(smi), "QUEUE_POLL_S": "0.05", "QUEUE_WAIT_S": "0.1", "QUEUE_MIN_FREE_GB": "0",
           "FAKE_LOG": str(tmp_path / "calls.jsonl"), "FAKE_STATE": str(tmp_path / "state.json"),
           "FAKE_TEMPS": str(tmp_path / "temps.txt")}
    return {"home": home, "code": code, "xd": xd, "env": env, "names": names, "tmp": tmp_path,
            "out": home / "runs" / "E2"}


def commit(c):
    g = ["git", "-C", str(c["code"])]
    subprocess.run(g + ["add", "-A", "experiments"], check=True)
    subprocess.run(g + ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "t2"], check=True)


def run(c, plan="test.txt", fake=None, refuse="", temps=None, mem=None, during=None, timeout=90, **kw):
    env = dict(c["env"], FAKE_PLAN=json.dumps(fake or {}), FAKE_REFUSE=refuse, **kw)
    if mem is not None:
        env["FAKE_MEM"] = str(mem)
    if temps:
        (c["tmp"] / "temps.txt").write_text("".join(f"{t}\n" for t in temps))
    th = threading.Thread(target=during) if during else None
    if th:
        th.start()
    p = subprocess.run(["bash", str(c["xd"] / "queue_e2.sh"), "--code", str(c["code"]), "--plan",
                        str(c["xd"] / "plans" / plan)], env=env, capture_output=True, text=True, timeout=timeout)
    if th:
        th.join()
    return p.returncode


def calls(c, tool):
    p = c["tmp"] / "calls.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
    return [r for r in rows if r["tool"] == tool]


def trained(c):
    return [os.path.basename(r["args"][0])[:-5] for r in calls(c, "train.py")]


def outcomes(c):
    p = c["out"] / "status.jsonl"
    return {json.loads(x)["run"]: json.loads(x)["outcome"] for x in p.read_text().splitlines()} if p.exists() else {}


def qlog(c):
    return (c["out"] / "queue_e2.txt").read_text()


def lock_free(c):
    with open(c["home"] / "locks" / "gpu.lock", "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False


def test_arm_trains_in_order_scores_branches_and_frees_the_lock(ctx):
    trunk, *branches = ctx["names"]
    assert run(ctx) == 0, qlog(ctx)
    assert trained(ctx) == ctx["names"]
    for r in calls(ctx, "train.py"):
        assert r["args"][1:] == ["--device", "cuda", "--require-committed"] and r["cwd"].endswith("/harness")
    assert calls(ctx, "preflight.py") and all("--strict" in r["args"] for r in calls(ctx, "preflight.py"))
    assert all(r["gpu_lock_held"] for r in calls(ctx, "train.py") + calls(ctx, "bpb.py"))
    assert len(calls(ctx, "bpb.py")) == 3 * len(branches)
    assert outcomes(ctx) == {trunk: "done", **{b: "done and scored" for b in branches}}
    assert not (ctx["out"] / trunk / "bpb").exists()
    for b in branches:
        lines = [json.loads(x) for x in (ctx["out"] / b / "bpb.jsonl").read_text().splitlines()]
        final = [x for x in lines if x["ckpt"].startswith("final_")]
        prose = [x for x in final if x["set"] == "PROSE"][0]
        assert prose["bpb"] == pytest.approx((100 + 200 + 400) / 300)
        assert [x["bpb"] for x in final if x["set"] == "CHAT"] == [3.0]
        assert len({x["ckpt"] for x in lines}) == 3
    assert lock_free(ctx)


def test_restart_skips_finished_runs(ctx):
    assert run(ctx) == 0
    n_train, n_bpb = len(calls(ctx, "train.py")), len(calls(ctx, "bpb.py"))
    assert run(ctx) == 0
    assert (len(calls(ctx, "train.py")), len(calls(ctx, "bpb.py"))) == (n_train, n_bpb)
    assert "skip" in qlog(ctx)


def test_trunk_divergence_never_resumed_and_later_branches_inherit_it(ctx):
    trunk, b62, b125, b250 = ctx["names"]
    assert run(ctx, fake={trunk: ["diverge@2000"]}) == 0, qlog(ctx)
    assert trained(ctx) == [trunk, b62]
    assert (ctx["out"] / trunk / "DIVERGED").exists()
    assert outcomes(ctx) == {trunk: "diverged", b62: "done and scored",
                             b125: "diverged (inherited from its trunk)", b250: "diverged (inherited from its trunk)"}


def test_crash_resumes_once_then_second_crash_is_a_gap(ctx):
    trunk, b62, b125, b250 = ctx["names"]
    assert run(ctx, fake={trunk: ["crash", "ok"], b62: ["crash", "crash"]}) == 0, qlog(ctx)
    assert trained(ctx) == [trunk, trunk, b62, b62, b125, b250]
    assert outcomes(ctx)[trunk] == "done" and outcomes(ctx)[b62] == "gap"
    assert (ctx["out"] / b62 / "GAP").exists() and outcomes(ctx)[b125] == "done and scored"


def test_stop_file_before_start(ctx):
    (ctx["home"] / "STOP").touch()
    assert run(ctx) == 1
    assert trained(ctx) == [] and "STOP present" in qlog(ctx)


def test_stop_during_a_run_checkpoints_then_resumes_on_restart(ctx):
    trunk = ctx["names"][0]

    def stop_soon():            # once train.py has started
        t0 = time.time()
        while not calls(ctx, "train.py") and time.time() - t0 < 30:
            time.sleep(0.05)
        (ctx["home"] / "STOP").touch()
    assert run(ctx, fake={trunk: ["wait_stop", "ok"]}, during=stop_soon) == 1
    assert trained(ctx) == [trunk] and not list((ctx["out"] / trunk).glob("final_*"))
    (ctx["home"] / "STOP").unlink()
    assert run(ctx) == 0
    assert trained(ctx)[:2] == [trunk, trunk] and (ctx["out"] / trunk / "TRUNK_DONE").exists()


def test_heat_stops_the_run_waits_and_retries(ctx):
    trunk = ctx["names"][0]
    assert run(ctx, fake={trunk: ["wait_stop", "ok"]}, temps=[50, 90, 90, 90, 80, 80, 80, 80, 50]) == 0, qlog(ctx)
    assert trained(ctx)[:2] == [trunk, trunk]
    assert "for 3 samples" in qlog(ctx) and "not <= 75" in qlog(ctx)


def test_preflight_refusal_stops_the_queue(ctx):
    assert run(ctx, refuse=ctx["names"][0]) == 1
    assert trained(ctx) == [] and "preflight refused" in qlog(ctx) and lock_free(ctx)


def test_dirty_checkout_is_refused(ctx):
    (ctx["code"] / "harness" / "train.py").write_text("# edited\n")
    assert run(ctx) == 1
    assert calls(ctx, "preflight.py") == [] and "uncommitted" in qlog(ctx)


def test_busy_gpu_memory_blocks_the_start(ctx):
    assert run(ctx, mem=5000) == 1
    assert trained(ctx) == [] and "> 3000" in qlog(ctx)


def test_marks_wait_mark_and_stop(ctx):
    trunk, b62 = ctx["names"][:2]
    (ctx["xd"] / "plans" / "m.txt").write_text(
        f"# c\nmark E2 5M DONE\nwait_mark E2 E2 5M DONE\ntrain {trunk}\nstop\ntrain {b62}\n")
    assert run(ctx, plan="m.txt") == 1 and "uncommitted" in qlog(ctx)     # plans must be committed too
    commit(ctx)
    assert run(ctx, plan="m.txt") == 0, qlog(ctx)
    assert trained(ctx) == [trunk] and (ctx["out"] / "marks" / "E2_5M_DONE").exists()
    assert "MARK E2 5M DONE" in qlog(ctx) and "plan says stop" in qlog(ctx)


def test_low_disk_stops_the_queue(ctx):
    assert run(ctx, QUEUE_MIN_FREE_GB="100000000") == 1
    assert trained(ctx) == [] and "GB free" in qlog(ctx)


def test_wait_mark_waits_and_honours_stop(ctx):
    (ctx["xd"] / "plans" / "w.txt").write_text(f"wait_mark E3 E3 PART 1 DONE\ntrain {ctx['names'][0]}\n")
    commit(ctx)

    def stop_when_waiting():
        t0 = time.time()
        while time.time() - t0 < 30:
            if (ctx["out"] / "queue_e2.txt").exists() and "waiting for E3" in qlog(ctx):
                break
            time.sleep(0.05)
        (ctx["home"] / "STOP").touch()
    assert run(ctx, plan="w.txt", during=stop_when_waiting) == 1
    assert trained(ctx) == [] and "STOP while waiting for E3" in qlog(ctx)


def test_prereg_refusal_by_train_py_stops_the_queue(ctx):
    trunk = ctx["names"][0]
    (ctx["xd"] / "plans" / "p.txt").write_text(f"train {trunk}\nmark AFTER\n")
    commit(ctx)
    assert run(ctx, plan="p.txt", fake={trunk: ["prereg"]}) == 1
    assert trained(ctx) == [trunk] and "prereg refused" in qlog(ctx) and lock_free(ctx)
    assert not (ctx["out"] / "marks" / "AFTER").exists()
