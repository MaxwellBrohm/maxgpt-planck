"""queue_k.sh with a fake python (fake_kpy.py) and a fake nvidia-smi: no model, no GPU (K3 round 16, 2026-11-21).
Needs bash >= 4; where flock is missing (the Mac) tests/flock_shim.py stands in for its fd forms."""
import json
import os
import shutil
import subprocess
import time

import pytest

import kcommon as K

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = K.ROOT          # the code root (K_CODE_ROOT under the mutant runner's scratch copies)
BASH = next((b for b in ("/opt/homebrew/bin/bash", "/usr/local/bin/bash", "/bin/bash") if os.path.exists(b) and
             int(subprocess.run([b, "-c", "echo ${BASH_VERSINFO[0]}"], capture_output=True, text=True).stdout) >= 4),
            None)
pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash >= 4")
SMI = '#!/bin/bash\ncase "$*" in *temperature*) echo 50 ;; *memory.used*) echo 1000 ;; esac\n'
RUNS = ("k0_smoke_s200", "k0g_g1_s199", "k0g_g2_s199", "k1_fh_s201")


def env0():
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


@pytest.fixture
def ctx(tmp_path):
    home, code = tmp_path / "home", tmp_path / "code"
    kx, cf, e2 = code / "experiments" / "K" / "code", code / "experiments" / "K" / "configs", \
        code / "experiments" / "E2_lr_transfer"
    for d in (home / "locks", home / "runs", kx, cf, e2, code / "harness", code / "data_prep", code / "tokenizer" / "v0",
              tmp_path / "bin", tmp_path / "data" / "evals"):
        d.mkdir(parents=True)
    shutil.copy(os.path.join(CODE, "queue_k.sh"), kx / "queue_k.sh")
    for f in ("queue_lib.sh", "bpb_lines.py"):
        shutil.copy(os.path.join(ROOT, "experiments", "E2_lr_transfer", f), e2 / f)
    for f in ("ktrain.py", "kconfig.py", "kpreflight.py", "probes.py"):
        (kx / f).write_text("# placeholder: fake_kpy.py answers\n")
    (code / "data_prep" / "bpb.py").write_text("# placeholder\n")
    (code / "tokenizer" / "v0" / "tok_v0_8k.json").write_text("{}\n")
    for n in RUNS:
        steps = 306 if "smoke" in n else 7630
        (cf / f"{n}.yaml").write_text(f"name: {n}\nout_dir: ../../../planck_root/runs/K/{n}\n"
                                      f"train: {{total_steps: {steps}}}\n")
    (tmp_path / "plan.txt").write_text("train k0_smoke_s200\ntrain k0g_g1_s199\ntrain k1_fh_s201\nmark grid done\n"
                                       "stop\ntrain k0g_g2_s199\n")
    g = ["git", "-C", str(code)]
    for a in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "t"]):
        subprocess.run(g + a, check=True, env=env0())
    (tmp_path / "smi.sh").write_text(SMI)
    fake = tmp_path / "fake_kpy.py"
    shutil.copy(os.path.join(HERE, "fake_kpy.py"), fake)
    for p in (tmp_path / "smi.sh", fake):
        p.chmod(0o755)
    path = os.environ["PATH"]
    if shutil.which("flock") is None:
        (tmp_path / "bin" / "flock").write_text(f"#!/bin/sh\nexec {shutil.which('python3')} {HERE}/flock_shim.py \"$@\"\n")
        (tmp_path / "bin" / "flock").chmod(0o755)
        path = f"{tmp_path / 'bin'}:{path}"
    env = {**env0(), "PATH": path, "HOME": str(tmp_path), "PLANCK_HOME": str(home), "PLANCK_PY": str(fake),
           "PLANCK_NVIDIA_SMI": str(tmp_path / "smi.sh"), "QUEUE_POLL_S": "0.05", "QUEUE_WAIT_S": "0.1",
           "QUEUE_MIN_FREE_GB": "0", "FAKE_LOG": str(tmp_path / "calls.jsonl"), "FAKE_STATE": str(tmp_path / "st.json")}
    return {"tmp": tmp_path, "code": code, "home": home, "env": env, "out": home / "runs" / "K", "cf": cf}


def run(c, **kw):
    env = dict(c["env"], **{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in kw.items()})
    r = subprocess.run([BASH, str(c["code"] / "experiments" / "K" / "code" / "queue_k.sh"), "--code", str(c["code"]),
                        "--plan", str(c["tmp"] / "plan.txt"), "--data", str(c["tmp"] / "data")],
                       env=env, capture_output=True, text=True, timeout=120)
    p = c["tmp"] / "calls.jsonl"
    calls = [json.loads(x) for x in open(p)] if p.exists() else []
    log = (c["out"] / "queue_k.txt").read_text() if (c["out"] / "queue_k.txt").exists() else ""
    return r.returncode, calls, log


def test_a_plan_runs_in_order_checked_locked_scored_and_marked(ctx):
    rc, calls, log = run(ctx)
    assert rc == 0, log
    seq = [(x["tool"], x["args"][1] if x["tool"] == "kconfig.py" else os.path.basename(x["args"][0])) for x in calls
           if x["tool"] in ("kconfig.py", "kpreflight.py", "ktrain.py")]
    assert seq == [("kconfig.py", "k0_smoke_s200"), ("kpreflight.py", "k0_smoke_s200.yaml"),
                   ("ktrain.py", "k0_smoke_s200.yaml"), ("kconfig.py", "k0g_g1_s199"),
                   ("kpreflight.py", "k0g_g1_s199.yaml"), ("ktrain.py", "k0g_g1_s199.yaml"),
                   ("kconfig.py", "k1_fh_s201"), ("kpreflight.py", "k1_fh_s201.yaml"),
                   ("ktrain.py", "k1_fh_s201.yaml")], seq
    tr = [x for x in calls if x["tool"] == "ktrain.py"]
    assert all(x["gpu_lock_held"] and "--require-committed" in x["args"] and "cuda" in x["args"] for x in tr)
    assert all(not x["gpu_lock_held"] for x in calls if x["tool"] in ("kconfig.py", "kpreflight.py"))
    pr = [x for x in calls if x["tool"] == "probes.py"]
    assert len(pr) == 2 and all(x["gpu_lock_held"] for x in pr)
    assert [x["args"][x["args"].index("--ckpt") + 1].split("/")[-2:] for x in pr] == [
        ["k0g_g1_s199", "final_00007630.pt"], ["k1_fh_s201", "final_00007630.pt"]]
    items = [x["args"][x["args"].index("--items") + 1:] for x in pr]
    # 2026-11-22 (REVIEW 17b): a K0-g run scores K-DEV alone (SPEC 8: K-DEV for g picks); a K1 run every SPEC 9 set
    assert [os.path.basename(p) for p in items[0]] == ["k_dev.jsonl"]
    assert [os.path.basename(p) for p in items[1]] == [f"{s}.jsonl" for s in
                                                      ("k_eval_id", "k_eval_s", "k_eval_ood", "fb_probe", "conflict",
                                                       "k_dev")]
    assert all(os.path.dirname(p) == str(ctx["tmp"] / "data" / "evals") for p in items[0] + items[1])
    assert len([x for x in calls if x["tool"] == "bpb.py"]) == 6 and all(
        x["gpu_lock_held"] for x in calls if x["tool"] == "bpb.py")
    g1 = ctx["out"] / "k0g_g1_s199"
    assert (g1 / "SCORED").exists() and (g1 / "k_eval.jsonl").exists() and (g1 / "k_eval.summary.json").exists()
    assert (ctx["out"] / "k0_smoke_s200" / "SCORED").exists() and not (ctx["out"] / "k0_smoke_s200" / "k_eval.jsonl").exists()
    assert (ctx["out"] / "marks" / "grid_done").exists() and not (ctx["out"] / "k0g_g2_s199").exists()
    rc, calls2, _ = run(ctx)                    # a restart skips every scored run: no check, train or score again
    assert rc == 0 and calls2[len(calls):] == []


def test_a_refusal_stops_the_queue_before_the_gpu(ctx):
    rc, calls, log = run(ctx, FAKE_KCHECK_REFUSE="k0g_g1_s199")
    assert rc == 1 and "kconfig check refused k0g_g1_s199" in log
    assert [os.path.basename(x["args"][0]) for x in calls if x["tool"] == "ktrain.py"] == ["k0_smoke_s200.yaml"]
    shutil.rmtree(ctx["out"])
    os.unlink(ctx["tmp"] / "calls.jsonl")
    rc, calls, log = run(ctx, FAKE_REFUSE="k0_smoke_s200")
    assert rc == 1 and "kpreflight refused k0_smoke_s200" in log and not [x for x in calls if x["tool"] == "ktrain.py"]


def test_diverged_goes_on_a_crash_resumes_once_and_a_failed_score_stops(ctx):
    (ctx["tmp"] / "plan.txt").write_text("train k0g_g1_s199\ntrain k0g_g2_s199\n")
    rc, calls, log = run(ctx, FAKE_PLAN={"k0g_g1_s199": ["diverge"], "k0g_g2_s199": ["crash", "ok"]})
    assert rc == 0, log
    assert (ctx["out"] / "k0g_g1_s199" / "DIVERGED").exists() and not (ctx["out"] / "k0g_g1_s199" / "SCORED").exists()
    assert [os.path.basename(x["args"][0]) for x in calls if x["tool"] == "ktrain.py"] == [
        "k0g_g1_s199.yaml", "k0g_g2_s199.yaml", "k0g_g2_s199.yaml"]
    assert (ctx["out"] / "k0g_g2_s199" / "SCORED").exists()
    st = [json.loads(x)["outcome"] for x in open(ctx["out"] / "status.jsonl")]
    assert st == ["diverged", "done and scored"]
    shutil.rmtree(ctx["out"])
    rc, _, log = run(ctx, FAKE_PROBE_FAIL="1")
    assert rc == 1 and "K scoring failed for k0g_g1_s199" in log and not (ctx["out"] / "k0g_g1_s199" / "SCORED").exists()


def test_an_uncommitted_k_file_refuses_the_queue(ctx):
    (ctx["cf"] / "k0g_g1_s199.yaml").write_text("name: k0g_g1_s199\nout_dir: x\n")
    rc, calls, log = run(ctx)
    assert rc == 1 and "uncommitted" in log and calls == []


def test_the_k0g_grid_waits_for_the_smoke_pass_mark(ctx):
    """2026-11-22 (REVIEW 17b): SPEC 13 makes "the K smoke run passed" a precondition of every K run, so plans/k0.txt
    puts "wait_mark K K0-smoke passed" between the smoke runs and the grid. Here the queue trains the smoke, then
    waits without the GPU lock and trains nothing more; k0smoke.mark (the file k0smoke.py writes on a pass, under
    the same slug) releases it, and the grid run trains after."""
    import time
    import k0smoke
    import kconfig as KC
    (ctx["tmp"] / "plan.txt").write_text(f"train k0_smoke_s200\nwait_mark K {KC.SMOKE_MARK}\ntrain k0g_g1_s199\n")
    env = dict(ctx["env"])
    p = subprocess.Popen([BASH, str(ctx["code"] / "experiments" / "K" / "code" / "queue_k.sh"), "--code",
                          str(ctx["code"]), "--plan", str(ctx["tmp"] / "plan.txt"), "--data", str(ctx["tmp"] / "data")],
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = ctx["out"] / "queue_k.txt"
    try:
        t0 = time.time()
        while not (log.exists() and "waiting for K to record: K0-smoke passed" in log.read_text()):
            assert p.poll() is None and time.time() - t0 < 60, log.read_text() if log.exists() else "no log"
            time.sleep(0.05)
        time.sleep(0.5)                              # several wait cycles (QUEUE_WAIT_S 0.1): still nothing trains
        calls = [json.loads(x) for x in open(ctx["tmp"] / "calls.jsonl")]
        assert [os.path.basename(x["args"][0]) for x in calls if x["tool"] == "ktrain.py"] == ["k0_smoke_s200.yaml"]
        assert p.poll() is None
        k0smoke.mark(str(ctx["out"]))
        assert p.wait(timeout=60) == 0, log.read_text()
    finally:
        if p.poll() is None:                         # never leave a waiting queue behind a failed assert
            p.kill()
            p.wait()
    calls = [json.loads(x) for x in open(ctx["tmp"] / "calls.jsonl")]
    assert [os.path.basename(x["args"][0]) for x in calls if x["tool"] == "ktrain.py"] == [
        "k0_smoke_s200.yaml", "k0g_g1_s199.yaml"]
    assert "seen K: K0-smoke passed" in log.read_text()


def _start(c, plan: str, **kw):
    import subprocess as sp
    (c["tmp"] / "plan.txt").write_text(plan)
    env = dict(c["env"], **{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in kw.items()})
    return sp.Popen([BASH, str(c["code"] / "experiments" / "K" / "code" / "queue_k.sh"), "--code", str(c["code"]), "--plan",
                     str(c["tmp"] / "plan.txt"), "--data", str(c["tmp"] / "data")], env=env, stdout=sp.PIPE,
                    stderr=sp.STDOUT, start_new_session=True)


def _end(p):
    """Never leave a queue (or its fake ktrain) behind a failed assert: the queue runs in its own session."""
    import signal
    if p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait()


def _until(p, cond, what):
    import time
    t0 = time.time()
    while not cond():
        assert p.poll() is None and time.time() - t0 < 60, what
        time.sleep(0.05)


def _trained(c):
    p = c["tmp"] / "calls.jsonl"
    return [os.path.basename(x["args"][0]) for x in map(json.loads, open(p)) if x["tool"] == "ktrain.py"] \
        if p.exists() else []


@pytest.mark.parametrize("which", ["planck", "k"])
def test_stop_ends_a_queue_waiting_at_a_mark(ctx, which):
    """2026-11-22 (port review finding 3): the plan's failed-smoke path ends the queue that waits at "wait_mark K
    K0-smoke passed" with a STOP file. ~/planck/STOP does it (queue_k.sh's wait loop) and so does K's own
    ~/planck/runs/K/STOP (port review note 6: ~/planck/STOP stops every Planck queue); nothing more trains, exit 1."""
    import kconfig as KC
    p = _start(ctx, f"train k0_smoke_s200\nwait_mark K {KC.SMOKE_MARK}\ntrain k0g_g1_s199\n")
    log = ctx["out"] / "queue_k.txt"
    try:
        _until(p, lambda: log.exists() and "waiting for K to record: K0-smoke passed" in log.read_text(), "no wait")
        stop = ctx["home"] / "STOP" if which == "planck" else ctx["out"] / "STOP"
        stop.write_text("")
        assert p.wait(timeout=60) == 1, log.read_text()
    finally:
        _end(p)
    text = log.read_text()
    assert "STOP while waiting for K: K0-smoke passed" in text and "queue stops at: wait_mark K K0-smoke passed" in text
    assert _trained(ctx) == ["k0_smoke_s200.yaml"]
    # 2026-11-23 (K review finding 3): K's STOP never creates ~/planck/STOP, which stops every Planck queue
    assert (ctx["home"] / "STOP").exists() == (which == "planck")


def test_k_stop_stops_a_running_k_run_at_its_checkpoint(ctx):
    """2026-11-22 (port review note 6): K's STOP, like ~/planck/STOP, makes the running ktrain.py stop at its
    checkpoint (the queue touches the run's STOP file), and the queue stops there."""
    p = _start(ctx, "train k0_smoke_s200\ntrain k0g_g1_s199\n", FAKE_PLAN={"k0_smoke_s200": ["wait_stop"]})
    log = ctx["out"] / "queue_k.txt"
    try:
        _until(p, lambda: _trained(ctx) == ["k0_smoke_s200.yaml"], "the smoke never started")
        (ctx["out"] / "STOP").write_text("")
        assert p.wait(timeout=60) == 1, log.read_text()
    finally:
        _end(p)
    text = log.read_text()
    assert "stopped k0_smoke_s200 at its checkpoint" in text and "queue stops at: train k0_smoke_s200" in text
    assert (ctx["out"] / "k0_smoke_s200" / "ckpt_00000005.pt").exists()
    assert not list((ctx["out"] / "k0_smoke_s200").glob("final_*.pt")) and _trained(ctx) == ["k0_smoke_s200.yaml"]
    assert not (ctx["home"] / "STOP").exists()          # 2026-11-23 (K review finding 3): no ~/planck/STOP leak


def test_k_stop_before_a_train_line_stops_the_queue(ctx):
    (ctx["out"]).mkdir(parents=True)
    (ctx["out"] / "STOP").write_text("")
    rc, calls, log = run(ctx)
    assert rc == 1 and "K STOP present" in log and "queue stops at: train k0_smoke_s200" in log
    assert calls == []


# 2026-11-23 (K review finding 3): queue_k.sh's own safety paths (header items 2 and 4, notes N-6), each pinned by a
# mutant that survived the frozen tests (mutation_k16.py, the rv_q and rv2_q ids). Built on the review's probe tests
# (k_review scripts/test_review_queue.py) with the heat spell counted out by the fake nvidia-smi instead of timed.
SMI_FILES = """#!/bin/bash
# temperature: 90 C while $D/hot holds a count above 0 (one sample used per call), else 50 C; memory from $D/mem
D={d}
case "$*" in
    *temperature*) n=$(cat "$D/hot" 2>/dev/null); if [ "${{n:-0}}" -gt 0 ]; then echo $((n - 1)) > "$D/hot"; echo 90;
                   else echo 50; fi ;;
    *memory.used*) cat "$D/mem" 2>/dev/null || echo 1000 ;;
esac
"""


def _smi_files(c):
    (c["tmp"] / "smi.sh").write_text(SMI_FILES.format(d=c["tmp"]))


def test_planck_stop_stops_a_running_k_run_at_its_checkpoint(ctx):
    """N-6: ~/planck/STOP still makes a running K run stop at its checkpoint, and the queue stops there."""
    p = _start(ctx, "train k0_smoke_s200\ntrain k0g_g1_s199\n", FAKE_PLAN={"k0_smoke_s200": ["wait_stop"]})
    log = ctx["out"] / "queue_k.txt"
    try:
        _until(p, lambda: _trained(ctx) == ["k0_smoke_s200.yaml"], "the smoke never started")
        (ctx["home"] / "STOP").write_text("")
        assert p.wait(timeout=30) == 1, log.read_text()
    finally:
        _end(p)
    text = log.read_text()
    assert "stopped k0_smoke_s200 at its checkpoint" in text and "queue stops at: train k0_smoke_s200" in text
    assert _trained(ctx) == ["k0_smoke_s200.yaml"]


def test_a_hot_gpu_stops_the_run_and_it_trains_again_after_cooling(ctx):
    """Header item 2's exit 14: three samples at 87 C or more stop the running K run at its checkpoint; the queue
    lets go of gpu.lock, waits in queue_lib's guards for 75 C, and trains the run again. Four hot samples here: a
    queue that never counts them, or needs more than three, never stops the run (the fake waits for its STOP)."""
    _smi_files(ctx)
    p = _start(ctx, "train k0_smoke_s200\n", FAKE_PLAN={"k0_smoke_s200": ["wait_stop", "ok"]})
    log = ctx["out"] / "queue_k.txt"
    try:
        _until(p, lambda: _trained(ctx) == ["k0_smoke_s200.yaml"], "the smoke never started")
        (ctx["tmp"] / "hot").write_text("4")
        assert p.wait(timeout=30) == 0, log.read_text()
    finally:
        _end(p)
    text = log.read_text()
    assert "GPU at 90 C for 3 samples: stopping the run" in text and "stopped k0_smoke_s200" not in text
    assert _trained(ctx) == ["k0_smoke_s200.yaml", "k0_smoke_s200.yaml"]
    out = ctx["out"] / "k0_smoke_s200"
    assert list(out.glob("final_*.pt")) and not (out / "HEAT").exists() and not (out / "crashes.txt").exists()
    assert [json.loads(x)["outcome"] for x in open(ctx["out"] / "status.jsonl")] == ["done (smoke: never scored)"]


def test_the_queue_records_the_largest_gpu_memory_of_a_run(ctx):
    """k0smoke.py's memory check reads <run>/gpu_mem_max_mib.txt and passes when the file is missing, so the queue
    must write it: the largest memory.used seen while the run trains, not the last."""
    _smi_files(ctx)
    mem, mx = ctx["tmp"] / "mem", ctx["out"] / "k0_smoke_s200" / "gpu_mem_max_mib.txt"
    mem.write_text("1200")
    p = _start(ctx, "train k0_smoke_s200\n", FAKE_PLAN={"k0_smoke_s200": ["wait_stop"]})
    try:
        _until(p, lambda: _trained(ctx) == ["k0_smoke_s200.yaml"], "the smoke never started")
        mem.write_text("3400")
        _until(p, lambda: mx.exists() and mx.read_text().strip() == "3400", "the maximum never written")
        mem.write_text("2100")
        time.sleep(0.5)                                  # about ten samples at QUEUE_POLL_S 0.05
        (ctx["out"] / "STOP").write_text("")
        assert p.wait(timeout=30) == 1
    finally:
        _end(p)
    assert mx.read_text().strip() == "3400"


def test_pause_holds_the_queue_before_a_run(ctx):
    """queue_lib's guards run before every run: ~/planck/PAUSE holds the queue (nothing trains) until removed."""
    (ctx["home"] / "PAUSE").write_text("")
    p = _start(ctx, "train k0_smoke_s200\n")
    log = ctx["out"] / "queue_k.txt"
    try:
        _until(p, lambda: log.exists() and "PAUSE present: waiting" in log.read_text(), "no pause")
        time.sleep(0.5)
        assert _trained(ctx) == [] and p.poll() is None
        (ctx["home"] / "PAUSE").unlink()
        assert p.wait(timeout=30) == 0, log.read_text()
    finally:
        _end(p)
    assert _trained(ctx) == ["k0_smoke_s200.yaml"]


def test_a_refused_start_stops_the_queue_and_is_never_resumed(ctx):
    """Exit 2 from ktrain.py (train.py's prereg gate or ktrain's k block refused the config) is exit 13: the queue
    stops at that line. It is not a crash: no crashes.txt, no resume, no GAP, and the plan does not go on."""
    (ctx["tmp"] / "plan.txt").write_text("train k0_smoke_s200\ntrain k0g_g1_s199\n")
    rc, calls, log = run(ctx, FAKE_PLAN={"k0_smoke_s200": ["refuse", "ok"]})
    assert rc == 1 and "prereg or k block refused for k0_smoke_s200.yaml" in log, log
    assert "queue stops at: train k0_smoke_s200" in log and _trained(ctx) == ["k0_smoke_s200.yaml"]
    out = ctx["out"] / "k0_smoke_s200"
    assert not (out / "crashes.txt").exists() and not (out / "GAP").exists() and not (out / "SCORED").exists()


def test_the_code_hash_list_covers_experiments_k_and_the_shared_code(ctx):
    """Header item 4: code_sha256_<time>.txt holds the sha256 of every K file the queue runs (code, configs) beside
    queue_lib's list (harness, data_prep, E2, tokenizer, corpus), each equal to the file's own hash."""
    import hashlib
    rc, _, log = run(ctx)
    assert rc == 0, log
    files = sorted(ctx["out"].glob("code_sha256_*.txt"))
    assert len(files) == 1
    got = {x.split()[1]: x.split()[0] for x in files[0].read_text().splitlines()}
    want = [os.path.relpath(os.path.join(d, f), ctx["code"]) for d, _, fs in os.walk(ctx["code"] / "experiments" / "K")
            for f in fs if f.endswith((".py", ".yaml", ".sh", ".txt"))]
    want += ["data_prep/bpb.py", "experiments/E2_lr_transfer/queue_lib.sh", "tokenizer/v0/tok_v0_8k.json"]
    assert len(want) >= 12
    for f in want:
        assert got.get(f) == hashlib.sha256((ctx["code"] / f).read_bytes()).hexdigest(), f


def test_the_disk_guard_keeps_40_gb_free(ctx):
    """Header item 5 (2026-11-23, K review): with QUEUE_MIN_FREE_GB unset the queue stops before a run at 39 GB free
    and trains at 40 (K OPEN step 3: keep 40 GB free; queue_lib's own floor is 25). df is a stand-in here."""
    b = ctx["tmp"] / "dfbin"
    b.mkdir()
    (b / "df").write_text('#!/bin/sh\necho Avail\necho " $(cat %s)G"\n' % (ctx["tmp"] / "free_gb"))
    (b / "df").chmod(0o755)
    (ctx["tmp"] / "plan.txt").write_text("train k0_smoke_s200\n")
    ctx["env"] = {k: v for k, v in ctx["env"].items() if k != "QUEUE_MIN_FREE_GB"}
    ctx["env"]["PATH"] = f"{b}:{ctx['env']['PATH']}"
    (ctx["tmp"] / "free_gb").write_text("39")
    rc, calls, log = run(ctx)
    assert rc == 1 and "disk: 39 GB free (< 40): queue stops" in log and _trained(ctx) == [], log
    (ctx["tmp"] / "free_gb").write_text("40")
    rc, calls, log = run(ctx)
    assert rc == 0 and _trained(ctx) == ["k0_smoke_s200.yaml"], log
