"""diag_queue.sh (the PC queue of the C6 diagnostics runner) under the Mac's bash 3.2, with stubs for flock,
nvidia-smi, python3's disk reading, sleep and the runner (macOS has no flock; the PC dry run used the real ones), plus
one pass with the real runner on a fixture tree (CPU). Holds stay under 35 min: a hard timeout of 1,980 s (+30 s kill)
wraps every runner call that holds gpu.lock; a 2 h lock wait is logged and waited again, never an exit.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_diag_queue.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

import diag_fixtures as FX

Q = os.path.join(FX.SX, "diag_queue.sh")
STUBS = {
    "flock": 'echo "flock $*" >> "$C6T/calls"\ncase $1 in -n) exit ${C6T_FLOCK_N:-0};; -u) exit 0;; esac\n'
             'n=$(head -1 "$C6T/flock_w" 2>/dev/null); sed -i.bak 1d "$C6T/flock_w" 2>/dev/null; exit ${n:-0}\n',
    "nvidia-smi": 'f=$C6T/temp; case "$*" in *memory*) f=$C6T/mem;; esac\n'
                  'n=$(head -1 "$f" 2>/dev/null); [ "$(wc -l < "$f" 2>/dev/null)" -gt 1 ] && sed -i.bak 1d "$f"\n'
                  'echo "${n:-40}"\n',
    "python3": 'n=$(head -1 "$C6T/free" 2>/dev/null); [ "$(wc -l < "$C6T/free" 2>/dev/null)" -gt 1 ] && '
               'sed -i.bak 1d "$C6T/free"\necho "${n:-500}"\n',
    "sleep": 'echo "sleep $*" >> "$C6T/calls"; rm -f "$PLANCK_HOME/PAUSE"\n',
    "runner": 'echo "runner $*" >> "$C6T/calls"\ncase "$*" in *--cpu-only*) exit ${C6T_CPU_RC:-0};; esac\n'
              'n=$(head -1 "$C6T/rc" 2>/dev/null); sed -i.bak 1d "$C6T/rc" 2>/dev/null\n'
              'case $n in slow) /bin/sleep 5; exit 0;; esac; exit ${n:-0}\n',
}


def setup(tmp, flock_w=(), rc=(), temp=(), mem=(), free=(), files=()):
    (tmp / "bin").mkdir()
    for name, body in STUBS.items():
        p = tmp / "bin" / name
        p.write_text("#!/bin/bash\n" + body)
        p.chmod(0o755)
    for name, seq in (("flock_w", flock_w), ("rc", rc), ("temp", temp), ("mem", mem), ("free", free)):
        (tmp / name).write_text("".join(f"{x}\n" for x in seq))
    home, code = tmp / "planck", tmp / "code"
    (home / "runs").mkdir(parents=True)
    (code / "experiments" / "screens").mkdir(parents=True)
    (code / "experiments" / "screens" / "diag_runner.py").write_text("# stub\n")
    for f in files:
        p = home / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x\n")
    return home, code


def go(tmp, args=None, py=None, timeout=60, cmd=Q, extra=(), **env):
    home, code = tmp / "planck", tmp / "code"
    e = {"PATH": f"{tmp / 'bin'}:{os.environ['PATH']}", "C6T": str(tmp), "PLANCK_HOME": str(home),
         "PLANCK_PY": py or str(tmp / "bin" / "runner"), "C6_SMI": "nvidia-smi", "C6_POLL_S": "0",
         "HOME": str(tmp), "DEVELOPER_DIR": "/Library/Developer/CommandLineTools", **env}
    argv = list(extra) if cmd != Q else (args if args is not None else ["--code", str(code)])
    p = subprocess.run(["/bin/bash", cmd] + argv, env=e,
                       capture_output=True, text=True, timeout=timeout)
    log = (home / "logs" / "c6diag.log").read_text() if (home / "logs" / "c6diag.log").exists() else ""
    calls = (tmp / "calls").read_text().splitlines() if (tmp / "calls").exists() else []
    return p.returncode, log, calls, home


def runner_calls(calls):
    return [c for c in calls if c.startswith("runner ")]


def test_cpu_pass_then_holds_until_done(tmp_path):
    setup(tmp_path, rc=[3, 0])
    rc, log, calls, home = go(tmp_path, args=["--code", str(tmp_path / "code"), "--only", "base_s101"])
    r = runner_calls(calls)
    assert all(x.split("--evalset ")[1].split()[1:3] == ["--only", "base_s101"] for x in r), r
    assert rc == 0 and (home / "c6diag" / "DONE").exists() and log.rstrip().endswith("c6diag: DONE"), log
    assert len(r) == 3 and "--cpu-only" in r[0] and all("--cpu-only" not in x for x in r[1:])
    assert all(f"--runs-root {home}/runs --evalset {home}/data/evalsets/v0" in x for x in r)
    assert all(x.endswith("--device cuda --budget-s 1500") for x in r[1:])
    assert log.count("gpu.lock held") == 2 and log.count("gpu.lock released") == 2
    w = [c for c in calls if c.startswith("flock -w")]
    seq = [c.split()[0] if c.startswith("runner") else c for c in calls if not c.startswith("sleep")]
    assert w == ["flock -w 7200 9"] * 2 and seq == ["flock -n 8", "runner", "flock -w 7200 9", "runner", "flock -u 9",
                                                    "flock -w 7200 9", "runner", "flock -u 9"], seq


def test_a_two_hour_lock_wait_waits_again_and_never_exits(tmp_path):
    setup(tmp_path, flock_w=[1, 1, 0], rc=[0])
    rc, log, calls, _ = go(tmp_path)
    assert rc == 0 and log.count("gpu.lock not free after 7200 s: waiting again") == 2 and "DONE" in log


@pytest.mark.parametrize("stop", ["STOP", "c6diag/CANCEL"])
def test_stop_or_cancel_ends_the_queue_before_any_hold(tmp_path, stop):
    setup(tmp_path, files=[stop])
    rc, log, calls, home = go(tmp_path)
    assert rc == 1 and runner_calls(calls) == [] and "present: exiting" in log and not (home / "c6diag" / "DONE").exists()


@pytest.mark.parametrize("gpu_rc", [1, 2])
def test_a_failed_or_refused_runner_stops_for_a_person(tmp_path, gpu_rc):
    setup(tmp_path, rc=[gpu_rc, 0])
    rc, log, calls, _ = go(tmp_path)
    assert rc == 1 and len(runner_calls(calls)) == 2 and f"runner exit {gpu_rc}: stopping for a person" in log


def test_a_failed_cpu_pass_takes_no_hold(tmp_path):
    setup(tmp_path)
    rc, log, calls, _ = go(tmp_path, C6T_CPU_RC="2")
    assert rc == 1 and len(runner_calls(calls)) == 1 and "CPU pass: runner exit 2" in log and "flock -w" not in "".join(calls)


def test_the_hold_limit_kills_the_runner_and_twice_stops(tmp_path):
    setup(tmp_path, rc=["slow", "slow", 0])
    rc, log, calls, _ = go(tmp_path, C6_HOLD_S="1")
    assert rc == 1 and log.count("runner exit 124") == 2 and "hold limit twice" in log and len(runner_calls(calls)) == 3


def test_one_hold_limit_kill_is_retried(tmp_path):
    setup(tmp_path, rc=["slow", 0])
    rc, log, calls, _ = go(tmp_path, C6_HOLD_S="1")
    assert rc == 0 and log.count("runner exit 124") == 1 and "DONE" in log


def test_waits_for_memory_heat_disk_and_pause(tmp_path):
    setup(tmp_path, rc=[0], temp=[80, 40], mem=[5000, 100], free=[12, 500], files=["PAUSE"])
    rc, log, calls, _ = go(tmp_path)
    assert rc == 0 and "PAUSE present: waiting" in log and "disk: [12] GB free (< 40): waiting" in log
    assert "GPU temperature [80] not <= 75: waiting" in log
    assert "GPU memory in use [5000] MiB > 3000 with the lock held: released, waiting" in log
    i = calls.index("flock -u 9")
    assert calls[i - 1] == "flock -w 7200 9" and len(runner_calls(calls)) == 2      # released before any runner call


def test_a_wait_reason_is_logged_once_per_change_not_once_per_poll(tmp_path):
    setup(tmp_path, rc=[0], temp=[80, 80, 80, 40])
    rc, log, calls, _ = go(tmp_path)
    assert rc == 0 and log.count("GPU temperature [80] not <= 75: waiting") == 1
    assert sum(c.startswith("sleep") for c in calls) == 3


def test_one_queue_at_a_time_and_argument_refusals(tmp_path):
    setup(tmp_path)
    rc, log, calls, _ = go(tmp_path, C6T_FLOCK_N="1")
    assert rc == 1 and "another c6diag queue holds" in log and runner_calls(calls) == []
    for args, want in ((["--bundle", "planck.bundle"], 2), (["--code", str(tmp_path / "nowhere")], 2), (["--x"], 2)):
        assert go(tmp_path, args=args)[0] == want


def test_hold_constants_keep_every_hold_under_35_minutes():
    txt = open(Q).read()
    hold, budget, wait = (int(re.search(rf"{k}:-(\d+)", txt).group(1)) for k in ("C6_HOLD_S", "C6_BUDGET_S", "C6_LOCK_WAIT_S"))
    kill = int(re.search(r'timeout -k (\d+) "\$HOLD"', txt).group(1))
    assert hold + kill < 35 * 60 and budget < hold and wait == 7200 and "MINFREE=40" in txt
    assert re.search(r'timeout -k \d+ "\$HOLD" "\$PY" "\$CODE/experiments/screens/diag_runner.py" "\$\{R\[@\]\}" '
                     r'--device "\$DEVICE" \\\n\s+--budget-s "\$BUDGET"', txt)


def test_bundle_launch_clones_and_runs_the_queue_from_the_clone(tmp_path):
    setup(tmp_path)
    kit, src = tmp_path / "kit", tmp_path / "src"
    (src / "experiments" / "screens").mkdir(parents=True)
    kit.mkdir()
    (src / "experiments" / "screens" / "diag_queue.sh").write_text('#!/bin/bash\necho "clone $*" >> "$C6T/calls"\n')
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(src)]
    env = {**os.environ, "DEVELOPER_DIR": "/Library/Developer/CommandLineTools"}
    for c in (["init", "-q", "-b", "main"], ["add", "."], ["commit", "-qm", "x"],
              ["bundle", "create", str(kit / "planck_c6diag.bundle"), "HEAD", "main"]):
        subprocess.run(git + c, check=True, capture_output=True, env=env)
    ref = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, env=env).stdout.strip()
    rc, log, calls, home = go(tmp_path, args=["--bundle", "planck_c6diag.bundle", "--groups", "stage2", "--only", "a,b"],
                              C6_KIT=str(kit))
    assert rc == 0 and calls == [f"clone --code {home}/qcode/{ref[:12]} --groups stage2 --only a,b"], (log, calls)


def test_start_wrapper_runs_the_kit_queue_in_bundle_mode(tmp_path):
    setup(tmp_path)
    kit = tmp_path / "kit"
    kit.mkdir()
    (kit / "diag_queue.sh").write_text('#!/bin/bash\necho "kitqueue $*" >> "$C6T/calls"\n')
    rc, log, calls, _ = go(tmp_path, args=None, C6_KIT=str(kit), cmd=os.path.join(FX.SX, "diag_start.sh"),
                           extra=["--groups", "stage2"])
    assert rc == 0 and calls == ["kitqueue --bundle planck_c6diag.bundle --groups stage2"], calls


def test_with_the_real_runner_on_a_fixture_tree(tmp_path):
    home, _ = setup(tmp_path, rc=[])
    FX.root_with(tmp_path / "planck", {("E3", "e3_5m_A_s1"): 6})
    FX.evalset(str(home / "data" / "evalsets" / "v0"))
    os.remove(tmp_path / "bin" / "python3")                     # the real disk reading
    rc, log, calls, _ = go(tmp_path, args=["--code", FX.ROOT, "--groups", "replay"], py=sys.executable, C6_DEVICE="cpu",
                           timeout=300)
    assert rc == 0 and "DONE" in log and "done E3/e3_5m_A_s1 hd" in log, log
    assert sorted(os.listdir(home / "runs" / "E3" / "e3_5m_A_s1" / "c6")) == ["hd.replay.json", "ind.replay.json"]
