"""Single-process guard runner for model jobs on this Mac (24 GB, kernel panic on 2026-09-23).

usage: guard.py --name NAME --ceiling SECONDS [--min-free 35] [--online] -- <cmd> [args...]

Before start:
  - refuses if another model job (run_battery.py / run_chat.py / ft_test.py / run_capacity.py /
    run_model.py / uprobe.py / khard.py / copysplit.py) is running
  - unloads LM Studio if anything is loaded, refuses if ollama has a model resident
  - waits until `memory_pressure` free >= min-free (polls every 30 s, gives up after 30 min)
During the run (every 5 s):
  - kills the child if wall clock > ceiling, free memory < 15 %, or swap used grew by > 768 MB
Writes logs/<name>.log (child output), logs/<name>.guard.log (samples) and logs/<name>.guard.json (verdict).
Sets PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7, PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6, HF_HUB_OFFLINE=1 (unless --online).
"""
import argparse, json, os, re, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
LOGS = os.path.join(EXP, "logs")
LMS = os.path.expanduser("~/.lmstudio/bin/lms")
MODEL_SCRIPTS = ("run_battery.py", "run_chat.py", "check_shared.py", "ft_test.py", "run_capacity.py", "run_model.py", "uprobe.py",
                 "khard.py", "copysplit.py", "exp_fixed_history.py")


def free_pct():
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage:\s*(\d+)%", out)
    return int(m.group(1)) if m else -1


def swap_used_mb():
    out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) if m else -1.0


def rss_mb(pid):
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) / 1024 if out else 0.0


def other_model_jobs():
    out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    me = os.getpid()
    hits = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        pid, cmd = line.split(None, 1)
        if int(pid) == me or "guard.py" in cmd:
            continue
        if "python" in cmd and any(s in cmd for s in MODEL_SCRIPTS):
            hits.append(line[:200])
    return hits


def preflight(log):
    jobs = other_model_jobs()
    if jobs:
        log(f"REFUSE: other model jobs running: {jobs}")
        return False
    try:
        lms_ps = lambda: (lambda r: r.stdout + r.stderr)(subprocess.run([LMS, "ps"], capture_output=True, text=True, timeout=60))
        ps = lms_ps()
        if "No models are currently loaded" not in ps:
            log(f"LM Studio reports: {ps.strip()[:200]!r}; unloading all")
            subprocess.run([LMS, "unload", "--all"], capture_output=True, text=True, timeout=120)
            ps = lms_ps()
            if "No models are currently loaded" not in ps:
                log("REFUSE: LM Studio still has a model loaded")
                return False
    except Exception as e:  # lms missing or hung: be conservative
        log(f"REFUSE: could not check LM Studio ({e})")
        return False
    try:
        op = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        if len(op) > 1:
            log(f"REFUSE: ollama has a model resident: {op[1:]}")
            return False
    except FileNotFoundError:
        pass
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ceiling", type=int, required=True)
    ap.add_argument("--min-free", type=int, default=35)
    ap.add_argument("--online", action="store_true")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    os.makedirs(LOGS, exist_ok=True)
    glog = open(os.path.join(LOGS, f"{a.name}.guard.log"), "a")

    def log(msg):
        glog.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        glog.flush()

    verdict = dict(name=a.name, cmd=cmd, started=None, ended=None, exit=None, killed=None,
                   min_free=None, max_swap_mb=None, peak_rss_mb=0.0, elapsed_s=None)
    vpath = os.path.join(LOGS, f"{a.name}.guard.json")
    if not preflight(log):
        verdict["killed"] = "preflight_refused"
        json.dump(verdict, open(vpath, "w"), indent=1)
        sys.exit(3)
    t_wait = time.time()
    while True:
        f = free_pct()
        if f >= a.min_free:
            break
        if time.time() - t_wait > 1800:
            log(f"REFUSE: free memory stayed below {a.min_free}% for 30 min (last {f}%)")
            verdict["killed"] = "memory_never_free"
            json.dump(verdict, open(vpath, "w"), indent=1)
            sys.exit(4)
        log(f"waiting for memory: free {f}%")
        time.sleep(30)

    env = dict(os.environ)
    env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.7"
    env["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.6"
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    if not a.online:
        env["HF_HUB_OFFLINE"] = "1"
    out = open(os.path.join(LOGS, f"{a.name}.log"), "w")
    swap0 = swap_used_mb()
    t0 = time.time()
    verdict["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
    log(f"start free={free_pct()}% swap={swap0:.0f}MB cmd={' '.join(cmd)}")
    p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env, cwd=HERE, start_new_session=True)
    min_free, max_swap, peak = 100, swap0, 0.0
    killed = None
    last_log = 0
    while p.poll() is None:
        time.sleep(5)
        f, s, r = free_pct(), swap_used_mb(), rss_mb(p.pid)
        min_free, max_swap, peak = min(min_free, f), max(max_swap, s), max(peak, r)
        el = time.time() - t0
        if el - last_log >= 30:
            log(f"t={el:5.0f}s free={f}% swap={s:.0f}MB rss={r:.0f}MB")
            last_log = el
        reason = None
        if el > a.ceiling:
            reason = f"wall_clock_ceiling_{a.ceiling}s"
        elif 0 <= f < 15:
            reason = f"free_memory_{f}pct"
        elif s - swap0 > 768:
            reason = f"swap_grew_{s - swap0:.0f}MB"
        if reason:
            log(f"KILL: {reason}")
            killed = reason
            try:
                os.killpg(p.pid, signal.SIGTERM)
                time.sleep(5)
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            break
    p.wait()
    verdict.update(ended=time.strftime("%Y-%m-%d %H:%M:%S"), exit=p.returncode, killed=killed,
                   min_free=min_free, max_swap_mb=max_swap, peak_rss_mb=round(peak), elapsed_s=round(time.time() - t0))
    log(f"end exit={p.returncode} killed={killed} min_free={min_free}% max_swap={max_swap:.0f}MB peak_rss={peak:.0f}MB")
    json.dump(verdict, open(vpath, "w"), indent=1)
    sys.exit(0 if (p.returncode == 0 and not killed) else 1)


if __name__ == "__main__":
    main()
