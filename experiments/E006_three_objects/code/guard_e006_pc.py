"""E006 single-job guard on the PC (WSL, RTX 5070). The queue runs every GPU job as
  flock -w 7200 ~/planck/locks/gpu.lock python guard_e006_pc.py --name NAME --ceiling S [--sha-from RUN.json] -- cmd
Before start: refuses when ../logs/NAME.log or NAME.guard.json exists (logs are never overwritten); waits (up to
--start-wait s, polling every 30 s; NAME.wait_<time>.log) while another E006/E005 model process runs, or while the GPU
is hotter than 75 C or has less than NEED_MIB free (pc/wsl/gpuguard.may_start with that start limit); then refuses
(NAME.refused_<time>.json, exit 3, no NAME.log: the queue may retry the same name).
During the run, every 5 s: pc/wsl/gpuguard's Guard (temp, gpu_mem, host_mem, smi verdicts, its default thresholds)
and the wall-clock ceiling. A verdict or the ceiling kills the child's process group (TERM, then KILL after 30 s).
Writes ../logs/NAME.log (child output, the home directory written as ~), NAME.guard.log (samples) and
NAME.guard.json (command, start, end, elapsed_s, exit, killed, verdict, peaks, weights_sha256 from --sha-from).
Exit code: the child's; 9 if killed; 3 if refused before start."""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(EXP)), "pc", "wsl"))
MODEL_PROCS = ("e006_ft_test.py", "chat_e006.py", "e005_ft_test.py", "run_chat.py", "numerics_e006.py")
NEED_MIB, POLL = 6000.0, 5.0
HOME = os.path.expanduser("~")


def scrub(s):
    return s.replace(HOME, "~")


def host_avail_mib():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def other_model_procs(me):
    out = subprocess.run(["ps", "-eo", "pid=,ppid=,args="], capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and "python" in parts[2] and any(p in parts[2] for p in MODEL_PROCS):
            if int(parts[0]) != me and "guard_e006_pc.py" not in parts[2]:
                rows.append(scrub(parts[2])[:160])
    return rows


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--" not in argv:
        raise SystemExit("usage: guard_e006_pc.py --name N --ceiling S [--start-wait S] [--sha-from F] -- cmd...")
    i = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ceiling", type=float, required=True)
    ap.add_argument("--start-wait", type=float, default=7200)
    ap.add_argument("--sha-from", default="")
    ap.add_argument("--logs", default=os.path.join(EXP, "logs"))
    a, cmd = ap.parse_args(argv[:i]), argv[i + 1:]
    import gpuguard as GG
    base = os.path.join(a.logs, a.name)
    if any(os.path.exists(base + x) for x in (".log", ".guard.json", ".guard.log")):
        print(f"REFUSE: {scrub(base)}.* exists (logs are never overwritten)", file=sys.stderr)
        return 3
    os.makedirs(a.logs, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    wlog = None
    rec = {"name": a.name, "command": scrub(" ".join(cmd)), "ceiling_s": a.ceiling, "killed": None, "verdict": None}
    cfg = GG.GuardConfig()
    t_wait = time.time()
    while True:
        others = other_model_procs(os.getpid())
        s = GG.query()
        total = (s or {}).get("mem_total_mib") or 12227.0
        ok, why = GG.may_start(s, cfg.merged({"start_max_mem_mib": total - NEED_MIB}))
        if not others and ok:
            break
        wlog = wlog or open(f"{base}.wait_{stamp}.log", "x")
        wlog.write(f"{time.strftime('%F %T')} WAIT others={others[:2]} gpu={why}\n")
        wlog.flush()
        if time.time() - t_wait > a.start_wait:
            rec.update(killed="preflight_refused", verdict=f"others={others[:2]} gpu={why}")
            json.dump(rec, open(f"{base}.refused_{stamp}.json", "x"), indent=1)
            return 3
        time.sleep(30)
    glog = open(base + ".guard.log", "x")
    guard, t0 = GG.Guard(cfg), time.time()
    rec["start"] = time.strftime("%F %T")
    logf = open(base + ".log", "x")
    child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                             start_new_session=True)

    def pump():
        for line in child.stdout:
            logf.write(scrub(line))
            logf.flush()
    th = threading.Thread(target=pump, daemon=True)
    th.start()
    while child.poll() is None:
        time.sleep(POLL)
        s, host = GG.query(), host_avail_mib()
        v = guard.check(s, host)
        el = time.time() - t0
        glog.write(f"{time.strftime('%F %T')} el={el:.0f}s temp={(s or {}).get('temp_c')} "
                   f"mem={(s or {}).get('mem_used_mib')} util={(s or {}).get('util_pct')} host_avail={host} v={v}\n")
        glog.flush()
        if v is None and el > a.ceiling:
            v = "ceiling"
        if v and child.poll() is None:
            rec.update(killed=v, verdict=v)
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
            break
    child.wait()
    th.join(10)
    logf.close()
    rec.update(end=time.strftime("%F %T"), elapsed_s=round(time.time() - t0, 1), exit=child.returncode,
               peak_temp_c=guard.peak["temp_c"], peak_gpu_mem_mib=guard.peak["mem_used_mib"])
    if a.sha_from and os.path.exists(a.sha_from):
        try:
            rec["weights_sha256"] = json.load(open(a.sha_from)).get("weights_sha256")
        except ValueError:
            rec["weights_sha256"] = None
    json.dump(rec, open(base + ".guard.json", "x"), indent=1)
    return 9 if rec["killed"] else child.returncode


if __name__ == "__main__":
    sys.exit(main())
