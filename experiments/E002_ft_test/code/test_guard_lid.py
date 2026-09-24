"""Self-test for guard.py's lid pause (no model loaded; the child is a counter script).
Scripted lid states replace the real ioreg/pmset check. Checks:
  1. the child is SIGSTOPped while 'lid closed on battery' (its counter does not advance) and resumes after
  2. paused time is not counted as active time; the verdict records the pause
  3. a job paused past the wall-clock cap (3 x ceiling) is still killed promptly (SIGCONT before SIGTERM)
  4. an active-time ceiling kill still works with no pause
usage: python test_guard_lid.py   (exit 0 = all pass)"""
import json, os, sys, time, threading, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import guard as G

G.LOGS = os.path.join(G.LOGS, "selftest")   # keep test logs out of the queue status
os.makedirs(G.LOGS, exist_ok=True)
G.preflight = lambda log: True
G.free_pct = lambda: 80
PY = sys.executable
COUNTER = "import time,sys\nfor i in range(10**6):\n    print(i, flush=True); time.sleep(0.2)\n"
fails = []

def run(name, ceiling, lid_fn):
    G.lid_closed_on_battery = lid_fn
    sys.argv = ["guard.py", "--name", name, "--ceiling", str(ceiling), "--", PY, "-c", COUNTER]
    try:
        G.main()
    except SystemExit:
        pass
    v = json.load(open(os.path.join(G.LOGS, f"{name}.guard.json")))
    lines = open(os.path.join(G.LOGS, f"{name}.log")).read().split()
    return v, lines

# 1+2: lid closed from t=3s to t=9s, ceiling 12 s of ACTIVE time
t0 = [None]
def lid_a():
    if t0[0] is None:
        t0[0] = time.time()
    e = time.time() - t0[0]
    return 3 <= e < 9
v, lines = run("selftest_lid_pause", 12, lid_a)
n = len(lines)
# 12 s active at 5 counts/s ~ 60 counts; if the pause did not stop the child, ~90 counts (18 s wall)
if v["n_pauses"] != 1: fails.append(f"expected 1 pause, got {v['n_pauses']}")
if not (5 <= v["paused_s"] <= 7): fails.append(f"paused_s {v['paused_s']} not ~6")
if not (v["killed"] or "").startswith("active_time_ceiling"): fails.append(f"expected active ceiling kill, got {v['killed']}")
# the counter ticks 5/s only while the child runs: counts ~ 5 x active_s if paused, ~ 5 x (active_s + paused_s) if not
if abs(n - 5 * v["active_s"]) > 0.15 * 5 * v["active_s"] + 3: fails.append(f"child counted {n}, expected ~{5*v['active_s']} (5 x active)")
if v["elapsed_s"] < 16: fails.append(f"elapsed {v['elapsed_s']} < 16 s: pause not excluded from the ceiling")
print("case1", {k: v[k] for k in ("killed", "elapsed_s", "active_s", "paused_s", "n_pauses")}, "counts", n)

# 3: lid closed forever after 2 s, ceiling 4 -> wall cap 12 s must kill a STOPPED child
t0 = [None]
def lid_b():
    if t0[0] is None:
        t0[0] = time.time()
    return time.time() - t0[0] >= 2
v, lines = run("selftest_lid_wallcap", 4, lid_b)
if not (v["killed"] or "").startswith("wall_clock_ceiling_12"): fails.append(f"expected wall cap kill, got {v['killed']}")
# wall cap 12 s + up to 5 s check cadence + 5 s SIGTERM grace + 1 s slack
if v["elapsed_s"] > 23: fails.append(f"stopped child not killed promptly ({v['elapsed_s']} s)")
if len(lines) > 5 * v["active_s"] + 5: fails.append(f"child kept counting while paused ({len(lines)})")
print("case3", {k: v[k] for k in ("killed", "elapsed_s", "active_s", "paused_s", "n_pauses")}, "counts", len(lines))

# 4: never closed, ceiling 3 -> active ceiling kill at ~3-5 s
v, lines = run("selftest_lid_none", 3, lambda: False)
if not (v["killed"] or "").startswith("active_time_ceiling"): fails.append(f"expected active ceiling, got {v['killed']}")
if v["n_pauses"] != 0: fails.append("paused without a closed lid")
print("case4", {k: v[k] for k in ("killed", "elapsed_s", "active_s", "paused_s", "n_pauses")}, "counts", len(lines))

print("FAILS:", fails if fails else "none")
sys.exit(1 if fails else 0)
