"""Mutation test of queue_dev_baselines.sh + queue_status.py against test_queue.py (Max's rule: every test claim is
watched failing; notes STEP 9 QUEUE). Each mutant replaces one text in a scratch copy of rc12 (the *.py files, the
queue script and dev/), then test_queue.py runs there; KILLED = it exits non-zero. The unmutated copy must pass;
malformed = the text is not found exactly once. No model, Mac only.
  python3 -B mutation_queue.py    (writes logs/mutation_queue.txt; exit 1 unless the baseline passes and all die)"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
Q, S = "queue_dev_baselines.sh", "queue_status.py"
E004 = os.path.join(HERE, "..", "experiments", "E004_general_updating", "code")   # quick_shortcuts' import, by path
ENV = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (E004, os.environ.get("PYTHONPATH")) if p))
MUTANTS = [
    (Q, '[ -e "$d/DONE" ] && [ -e "${d}_owncf/DONE" ] && continue', 'false && continue', "finished seeds rerun"),
    (Q, 'RENDERS=${Q_RENDERS:-template plain}', 'RENDERS=${Q_RENDERS:-plain template}', "plain before template"),
    (Q, 'SEEDS=${Q_SEEDS:-greedy 1 2 3}', 'SEEDS=${Q_SEEDS:-1 2 3 greedy}', "greedy last"),
    (Q, 'SEEDS=${Q_SEEDS:-greedy 1 2 3}', 'SEEDS=${Q_SEEDS:-greedy 1 2}', "seed 3 missing"),
    (Q, 'MODELS=${Q_MODELS:-"Qwen/Qwen2.5-0.5B-Instruct\ntiiuae/Falcon-H1-Tiny-90M-Instruct',
     'MODELS=${Q_MODELS:-"tiiuae/Falcon-H1-Tiny-90M-Instruct\nQwen/Qwen2.5-0.5B-Instruct', "comparator not first"),
    (Q, ' Qwen/Qwen3.5-0.8B LiquidAI/LFM2-2.6B\n', ' LiquidAI/LFM2-2.6B Qwen/Qwen3.5-0.8B\n', "core not by size"),
    (Q, ' LiquidAI/LFM2-700M LiquidAI/LFM2-1.2B"}', ' LiquidAI/LFM2-700M"}', "an extra missing"),
    (Q, '--seeds "$seed"', '--seeds greedy', "the seed does not reach the run"),
    (Q, '--render "$r" --seeds', '--render template --seeds', "the render does not reach the run"),
    (Q, '"$LOCKS/gpu.lock" bash', '"$LOCKS/other.lock" bash', "another lock file"),
    (Q, '-E 75 -w "$LOCK_WAIT"', '-E 75 -w 60', "lock wait not 7200 s"),
    (Q, 'for try in 1 2 3; do', 'for try in 1; do', "no retry on a lock timeout"),
    (Q, '[ $rc -ne 75 ] && break', 'break', "a lock timeout is not retried"),
    (Q, '[ $rc -ne 0 ] && failed=$seed', 'true', "a failure does not skip the rest of the render"),
    (Q, '    failed=""\n', '', "one failure skips every later render and model"),
    (Q, 'if [ -e "$LOGS/$NAME.STOP" ]; then', 'if false; then', "STOP file ignored"),
    (Q, 'if [ -z "$eng" ]; then status', 'if false; then status', "a model without an engine is run"),
    (Q, '[ -e "$d/DONE" ] || runs=dev;', 'runs=dev;', "status for a run finished earlier"),
    (Q, '[ -e "${d}_owncf/DONE" ] || runs="$runs owncf"', 'true', "no status for the own-cf twin"),
    (Q, '[ -n "$Q_LIMIT" ] && extra="$extra --limit $Q_LIMIT"', 'true', "(test speed) limit dropped: full dev"),
    (Q, '  > "$LOGS/$NAME.DONE" 2>&1', '  > /dev/null 2>&1', "no DONE marker"),
    (Q, '"$eng:$m"', '"fake:IDEAL"', "the engine or model does not reach the run"),
    (Q, "print(json.load(open(sys.argv[1]))", "print(json.load(open(sys.argv[1] + 'x'))", "engines.json unread"),
    (S, 'stops[t["stop"]] += 1', 'stops[t["stop"]] += 1 if t["i"] > 1 else 0', "first turn not counted"),
    (S, 'RUNS = {"dev": "", "owncf": "_owncf"}', 'RUNS = {"dev": "_owncf", "owncf": ""}', "runs swapped"),
    (S, 'start = end - secs', 'start = end', "start is the end"),
    (S, 'sorted(stops.items())', 'stops.items()', "stop reasons unsorted"),
    (S, 'total += 1', 'total += 2', "progress total wrong"),
    (S, 'elif os.path.isdir(d + ".partial"):', 'elif False:', "run in progress not shown"),
    (S, 'lock_secs={int(t1) - int(t0)}', 'lock_secs={int(t1)}', "lock seconds wrong"),
]


def copy_tree(dst):
    for f in os.listdir(HERE):
        if f.endswith(".py") or f == Q:
            shutil.copy(os.path.join(HERE, f), dst)
    shutil.copytree(os.path.join(HERE, "dev"), os.path.join(dst, "dev"))


def run(src_file=None, old=None, new=None):
    d = tempfile.mkdtemp(prefix="rc12_mq_")
    copy_tree(d)
    if src_file:
        p = os.path.join(d, src_file)
        text = open(p).read()
        if text.count(old) != 1:
            return "malformed"
        open(p, "w").write(text.replace(old, new))
    try:
        rc = subprocess.run([sys.executable, "-B", os.path.join(d, "test_queue.py")], capture_output=True,
                            timeout=240, env=ENV).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    shutil.rmtree(d, ignore_errors=True)
    return "pass" if rc == 0 else f"fail({rc})"


def main():
    t = time.time()
    out = [f"baseline: {run()}"]
    killed = 0
    for f, old, new, why in MUTANTS:
        r = run(f, old, new)
        k = r.startswith("fail")
        killed += k
        out.append(f"{'KILLED ' if k else 'SURVIVED' if r == 'pass' else r.upper():9s} {f}: {why}")
        print(out[-1], flush=True)
    out.append(f"MUTANTS killed {killed}/{len(MUTANTS)}; {time.time() - t:.0f} s")
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    open(os.path.join(HERE, "logs", "mutation_queue.txt"), "w").write("\n".join(out) + "\n")
    print(out[0], out[-1], sep="\n")
    return 0 if out[0] == "baseline: pass" and killed == len(MUTANTS) else 1


if __name__ == "__main__":
    sys.exit(main())
