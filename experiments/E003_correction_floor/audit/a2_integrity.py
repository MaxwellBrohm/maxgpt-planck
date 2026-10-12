"""E003 audit A2: run completeness, guard records, commands, file timing and queue attempts (own code, stdlib).

For every baseline, LR-search and scored run: run.json fields, record counts against the pre-registered sizes and
against the actual line counts, 400 finite losses, the loss self-check, the final guard record (exit 0, not killed),
the command against the one queue_e003.sh builds, every out/ file's mtime inside the final guard window, and the
attempt history in logs/queue.txt (killed attempts, exactly one completed attempt, nothing re-run after it).
usage: python3 -B a2_integrity.py
"""
import glob, json, math, os, re, time
from collections import defaultdict
from datetime import datetime

AUD = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(AUD)
OUT, LOGS = os.path.join(EXP, "out"), os.path.join(EXP, "logs")
PY = os.path.expanduser("~/Documents/Projects/video-editor/.venv/bin/python")
MODELS = [("ts1m", "roneneldan/TinyStories-1M"), ("p14m", "EleutherAI/pythia-14m"), ("ts3m", "roneneldan/TinyStories-3M"),
          ("p31m", "EleutherAI/pythia-31m"), ("ts8m", "roneneldan/TinyStories-8M"), ("p70m", "EleutherAI/pythia-70m"),
          ("ts33m", "roneneldan/TinyStories-33M"), ("p160m", "EleutherAI/pythia-160m")]
COMMON = ["--bs", "4", "--accum", "4", "--max-len", "768", "--grad-ckpt"]
EVAL = {"new/plain": 1728, "extra/plain": 192, "old/plain": 1116, "uprobe/plain": 576, "khard/plain": 80,
        "kbig/plain": 441, "cross/plain": 384}
DEV = {"dev/plain": 320}


def jobs():
    """-> list of (job name, model, tag, expected args after the model id, expected eval_counts, steps)."""
    out = []
    for short, mid in MODELS:
        out.append((f"base_{short}", mid, "base", COMMON + ["--seed", "0", "--steps", "0", "--sets", "eval,dev"],
                    {**EVAL, **DEV}, 0))
        s = mid.replace("/", "__")
        for p in sorted(glob.glob(os.path.join(OUT, f"{s}__lr*_s0__run.json"))):
            tag = os.path.basename(p).split("__")[2]
            lr = tag[2:-3]
            out.append((f"lr_{short}_{lr}", mid, tag, COMMON + ["--seed", "0", "--sets", "dev", "--save", "0", "--lr", lr],
                        DEV, 400))
        lr = json.load(open(os.path.join(OUT, f"{s}__s1__run.json")))["lr"]
        for k in (1, 2, 3):
            out.append((f"ft_{short}_s{k}", mid, f"s{k}",
                        COMMON + ["--sets", "eval", "--save", "1", "--seed", str(k), "--lr", f"{lr:.0e}"], EVAL, 400))
    return out


def ts(s):
    return time.mktime(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple())


def queue_history():
    hist = defaultdict(list)
    for line in open(os.path.join(LOGS, "queue.txt"), errors="replace"):
        m = re.match(r"(\S+ \S+) (\S+) exit=(\S+) killed=(\S+)", line)
        if m and "already finished" not in line:
            hist[m.group(2)].append((m.group(1), m.group(3), m.group(4)))
    return hist


def main():
    problems, killed_total, rows = [], defaultdict(list), []
    hist = queue_history()
    J = jobs()
    for name, mid, tag, args, counts, steps in J:
        stem = os.path.join(OUT, f"{mid.replace('/', '__')}__{tag}")
        meta = json.load(open(stem + "__run.json"))
        g = json.load(open(os.path.join(LOGS, f"{name}.guard.json")))
        want = [PY, "-B", "e003_ft_test.py", mid] + args + ["--tag", tag]
        if g["cmd"] != want:
            problems.append(f"{name}: guard cmd differs: {g['cmd'][4:]} vs {want[4:]}")
        if g["exit"] != 0 or g["killed"] is not None:
            problems.append(f"{name}: final guard record exit {g['exit']} killed {g['killed']}")
        if meta.get("fatal"):
            problems.append(f"{name}: fatal {meta['fatal']}")
        if meta.get("steps") != steps or meta.get("tag") != tag or meta.get("model") != mid:
            problems.append(f"{name}: steps/tag/model {meta.get('steps')} {meta.get('tag')} {meta.get('model')}")
        if meta.get("eval_counts") != counts:
            problems.append(f"{name}: eval_counts {meta.get('eval_counts')}")
        L = meta.get("losses") or []
        if steps and (len(L) != 400 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in L)):
            problems.append(f"{name}: {len(L)} losses or non-finite")
        if steps:
            sc = meta.get("loss_selfcheck") or {}
            d = abs(sc.get("hf", 1e9) - sc.get("answer_only", -1e9))
            if d > 1e-3:
                problems.append(f"{name}: loss self-check gap {d}")
            if meta.get("lock_in_step") is not None:
                rows.append(f"{name}: lock_in_step {meta['lock_in_step']}")
        # record files: line counts and mtimes inside the final guard window
        t0, t1 = ts(g["started"]) - 1, ts(g["ended"]) + 2
        files = glob.glob(stem + "__*")
        for key, n in counts.items():
            s_, r_ = key.split("/")
            p = f"{stem}__{s_}__{r_}.jsonl"
            if not os.path.exists(p):
                problems.append(f"{name}: missing {os.path.basename(p)}")
                continue
            k = sum(1 for l in open(p) if l.strip())
            if k != n:
                problems.append(f"{name}: {os.path.basename(p)} has {k} records, want {n}")
        extra = {os.path.basename(f) for f in files} - {f"{os.path.basename(stem)}__{k.replace('/', '__')}.jsonl" for k in counts} \
            - {os.path.basename(stem) + "__run.json"}
        if extra:
            problems.append(f"{name}: unexpected files {sorted(extra)}")
        for f in files:
            mt = os.path.getmtime(f)
            if not (t0 <= mt <= t1):
                problems.append(f"{name}: {os.path.basename(f)} mtime {time.ctime(mt)} outside {g['started']}..{g['ended']}")
        if tag.startswith("s"):
            wd = os.path.join(EXP, "weights", f"{mid.replace('/', '__')}__{tag}")
            wf = [f for f in glob.glob(os.path.join(wd, "*")) if f.endswith(".safetensors") or f.endswith(".bin")]
            if not wf:
                problems.append(f"{name}: no weights file in {wd}")
            for f in wf:
                if not (t0 <= os.path.getmtime(f) <= t1):
                    problems.append(f"{name}: weights mtime outside final window")
        # queue history
        h = hist.get(name, [])
        done = [x for x in h if x[1] == "0" and x[2] == "None"]
        if len(done) != 1:
            problems.append(f"{name}: {len(done)} completed attempts in queue.txt")
        elif h[-1] != done[0]:
            problems.append(f"{name}: an attempt after the completed one")
        elif abs(ts(done[0][0]) - ts(g["ended"])) > 2:
            problems.append(f"{name}: queue completion {done[0][0]} != guard end {g['ended']}")
        for x in h:
            if x not in done:
                killed_total[name].append(x[2])
    stray = sorted(set(hist) - {j[0] for j in J} - {f"dry_{s}" for s, _ in MODELS})
    if stray:
        problems.append(f"queue jobs not accounted for: {stray}")
    print(f"jobs checked: {len(J)} (base {sum(1 for j in J if j[2]=='base')}, lr {sum(1 for j in J if j[2].startswith('lr'))},"
          f" seeds {sum(1 for j in J if re.fullmatch(r's[123]', j[2]))})")
    n = sum(len(v) for v in killed_total.values())
    sw = sum(1 for v in killed_total.values() for x in v if x.startswith("swap_grew"))
    print(f"killed attempts: {n} (swap {sw}, other {n - sw})")
    for k, v in killed_total.items():
        print(f"  {k}: {len(v)} {sorted(set(re.sub(r'_[0-9]+MB', '', x) for x in v))}")
    for r in rows:
        print("  " + r)
    print(f"PROBLEMS: {len(problems)}")
    for p in problems:
        print("  " + p)


if __name__ == "__main__":
    main()
