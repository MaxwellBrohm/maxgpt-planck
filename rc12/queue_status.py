"""RC-12 dev queue status (queue_dev_baselines.sh; notes STEP 9 QUEUE). Two modes.
Line mode: one status line per run of one dev_batch.py process (runs "dev" = <seed>/, "owncf" = <seed>_owncf/):
  <model> <render> <seed> <engine> <run> start=<iso> end=<iso> exit=<code> done=<0|1> convs=<n> replies=<n>
  stops=<reason:count,...> secs=<run seconds> lock_secs=<process seconds under the lock> [note=<text>]
  start / end / secs of a finished run come from its meta.json (the run itself, model load excluded); an unfinished
  run gets the process bounds and replies=0. Replies and stop reasons are counted in its transcripts.jsonl.
Progress mode (--progress): "complete X of Y runs" over the queue's plan, then per model and render the finished
  runs, the run in progress (a .partial directory) and the mean run seconds so far.
Pure Python, no model."""
import argparse
import collections
import json
import os
import time

RUNS = {"dev": "", "owncf": "_owncf"}


def iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t))


def slug(model):
    return model.rstrip("/").split("/")[-1]


def counts(d):
    """(conversations, replies, Counter of stop reasons) over d/transcripts.jsonl."""
    convs, stops = 0, collections.Counter()
    with open(os.path.join(d, "transcripts.jsonl")) as f:
        for line in f:
            row = json.loads(line)
            convs += 1
            for t in row["turns"]:
                stops[t["stop"]] += 1
    return convs, sum(stops.values()), stops


def run_line(root, model, render, seed, engine, run, code, t0, t1, note=""):
    d = os.path.join(root, slug(model), render, seed + RUNS[run])
    done = os.path.exists(os.path.join(d, "DONE"))
    convs, replies, stops, secs, start, end = 0, 0, collections.Counter(), None, t0, t1
    if done:
        meta = json.load(open(os.path.join(d, "meta.json")))
        convs, replies, stops = counts(d)
        secs = meta["seconds"]
        end = time.mktime(time.strptime(meta["finished"], "%Y-%m-%d %H:%M:%S"))
        start = end - secs
    stop_s = ",".join(f"{k}:{v}" for k, v in sorted(stops.items())) or "-"
    out = (f"{slug(model)} {render} {seed} {engine} {run} start={iso(start)} end={iso(end)} exit={code} "
           f"done={int(done)} convs={convs} replies={replies} stops={stop_s} secs={secs if secs is not None else '-'} "
           f"lock_secs={int(t1) - int(t0)}")
    return out + (f" note={note.replace(' ', '_')}" if note else "")


def progress(root, models, renders, seeds):
    total = done = 0
    lines = []
    for m in models:
        for r in renders:
            fin, secs, partial = 0, [], []
            for s in seeds:
                for run, suf in RUNS.items():
                    d = os.path.join(root, slug(m), r, s + suf)
                    total += 1
                    if os.path.exists(os.path.join(d, "DONE")):
                        fin += 1
                        if run == "dev":
                            secs.append(json.load(open(os.path.join(d, "meta.json")))["seconds"])
                    elif os.path.isdir(d + ".partial"):
                        partial.append(s + suf)
            done += fin
            mean = f"{sum(secs) / len(secs):.0f}" if secs else "-"
            lines.append(f"  {slug(m):32s} {r:8s} {fin}/{2 * len(seeds)} done, dev run mean {mean} s"
                         + (f", in progress: {' '.join(partial)}" if partial else ""))
    return [f"complete {done} of {total} runs"] + lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--models", default="")
    ap.add_argument("--renders", default="template plain")
    ap.add_argument("--seeds", default="greedy 1 2 3")
    ap.add_argument("--model")
    ap.add_argument("--render")
    ap.add_argument("--seed")
    ap.add_argument("--engine", default="-")
    ap.add_argument("--exit", default="-")
    ap.add_argument("--t0", type=float, default=0)
    ap.add_argument("--t1", type=float, default=0)
    ap.add_argument("--runs", default="dev owncf")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    root = os.path.expanduser(a.root)
    if a.progress:
        print("\n".join(progress(root, a.models.split(), a.renders.split(), a.seeds.split())))
        return 0
    for run in a.runs.split():
        print(run_line(root, a.model, a.render, a.seed, a.engine, run, a.exit, a.t0, a.t1, a.note), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
