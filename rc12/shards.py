"""RC-12 dev-run shards (rc12/notes.txt STEP 14X, 2026-10-10 deviation for cRia-LM-75M-Instruct). engines.json may
name "shards": N (an int >= 1) for a model; with N > 1 every run of that model (a seed's dev run and its --own-cf
twin) is played by N dev_batch.py processes, process i taking the records i, i + N, i + 2N, ... of the run's record
list (round robin, so every family is spread over the shards and the shards take about the same time). Each process
plays its slice of BOTH runs of its seed and writes each to <final>.shards/<i>of<N>/ (transcripts.jsonl, part.json,
DONE; the same durable publish as a whole run). The process that finds all N parts of a run DONE merges them: the
rows back in the record order (record k from part k mod N, position k div N), transcripts.jsonl, scores.jsonl and
meta.json written exactly as one dev_batch.py process writes them, meta.json adding shards, shard_seconds and
shard_batches (seconds = their sum, batches = theirs in shard order), then DONE, the publish, and <final>.shards/
removed. The merge refuses (and leaves every part in place) unless every part names the same N, the expected index,
the run's own ids in order, and the same run config (engine, dtype, decoding, audit, versions, executable, ...).
What changes against one process: only which conversations share a lockstep batch. Sampling draws are per
conversation and turn (hf_responder.conv_seed), so with equal logits a sharded run is the unsharded one; batch makeup
can move numerics only where padding does, which the model's parity measures (cRia: IDENTICAL, batched vs serial).
Pure Python, no model."""
import json
import os
import shutil
import time

import score as S


def parse(text):
    """'I/N' -> (I, N) with N >= 2 and 0 <= I < N; ValueError otherwise."""
    a, sep, b = str(text).partition("/")
    if not sep or not a.isdigit() or not b.isdigit():
        raise ValueError(f"--shard {text!r}: I/N, two integers")
    i, n = int(a), int(b)
    if n < 2 or not 0 <= i < n:
        raise ValueError(f"--shard {text!r}: N >= 2 and 0 <= I < N")
    return i, n


def of_engines(model_id, path):
    """engines.json "shards" for model_id: 1 when absent; ValueError unless an int >= 1."""
    e = json.load(open(path))["models"].get(model_id, {}) if os.path.exists(path) else {}
    v = e.get("shards", 1)
    if type(v) is not int or v < 1:
        raise ValueError(f"engines.json shards {v!r} for {model_id}: an int >= 1")
    return v


def part_dir(final, i, n):
    return os.path.join(final + ".shards", f"{i}of{n}")


def part_done(final, i, n):
    return os.path.exists(os.path.join(part_dir(final, i, n), "DONE"))


def all_done(final, n):
    return all(part_done(final, i, n) for i in range(n))


def take(recs, i, n):
    return recs[i::n]


def interleave(parts):
    """the rows of N parts (part i holding records i, i + N, ...) back in record order."""
    n, out = len(parts), []
    for k in range(sum(len(p) for p in parts)):
        out.append(parts[k % n][k // n])
    return out


def write_part(tmp, rows, part):
    """a part's files in tmp (the caller publishes tmp as the part dir): transcripts.jsonl, part.json, DONE."""
    os.makedirs(tmp, exist_ok=True)
    with open(os.path.join(tmp, "transcripts.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    json.dump(part, open(os.path.join(tmp, "part.json"), "w"), indent=1)
    open(os.path.join(tmp, "DONE"), "w").close()


SAME = ("config", "versions", "model", "render", "seed", "own_cf", "ctx")


def read_parts(final, n, recs):
    """[(part.json, rows)] for the N parts of final; SystemExit naming what disagrees (nothing is removed)."""
    out = []
    for i in range(n):
        d = part_dir(final, i, n)
        p = json.load(open(os.path.join(d, "part.json")))
        rows = [json.loads(line) for line in open(os.path.join(d, "transcripts.jsonl"))]
        want = [r["id"] for r in take(recs, i, n)]
        bad = [k for k in SAME if out and p.get(k) != out[0][0].get(k)]
        if p.get("shard") != [i, n] or p.get("ids") != want or [r["id"] for r in rows] != want or bad:
            raise SystemExit(f"shards of {final}: part {i}of{n} does not fit (shard {p.get('shard')}, ids "
                             f"{len(p.get('ids') or [])} of {len(want)} expected, rows {len(rows)}, differs in {bad}); "
                             "nothing merged")
        out.append((p, rows))
    return out


def merge(final, n, recs, publish):
    """merge the N DONE parts of final into the run dir final (publish(tmp, final) makes it durable), then remove
    final.shards. Returns the merged meta."""
    parts = read_parts(final, n, recs)
    rows = interleave([r for _, r in parts])
    tmp = final + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    with open(os.path.join(tmp, "transcripts.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    with open(os.path.join(tmp, "scores.jsonl"), "w") as f:
        for line in S.score_lines(rows):
            f.write(json.dumps(line) + "\n")
    p0 = parts[0][0]
    summ = S.summarize(rows)
    meta = dict(p0["config"], model=p0["model"], render=p0["render"], seed=p0["seed"], own_cf=p0["own_cf"],
                conversations=len(rows), ctx=p0["ctx"], seconds=round(sum(p["seconds"] for p, _ in parts), 1),
                batches=[b for p, _ in parts for b in p["batches"]], versions=p0["versions"],
                R_ungated=summ.get("R_ungated"), loop_rate=summ.get("loop_rate"),
                finished=time.strftime("%Y-%m-%d %H:%M:%S"), shards=n,
                shard_seconds=[p["seconds"] for p, _ in parts], shard_batches=[p["batches"] for p, _ in parts],
                shard_finished=[p["finished"] for p, _ in parts])
    json.dump(meta, open(os.path.join(tmp, "meta.json"), "w"), indent=1)
    open(os.path.join(tmp, "DONE"), "w").close()
    publish(tmp, final)
    shutil.rmtree(final + ".shards", ignore_errors=True)
    print(f"{final}: merged {n} shards, {len(rows)} conversations, {meta['seconds']} s, R_ungated {meta['R_ungated']}",
          flush=True)
    return meta
