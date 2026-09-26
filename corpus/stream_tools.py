"""Side tools of stream_core.py: the dry run, sealing open shards on demand, the shard list
(SPEC.json) that neardedup_lsh.py takes, and the late-commit audit. All of them run without the
heavy lock (light work).

    python corpus/stream_tools.py late-commits --out STAGE1 [--manifest PATH]
"""
import argparse
import json
import os
import sys

import stream_core as C
import stream_fetch as F
import stream_rank as R
import stream_recover as SR
import stream_work as W


def dry_run(cfg, log=print):
    """What a run would do now: pending files, bytes, readers; changes nothing."""
    cfg = C.check_config(cfg)
    manifest, _, _ = C._load(cfg)
    done, lp = set(), os.path.join(cfg["out"], "LEDGER.jsonl")
    if os.path.exists(lp):
        with open(lp, encoding="utf-8") as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("event") == "file_done":
                    done.add(ev["fid"])
    items, unready = C.select(cfg, manifest, done)
    by = {}
    for it in items:
        b = by.setdefault(it["source"], {"files": 0, "gb": 0.0, "local": 0, "reader": it["reader"]})
        b["files"] += 1
        b["gb"] = round(b["gb"] + it["entry"]["size"] / 1e9, 3)
        b["local"] += it["local"]
    for s, b in by.items():
        log(f"{s:14s} {b['files']:4d} files {b['gb']:8.2f} GB local {b['local']} {b['reader']}")
    log(f"done already {len(done)}; unready {unready}; free {F.disk_free(cfg['raw']) / F.GIB:.0f} GiB")
    return {"by_source": by, "unready": unready, "done": len(done)}


def neardedup_spec(out):
    """The shard list neardedup_lsh.py takes (SPEC.json), for the sealed shards that have a
    <stem>.mh.npy, in commit order: tier, source, shard index."""
    ledger = [json.loads(x) for x in open(os.path.join(out, "LEDGER.jsonl"), encoding="utf-8")]
    tier = {e["source"]: e["tier"] for e in ledger if e["event"] == "file_done"}
    spec = []
    for e in ledger:
        if e["event"] == "shard_sealed" and "mh" in (e.get("sidecars") or {}):
            spec.append({"stem": os.path.join(out, e["shard"].rsplit(".jsonl.", 1)[0]),
                         "tier": tier[e["source"]], "label": e["source"], "frozen": False,
                         "include": None})
    return sorted(spec, key=lambda s: (s["tier"], s["label"], s["stem"]))


def seal_open(cfg):
    """Seal the open shard of every source (or of --sources) after recovering the output
    directory; a source whose last files failed for good would otherwise never be sealed, and
    near-dedup reads sealed shards only. A later file of the source starts a new shard.
    -> the shards sealed."""
    cfg = C.check_config(cfg)
    _, by_fid, _ = C._load(cfg)
    ctx, done = W.Context(cfg), []
    try:
        C.check_sidecars(ctx, cfg)
        SR.recover(ctx, by_fid)
        for src in sorted(ctx.states):
            if cfg["sources"] and src not in cfg["sources"]:
                continue
            rec = ctx.shards(src).seal_open()
            if rec:
                ctx.ledger.append("shard_sealed", **rec)
                SR.finalize_sidecars(cfg["out"], rec, ctx.dtypes(src))
                done.append(rec["shard"])
    finally:
        ctx.close()
    return done


def late_commits(out, manifest_path):
    """Files committed after a file ranked below them (a retry in a later run, or a narrowed run),
    in ledger order, under the stream_rank order ("rank") and plain manifest order ("manifest",
    the order before stream_rank.py). For each: how many lower-ranked files were committed first,
    its dup_exact and dup_id drops, and whether the committing code was rank-aware
    (stream_rank.py in the run's code hashes). A late file committed by older code may have lost
    up to dup_exact + dup_id documents to lower-ranked copies; a rank-aware one lost none."""
    with open(manifest_path) as f:
        files = json.load(f)["files"]
    ranks = {"rank": R.manifest_ranks({"files": files}),
             "manifest": {R.fid_of(e): i for i, e in enumerate(files)}}
    before, aware, out_rows, events = [], False, [], []
    with open(os.path.join(out, "LEDGER.jsonl"), encoding="utf-8") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                break                          # a torn last line of a running job
    for e in events:
        if e["event"] == "run_start":
            aware = "stream_rank.py" in (e.get("code_sha256") or {})
        if e["event"] != "file_done" or e["fid"] not in ranks["rank"]:
            continue
        late = {k: n for k, rk in ranks.items()
                if (n := sum(rk[x] > rk[e["fid"]] for x in before))}
        before.append(e["fid"])
        if late:
            drops = e["stats"]["dropped"]
            out_rows.append({
                "fid": e["fid"], "late_after": late, "rank_aware": aware,
                "dup_exact": drops.get("dup_exact", {}).get("docs", 0),
                "dup_id": drops.get("dup_id", {}).get("docs", 0),
                "displaced": sum(len(d["keys"]) + len(d["ids"]) for d in e.get("displaced") or ())})
    return out_rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="stream_core side tools")
    ap.add_argument("tool", choices=["late-commits"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=C.DEFAULTS["manifest"])
    a = ap.parse_args(argv)
    rows = late_commits(os.path.expanduser(a.out), os.path.expanduser(a.manifest))
    for r in rows:
        print(json.dumps(r))
    old = [r for r in rows if not r["rank_aware"]]
    print(f"{len(rows)} late commits; {len(old)} by code before stream_rank.py, which may have "
          f"dropped up to {sum(r['dup_exact'] + r['dup_id'] for r in old):,} documents")
    return 0


if __name__ == "__main__":
    sys.exit(main())
