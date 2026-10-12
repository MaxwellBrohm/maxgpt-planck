"""stream_core.py: resumable stream-and-delete runner for the strict-open core v0.
CORPUS.md 3.1 gates and 3.2 step 1 (exact dedup); the plan's pass A. On the PC only.

    python corpus/stream_core.py --out ~/planck/data/core_v0 --raw ~/planck/data/raw/core_v0 \\
        --starter ~/planck/data/raw/starter --tiers 1 [--sources irc ...] [--dry-run]

For every selected file of the manifest (corpus/fetch/core_v0_manifest.json, role core), in rank
order (stream_rank.py: tier, source priority, manifest order = dedup priority): download with
resume and verify the checksum (stream_fetch.py), extract with the existing readers and hygiene in
a worker, then in the parent drop exact duplicates of anything a better-ranked file committed
(text sha1 or doc id, global across files and runs, stream_ledger.KeyStore), append the rest to
OUT/<source>/ shards of about 256 MB (stream_io.py), commit, record the file in OUT/LEDGER.jsonl,
and delete the raw file. A file retried in a later run still wins over lower-ranked copies.

Optional per-document sidecars: --sidecar mh=stream_work:mh_row writes neardedup.py's MinHash
rows as <shard stem>.mh.npy once a shard is sealed; --neardedup-spec PATH then writes the SPEC.json
neardedup_lsh.py takes. Modules: stream_fetch (download, verify, disk guard, prefetch), stream_work
(readers, worker, commit), stream_io (shards), stream_ledger (ledger, dedup store), stream_recover
(restart), stream_lock (pc_heavy.lock), stream_loop (the chunk loop).

Restart: run the same command again. The ledger is the commit record: shards are cut back to
their last committed frame, the store drops rows of a file the ledger never recorded, leftover raw
files of done files are deleted, and done files are skipped. OUT/STATUS.json is rewritten after
every chunk; touching OUT/STOP stops the run cleanly between chunks.

Etiquette (the PC is shared with timed GPU runs): the process runs at nice 15; extraction and
commit of each chunk (at most `workers` files, <= 20) hold pc_heavy.lock (flock, released between
chunks; waiting longer than --lock-timeout stops the run cleanly); downloads do not take the lock.
D8: a source whose documents carry no license of their own (CCCC) runs only when named in
--allow-unrecorded-license, as make_tok_sample.py requires; the worker drops any such document
from another source (license_unrecorded).
Disk: a chunk or download that would leave under --min-free-gb (default 90, never below 80) stops
the run cleanly; writes abort below the 80 GiB floor.
Code: the CLI runs from inside OUT (a replaced code directory cannot pull the working directory
away); before every chunk the corpus/*.py and data files hashed at run start are hashed again, and
any change stops the run cleanly (code_changed), so workers never run other code.
Exit codes: 0 done, 2 done except failed files, 3 disk_low, 4 lock_timeout, 5 stopped,
6 code_changed, 1 error.
"""
import argparse
import json
import os
import re
import signal
import sys

import extract
import stream_fetch as F
import stream_io as SIO
import stream_loop as L
import stream_rank as R
import stream_recover as SR
import stream_work as W

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_WORKERS, HARD_FLOOR_GB = 20, 80
EXIT = {"done": 0, "failed": 2, "disk_low": 3, "lock_timeout": 4, "stop_file": 5,
        "code_changed": 6}
DEFAULTS = dict(
    manifest=os.path.join(HERE, "fetch", "core_v0_manifest.json"), out=None, raw=None,
    starter=None, lock_file="~/planck/locks/pc_heavy.lock", lock_timeout=7200, workers=16,
    source_workers={}, mem_per_worker_gb=0.35, mem_reserve_gb=3, nice=15,
    min_free_gb=90, floor_gb=HARD_FLOOR_GB, reserve_gb=20, window_gb=30, dl_threads=2,
    shard_bytes=256 << 20, codec=SIO.best_codec(), level=3, tiers=None, sources=None, match=None,
    roles=["core"], max_files=0, readers={}, opts={}, sidecars={}, skip_unready=False, batch_wait=20,
    allow_unrecorded_license=[], url_rewrite={}, blocked_grace=60, curl_attempts=5, retry_sleep=10,
    seal_done_sources=True)


def check_config(cfg):
    c = dict(DEFAULTS, **cfg)
    for k in ("out", "raw", "starter", "manifest", "lock_file"):
        c[k] = os.path.abspath(os.path.expanduser(c[k])) if c[k] else c[k]
    if not c["out"] or not c["raw"]:
        raise ValueError("out and raw directories are required")
    F.check_disk_config(c, HARD_FLOOR_GB)
    if not 1 <= c["workers"] <= MAX_WORKERS:
        raise ValueError(f"workers must be 1..{MAX_WORKERS}")
    if any(not re.fullmatch(r"[a-z0-9_]+", n) or n == "keys" for n in c["sidecars"]):
        raise ValueError(f"bad sidecar names {sorted(c['sidecars'])}")
    if not SIO.codec_ok(c["codec"]):
        raise ValueError(f"codec {c['codec']} is not available in this Python")
    return c


def public(cfg):
    home = os.path.expanduser("~")
    return {k: (v.replace(home, "~", 1) if isinstance(v, str) else v) for k, v in cfg.items()}


def select(cfg, manifest, done):
    """-> (pending items in rank order, {source: n files without a reader})."""
    items, unready, rank = [], {}, R.manifest_ranks(manifest)
    for e in manifest["files"]:
        fid = f"{e['dataset']}/{e['path']}"
        if (e.get("role", "core") not in cfg["roles"] or fid in done
                or (cfg["tiers"] and e.get("tier") not in cfg["tiers"])
                or (cfg["sources"] and e["source"] not in cfg["sources"])
                or (cfg["match"] and not any(m in e["path"] for m in cfg["match"]))):
            continue
        spec = W.resolve_reader(e, cfg["readers"])
        if spec is None:
            unready[e["source"]] = unready.get(e["source"], 0) + 1
            continue
        rel = F.raw_rel(e)
        local = bool(e.get("starter_local") and cfg["starter"]
                     and os.path.exists(os.path.join(cfg["starter"], rel)))
        dl = os.path.join(cfg["raw"], rel)
        items.append(dict(entry=e, fid=fid, source=e["source"], reader=spec, local=local,
                          dl_path=dl, path=os.path.join(cfg["starter"], rel) if local else dl))
    items.sort(key=lambda it: rank[it["fid"]])
    return items[:cfg["max_files"]] if cfg["max_files"] else items, unready


def _load(cfg):
    with open(cfg["manifest"]) as f:
        manifest = json.load(f)
    by_fid = {f"{e['dataset']}/{e['path']}": e for e in manifest["files"]}
    by_source = {}
    for fid, e in by_fid.items():
        if e.get("role", "core") in cfg["roles"]:
            by_source.setdefault(e["source"], []).append(fid)
    return manifest, by_fid, by_source


def unrecorded_license(source) -> bool:
    """True for a source whose documents carry no license of their own (readers_cp: CCCC)."""
    import readers_cp
    return readers_cp.CP_SOURCES.get(source, {}).get("license_required") is False


def check_sidecars(ctx, cfg):
    first = next(iter(ctx.ledger.of("run_start")), None)
    if first and first["config"].get("sidecars", {}) != cfg["sidecars"]:
        raise SystemExit(f"--sidecar {cfg['sidecars']} differs from this output's first run "
                         f"{first['config'].get('sidecars')}: rows would not line up")


def run(cfg, log=print):
    """Stream every selected file not yet done. -> {"reason", "failed", "files"}."""
    cfg = check_config(cfg)
    os.makedirs(cfg["out"], exist_ok=True)
    os.makedirs(cfg["raw"], exist_ok=True)
    cur = os.nice(0)
    if cfg["nice"] > cur:
        os.nice(cfg["nice"] - cur)
    manifest, by_fid, by_source = _load(cfg)
    ctx = W.Context(cfg)
    ctx.rank = R.manifest_ranks(manifest)
    reason, failed, items, pf = "error", [], [], None
    try:
        check_sidecars(ctx, cfg)
        notes = SR.recover(ctx, by_fid)
        items, unready = select(cfg, manifest, ctx.ledger.done())
        blocked = sorted({it["source"] for it in items if unrecorded_license(it["source"])
                          and it["source"] not in cfg["allow_unrecorded_license"]})
        if blocked:
            raise SystemExit(f"{blocked} record no per-document license (CORPUS addendum, CCCC "
                             f"ruling): pass --allow-unrecorded-license {' '.join(blocked)}")
        if unready and not cfg["skip_unready"]:
            raise SystemExit(f"no reader yet for {unready}: add one, pass --reader "
                             "SOURCE=module:function, narrow --sources, or --skip-unready")
        ctx.code = extract.code_hashes()
        ctx.ledger.append("run_start", pid=os.getpid(), config=public(cfg), recovery=notes,
                          code_sha256=ctx.code, pending=len(items), unready=unready)
        log(f"stream_core: {len(items)} files pending, recovery {notes}, unready {unready}")
        pf = F.Prefetcher(items, cfg, log)
        pf.start()
        reason, failed = L.run_loop(ctx, cfg, items, pf, by_source, log)
    except BaseException as e:
        sig = isinstance(e, KeyboardInterrupt) or (isinstance(e, SystemExit) and e.code == 143)
        reason = "signal" if sig else f"error: {type(e).__name__}: {e}"
        raise
    finally:
        if pf is not None:
            pf.stop()
        ctx.ledger.append("run_stop", reason=reason, failed=failed,
                          pending=sum(it.get("state") != "consumed" for it in items))
        L.write_status(ctx, cfg, reason, items)
        ctx.close()
    log(f"stream_core: stop ({reason}), {len(failed)} failed")
    return {"reason": "failed" if reason == "done" and failed else reason, "failed": failed,
            "files": len(items)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--starter")
    ap.add_argument("--manifest", default=DEFAULTS["manifest"])
    ap.add_argument("--tiers", type=int, nargs="+")
    ap.add_argument("--sources", nargs="+")
    ap.add_argument("--match", nargs="+", help="only files whose path holds one of these")
    ap.add_argument("--reader", action="append", default=[], help="SOURCE=module:function")
    ap.add_argument("--opt", action="append", default=[], help="reader option KEY=JSON")
    ap.add_argument("--url-rewrite", action="append", default=[], metavar="FROM=TO",
                    help="fetch URLs starting with FROM from TO instead (a mirror)")
    ap.add_argument("--sidecar", action="append", default=[], help="NAME=module:function, e.g. "
                    "mh=stream_work:mh_row (neardedup.py's MinHash rows)")
    for k in ("workers", "dl_threads", "shard_mb", "level", "lock_timeout", "max_files",
              "batch_wait", "curl_attempts", "retry_sleep"):
        ap.add_argument("--" + k.replace("_", "-"), type=int)
    for k in ("window_gb", "min_free_gb", "reserve_gb"):
        ap.add_argument("--" + k.replace("_", "-"), type=float)
    ap.add_argument("--codec", choices=["zst", "gz"])
    ap.add_argument("--lock-file")
    ap.add_argument("--skip-unready", action="store_true")
    ap.add_argument("--allow-unrecorded-license", nargs="+", metavar="SOURCE",
                    help="admit these sources' documents without a per-document license "
                    "(the Sep 25 CCCC ruling; recorded in the ledger)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--neardedup-spec", metavar="PATH", help="write neardedup_lsh's SPEC.json "
                    "for the sealed shards of --out, then exit")
    ap.add_argument("--seal-open", action="store_true", help="seal the open shard of each "
                    "(--sources) source now, e.g. when a file failed for good, then exit")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)       # the log is a file: flush every line
    cfg = {k: v for k, v in vars(a).items() if v is not None and k not in
           ("reader", "opt", "sidecar", "dry_run", "shard_mb", "neardedup_spec", "seal_open",
            "url_rewrite")}
    cfg["readers"] = dict(x.split("=", 1) for x in a.reader)
    cfg["sidecars"] = dict(x.split("=", 1) for x in a.sidecar)
    cfg["opts"] = {k: json.loads(v) for k, v in (x.split("=", 1) for x in a.opt)}
    cfg["url_rewrite"] = dict(x.split("=", 1) for x in a.url_rewrite)
    if a.shard_mb:
        cfg["shard_bytes"] = a.shard_mb << 20
    import stream_tools as T
    if a.neardedup_spec:
        with open(a.neardedup_spec, "w") as f:
            json.dump(T.neardedup_spec(check_config(cfg)["out"]), f, indent=1)
        return 0
    if a.dry_run:
        T.dry_run(cfg)
        return 0
    if a.seal_open:
        print(T.seal_open(cfg))
        return 0
    cfg = check_config(cfg)                          # every path absolute before the chdir
    os.makedirs(cfg["out"], exist_ok=True)
    sys.path[:] = [os.path.abspath(p) for p in sys.path]      # '' would follow the chdir
    os.chdir(cfg["out"])
    for sig in (signal.SIGTERM, signal.SIGHUP):      # a clean run_stop, not a silent death
        signal.signal(sig, lambda *_: sys.exit(143))
    return EXIT.get(run(cfg)["reason"], 1)


if __name__ == "__main__":
    sys.exit(main())
