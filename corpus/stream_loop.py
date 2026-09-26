"""The chunk loop of stream_core.py: wait for downloaded files, take the heavy-CPU lock, extract
a chunk in parallel workers, commit each file in rank order, delete raw files, seal shards of
finished sources, rewrite STATUS.json.

A chunk is a run of consecutive ready files of one tier (sources may mix, so a tier of many small
sources takes few lock holds), at most min(workers, the MemAvailable cap, the worker cap of each
source in it) long; it starts when full, at a tier boundary, when the next file is not coming soon
(failed, disk-blocked, end), or after batch_wait seconds.
"""
import json
import multiprocessing as mp
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import extract
import stream_fetch as F
import stream_io as SIO
import stream_recover as SR
import stream_work as W
from stream_ledger import now
from stream_lock import (MAX_WORKERS, HeavyLock, LockTimeout, exit_with_parent, lock_free_now,
                         mem_cap)


def wait_chunk(cfg, pf, items, pos):
    """Block until a run of ready files of one tier (any sources) can start. -> list of items,
    [] when the front file failed, or a stop reason."""
    t0, stop = time.time(), os.path.join(cfg["out"], "STOP")
    with pf.cond:
        while True:
            tier = items[pos]["entry"].get("tier")
            cap, chunk, full = min(cfg["workers"], mem_cap(cfg)), [], False
            for it in items[pos:]:
                if it["entry"].get("tier") != tier:
                    full = True                              # a tier boundary ends the chunk
                    break
                cap = min(cap, cfg["source_workers"].get(it["source"], MAX_WORKERS))
                if len(chunk) >= cap:
                    full = True
                    break
                if it["state"] != "ready":
                    break
                chunk.append(it)
            j = pos + len(chunk)
            nxt = items[j] if j < len(items) else None
            if chunk and (full or nxt is None or nxt["state"] in ("failed", "blocked")
                          or time.time() - t0 >= cfg["batch_wait"]):
                return chunk
            front = items[pos]
            if front["state"] == "failed":
                return []
            if front["state"] == "blocked" and time.time() - front.get(
                    "blocked_since", time.time()) >= cfg["blocked_grace"]:
                return "disk_low"
            if os.path.exists(stop):
                return "stop_file"
            pf.cond.wait(1)


def run_loop(ctx, cfg, items, pf, by_source, log):
    """-> (stop reason, fids that failed this run)."""
    pos, pool, failed = 0, {}, []
    try:
        while pos < len(items):
            if os.path.exists(os.path.join(cfg["out"], "STOP")):
                return "stop_file", failed
            changed = code_changed(getattr(ctx, "code", None))
            if changed:
                log(f"stream_core: code changed under the run: {changed[:5]}")
                return "code_changed", failed
            chunk = wait_chunk(cfg, pf, items, pos)
            if isinstance(chunk, str):
                return chunk, failed
            if not chunk:                                   # the front file failed to download
                it = items[pos]
                ctx.ledger.append("file_failed", fid=it["fid"], stage="fetch", error=it["error"])
                failed.append(it["fid"])
                pf.consumed(it)
                pos += 1
                continue
            est = sum(it["entry"]["size"] * F.expand(it["entry"]) for it in chunk) * 1.4
            if F.disk_free(cfg["out"]) - est < cfg["min_free_gb"] * F.GIB:
                return "disk_low", failed
            tmp = os.path.join(ctx.tmp_root, f"c{pos:05d}")
            os.makedirs(tmp, exist_ok=True)
            jobs = []
            for k, it in enumerate(chunk):
                it["tmp"] = os.path.join(tmp, f"{k:03d}")
                e = it["entry"]
                extra = {kk: e[kk] for kk in ("revision", "item") if e.get(kk)}
                jobs.append(dict(reader=it["reader"], source=it["source"], path=it["path"],
                                 rel=F.raw_rel(e), tmp=it["tmp"], extra_meta=extra,
                                 sidecars=cfg["sidecars"],
                                 allow_unrecorded=cfg["allow_unrecorded_license"],
                                 floor=cfg["floor_gb"] * F.GIB,
                                 o=dict(extract.DEFAULTS, **dict(
                                     {"gutenberg_cap": 0}, **cfg["opts"],
                                     scratch_dir=it["tmp"] + ".scratch"))))
            disk_low = False
            if pool.get("ex") is not None and not lock_free_now(cfg["lock_file"]):
                pool["ex"].shutdown(wait=True)   # idle workers keep ~50 MB each through the wait
                pool["ex"] = None
            with HeavyLock(cfg["lock_file"], cfg["lock_timeout"]) as lk:
                results = run_jobs(pool, cfg, jobs)
                for it, res in zip(chunk, results):
                    if "error" in res:
                        ctx.ledger.append("file_failed", fid=it["fid"], stage="extract",
                                          error=res["error"], trace=res.get("trace"))
                        failed.append(it["fid"])
                        disk_low |= bool(res.get("disk_low"))
                        continue
                    W.commit_file(ctx, it, res, {"lock_wait": lk.waited, "chunk_files": len(chunk)})
                    it["committed"] = True
                    st = res["stats"]
                    log(f"[{pos + 1 + chunk.index(it)}/{len(items)}] {it['fid']} kept "
                        f"{st['docs_kept']:,}/{st['docs_in']:,} docs "
                        f"{st['bytes_kept'] / 1e6:,.1f} MB, extract {res['seconds']} s, "
                        f"lock wait {lk.waited} s")
            for it in chunk:
                if it.get("committed"):
                    W.delete_raw(ctx, it)
                pf.consumed(it)
            shutil.rmtree(tmp, ignore_errors=True)
            pos += len(chunk)
            if cfg["seal_done_sources"]:
                seal_done(ctx, by_source, {it["source"] for it in chunk})
            write_status(ctx, cfg, "running", items)
            if disk_low:
                return "disk_low", failed
        return "done", failed
    except SIO.DiskLow:
        return "disk_low", failed
    except LockTimeout:
        return "lock_timeout", failed
    finally:
        if pool.get("ex") is not None:
            pool["ex"].shutdown(wait=True, cancel_futures=True)


def code_changed(start):
    """Files hashed at run start that are now different or gone (all of them if the code
    directory itself is gone). New files are ignored."""
    if not start:
        return []
    try:
        now = extract.code_hashes()
    except OSError:
        return ["<code directory missing>"]
    return sorted(f for f, h in start.items() if now.get(f) != h)


def _executor(cfg, n):
    return ProcessPoolExecutor(n, mp_context=mp.get_context("spawn"), max_tasks_per_child=8,
                               initializer=exit_with_parent, initargs=(os.getpid(),))


def run_jobs(pool, cfg, jobs):
    """Extract jobs in spawned workers (inline when workers == 1). A worker that dies (the OOM
    killer, a crash in C code) breaks the pool; the jobs without a result are then re-run one at
    a time in a fresh process, so only the file that kills its worker fails."""
    if cfg["workers"] == 1:
        return [W.extract_file(j) for j in jobs]
    if pool.get("ex") is None:
        pool["ex"] = _executor(cfg, cfg["workers"])
    futs, out = [pool["ex"].submit(W.extract_file, j) for j in jobs], []
    for f in futs:
        try:
            out.append(f.result())
        except BrokenProcessPool:
            out.append(None)
    if None in out:
        pool["ex"].shutdown(wait=True, cancel_futures=True)
        pool["ex"] = None
        for k, j in enumerate(jobs):
            if out[k] is None:
                with _executor(cfg, 1) as ex:
                    try:
                        out[k] = ex.submit(W.extract_file, j).result()
                    except BrokenProcessPool as e:
                        out[k] = {"error": f"worker process died ({e}); out of memory?"}
    return out


def seal_done(ctx, by_source, sources):
    """Seal the open shard of every source whose manifest files are all done."""
    done = ctx.ledger.done()
    for src in sources:
        if all(fid in done for fid in by_source.get(src, ())):
            rec = ctx.shards(src).seal_open()
            if rec:
                ctx.ledger.append("shard_sealed", **rec)
                SR.finalize_sidecars(ctx.cfg["out"], rec, ctx.dtypes(src))


def write_status(ctx, cfg, state, items):
    """OUT/STATUS.json: progress by source, rewritten atomically."""
    by = {}
    for e in ctx.ledger.done().values():
        b = by.setdefault(e["source"], {"files": 0, "docs_in": 0, "docs_kept": 0,
                                        "bytes_in": 0, "bytes_kept": 0, "dropped": {}})
        b["files"] += 1
        for k in ("docs_in", "docs_kept", "bytes_in", "bytes_kept"):
            b[k] += e["stats"][k]
        for r, d in e["stats"]["dropped"].items():
            b["dropped"][r] = b["dropped"].get(r, 0) + d["docs"]
    states = {}
    for it in items:
        states[it.get("state", "wait")] = states.get(it.get("state", "wait"), 0) + 1
    st = {"updated": now(), "state": state, "free_gib": round(F.disk_free(cfg["out"]) / F.GIB, 1),
          "files_done_total": sum(b["files"] for b in by.values()), "this_run": states,
          "sealed_shards": len(ctx.ledger.sealed()), "by_source": by}
    tmp = os.path.join(cfg["out"], "STATUS.json.tmp")
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, os.path.join(cfg["out"], "STATUS.json"))
