"""Crash recovery of stream_core.py's output shards, and the sealed-shard sidecar step.

recover_shards() brings OUT/<source>/ back to what the ledger committed:
- a sealed shard (shard_sealed event) must still have its recorded sizes; its sidecar .bin files
  are left for finalize_sidecars() (a crash can land between the seal event and the conversion);
- an unsealed shard that committed segments name is cut back to the end of its last committed
  frame, and so are its keys file and sidecar .bin files; a stray .npy is deleted (a seal that
  never reached the ledger);
- every other file of the source's shard pattern is deleted (output of a file that never
  committed);
- a file the ledger needs that is missing, or a shard written with another codec, stops the run.
"""
import os
import re
import shutil

import numpy as np

from stream_io import seal_record, shard_rel


def dtype_from_json(descr) -> np.dtype:
    """Inverse of numpy's dtype.descr after a JSON round trip (lists instead of tuples)."""
    if isinstance(descr, str):
        return np.dtype(descr)
    return np.dtype([tuple(x[:2]) + ((tuple(x[2]),) if len(x) > 2 else ()) for x in descr])


def finalize_sidecars(root, sealed, dtypes):
    """After a shard_sealed event is in the ledger: <stem>.NAME.bin -> <stem>.NAME.npy (written
    to a temp name, fsynced, renamed), then the .bin is deleted. Idempotent (recovery reruns it)."""
    stem = os.path.join(root, sealed["shard"].rsplit(".jsonl.", 1)[0])
    for name, info in (sealed.get("sidecars") or {}).items():
        bp, npy = f"{stem}.{name}.bin", f"{stem}.{name}.npy"
        if not os.path.exists(bp):
            if not os.path.exists(npy):
                raise SystemExit(f"{stem}: sidecar {name} missing (neither .bin nor .npy)")
            continue
        if os.path.getsize(bp) != info["bytes"]:
            raise SystemExit(f"{bp}: {os.path.getsize(bp)} bytes, sealed at {info['bytes']}")
        rows = np.fromfile(bp, dtype=dtype_from_json(dtypes[name]))
        if len(rows) != sealed["docs"]:
            raise SystemExit(f"{bp}: {len(rows)} rows for {sealed['docs']} documents")
        with open(npy + ".part", "wb") as f:
            np.save(f, rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(npy + ".part", npy)
        os.remove(bp)


def shard_index(rel) -> int:
    """'cccc/cccc-00012.jsonl.zst' -> 12."""
    return int(rel.rsplit("-", 1)[1][:5])


def _ref(segs):
    ref = {}
    for s in segs:
        r = ref.setdefault(s["shard"], {"end": 0, "kend": 0, "docs": 0, "ubytes": 0, "sc": {}})
        r["end"], r["kend"] = max(r["end"], s["end"]), max(r["kend"], s["kend"])
        r["docs"] += s["docs"]
        r["ubytes"] += s["ubytes"]
        for n, (_, e) in (s.get("sc") or {}).items():
            r["sc"][n] = max(r["sc"].get(n, 0), e)
    return ref


def _cut(path, want):
    have = os.path.getsize(path)
    if have < want:
        raise SystemExit(f"{path}: shorter than its committed {want} bytes")
    if have > want:
        os.truncate(path, want)


def recover_shards(root, source, codec, segs, sealed):
    """segs: every committed segment of the source (file_done events); sealed: {shard rel:
    shard_sealed event}. -> (ShardSet state, [states of older unsealed shards to seal now])."""
    ref, d = _ref(segs), os.path.join(root, source)
    pat = re.compile(rf"^{re.escape(source)}-(\d{{5}})\.(.+)$")
    for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        m = pat.match(name)
        if not m:
            continue
        kind, path = m.group(2), os.path.join(d, name)
        if kind.startswith("jsonl.") and kind != f"jsonl.{codec}":
            raise SystemExit(f"{path}: written with another codec; use --codec {kind[6:]} or "
                             "a new output directory")
        rel = shard_rel(source, int(m.group(1)), codec)
        side = kind[:-4] if kind.endswith(".bin") else None
        if rel in sealed:
            s = sealed[rel]
            want = {f"jsonl.{codec}": s["bytes"], "keys.u64": s["keys_bytes"]}.get(kind)
            if side is not None:
                want = (s.get("sidecars") or {}).get(side, {}).get("bytes")
            if kind.endswith(".part"):
                os.remove(path)
            elif want is not None and os.path.getsize(path) != want:
                raise SystemExit(f"{path}: sealed at {want} bytes, now {os.path.getsize(path)}")
        elif rel in ref:
            r = ref[rel]
            want = {f"jsonl.{codec}": r["end"], "keys.u64": r["kend"]}.get(kind)
            if side is not None and side in r["sc"]:
                want = r["sc"][side]
            if want is None:
                os.remove(path)                   # .npy or .part of a seal never recorded
            else:
                _cut(path, want)
        else:
            os.remove(path)
    for rel, r in ref.items():
        i = shard_index(rel)
        need = [rel, shard_rel(source, i, codec, "keys")]
        need += [shard_rel(source, i, codec, n + ".bin") for n in r["sc"] if rel not in sealed]
        for p in need:
            if not os.path.exists(os.path.join(root, p)):
                raise SystemExit(f"{root}/{p}: committed in the ledger but missing on disk")
    unsealed = sorted((shard_index(r), r) for r in ref if r not in sealed)
    states = [{"index": i, "docs": ref[r]["docs"], "ubytes": ref[r]["ubytes"]} for i, r in unsealed]
    known = [shard_index(r) for r in list(ref) + list(sealed)]
    return {"next": max(known) + 1 if known else 0, "open": states[-1] if states else None}, \
        states[:-1]


def recover(ctx, manifest_by_fid):
    """Make the output tree, the store and the raw directory match the ledger; finish pending
    sidecar conversions of sealed shards. -> notes."""
    import stream_fetch as F
    import stream_work as W                       # lazy: stream_work imports this module
    led, notes = ctx.ledger, {"ledger_torn_bytes": ctx.ledger.repaired}
    done = led.done()
    notes["store_rolled_back"] = ctx.store.rollback_uncommitted(set(done))
    sources = {e["source"] for e in done.values()}
    root = ctx.cfg["out"]
    sources |= {d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
                and not d.startswith("_")}
    sealed = led.sealed()
    for src in sorted(sources):
        state, stale = recover_shards(root, src, ctx.cfg["codec"], led.segments(src),
                                      {k: v for k, v in sealed.items() if v["source"] == src})
        names = sorted(ctx.cfg["sidecars"])
        for st in stale:
            led.append("shard_sealed", **seal_record(root, src, ctx.cfg["codec"], st, names))
        ctx.states[src] = state
        if state["open"] and state["open"]["ubytes"] >= ctx.cfg["shard_bytes"]:
            led.append("shard_sealed", **ctx.shards(src).seal_open())
        for rec in led.sealed().values():
            if rec["source"] == src:
                finalize_sidecars(root, rec, ctx.dtypes(src))
    gone = led.deleted()
    for fid, ev in done.items():
        if fid in gone or ev.get("local"):
            continue
        entry = manifest_by_fid.get(fid) or {"dataset": ev["dataset"], "path": ev["path"]}
        path = os.path.join(ctx.cfg["raw"], F.raw_rel(entry))
        W.delete_raw(ctx, {"local": False, "path": path, "fid": fid, "entry": entry})
    shutil.rmtree(ctx.tmp_root, ignore_errors=True)
    return notes
