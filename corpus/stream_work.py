"""Per-file work of stream_core.py: reader lookup, the extraction worker, the in-order commit with
global exact dedup, and crash recovery.

Reader protocol (readers_cp.py): reader(source, path, rel, o) yields ("keep", record, raw_bytes),
("drop", reason, raw_bytes) and ("note", name, n). Readers are named "module:function" so spawned
workers can import them. Default lookup (resolve_reader), unless --reader SOURCE=module:function:
- oasst2, dolly: readers_chat.read_oasst / read_dolly;
- a .7z file (the Stack Exchange dump): readers_se_dump.read_se_dump, if that module exists;
- any source in readers_core.READERS (gutenberg as whole books, loc, youtube, news, pressbooks,
  oercommons, foodista, pdr, and cccc/irc/wikimedia through readers_cp.cp_doc): read_core;
- else any source in readers_cp.CP_SOURCES: readers_cp.read_common_pile.
The reader gets `o` = extract.DEFAULTS with gutenberg_cap 0 (whole books), overridden by --opt,
plus o["scratch_dir"], an empty per-file directory it may use (deleted afterwards).

Worker (extract_file, in a spawned process): raw file -> <tmp>.jsonl (records that passed the
reader, meta.sha1 and provenance added) and <tmp>.docs.npy (per record: text key, id hash, text
bytes, raw bytes, undated flag). Commit (commit_file, in the parent, in rank order): a record
whose text key or id hash a better-ranked file committed (in any run), or seen earlier in the same
file, is dropped as dup_exact / dup_id; the rest go to the source's shards. A record whose copies
all belong to lower-ranked files (a late file, stream_rank.py) is kept, and its file_done event
names those copies ("displaced": [{"fid", "keys", "ids"}]) for core_finalize.py to drop.
"""
import hashlib
import importlib
import json
import os
import shutil
import time
import traceback
from array import array

import numpy as np

import stream_fetch as F
import stream_io as SIO
import stream_recover as SR
import stream_rank as R
from extract_merge import add_drop
from stream_ledger import KeyStore, Ledger, id_hash

CHAT = {"oasst2": "readers_chat:read_oasst", "dolly": "readers_chat:read_dolly"}
SE_DUMP = "readers_se_dump:read_se_dump"


class Context:
    """The open ledger, dedup store and shard sets of one output directory."""

    def __init__(self, cfg):
        self.cfg = cfg
        state_dir = os.path.join(cfg["out"], "_state")
        os.makedirs(state_dir, exist_ok=True)
        self.ledger = Ledger(os.path.join(cfg["out"], "LEDGER.jsonl"))
        self.store = KeyStore(os.path.join(state_dir, "exact.sqlite"))
        self.tmp_root = os.path.join(state_dir, "tmp")
        self.states, self._sets, self.rank = {}, {}, {}

    def shards(self, source):
        if source not in self._sets:
            c = self.cfg
            self._sets[source] = SIO.ShardSet(
                c["out"], source, c["codec"], c["level"], c["shard_bytes"],
                self.states.get(source) or {"next": 0, "open": None},
                lambda p: F.disk_free(p), c["floor_gb"] * F.GIB, sorted(c["sidecars"]))
        return self._sets[source]

    def dtypes(self, source):
        """{sidecar name: dtype descr} from this source's committed files (all agree)."""
        out = {}
        for e in self.ledger.of("file_done"):
            if e["source"] == source:
                out.update(e.get("sidecar_dtypes") or {})
        return out

    def close(self):
        self.store.close()
        self.ledger.close()


def load_reader(spec):
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


def resolve_reader(entry, overrides):
    """-> 'module:function' for this manifest entry, or None when no reader exists yet."""
    src = entry["source"]
    spec = overrides.get(src) or CHAT.get(src)
    if spec is None and entry["path"].endswith(".7z"):
        spec = SE_DUMP
    if spec is None:
        try:
            import readers_core
            spec = "readers_core:read_core" if src in readers_core.READERS else None
        except ImportError:
            spec = None
    if spec is None:
        import readers_cp
        spec = "readers_cp:read_common_pile" if src in readers_cp.CP_SOURCES else None
    if spec is None:
        return None
    try:
        load_reader(spec)
    except (ImportError, AttributeError):
        return None
    return spec


def new_stats():
    return {"docs_in": 0, "bytes_in": 0, "docs_kept": 0, "bytes_kept": 0, "bytes_trimmed": 0,
            "undated_kept": 0, "dropped": {}, "notes": {}}


def extract_file(job):
    """Worker: one raw file -> tmp records + per-doc columns. -> {"stats", "docs", "seconds"} or
    {"error", "trace"}. Never raises."""
    t0 = time.time()
    try:
        reader = load_reader(job["reader"])
        st, cols, n, written, check = new_stats(), [array("Q") for _ in range(5)], 0, 0, SIO.CHECK_EVERY
        os.makedirs(job["o"]["scratch_dir"], exist_ok=True)
        hooks = {k: load_reader(v) for k, v in sorted(job["sidecars"].items())}
        side = {k: open(f"{job['tmp']}.{k}.bin", "wb") for k in hooks}
        dtypes = {}
        with open(job["tmp"] + ".jsonl", "wb") as out:
            for ev in reader(job["source"], job["path"], job["rel"], job["o"]):
                if ev[0] == "note":
                    st["notes"][ev[1]] = st["notes"].get(ev[1], 0) + ev[2]
                    continue
                st["docs_in"] += 1
                st["bytes_in"] += ev[2]
                if ev[0] == "drop":
                    add_drop(st, ev[1], ev[2])
                    continue
                rec = ev[1]
                if (rec["meta"].get("license_basis") == "unrecorded"
                        and job["source"] not in job["allow_unrecorded"]):
                    add_drop(st, "license_unrecorded", ev[2])       # D8: explicit allow only
                    continue
                text = rec["text"].encode("utf-8")
                h = hashlib.sha1(text).digest()
                rec["meta"]["sha1"] = h.hex()
                rec["meta"].update(job["extra_meta"])
                line = (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")
                for k, fn in hooks.items():
                    row = np.asarray(fn(rec))
                    if dtypes.setdefault(k, row.dtype) != row.dtype or row.size != 1:
                        raise ValueError(f"sidecar {k}: rows must be one {dtypes[k]} each")
                    side[k].write(row.tobytes())
                out.write(line)
                for c, v in zip(cols, (int.from_bytes(h[:8], "big"), id_hash(rec["id"]),
                                       len(text), ev[2],
                                       int(rec["meta"].get("date_status") == "undated"))):
                    c.append(v & 0xFFFFFFFFFFFFFFFF)
                n += 1
                written += len(line)
                if written >= check:
                    check = written + SIO.CHECK_EVERY
                    if F.disk_free(job["tmp"]) < job["floor"]:
                        raise SIO.DiskLow("free space under the floor while extracting")
        for f in side.values():
            f.close()
        np.save(job["tmp"] + ".docs.npy",
                np.stack([np.frombuffer(c, dtype=np.uint64) for c in cols], axis=1))
        return {"stats": st, "docs": n, "seconds": round(time.time() - t0, 1),
                "sidecar_dtypes": {k: v.descr for k, v in dtypes.items()}}
    except SIO.DiskLow as e:
        return {"error": f"disk_low: {e}", "disk_low": True}
    except Exception as e:  # noqa: BLE001 (reported per file; the run goes on)
        return {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]}
    finally:
        shutil.rmtree(job["o"]["scratch_dir"], ignore_errors=True)


def commit_file(ctx, item, res, extra):
    """Dedup one worker result, append survivors to the shards, commit the store, then the
    ledger (the commit point). Raises on any error: the caller stops the run and the next start
    recovers."""
    e, tmp, st = item["entry"], item["tmp"], res["stats"]
    t0 = time.time()
    docs = np.load(tmp + ".docs.npy")
    keys, ids = docs[:, 0].view(np.int64), docs[:, 1].view(np.int64)
    mine, own_k, own_i = (ctx.rank.get(item["fid"], -1), R.best_owner(ctx.store, "keys", keys,
                          ctx.rank), R.best_owner(ctx.store, "ids", ids, ctx.rank))
    seen_k, seen_i, disp, over = set(), set(), {}, {"keys": [], "ids": []}
    dts = {k: SR.dtype_from_json(v) for k, v in res.get("sidecar_dtypes", {}).items()}
    rows = {k: np.fromfile(f"{tmp}.{k}.bin", dtype=dts[k]) if k in dts else [] for k in
            sorted(ctx.cfg["sidecars"])}
    if any(len(r) != len(docs) for r in rows.values()):
        raise RuntimeError(f"{tmp}: sidecar rows out of step with records")
    shards = ctx.shards(item["source"])
    shards.begin()
    n = 0
    with open(tmp + ".jsonl", "rb") as fj:
        for i, line in enumerate(fj):
            n += 1
            k, d, kb, rb, und = (int(keys[i]), int(ids[i]), int(docs[i, 2]), int(docs[i, 3]),
                                 int(docs[i, 4]))
            ok, oi = own_k.get(k), own_i.get(d)
            if k in seen_k or (ok and ok[0] <= mine):
                add_drop(st, "dup_exact", rb)
                continue
            if d in seen_i or (oi and oi[0] <= mine):
                add_drop(st, "dup_id", rb)
                continue
            seen_k.add(k)
            seen_i.add(d)
            for t, v, o in (("keys", k, ok), ("ids", d, oi)):
                if o:                              # only lower-ranked files own it: displace
                    over[t].append(v)
                    disp.setdefault(o[1], {"fid": o[1], "keys": [], "ids": []})[t].append(v)
            shards.add(line, k, {n: r[i].tobytes() for n, r in rows.items()})
            st["docs_kept"] += 1
            st["bytes_kept"] += kb
            st["bytes_trimmed"] += rb - kb
            st["undated_kept"] += und
    if n != len(docs) or n != res["docs"]:
        raise RuntimeError(f"{tmp}: {n} records but {len(docs)} key rows")
    segs, sealed = shards.finish()
    ctx.store.commit_file(item["fid"], list(seen_k - set(over["keys"])),
                          list(seen_i - set(over["ids"])), over)
    extra = dict(extra, displaced=[disp[f] for f in sorted(disp)]) if disp else extra
    alg, digest = F.checksum(e)
    ctx.ledger.append("file_done", fid=item["fid"], source=item["source"], dataset=e["dataset"],
                      path=e["path"], revision=e.get("revision"), tier=e.get("tier"),
                      size=e["size"], checksum={alg: digest}, local=item["local"],
                      reader=item["reader"], fetch=item.get("fetch"), stats=st, segments=segs,
                      t_extract=res.get("seconds"), t_commit=round(time.time() - t0, 1),
                      sidecar_dtypes=res.get("sidecar_dtypes") or {}, **extra)
    for s in sealed:
        ctx.ledger.append("shard_sealed", **s)
        SR.finalize_sidecars(ctx.cfg["out"], s, ctx.dtypes(item["source"]))


def delete_raw(ctx, item):
    """Remove a committed raw file (never a starter-local one) and record it."""
    if item["local"]:
        return
    for p in (item["path"], item["path"] + ".part"):
        if os.path.exists(p):
            os.remove(p)
    ctx.ledger.append("raw_deleted", fid=item["fid"], path=F.raw_rel(item["entry"]))


def mh_row(rec):
    """Sidecar hook for neardedup.py (--sidecar mh=stream_work:mh_row): the document's MinHash
    row (neardedup.SIG_DTYPE), dated by meta.date, else meta.created, for its keep rule."""
    import neardedup as ND
    m = rec["meta"]
    return ND.sketch([rec["text"]], [m.get("date") or m.get("created")])[0]
