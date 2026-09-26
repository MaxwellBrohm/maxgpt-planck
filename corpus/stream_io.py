"""Append-only compressed output shards for stream_core.py (the core v0 stream-and-delete runner).

Layout under OUT/<source>/:
- <source>-NNNNN.jsonl.<zst|gz>: one JSON record per line ({"id", "source", "text", "meta"}, plus
  "turns" for chat), stored as a sequence of complete compressed frames (zstd frames or gzip
  members). zstd -d, gzip -d, compression.zstd.open and gzip.open all read them as one stream
  (checked on the PC for compression.zstd, Python 3.14.4). Each raw file adds one frame to every
  shard it touches, so after a crash a shard is cut back to the end of its last committed frame.
- <source>-NNNNN.keys.u64: the exact-dedup key of every line, in line order, little-endian uint64
  (the first 8 bytes of the sha1 of the record's text, read big-endian; meta.sha1 holds all 20).
- optional per-document sidecars (stream_core --sidecar NAME=module:function): one fixed-dtype row
  per line, appended to <source>-NNNNN.NAME.bin while the shard is open; once its seal is in the
  ledger, stream_recover.finalize_sidecars() writes <source>-NNNNN.NAME.npy (np.load gives one row per line) and
  deletes the .bin. With NAME=mh and stream_work:mh_row that is neardedup.py's <stem>.mh.npy.

A shard is sealed (closed for good, whole-file sha256 recorded in the ledger) at the first document
boundary at or past `limit` bytes of uncompressed JSONL, so a sealed shard holds at least `limit`
bytes and at most one document more; only the newest shard of a source is ever open.
"""
import gzip
import hashlib
import io
import json
import os
import zlib

import numpy as np

try:
    from compression import zstd as _zstd             # Python 3.14+
except ImportError:
    _zstd = None
try:
    import zstandard as _zstandard
except ImportError:
    _zstandard = None

CHECK_EVERY = 64 << 20        # bytes written between disk-space checks


class DiskLow(RuntimeError):
    """Free space fell under the hard floor while writing."""


def codec_ok(codec: str) -> bool:
    return codec == "gz" or (codec == "zst" and (_zstd is not None or _zstandard is not None))


def best_codec() -> str:
    return "zst" if codec_ok("zst") else "gz"


class Frame:
    """One compressed frame: compress() pieces, then end() for the bytes that complete it.
    Output is deterministic for a given input, codec and level (gzip header mtime is 0)."""

    def __init__(self, codec, level):
        if codec == "gz":
            c = zlib.compressobj(level, zlib.DEFLATED, 31)
            self._end = lambda: c.flush(zlib.Z_FINISH)
        elif _zstd is not None:
            c = _zstd.ZstdCompressor(level=level)
            self._end = lambda: c.flush(_zstd.ZstdCompressor.FLUSH_FRAME)
        else:
            c = _zstandard.ZstdCompressor(level=level).compressobj()
            self._end = lambda: c.flush(_zstandard.COMPRESSOBJ_FLUSH_FINISH)
        self.compress = c.compress

    def end(self) -> bytes:
        return self._end()


def open_shard(path):
    """Binary file object over the decompressed lines of a shard (all frames)."""
    if path.endswith(".gz"):
        return gzip.open(path, "rb")
    if _zstd is not None:
        return _zstd.open(path, "rb")
    fh = open(path, "rb")
    return io.BufferedReader(_zstandard.ZstdDecompressor().stream_reader(
        fh, read_across_frames=True, closefd=True))


def iter_docs(path):
    """The records of one shard, in order."""
    with open_shard(path) as f:
        for line in f:
            yield json.loads(line)


def read_keys(path) -> np.ndarray:
    return np.fromfile(path, dtype="<u8")          # one uint64 per line of the shard


def sha256_file(path, start=0, end=None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(start)
        left = (os.path.getsize(path) if end is None else end) - start
        while left > 0:
            b = f.read(min(left, 1 << 22))
            if not b:
                break
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def shard_rel(source, idx, codec, kind="data") -> str:
    """Relative path of a shard ('data'), its keys ('keys') or a sidecar ('NAME.bin'/'NAME.npy')."""
    ext = {"data": f"jsonl.{codec}", "keys": "keys.u64"}.get(kind, kind)
    return f"{source}/{source}-{idx:05d}.{ext}"


def seal_record(root, source, codec, st, sidecars=()) -> dict:
    """The shard_sealed ledger event body for shard state st = {"index", "docs", "ubytes"}."""
    d, k = shard_rel(source, st["index"], codec), shard_rel(source, st["index"], codec, "keys")
    dp, kp = os.path.join(root, d), os.path.join(root, k)
    rec = {"source": source, "shard": d, "docs": st["docs"], "ubytes": st["ubytes"],
           "bytes": os.path.getsize(dp), "sha256": sha256_file(dp), "keys": k,
           "keys_bytes": os.path.getsize(kp), "keys_sha256": sha256_file(kp)}
    if sidecars:
        rec["sidecars"] = {}
        for name in sidecars:
            bp = os.path.join(root, shard_rel(source, st["index"], codec, name + ".bin"))
            rec["sidecars"][name] = {"bytes": os.path.getsize(bp), "sha256": sha256_file(bp)}
    return rec


class ShardSet:
    """The output shards of one source. state = {"next": index of the next new shard, "open":
    {"index", "docs", "ubytes"} of the open shard, or None}, as stream_recover returns it.
    sidecars: names of the per-document sidecars (fixed for an output directory).
    Per raw file: begin(), add() each surviving line, finish() -> (segments, sealed)."""

    def __init__(self, root, source, codec, level, limit, state, free_fn=None, floor=0,
                 sidecars=()):
        self.root, self.source, self.codec, self.level, self.limit = root, source, codec, level, limit
        self.next, self.open, self.sidecars = state["next"], state["open"], tuple(sidecars)
        self.free_fn, self.floor = free_fn, floor
        self.fh = None
        self.segs, self.sealed, self.written, self.check_at = [], [], 0, CHECK_EVERY

    def begin(self):
        assert self.fh is None, "previous file not finished"
        self.segs, self.sealed = [], []

    def _path(self, kind):
        return os.path.join(self.root, shard_rel(self.source, self.open["index"], self.codec, kind))

    def _start(self):
        if self.open is None:
            self.open = {"index": self.next, "docs": 0, "ubytes": 0}
            self.next += 1
        os.makedirs(os.path.join(self.root, self.source), exist_ok=True)
        self.fh = open(self._path("data"), "ab")
        self.side = {n: open(self._path("keys" if n is None else n + ".bin"), "ab")
                     for n in (None,) + self.sidecars}
        self.seg = {"shard": shard_rel(self.source, self.open["index"], self.codec),
                    "start": self.fh.tell(), "kstart": self.side[None].tell(), "docs": 0,
                    "ubytes": 0, "sc": {n: [self.side[n].tell()] for n in self.sidecars}}
        self.frame, self.h = Frame(self.codec, self.level), hashlib.sha256()
        self.keys, self.rows = [], {n: [] for n in self.sidecars}

    def _write(self, b):
        if not b:
            return
        self.fh.write(b)
        self.h.update(b)
        self.written += len(b)
        if self.free_fn is not None and self.written >= self.check_at:
            self.check_at = self.written + CHECK_EVERY
            if self.free_fn(self.root) < self.floor:
                raise DiskLow(f"free space under {self.floor / 2**30:.0f} GiB while writing")

    def add(self, line: bytes, key: int, rows=None):
        """rows: {sidecar name: that document's row bytes}, one per configured sidecar."""
        if self.fh is None:
            self._start()
        self._write(self.frame.compress(line))
        self.keys.append(key)
        for n in self.sidecars:
            self.rows[n].append(rows[n])
        for c in (self.seg, self.open):
            c["docs"] += 1
            c["ubytes"] += len(line)
        if self.open["ubytes"] >= self.limit:
            self._end_frame()
            self.sealed.append(seal_record(self.root, self.source, self.codec, self.open,
                                           self.sidecars))
            self.open = None

    def _end_frame(self):
        self._write(self.frame.end())
        self.side[None].write(np.asarray(self.keys, dtype="<i8").tobytes())   # same bytes as <u8
        for n in self.sidecars:
            self.side[n].write(b"".join(self.rows[n]))
        files = [self.fh] + list(self.side.values())
        for f in files:
            f.flush()
            os.fsync(f.fileno())
        self.seg.update(end=self.fh.tell(), kend=self.side[None].tell(), sha256=self.h.hexdigest())
        for n in self.sidecars:
            self.seg["sc"][n].append(self.side[n].tell())
        for f in files:
            f.close()
        self.segs.append(self.seg)
        self.fh = None

    def finish(self):
        if self.fh is not None:
            self._end_frame()
        return self.segs, self.sealed

    def seal_open(self):
        """Seal the open shard now (all of its source is done); -> the sealed record or None."""
        assert self.fh is None
        if self.open is None or self.open["docs"] == 0:
            return None
        rec = seal_record(self.root, self.source, self.codec, self.open, self.sidecars)
        self.open = None
        return rec
