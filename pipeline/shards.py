"""Append-only jsonl shards that survive a kill at any moment (SPEC 11 output files).

Each record is one line, written with a single write() and flushed (fsync optional). A process killed mid-write can
leave one torn last line; repair() cuts a shard back to its last newline before it is read or appended to, and
reports how many bytes it dropped. Shards rotate at `size` records: <dir>/<prefix>-00000.jsonl, -00001, ..."""
import glob
import json
import os


def paths(d, prefix):
    return sorted(glob.glob(os.path.join(d, f"{prefix}-[0-9][0-9][0-9][0-9][0-9].jsonl")))


def repair(path):
    """truncate a torn last line; -> bytes dropped."""
    size = os.path.getsize(path)
    if size == 0:
        return 0
    with open(path, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return 0
        pos = size - 1
        chunk = 1 << 16
        while pos > 0:
            start = max(0, pos - chunk)
            f.seek(start)
            buf = f.read(pos - start)
            k = buf.rfind(b"\n")
            if k >= 0:
                cut = start + k + 1
                break
            pos = start
        else:
            cut = 0
        f.truncate(cut)
        return size - cut


def read(d, prefix, fix=True):
    """every record of every shard in order. fix=True repairs torn last lines first; a line that still does not
    parse raises (a shard is never silently skipped)."""
    for p in paths(d, prefix):
        if fix:
            repair(p)
        with open(p, encoding="utf-8") as f:
            for n, line in enumerate(f):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except ValueError as e:
                        raise ValueError(f"{p}:{n + 1} is not JSON: {e}") from e


class Writer:
    def __init__(self, d, prefix, size=5000, fsync=True):
        os.makedirs(d, exist_ok=True)
        self.d, self.prefix, self.size, self.fsync = d, prefix, size, fsync
        existing = paths(d, prefix)
        self.dropped = sum(repair(p) for p in existing)
        if existing:
            self.idx = len(existing) - 1
            with open(existing[-1], "rb") as f:
                self.count = sum(1 for _ in f)
        else:
            self.idx, self.count = 0, 0
        self.total = sum(_lines(p) for p in existing)
        self.f = None

    def _open(self):
        if self.f is not None and self.count < self.size:
            return
        if self.f is not None:
            self.f.close()
            self.f = None
        if self.count >= self.size:
            self.idx += 1
            self.count = 0
        self.f = open(os.path.join(self.d, f"{self.prefix}-{self.idx:05d}.jsonl"), "a", encoding="utf-8")

    def write(self, rec):
        self._open()
        self.f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        self.f.flush()
        if self.fsync:
            os.fsync(self.f.fileno())
        self.count += 1
        self.total += 1

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None


def _lines(p):
    with open(p, "rb") as f:
        return sum(1 for _ in f)
