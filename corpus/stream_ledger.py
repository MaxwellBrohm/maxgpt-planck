"""The ledger and the global exact-dedup store of stream_core.py.

LEDGER.jsonl (append-only, one JSON event per line, fsynced after every line) is the single record
of what is done. A raw file counts as done only once its file_done line is on disk. Events:
- run_start:    config, code sha256s, pid.
- file_done:    fid (dataset/path), source, checksum, fetch and extract stats, and "segments": the
                byte ranges it appended to each shard (start, end, docs, ubytes, sha256 of the
                compressed bytes, and kstart/kend in the keys sidecar).
- shard_sealed: a shard closed for good, with its whole-file sha256 (see stream_io.py).
- raw_deleted:  the raw file of a done fid was removed from the raw directory.
- file_failed:  fetch or extract failed; the file is retried on the next run.
- run_stop:     why a run ended (done, disk_low, lock_timeout, stop_file, signal, error).
A crash can leave a torn last line; opening the ledger cuts it off (it was never committed).

KeyStore (sqlite, WAL, synchronous=FULL) remembers every committed document's exact key (first 8
bytes of the sha1 of its text) and id hash (8-byte blake2b of its id), with the file that won it,
so exact dedup is global across files and runs. Files commit in rank order (stream_rank.py), so
the first copy is normally the best-ranked one. A late file (retried after files ranked below it
were committed) that keeps a document already owned only by lower-ranked files adds an overrides
row (table, value, its file) instead of a keys/ids row; owners() returns both, and the caller
takes the best rank. Commit order per file: shard frames fsynced, then the store transaction
(keys, ids, overrides, files row, and meta.inflight = fid), then the ledger line. If a crash lands
between the last two, the next start deletes the in-flight file's rows, overrides included
(rollback_uncommitted), so a re-run of that file does not see its own documents as duplicates.
"""
import hashlib
import json
import os
import sqlite3
import time

import numpy as np

TABLES = ("keys", "ids")
BATCH = 5000


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def id_hash(doc_id: str) -> int:
    """8-byte blake2b of a document id, as a signed int64 (sqlite's integer type)."""
    return int.from_bytes(hashlib.blake2b(doc_id.encode("utf-8"), digest_size=8).digest(),
                          "big", signed=True)


class Ledger:
    def __init__(self, path):
        self.path, self.events, self.repaired = path, [], 0
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = f.read()
            cut = data.rfind(b"\n") + 1
            if cut < len(data):                         # torn last line: never committed
                self.repaired = len(data) - cut
                os.truncate(path, cut)
                data = data[:cut]
            for n, line in enumerate(data.splitlines(), 1):
                if line.strip():
                    try:
                        self.events.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise SystemExit(f"{path}:{n}: corrupt ledger line ({e})")
        self.fh = open(path, "a", encoding="utf-8")

    def append(self, event: str, **fields) -> dict:
        ev = {"event": event, "ts": now(), **fields}
        self.fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
        self.fh.flush()
        os.fsync(self.fh.fileno())
        self.events.append(ev)
        return ev

    def close(self):
        self.fh.close()

    def of(self, event):
        return [e for e in self.events if e["event"] == event]

    def done(self) -> dict:
        return {e["fid"]: e for e in self.of("file_done")}

    def sealed(self) -> dict:
        return {e["shard"]: e for e in self.of("shard_sealed")}

    def deleted(self) -> set:
        return {e["fid"] for e in self.of("raw_deleted")}

    def segments(self, source=None) -> list:
        return [s for e in self.of("file_done") if source is None or e["source"] == source
                for s in e["segments"]]


class KeyStore:
    def __init__(self, path):
        self.db = sqlite3.connect(path, isolation_level=None)
        for p in ("journal_mode=WAL", "synchronous=FULL", "cache_size=-262144",
                  "temp_store=MEMORY"):
            self.db.execute(f"PRAGMA {p}")
        for t in TABLES:
            self.db.execute(f"CREATE TABLE IF NOT EXISTS {t} (k INTEGER PRIMARY KEY, "
                            "f INTEGER NOT NULL) WITHOUT ROWID")
        self.db.execute("CREATE TABLE IF NOT EXISTS files (f INTEGER PRIMARY KEY, "
                        "fid TEXT UNIQUE NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS meta (name TEXT PRIMARY KEY, value TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS overrides (t TEXT NOT NULL, k INTEGER NOT "
                        "NULL, f INTEGER NOT NULL, PRIMARY KEY (t, k, f)) WITHOUT ROWID")
        self._fids = {}
        self._count_overrides()

    def _count_overrides(self):
        self.n_over = self.db.execute("SELECT count(*) FROM overrides").fetchone()[0]

    def close(self):
        self.db.close()

    def owners(self, table, values) -> list:
        """-> [(value, file number)] for every owner of the values already in the table: the
        file that first committed it, plus the late files that took it over (overrides)."""
        assert table in TABLES
        vals, out = [int(v) for v in values], []
        for i in range(0, len(vals), BATCH):
            part = vals[i:i + BATCH]
            qs = ",".join("?" * len(part))
            out += self.db.execute(f"SELECT k, f FROM {table} WHERE k IN ({qs})", part).fetchall()
            if self.n_over:
                out += self.db.execute(f"SELECT k, f FROM overrides WHERE t=? AND k IN ({qs})",
                                       [table] + part).fetchall()
        return out

    def fid(self, f) -> str:
        if f not in self._fids:
            row = self.db.execute("SELECT fid FROM files WHERE f=?", (f,)).fetchone()
            if row is None:
                raise KeyError(f"file number {f} has no files row")
            self._fids[f] = row[0]
        return self._fids[f]

    def commit_file(self, fid, keys, ids, over=None):
        """One transaction: the file's surviving keys and id hashes, its files row, inflight=fid;
        over = {"keys": [...], "ids": [...]}: values it keeps although lower-ranked files own
        them (overrides rows). A key already present raises (the caller checked first, so that
        would be a bug)."""
        db, over = self.db, over or {}
        db.execute("BEGIN IMMEDIATE")
        try:
            f = db.execute("INSERT INTO files (fid) VALUES (?)", (fid,)).lastrowid
            for t, vals in (("keys", keys), ("ids", ids)):
                db.executemany(f"INSERT INTO {t} (k, f) VALUES (?, ?)",
                               ((int(v), f) for v in np.sort(np.asarray(vals, dtype=np.int64))))
                db.executemany("INSERT INTO overrides (t, k, f) VALUES (?, ?, ?)",
                               ((t, int(v), f) for v in sorted(set(over.get(t) or ()))))
            db.execute("INSERT OR REPLACE INTO meta VALUES ('inflight', ?)", (fid,))
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
        self._count_overrides()

    def rollback_uncommitted(self, done: set) -> list:
        """Delete the rows of the in-flight file when the ledger never recorded it as done, then
        require the store's files to be exactly the ledger's done set. -> fids rolled back."""
        db, rolled = self.db, []
        row = db.execute("SELECT value FROM meta WHERE name='inflight'").fetchone()
        if row and row[0] and row[0] not in done:
            f = db.execute("SELECT f FROM files WHERE fid=?", (row[0],)).fetchone()
            db.execute("BEGIN IMMEDIATE")
            if f:
                for t in TABLES + ("overrides",):
                    db.execute(f"DELETE FROM {t} WHERE f=?", (f[0],))
                db.execute("DELETE FROM files WHERE f=?", (f[0],))
                self._fids.pop(f[0], None)
            db.execute("DELETE FROM meta WHERE name='inflight'")
            db.execute("COMMIT")
            rolled.append(row[0])
            self._count_overrides()
        have = {r[0] for r in db.execute("SELECT fid FROM files")}
        if have != set(done):
            extra, missing = sorted(have - set(done)), sorted(set(done) - have)
            raise SystemExit(f"dedup store and ledger disagree: {len(extra)} files only in the "
                             f"store {extra[:3]}, {len(missing)} only in the ledger {missing[:3]}. "
                             "They must be kept (and deleted) together.")
        return rolled
