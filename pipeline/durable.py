"""Durable publishing: never leave a published name pointing at bytes that a power cut can still lose.

Incident 2026-09-29 (experiments/E2_lr_transfer/notes.txt "CRASH RESUME"): a power cut during the last saves of an
E2 trunk left ckpt_00012200.pt, stable_00012207.pt and final_00012208.pt at 0 bytes, while latest.json (pointing at
the 0-byte final) and the queue's TRUNK_DONE survived. runio._atomic_save did torch.save to path.tmp and os.replace
with no fsync, so ext4 (WSL) committed the rename before the data. The rule here, as corpus/stream_io.py
ShardSet._end_frame already does for shard frames:
  1. write the bytes under a temporary name, flush, os.fsync       the data is on disk
  2. os.replace(tmp, path)                                         the name points at it
  3. fsync the directory that holds path                           the rename is on disk
Anything written after step 3 (latest.json, a DONE marker, the process exiting 0 so a queue touches TRUNK_DONE)
can no longer name an empty file. A directory built as <dir>.partial gets the same: every file and directory in it
fsynced, the rename, then the parent fsynced. Step 3 is best effort (fsync_dir says why); steps 1-2 raise on error.
os.replace across filesystems (EXDEV) raises as before: a copy is not atomic, so there is no copy fallback.

Stdlib only. Byte-identical copies live in pipeline/durable.py and pc/wsl/durable.py (each of those trees is synced
and tested on its own); harness/test_durable.py fails if a copy drifts. rc12/dev_batch.py (rc12 is synced alone)
carries its own few lines of the same rule.
"""
from __future__ import annotations

import contextlib
import errno
import os
import sys

_WARNED: set = set()      # error kinds already warned about by fsync_dir in this process


def _dir_refused(path: str, e: OSError) -> None:
    code = errno.errorcode.get(e.errno, str(e.errno))
    if code not in _WARNED:
        _WARNED.add(code)
        print(f"durable: directory fsync refused on {path} ({code}): files are fsynced, but a rename there may not "
              f"survive a power cut (warned once per error kind)", file=sys.stderr, flush=True)


def fsync_dir(path: str) -> None:
    """fsync a directory (opened read-only, always closed) so a create or rename in it reaches the disk: ext4 may
    commit a rename before the data otherwise. Best effort on every platform. Every caller fsyncs the data first, so
    a refused directory fsync can only lose the rename itself after a power cut (the old file, or none, comes back),
    never publish empty bytes. Some filesystems refuse it (WSL drvfs mounts such as /mnt/d, FAT/exFAT, some network
    filesystems; which errno they return is not measured here), and that must not crash a run whose data is already
    safe, so no error raises: it is warned once per error kind on stderr. A failing FILE fsync still raises."""
    try:
        fd = os.open(path or ".", os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError as e:
        _dir_refused(path, e)
        return
    try:
        os.fsync(fd)
    except OSError as e:
        _dir_refused(path, e)
    finally:
        os.close(fd)


def fsync_file(path: str) -> None:
    """fsync a file that something else wrote and closed (torch.save(obj, path), np.save(path, a))."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def replace(tmp: str, path: str) -> None:
    """os.replace(tmp, path), then fsync the directory now holding path (and tmp's directory when it differs).
    The caller has already fsynced the data under tmp (atomic_write and publish_file do)."""
    os.replace(tmp, path)
    d = os.path.dirname(os.path.abspath(path))
    fsync_dir(d)
    s = os.path.dirname(os.path.abspath(tmp))
    if s != d:
        fsync_dir(s)


@contextlib.contextmanager
def atomic_write(path: str, mode: str = "w", tmp: str | None = None, **open_kw):
    """with atomic_write(p) as f: f.write(...)   -> tmp written, flushed, fsynced, renamed onto p, directory fsynced.
    An exception inside the block publishes nothing (tmp is left behind, as a plain tmp-and-rename leaves it)."""
    tmp = tmp or path + ".tmp"
    with open(tmp, mode, **open_kw) as f:
        yield f
        f.flush()
        os.fsync(f.fileno())
    replace(tmp, path)


def publish_file(tmp: str, path: str) -> None:
    """tmp was written and closed by someone else (torch.save to a path): fsync it, rename it onto path, fsync the
    directory."""
    fsync_file(tmp)
    replace(tmp, path)


def fsync_tree(root: str) -> None:
    """fsync every regular file under root, then each directory bottom-up (root last)."""
    for d, _dirs, files in os.walk(root, topdown=False):
        for n in sorted(files):
            p = os.path.join(d, n)
            if os.path.isfile(p) and not os.path.islink(p):
                fsync_file(p)
        fsync_dir(d)


def publish_dir(tmp: str, final: str) -> None:
    """a directory built under a temporary name (<dir>.partial): fsync its whole tree, rename it to final, fsync
    final's parent. After a crash there is either no final or a final whose every file is complete."""
    fsync_tree(tmp)
    os.replace(tmp, final)
    fsync_dir(os.path.dirname(os.path.abspath(final)))
