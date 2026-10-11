"""SCREENS DEVIATION 1 (SCREENS.txt, 2026-10-10): durable scoring outputs and run markers for queue_screens.sh.
Stdlib plus harness/durable.py (fsync the data, rename, fsync the directory). Never loads a model.

  python score_sync.py whole JSON_FILE        exit 0 if the file parses as a JSON object, else 1 (a power cut can
                                              leave NUL bytes or a lost tail in a bpb.py output written in place)
  python score_sync.py publish OUT_DIR        every bpb/*.json must be whole; each is fsynced, then bpb/; then
                                              bpb.jsonl (E2's bpb_lines.py output, byte for byte) is written through
                                              a fsynced temporary file and a rename; then SCORED the same way. So
                                              SCORED never names score files a power cut can still lose (before,
                                              queue_lib.sh redirected bpb_lines.py into bpb.jsonl and the queue
                                              touched SCORED with no fsync)
  python score_sync.py mark FILE TEXT         FILE holds TEXT and a newline, written durably (QRUN, GAP)
  python score_sync.py append FILE TEXT       one line appended to FILE, then fsynced (crashes.txt)
Exit 2: usage. publish exits 1 (and writes nothing) when a bpb/*.json is not whole or there is none.
"""
from __future__ import annotations

import contextlib
import glob
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for d in (os.path.join(ROOT, "harness"), os.path.join(ROOT, "experiments", "E2_lr_transfer")):
    if d not in sys.path:
        sys.path.insert(0, d)
import bpb_lines  # noqa: E402  (E2's flattener, unchanged)
import durable  # noqa: E402  (harness/durable.py)


def whole(path: str) -> bool:
    try:
        with open(path, encoding="utf-8") as f:
            return isinstance(json.load(f), dict)
    except (OSError, ValueError):
        return False


def write_durably(path: str, text: str) -> None:
    with durable.atomic_write(path, "w", encoding="utf-8") as f:
        f.write(text)


def publish(out: str) -> int:
    files = sorted(glob.glob(os.path.join(out, "bpb", "*.json")))
    bad = [os.path.basename(p) for p in files if not whole(p)]
    if not files or bad:
        print(f"score_sync publish: {'no bpb/*.json' if not files else 'not whole: ' + ', '.join(bad)}; "
              f"nothing written in {os.path.basename(os.path.normpath(out))}", file=sys.stderr)
        return 1
    for p in files:
        durable.fsync_file(p)
    durable.fsync_dir(os.path.join(out, "bpb"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = bpb_lines.main([out])
    if rc:
        return 1
    write_durably(os.path.join(out, "bpb.jsonl"), buf.getvalue())
    write_durably(os.path.join(out, "SCORED"), "")
    return 0


def append(path: str, text: str) -> None:
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")
        f.flush()
        os.fsync(f.fileno())
    if new:
        durable.fsync_dir(os.path.dirname(os.path.abspath(path)))


def main(argv=None) -> int:
    a = sys.argv[1:] if argv is None else argv
    if a[:1] == ["whole"] and len(a) == 2:
        return 0 if whole(a[1]) else 1
    if a[:1] == ["publish"] and len(a) == 2:
        return publish(a[1])
    if a[:1] == ["mark"] and len(a) == 3:
        write_durably(a[1], a[2] + "\n")
        return 0
    if a[:1] == ["append"] and len(a) == 3:
        append(a[1], a[2])
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
