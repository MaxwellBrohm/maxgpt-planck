"""Flatten a run's data_prep/bpb.py results (OUT_DIR/bpb/<checkpoint>.json) into bpb.jsonl lines, one per
checkpoint x set x split, plus the two E2 metrics (E2 notes, METRIC):
  CHAT   the oasst2 set (all scored turns; its user and assistant splits are the per-role lines)
  PROSE  cccc + gutenberg + wikimedia pooled: total bits / total bytes (a micro-average over bytes)
Stdlib only.   python bpb_lines.py OUT_DIR > OUT_DIR/bpb.jsonl
"""
from __future__ import annotations

import glob
import json
import os
import sys

CHAT, PROSE = "oasst2", ("cccc", "gutenberg", "wikimedia")
KEYS = ("bits", "bytes", "tokens", "windows", "bpb")


def lines_for(path: str) -> list[dict]:
    r = json.load(open(path, encoding="utf-8"))
    head = {"ckpt": r["checkpoint"], "step": r["step"], "tokens_trained": r.get("tokens_trained"),
            "evalset_sha256": r["evalset_sha256"], "tokenizer_sha256": r["tokenizer"]["sha256"],
            "precision": r["precision"], "max_windows": r["max_windows"]}
    out = []
    for name, st in sorted(r["sets"].items()):
        out.append({**head, "set": name, "split": "all", "truncated": st.get("truncated"),
                    **{k: st.get(k) for k in KEYS}})
        for role, rs in sorted((st.get("by_role") or {}).items()):
            out.append({**head, "set": name, "split": role, **{k: rs.get(k) for k in KEYS}})
    if CHAT in r["sets"]:
        st = r["sets"][CHAT]
        out.append({**head, "set": "CHAT", "split": "all", **{k: st.get(k) for k in KEYS}})
    if all(s in r["sets"] for s in PROSE):
        bits = sum(r["sets"][s]["bits"] for s in PROSE)
        nb = sum(r["sets"][s]["bytes"] for s in PROSE)
        out.append({**head, "set": "PROSE", "split": "all", "bits": bits, "bytes": nb,
                    "tokens": sum(r["sets"][s]["tokens"] for s in PROSE),
                    "windows": sum(r["sets"][s]["windows"] for s in PROSE), "bpb": bits / nb})
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out_dir = argv[0]
    run = os.path.basename(os.path.normpath(out_dir))
    for p in sorted(glob.glob(os.path.join(out_dir, "bpb", "*.json"))):
        for ln in lines_for(p):
            print(json.dumps({"run": run, **ln}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
