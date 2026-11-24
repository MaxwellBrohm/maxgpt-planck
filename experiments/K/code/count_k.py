"""K10 count report (research/CUTS_AUDIT_2026-10.md row 135; K5 notes CHANGES AFTER CRITIQUE): the supervised-token
count of every K stream per arm, counted only (no arm, run or rule follows from it; the noisy-lookup arm stays out).

  python count_k.py DATA_ROOT [--arms F0 FL FH CAP L0 L50 L100] [--out counts.json]

For each arm, the K sources of its data: block (gen.data_block; the starter-corpus LANG sources are not K data and
are left out) are read from the written files, and every item is counted as the harness loader would feed it:
token sources by harness TokenShardSource's rule (a document ends at EOT; one longer than max_len is cut, each
piece its own item; every token but an item's first is a supervised target), chat sources rendered by harness
ChatTemplate with the run's role ids and assistant loss (supervised = the item's loss flags). Per source: items,
tokens, supervised targets and their share of tokens, split by kind: "fb" or "filler" (the first name token's pool:
F1 or Z1) for bio docs and QA chats, "lookup_hit" / "lookup_miss" for K5 chats (by the tool turn). Per arm, the FACT
stream (sources named fact_*): written tokens, supervised targets, the supervised share, and that share times the
FACT draw of SPEC 7 (FACT share x B x 0.9885 drawn tokens per slot; CAP: share 1.0) as the expected supervised
targets drawn. Reads files only; no model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import kcommon as K

if K.HARNESS not in sys.path:
    sys.path.insert(0, K.HARNESS)

ARMS = ("F0", "FL", "FH", "CAP", "L0", "L50", "L100")


def _add(d: dict, kind: str, n_tok: int, n_sup: int, n_items: int = 1) -> None:
    e = d.setdefault(kind, {"items": 0, "tokens": 0, "supervised": 0})
    e["items"] += n_items
    e["tokens"] += n_tok
    e["supervised"] += n_sup


def count_tokens(paths: list[str], eot: int, max_len: int, first_kind) -> dict:
    out: dict = {}
    for p in paths:
        ids = np.fromfile(p, dtype=np.uint16).astype(np.int64)
        bounds = [0] + (np.flatnonzero(ids == eot) + 1).tolist()
        if bounds[-1] < len(ids):
            bounds.append(len(ids))
        for a, b in zip(bounds[:-1], bounds[1:]):
            n = b - a
            pieces = -(-n // max_len)
            _add(out, first_kind(int(ids[a])), n, n - pieces, pieces)
    return out


def kind_of_chat(rec: dict, owner: dict) -> str:
    turns = rec["turns"]
    if turns and turns[0]["role"] == "tool":
        return "lookup_miss" if K.NONE in turns[0]["ids"] else "lookup_hit"
    return name_kind(int(turns[0]["ids"][0]), owner)


def name_kind(tok: int, owner: dict) -> str:
    return {"F1": "fb", "Z1": "filler"}.get(owner.get(tok), "other")


def count_chat(paths: list[str], tmpl, max_len: int, owner: dict, fixed: str | None = None) -> dict:
    out: dict = {}
    for p in paths:
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                ids, flags = tmpl.render(rec)
                if len(ids) > max_len or len(ids) < 2:      # ChatJsonlSource drops these
                    _add(out, "dropped", len(ids), 0)
                    continue
                _add(out, fixed or kind_of_chat(rec, owner), len(ids), int(flags.sum()))
    return out


def _total(by_kind: dict) -> dict:
    t = {k: sum(e[k] for e in by_kind.values()) for k in ("items", "tokens", "supervised")}
    t["sup_share"] = t["supervised"] / t["tokens"] if t["tokens"] else None
    return t


def count_arm(root: str, arm: str, max_len: int = 2048) -> dict:
    import gen
    import pools as P
    from chat_template import ChatTemplate
    from sources import expand
    owner = P.get().owner()
    block = gen.data_block(arm, root)
    tmpl = ChatTemplate(dict(block["chat"]["role_ids"]), int(block["chat"]["end_id"]), block["chat"]["loss"])
    out = {"sources": {}}
    for s in block["sources"]:
        if not s["paths"][0].startswith(root):
            continue                                        # LANG: the starter corpus, not K data
        paths = expand(s["paths"])
        if not paths:
            raise FileNotFoundError(f"{arm}: no files for source {s['name']} ({s['paths'][0]})")
        if s["kind"] == "tokens":
            bk = count_tokens(paths, int(s["eot_id"]), max_len, lambda t: name_kind(t, owner))
        else:
            bk = count_chat(paths, tmpl, max_len, owner, "skill" if s["name"] == "skill" else None)
        out["sources"][s["name"]] = {"share": s["share"], "files": len(paths), "by_kind": bk, "total": _total(bk)}
    fact = [v["total"] for n, v in out["sources"].items() if n.startswith("fact_")]
    ft = {k: sum(t[k] for t in fact) for k in ("items", "tokens", "supervised")}
    ft["sup_share"] = ft["supervised"] / ft["tokens"] if ft["tokens"] else None
    share = 1.0 if arm == "CAP" else K.SHARES["fact"]
    ft["draw_tokens"] = share * K.B_SLOTS * K.FILL
    ft["draw_supervised_expected"] = ft["sup_share"] * ft["draw_tokens"] if ft["sup_share"] is not None else None
    out["fact"] = ft
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)
    rep = {"root_name": os.path.basename(root), "arms": {arm: count_arm(root, arm) for arm in a.arms}}
    if a.out:
        K.write_json(a.out, rep)
    for arm, r in rep["arms"].items():
        f = r["fact"]
        print(f"{arm:5s} FACT tokens {f['tokens']:>12,} supervised {f['supervised']:>12,} share "
              f"{f['sup_share']:.4f} expected drawn supervised {f['draw_supervised_expected']:,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
