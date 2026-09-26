"""Tokenizer v0 training sample + disjoint held-out sets, from extract.py's shards.

    python corpus/make_tok_sample.py EXTRACTED_DIR OUT_DIR [--total-mb 1500] [--seed 1]
        [--share-chat .37 --share-web .28 --share-se .15 --share-books .10 --share-wiki .10]
        [--heldout-mb 5] [--heldout-max-frac 0.25] [--short-supply scale|fill]
        [--chat-max-repeat 16] [--irc-max-share 0.04] [--allow-unrecorded-license cccc]

Writes OUT_DIR/train/<source>.jsonl, OUT_DIR/heldout/<source>.jsonl (same records as the shards)
and OUT_DIR/manifest.json (config, plan, the chat repeat "weights" the trainer applies, achieved
bytes and shares counting repeats, sha256 of every output file, the checks below and the v0 notes).
Selection rules: sample_plan.py.

D8 guard: a record whose license_basis is 'unrecorded' (CCCC) is used only when its source is named
in --allow-unrecorded-license; otherwise the run stops unless that source's group share is 0. The
manifest records the sources allowed this way.

Held-out: per source, the lowest-u docs up to --heldout-mb (default 5 MB), but never more than
--heldout-max-frac of that source (default 0.25, so a small source such as Dolly is not mostly
eaten by its held-out set). OASST2 held-out threads come from non-reserve trees only (the extract
already removed the reserve; this script refuses to run if it finds a reserved tree) and share no
tree id with the training sample.

Control and markup token strings (<|user|>, <|end|>, <think> ... ; sample_plan.SPECIAL_STRINGS)
are replaced by a space in the output text, per the tokenizer sample contract in
tokenizer/sample_io.py; the manifest counts the docs touched.

Checks recorded (and enforced, the run fails otherwise): no doc id and no OASST2 tree id appears
in both train and held-out; no OASST2 tree in the OOD-H reserve appears anywhere in the output;
no doc id repeats in the input.
"""
import argparse
import hashlib
import json
import os
import sys
from array import array

import numpy as np

import sample_plan as P
from oodh import in_oodh_reserve

SOURCES = [s for ss in P.GROUPS.values() for s in ss]


def scan(ex_dir, seed):
    """-> ({source: {"shards", "u", "b"}}, extract_stats sha256, {source: unrecorded-license docs})."""
    with open(os.path.join(ex_dir, "extract_stats.json"), "rb") as f:
        raw = f.read()
    stats = json.loads(raw)
    idx, seen, unrec = {}, set(), {}
    for s in SOURCES:
        shards = sorted(o["path"] for o in stats["outputs"] if o["source"] == s)
        u, b = array("d"), array("q")
        for sh in shards:
            with open(os.path.join(ex_dir, sh), encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    if r["id"] in seen:
                        raise SystemExit(f"doc id repeats in the input: {r['id']}")
                    seen.add(r["id"])
                    key = r["meta"].get("split_key") or r["id"]
                    if s == "oasst2":
                        key = r["meta"]["tree_id"]
                        if in_oodh_reserve(key):
                            raise SystemExit(f"OOD-H reserved tree in the input: {key}")
                    if r["meta"].get("license_basis") == "unrecorded":
                        unrec[s] = unrec.get(s, 0) + 1
                    u.append(P.unit_hash(seed, s, key))
                    b.append(len(r["text"].encode("utf-8")))
        idx[s] = {"shards": shards, "u": np.frombuffer(u, dtype=np.float64),
                  "b": np.frombuffer(b, dtype=np.int64)}
    return idx, hashlib.sha256(raw).hexdigest(), unrec


def assign(idx, shares, total, heldout, heldout_frac, mode, max_repeat, irc_share):
    """-> ({source: labels int8 (0 unused, 1 train, 2 heldout)}, plan dict, chat repeat count)."""
    labels, pools, avail_s, held = {}, {}, {}, {}
    for s, d in idx.items():
        order = np.argsort(d["u"], kind="stable")
        us, bs = d["u"][order], d["b"][order]
        cap = min(heldout, heldout_frac * float(bs.sum()))
        n_h = P.extend_ties(us, P.take_prefix(bs, cap))
        lab = np.zeros(len(order), dtype=np.int8)
        lab[order[:n_h]] = 2
        labels[s], pools[s] = lab, order[n_h:]
        held[s] = int(bs[:n_h].sum())
        avail_s[s] = int(bs[n_h:].sum())
    gt, t = P.plan(avail_s, shares, total, mode, max_repeat, irc_share)
    st, w = P.source_targets(gt, avail_s, t, max_repeat, irc_share)
    for s, pool in pools.items():
        n = P.take_prefix(idx[s]["b"][pool], st[s])
        labels[s][pool[:n]] = 1
    plan = {"available_train_bytes": avail_s, "planned_total_bytes": t,
            "chat_capacity_bytes": P.chat_capacity(avail_s, t, max_repeat, irc_share),
            "heldout_target_bytes": {s: min(heldout, heldout_frac * float(idx[s]["b"].sum()))
                                     for s in idx},
            "group_targets": gt, "source_targets": st, "chat_repeat": w}
    return labels, plan, w


def write(ex_dir, out_dir, idx, labels):
    outs, ids, trees = {}, {1: set(), 2: set()}, {1: set(), 2: set()}
    reserved = stripped = 0
    for split in ("train", "heldout"):
        os.makedirs(os.path.join(out_dir, split), exist_ok=True)
    for s in SOURCES:
        i = 0
        for sh in idx[s]["shards"]:
            with open(os.path.join(ex_dir, sh), "rb") as fh:
                for line in fh:
                    lab = int(labels[s][i])
                    i += 1
                    if not lab:
                        continue
                    rel = f"{'train' if lab == 1 else 'heldout'}/{s}.jsonl"
                    if rel not in outs:
                        outs[rel] = {"fh": open(os.path.join(out_dir, rel), "wb"),
                                     "h": hashlib.sha256(), "docs": 0, "bytes": 0,
                                     "text_bytes": 0}
                    o = outs[rel]
                    r = json.loads(line)
                    if P.SPECIAL_RE.search(r["text"]):
                        r["text"] = P.SPECIAL_RE.sub(" ", r["text"])
                        for t in r.get("turns") or []:
                            t["text"] = P.SPECIAL_RE.sub(" ", t["text"])
                        line = (json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8")
                        stripped += 1
                    o["fh"].write(line)
                    o["h"].update(line)
                    o["docs"] += 1
                    o["bytes"] += len(line)
                    o["text_bytes"] += len(r["text"].encode("utf-8"))
                    ids[lab].add(r["id"])
                    if s == "oasst2":
                        trees[lab].add(r["meta"]["tree_id"])
                        reserved += in_oodh_reserve(r["meta"]["tree_id"])
        assert i == len(labels[s]), f"{s}: line count changed since the scan"
    files = []
    for rel, o in sorted(outs.items()):
        o["fh"].close()
        files.append({"path": rel, "sha256": o["h"].hexdigest(), "bytes": o["bytes"],
                      "docs": o["docs"], "text_bytes": o["text_bytes"]})
    checks = {"train_heldout_id_overlap": len(ids[1] & ids[2]),
              "train_heldout_tree_overlap": len(trees[1] & trees[2]),
              "oodh_reserved_trees_in_output": reserved}
    return files, checks, stripped


def build_sample(ex_dir, out_dir, shares=None, total_bytes=1500e6, heldout_bytes=5e6,
                 heldout_max_frac=0.25, seed=1, short_supply="scale", allow_archaic=False,
                 max_repeat=P.CHAT_MAX_REPEAT, irc_share=P.IRC_MAX_SHARE, allow_unrecorded=()):
    shares = dict(shares or P.DEFAULT_SHARES)
    P.check_shares(shares, allow_archaic)
    if os.path.exists(os.path.join(out_dir, "manifest.json")):
        raise SystemExit(f"{out_dir}/manifest.json exists; use a new OUT_DIR")
    idx, stats_sha, unrec = scan(ex_dir, seed)
    blocked = {s: n for s, n in unrec.items()
               if s not in allow_unrecorded and shares[P.SOURCE_GROUP[s]] > 0}
    if blocked:
        raise SystemExit(f"docs with no per-document license (D8): {blocked}; pass "
                         f"--allow-unrecorded-license {','.join(sorted(blocked))} to use them, or set "
                         f"their group share to 0")
    labels, plan, w = assign(idx, shares, total_bytes, heldout_bytes, heldout_max_frac,
                             short_supply, max_repeat, irc_share)
    weights = {s: w for s in P.WHOLE_CHAT} if w > 1 else {}
    files, checks, stripped = write(ex_dir, out_dir, idx, labels)
    by = {f["path"]: f["text_bytes"] for f in files}
    train = {s: by.get(f"train/{s}.jsonl", 0) for s in SOURCES}
    weighted = {s: train[s] * weights.get(s, 1) for s in SOURCES}
    tot = sum(weighted.values())
    achieved = {g: (sum(weighted[s] for s in ss) / tot if tot else 0.0)
                for g, ss in P.GROUPS.items()}
    manifest = {
        "config": {"shares": shares, "total_bytes": total_bytes, "heldout_bytes": heldout_bytes,
                   "heldout_max_frac": heldout_max_frac, "seed": seed,
                   "short_supply": short_supply, "groups": P.GROUPS,
                   "chat_max_repeat": max_repeat, "irc_max_share": irc_share,
                   "allow_unrecorded_license": sorted(allow_unrecorded)},
        "input": {"extract_stats_sha256": stats_sha,
                  "shards": sum(len(d["shards"]) for d in idx.values())},
        "plan": plan, "weights": weights,
        "license_unrecorded_docs": unrec,
        "achieved": {"train_text_bytes": train, "train_weighted_bytes": weighted,
                     "train_total_text_bytes": sum(train.values()),
                     "train_total_weighted_bytes": tot,
                     "heldout_text_bytes": {s: by.get(f"heldout/{s}.jsonl", 0) for s in SOURCES},
                     "group_shares": achieved,
                     "source_shares": {s: (weighted[s] / tot if tot else 0.0) for s in SOURCES},
                     "share_error_points": {g: round(100 * (achieved[g] - shares[g]), 3)
                                            for g in shares}},
        "docs_with_special_strings_stripped": stripped, "files": files, "checks": checks,
        "notes": P.V0_NOTES}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    if any(checks.values()):
        raise SystemExit(f"sample checks failed: {checks}")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("extracted_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--total-mb", type=float, default=1500)
    ap.add_argument("--heldout-mb", type=float, default=5)
    ap.add_argument("--heldout-max-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--short-supply", choices=["scale", "fill"], default="scale")
    ap.add_argument("--allow-archaic-over", action="store_true")
    ap.add_argument("--chat-max-repeat", type=int, default=P.CHAT_MAX_REPEAT)
    ap.add_argument("--irc-max-share", type=float, default=P.IRC_MAX_SHARE)
    ap.add_argument("--allow-unrecorded-license", default="",
                    help="comma list of sources whose docs lack a per-document license (D8)")
    for g, v in P.DEFAULT_SHARES.items():
        ap.add_argument(f"--share-{g}", type=float, default=v)
    a = ap.parse_args(argv)
    m = build_sample(a.extracted_dir, a.out_dir, {g: getattr(a, f"share_{g}") for g in P.GROUPS},
                     a.total_mb * 1e6, a.heldout_mb * 1e6, a.heldout_max_frac, a.seed,
                     a.short_supply, a.allow_archaic_over, a.chat_max_repeat, a.irc_max_share,
                     tuple(x for x in a.allow_unrecorded_license.split(",") if x))
    print(json.dumps({"achieved": m["achieved"], "checks": m["checks"]}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
