"""W5 dry build (BANKPASS s7 (1)): N skeletons built on a frozen bank set (PC CPU; no model, no teacher call), and
what they carry: the FAKE refs each skeleton records (assemble.fake_refs: the bank set, pools, list names, personas,
topic words), which FAKE refs those are, the author of every exact line (s7 (2): each teacher about a third of the
exact lines), how often the most used lines repeat (SPEC 4's cap is 50 uses per 100k chats; the sampler does not
enforce it), and the exact share. Counts only: no line text is printed or written.

    python -m bankpass.w5dry --banks BANKS_V1 --n 1000 --shard bankpass-dry-1005 [--out dry.json]"""
import argparse
import collections
import json
import os
import sys

import assemble as A
from bankpass import load, store


def line_authors(bank_dir, man):
    """{line id: author label} for every installed line bank, in the loader's order (kept records sorted by id)."""
    out = {}
    for bank, meta in man["banks"].items():
        recs = sorted((r for r in store.read_jsonl(os.path.join(bank_dir, meta["file"]), tolerate_torn_tail=False)
                       if r["status"] == "kept"), key=lambda r: r["id"])
        for i, r in enumerate(recs):
            out[f"{bank}.{i}"] = store.author_label(r["author"])
    return out


def summarize(sks, authors):
    fake = collections.Counter()
    exact, by_author, uses = 0, collections.Counter(), collections.Counter()
    user_turns = 0
    for sk in sks:
        for r in A.fake_refs(sk["provenance"]):
            fake[r.split("@")[0] if "@" in r else r] += 1
        for t in sk["turns"]:
            if t["role"] != "user":
                continue
            user_turns += 1
            if t["mode"] == "exact" and t.get("bank_ref"):
                exact += 1
                uses[t["bank_ref"]] += 1
                by_author[authors.get(t["bank_ref"], "FAKE")] += 1
        if sk["assistant"].get("system_ref"):
            uses[sk["assistant"]["system_ref"]] += 1
            by_author[authors.get(sk["assistant"]["system_ref"], "FAKE")] += 1
    n = len(sks) or 1
    top = uses.most_common(10)
    return {"skeletons": len(sks), "skeletons_fake": sum(bool(A.fake_refs(s["provenance"])) for s in sks),
            "fake_refs_by_name": dict(fake.most_common()), "fake_ref_names": len(fake),
            "user_turns": user_turns, "exact_user_turns": exact, "exact_share": round(exact / max(1, user_turns), 4),
            "exact_lines_by_author": dict(by_author.most_common()),
            "exact_author_shares": {k: round(v / max(1, sum(by_author.values())), 4) for k, v in by_author.items()},
            "top_line_uses_per_100k": [[lid.rsplit(".", 1)[0], round(c * 100000 / n)] for lid, c in top],
            "lines_used_5_or_more": sum(c >= 5 for c in uses.values()),       # 5 per 1,000 = 10x SPEC 4's cap
            "distinct_lines_used": len(uses)}


def build(bank_dir, n, shard, register="RM"):
    import skeleton_shard as SS
    bs = load.from_dir(bank_dir)
    undo = load.install(bs)
    try:
        sks = SS.shard(shard, n, register)
    finally:
        undo()
    rep = summarize(sks, line_authors(bank_dir, bs.manifest))
    rep.update(shard=shard, register=register, bank_manifest_sha256=store.sha256_file(
        os.path.join(bank_dir, "manifest.json")), fake_banks=bs.fake_banks, bankset_provenance=bs.provenance)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--shard", default="bankpass-dry-1005")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rep = build(a.banks, a.n, a.shard)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1, sort_keys=True)
    print(json.dumps(rep, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
