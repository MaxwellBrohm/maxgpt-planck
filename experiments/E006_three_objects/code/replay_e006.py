"""E006 arm G replay pool (notes.txt ARMS G, "Replay data"). Pure Python, stdlib + the repo's corpus/ modules; no
model, no tokenizer (the tokenizer-side drops happen at stream time in e006_sets.replay_stream).
Source: the OASST2 threads of the tokenizer sample's TRAIN side (tok_sample_v0c/train/oasst2.jsonl, 3,338 threads,
built by corpus/readers_chat.py from 2023-11-05_oasst2_all.trees.jsonl.gz: reserve trees dropped first, English,
ready_for_export, deleted/synthetic/model-written/review-failed/spam messages dropped, hygiene.aiism per message,
best-ranked path, trailing user turns cut). Drops, first matching reason wins:
  reserve       the tree is in the OOD-H reserve (oodh.in_oodh_reserve; expected 0)
  heldout       the tree is on the tokenizer sample's held-out side (expected 0)
  reserve_leak  a user turn (oodh.prompt_key, >= oodh.LEAK_MIN_CHARS) equals a user turn of a reserved tree of the
                raw file (the leak rule seen from the training side)
  aiism         any turn trips hygiene.aiism (expected 0)
  eval_8gram    a word 8-gram (text_e004.words) shared with any item text of draws 4004/4104/4204, BIG, H5L, AL,
                or a chat-probe user turn or system prompt
  probe_turn    a turn equal (prompt_key) to a chat-probe user turn, or containing one of 3+ words (critique 3)
  eval_surname  an eval, AL, BIG or H5L surname as a whole word, any case (critique 3)
Also written: the kbig441 exposure lists (strict: one turn holds the gold and every content word of the question;
loose: the gold and any capitalised content word of the question), fixed before any run.
usage (on the PC): python3 -B replay_e006.py --tok-sample DIR --raw FILE.gz --out DIR
writes OUT/replay_pool.jsonl (+ .sha256), OUT/replay_pool_ids.txt, OUT/replay_build.json, OUT/exposed_kbig.json;
refuses to overwrite any of them."""
import argparse
import gzip
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "corpus"))
import oodh                      # noqa: E402
import hygiene as H              # noqa: E402
from text_e004 import words, content_words  # noqa: E402

REASONS = ("reserve", "heldout", "reserve_leak", "aiism", "eval_8gram", "probe_turn", "eval_surname")
NGRAM, PROBE_MIN_WORDS = 8, 3


def tree_id(row):
    return row["meta"]["tree_id"]


def raw_trees(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def user_turns(m):
    if m.get("role") == "prompter":
        yield m.get("text") or ""
    for r in m.get("replies") or []:
        yield from user_turns(r)


def reserve_keys(path):
    keys = set()
    for t in raw_trees(path):
        if oodh.in_oodh_reserve(t["message_tree_id"]):
            keys |= {k for k in map(oodh.prompt_key, user_turns(t["prompt"])) if len(k) >= oodh.LEAK_MIN_CHARS}
    return keys


def item_texts(it):
    return [x for u, a in it["turns"] for x in (u, a)] + [it["question"], it.get("prefix") or ""]


def eval_side():
    """-> (texts, probe user turns, surnames): every eval text the replay must not share an 8-gram with."""
    import items as _I  # noqa: F401
    import items_e004 as I
    import items_al as AL
    import items_big_e006 as BG
    import battery as BT
    import heldout_e004 as HE
    texts = []
    for name in I.DRAWS:
        for its in I.draw(name).values():
            texts += [x for it in its for x in item_texts(it)]
    for its in BG.load().values():
        texts += [x for it in its for x in item_texts(it)]
    texts += [x for it in AL.load() for x in item_texts(it)]
    probe = [u for t in BT.TESTS for u in t["turns"]]
    texts += probe + [t["system"] for t in BT.TESTS if t["system"]]
    names = set(HE.ALIAS_SURNAMES) | set(AL.AL_SURNAMES)
    for its in list(I.draw("eval").values()) + list(BG.load().values()) + [AL.load()]:
        for it in its:
            for key in ("alias", "aliases", "aliases_all"):
                v = it.get(key) or []
                for a in ([v] if isinstance(v, str) else (list(v.values()) if isinstance(v, dict) else list(v))):
                    if isinstance(a, str) and a:
                        names.add(a.split()[-1])
    return texts, probe, sorted(names)


def grams(text, n=NGRAM):
    w = words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def probe_matchers(probe):
    eq = {oodh.prompt_key(p) for p in probe}
    sub = [re.compile(r"(?<![a-z0-9])" + re.escape(oodh.prompt_key(p)) + r"(?![a-z0-9])")
           for p in set(probe) if len(p.split()) >= PROBE_MIN_WORDS]
    return eq, sub


def surname_re(names):
    return re.compile(r"(?<![A-Za-z])(?:" + "|".join(map(re.escape, names)) + r")(?![A-Za-z])", re.I)


def drop_reason(row, ctx):
    tid = tree_id(row)
    turns = row["turns"]
    if oodh.in_oodh_reserve(tid):
        return "reserve"
    if tid in ctx["heldout_ids"]:
        return "heldout"
    if any(len(k) >= oodh.LEAK_MIN_CHARS and k in ctx["reserve_keys"]
           for k in (oodh.prompt_key(t["text"]) for t in turns if t["role"] == "user")):
        return "reserve_leak"
    if any(H.aiism(t["text"]) for t in turns):
        return "aiism"
    if any(grams(t["text"]) & ctx["eval_grams"] for t in turns):
        return "eval_8gram"
    eq, sub = ctx["probe"]
    for t in turns:
        k = oodh.prompt_key(t["text"])
        if k in eq or any(r.search(k) for r in sub):
            return "probe_turn"
    if any(ctx["surnames"].search(t["text"]) for t in turns):
        return "eval_surname"
    return None


def _has(text, phrase):
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(phrase) + r"(?![A-Za-z0-9])", text, re.I) is not None


def _has_word(text, w):
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(w) + r"(?:'s|s)?(?![A-Za-z0-9])", text, re.I) is not None


def exposure(pool):
    """-> {"strict": [qid...], "loose": [qid...]} over kbig441 and every turn of every pool thread."""
    import kbig_items as KB
    turns = [t["text"] for r in pool for t in r["turns"]]
    out = {"strict": [], "loose": []}
    for it in KB.build():
        gold, q = it["cands"]["gold"].strip(), it["question"]
        cw = content_words(q)
        capset = {x.lower() for x in re.findall(r"[A-Za-z][A-Za-z']*", q)[1:] if x[0].isupper()}
        caps = [w for w in cw if w in capset]
        hits = [t for t in turns if _has(t, gold)]
        if cw and any(all(_has_word(t, w) for w in cw) for t in hits):
            out["strict"].append(it["qid"])
        if any(any(_has_word(t, w) for w in (caps or cw)) for t in hits):
            out["loose"].append(it["qid"])
    return out


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def build(rows, heldout_ids, rkeys, eval_texts, probe, names):
    ctx = {"heldout_ids": heldout_ids, "reserve_keys": rkeys, "probe": probe_matchers(probe),
           "eval_grams": set().union(*(grams(t) for t in eval_texts)), "surnames": surname_re(names)}
    kept, reasons = [], []
    for r in sorted(rows, key=lambda r: r["id"]):
        why = drop_reason(r, ctx)
        reasons.append((tree_id(r), why or "keep"))
        if why is None:
            kept.append({"id": r["id"], "tree_id": tree_id(r), "turns": r["turns"]})
    return kept, reasons


def write_x(path, text):
    with open(path, "x") as f:
        f.write(text)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok-sample", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    src = os.path.join(a.tok_sample, "train", "oasst2.jsonl")
    rows = [json.loads(l) for l in open(src) if l.strip()]
    held = {json.loads(l)["meta"]["tree_id"] for l in open(os.path.join(a.tok_sample, "heldout", "oasst2.jsonl"))}
    rkeys = reserve_keys(a.raw)
    leaked = oodh.leaked_reserve_trees(raw_trees(a.raw))
    texts, probe, names = eval_side()
    kept, reasons = build(rows, held, rkeys, texts, probe, names)
    counts = {k: sum(1 for _, r in reasons if r == k) for k in REASONS + ("keep",)}
    pool_text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in kept)
    exp = exposure(kept)
    os.makedirs(a.out, exist_ok=True)
    write_x(os.path.join(a.out, "replay_pool.jsonl"), pool_text)
    h = sha_bytes(pool_text.encode())
    write_x(os.path.join(a.out, "replay_pool.sha256"), f"{h}  replay_pool.jsonl\n")
    write_x(os.path.join(a.out, "replay_pool_ids.txt"), "".join(f"{t}\t{r}\n" for t, r in reasons))
    expj = json.dumps(exp, sort_keys=True)
    write_x(os.path.join(a.out, "exposed_kbig.json"), json.dumps(dict(exp, sha256=sha_bytes(expj.encode())), indent=1))
    info = {"source": "tok_sample_v0c/train/oasst2.jsonl", "source_sha256": sha_bytes(open(src, "rb").read()),
            "raw_sha256": sha_bytes(open(a.raw, "rb").read()), "n_source": len(rows), "counts": counts,
            "reserve_keys": len(rkeys), "leaked_reserve_trees_in_raw": len(leaked), "eval_surnames": names,
            "probe_turns": len(probe), "pool_sha256": h, "n_pool": len(kept),
            "exposed_strict": len(exp["strict"]), "exposed_loose": len(exp["loose"])}
    write_x(os.path.join(a.out, "replay_build.json"), json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    return 0 if counts["keep"] >= 1604 and not any(counts[k] for k in ("reserve", "heldout", "aiism")) else 1


if __name__ == "__main__":
    sys.exit(main())
