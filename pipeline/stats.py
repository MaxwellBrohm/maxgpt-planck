"""Diversity statistics over accepted conversations (SPEC 7 measurement side, CORPUS 4.4). CPU only, read-only.

  python3 -B stats.py RUN_DIR [--json OUT.json]      (RUN_DIR/accepted/*.jsonl, or any dir of accepted-*.jsonl)

Reports, per role (user, assistant, all):
  distinct-1/2/3 (distinct n-grams / total n-grams over the corpus), gzip ratio (compressed / raw bytes);
slot coverage per value type: slot count, distinct values, pool size and the share of the pool used, top value
  share, nonce share;
length histograms: user turns per conversation, words per user and assistant turn, estimated tokens per chat;
coverage matrix: kind x variant x d bin, register, slice, user style, topic count;
repetition against the SPEC 7 caps, on teacher-written turns with slot values masked by type:
  top 8-grams by share of conversations (cap 0.1%), top assistant sentences (cap 0.1%), top first-user-turn 4-word
  openings (cap 2%). A cap alarm is raised only from MIN_N conversations up (below that one repeat is over 0.1%).
Self-BLEU and embedding clusters (CORPUS 4.4) are not computed here."""
import argparse
import collections
import gzip
import json
import os
import re
import sys

sys.dont_write_bytecode = True

import dedup  # noqa: E402
import pools  # noqa: E402

CAP_NGRAM, CAP_SENT, CAP_OPEN = 0.001, 0.001, 0.02
MIN_N = 1000
WORD_EDGES = [1, 4, 8, 12, 16, 20, 25, 30, 40, 1000]
TOKEN_EDGES = [0, 100, 150, 200, 250, 300, 400, 500, 700, 1000, 1800, 100000]
TURN_EDGES = list(range(1, 14)) + [100]
_SENT = re.compile(r"(?<=[.!?])\s+")


def load(path):
    """accepted records; tolerates a torn last line (a run may be writing) without modifying any file."""
    d = os.path.join(path, "accepted") if os.path.isdir(os.path.join(path, "accepted")) else path
    names = sorted(f for f in os.listdir(d) if re.fullmatch(r"accepted-\d{5}\.jsonl", f))
    out = []
    for name in names:
        with open(os.path.join(d, name), encoding="utf-8") as f:
            for line in f:
                if line.endswith("\n") and line.strip():
                    out.append(json.loads(line))
    return out


def words(text):
    return dedup.norm(text).split()


def distinct(token_lists, n):
    seen, total = set(), 0
    for toks in token_lists:
        for i in range(len(toks) - n + 1):
            seen.add(tuple(toks[i:i + n]))
            total += 1
    return round(len(seen) / total, 4) if total else None


def gzip_ratio(texts):
    raw = "\n".join(texts).encode()
    return round(len(gzip.compress(raw, 9)) / len(raw), 4) if raw else None


def hist(values, edges):
    counts = [0] * (len(edges) - 1)
    for v in values:
        for k in range(len(edges) - 1):
            if edges[k] <= v < edges[k + 1]:
                counts[k] += 1
                break
    return [[edges[k], edges[k + 1], counts[k]] for k in range(len(counts))]


def role_texts(recs, role):
    return [t["text"] for r in recs for t in r["turns"] if role in ("all", t["role"])]


def text_stats(recs):
    out = {}
    for role in ("user", "assistant", "all"):
        texts = role_texts(recs, role)
        toks = [words(t) for t in texts]
        out[role] = {"turns": len(texts), "words": sum(map(len, toks)), "distinct_1": distinct(toks, 1),
                     "distinct_2": distinct(toks, 2), "distinct_3": distinct(toks, 3), "gzip_ratio": gzip_ratio(texts)}
    return out


def slot_coverage(recs):
    by = collections.defaultdict(list)
    for r in recs:
        for s in r["slots"].values():
            by[s["type"]].append((s["value"], bool(s.get("nonce"))))
    out = {}
    for typ, vals in sorted(by.items()):
        c = collections.Counter(v for v, _ in vals)
        try:
            pool_n = len(pools.pool(typ).values)
        except (KeyError, ValueError, AttributeError):
            pool_n = None
        in_pool = len(set(c) & set(pools.pool(typ).values)) if pool_n else None
        out[typ] = {"slots": len(vals), "distinct": len(c), "pool": pool_n,
                    "pool_used": round(in_pool / pool_n, 4) if pool_n else None,
                    "top_value_share": round(c.most_common(1)[0][1] / len(vals), 4),
                    "nonce_share": round(sum(n for _, n in vals) / len(vals), 4)}
    return out


def lengths(recs):
    ut = [sum(t["role"] == "user" for t in r["turns"]) for r in recs]
    uw = [len(words(t["text"])) for r in recs for t in r["turns"] if t["role"] == "user"]
    aw = [len(words(t["text"])) for r in recs for t in r["turns"] if t["role"] == "assistant"]
    tk = [r.get("est_tokens", 0) for r in recs]
    mean = (lambda xs: round(sum(xs) / len(xs), 2) if xs else None)
    return {"user_turns_per_chat": hist(ut, TURN_EDGES), "words_per_user_turn": hist(uw, WORD_EDGES),
            "words_per_assistant_turn": hist(aw, WORD_EDGES), "est_tokens_per_chat": hist(tk, TOKEN_EDGES),
            "means": {"user_turns": mean(ut), "user_words": mean(uw), "assistant_words": mean(aw),
                      "est_tokens": mean(tk)}}


def coverage(recs):
    cells, reg, sl, style, topics = (collections.Counter() for _ in range(5))
    for r in recs:
        for e in r["events"]:
            d = e["params"].get("d")
            db = "-" if d is None else ("d1-2" if d <= 2 else ("d3-5" if d <= 5 else "d6-10"))
            cells["|".join((e["kind"], str(e["params"].get("variant", "-")), db))] += 1
        reg[r.get("register")] += 1
        sl[r.get("slice")] += 1
        style[(r.get("user") or {}).get("style", "?")] += 1
        for t in r.get("topic_path", []):
            topics[t] += 1
    return {"cells": dict(sorted(cells.items())), "register": dict(reg), "slice": dict(sl), "style": dict(style),
            "distinct_topics": len(topics)}


def _masked(r, t):
    return dedup.norm(dedup.mask(t["text"], dedup.slot_list(r["slots"]))).split()


def repetition(recs, top=10):
    n = len(recs)
    ng, sents, opens = collections.Counter(), collections.Counter(), collections.Counter()
    for r in recs:
        conv_ng, conv_s = set(), set()
        for t in r["turns"]:
            if not str(t.get("author", "")).startswith("teacher:"):
                continue
            w = _masked(r, t)
            conv_ng.update(" ".join(w[i:i + 8]) for i in range(len(w) - 7))
            if t["role"] == "assistant":
                for s in _SENT.split(t["text"]):
                    ms = " ".join(dedup.norm(dedup.mask(s, dedup.slot_list(r["slots"]))).split())
                    if len(ms.split()) >= 4:
                        conv_s.add(ms)
        ng.update(conv_ng)
        sents.update(conv_s)
        first = next((t for t in r["turns"] if t["role"] == "user"), None)
        if first:
            opens[" ".join(_masked(r, first)[:4])] += 1

    def rows(c, cap, denom):
        worst = c.most_common(1)[0][1] / denom if c and denom else 0.0
        return {"top": [[k, v, round(v / denom, 5)] for k, v in c.most_common(top)], "max_share": round(worst, 5),
                "cap": cap, "alarm": bool(n >= MIN_N and worst > cap)}
    return {"ngram8": rows(ng, CAP_NGRAM, n), "assistant_sentence": rows(sents, CAP_SENT, n),
            "opening4": rows(opens, CAP_OPEN, sum(opens.values()) or 1), "min_n_for_alarm": MIN_N}


def compute(recs):
    return {"conversations": len(recs), "text": text_stats(recs), "slots": slot_coverage(recs),
            "lengths": lengths(recs), "coverage": coverage(recs), "repetition": repetition(recs)}


def summary(st):
    t, rep = st["text"], st["repetition"]
    lines = [f"conversations {st['conversations']}, mean est tokens {st['lengths']['means']['est_tokens']}"]
    for role in ("user", "assistant", "all"):
        x = t[role]
        lines.append(f"{role:9s} turns {x['turns']:6d}  distinct-1/2/3 {x['distinct_1']}/{x['distinct_2']}/"
                     f"{x['distinct_3']}  gzip {x['gzip_ratio']}")
    for k in ("ngram8", "assistant_sentence", "opening4"):
        lines.append(f"{k}: max share {rep[k]['max_share']} (cap {rep[k]['cap']}){'  ALARM' if rep[k]['alarm'] else ''}")
    low = [f"{k} {v['pool_used']}" for k, v in st["slots"].items() if v["pool_used"] is not None and v["pool_used"] < 0.5]
    lines.append("slot types using under half their pool: " + (", ".join(low) or "none"))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    st = compute(load(a.run_dir))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(st, f, indent=1)
    print(summary(st))


if __name__ == "__main__":
    main()
