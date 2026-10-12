"""Compression measurements for the tokenizer family (Stage 0 table, P-098).

bytes/token of a tokenizer on a set of texts = total UTF-8 bytes / total tokens, where tokens are
counted by Tokenizer.encode(text, add_special_tokens=False) (nothing is added or stripped; the
held-out files of make_tok_sample.py already carry no special-token strings). Per source, and
pooled over all sources ("ALL": total bytes of every source / total tokens of every source, so a
large source weighs more than a small one).

P-098 compares the nested-truncated member of the family with a separately trained tokenizer of the
same size on the same sample: rel_diff_pct = (bpt_truncated / bpt_separate - 1) * 100 per source
and pooled. Positive means the truncated one compresses better. The ledger default holds when the
two are identical, or when |pooled rel_diff_pct| <= 0.5 (research/followup/tokenizer.md L3).
"""
from __future__ import annotations

import json

P098_TOLERANCE_PCT = 0.5
POOLED = "ALL"


def utf8_len(texts) -> int:
    return sum(len(t.encode("utf-8")) for t in texts)


def count_tokens(tok, texts, chunk: int = 512) -> int:
    """Total tokens of texts under tok (encode_batch in chunks, no special tokens added)."""
    texts = list(texts)
    n = 0
    for i in range(0, len(texts), chunk):
        n += sum(len(e.ids) for e in tok.encode_batch(texts[i: i + chunk], add_special_tokens=False))
    return n


def source_stats(tok, heldout: dict[str, list[str]]) -> dict[str, dict]:
    """{source: {bytes, tokens, docs, bpt}} plus the pooled row POOLED."""
    rows, tb, tt, td = {}, 0, 0, 0
    for name in sorted(heldout):
        texts = heldout[name]
        b, t = utf8_len(texts), count_tokens(tok, texts)
        rows[name] = {"bytes": b, "tokens": t, "docs": len(texts), "bpt": round(b / t, 4) if t else None}
        tb, tt, td = tb + b, tt + t, td + len(texts)
    rows[POOLED] = {"bytes": tb, "tokens": tt, "docs": td, "bpt": round(tb / tt, 4) if tt else None}
    return rows


def bpt_table(toks: dict[int, object], heldout: dict[str, list[str]]) -> dict[int, dict[str, dict]]:
    """{vocab size: source_stats} for every tokenizer of the family."""
    return {v: source_stats(toks[v], heldout) for v in sorted(toks)}


def rel_diff_pct(a_bytes: int, a_tokens: int, b_bytes: int, b_tokens: int) -> float | None:
    """(bpt_a / bpt_b - 1) * 100, from raw totals (no rounding before the division)."""
    if not (a_tokens and b_tokens and a_bytes and b_bytes):
        return None
    return round(((a_bytes / a_tokens) / (b_bytes / b_tokens) - 1.0) * 100.0, 4)


def same_model(path_a: str, path_b: str) -> dict:
    """Whether two tokenizer files have identical vocab (token -> id) and merge lists."""
    with open(path_a, encoding="utf-8") as f:
        a = json.load(f)["model"]
    with open(path_b, encoding="utf-8") as f:
        b = json.load(f)["model"]
    return {"vocab_identical": a["vocab"] == b["vocab"], "merges_identical": a["merges"] == b["merges"],
            "n_merges": [len(a["merges"]), len(b["merges"])]}


def p098(trunc_stats: dict[str, dict], sep_stats: dict[str, dict], identity: dict,
         tol: float = P098_TOLERANCE_PCT) -> dict:
    """P-098 from two source_stats tables computed on the same held-out texts."""
    assert set(trunc_stats) == set(sep_stats), "P-098 arms were measured on different sources"
    per = {}
    for name in sorted(set(trunc_stats) - {POOLED}) + [POOLED]:
        a, b = trunc_stats[name], sep_stats[name]
        assert a["bytes"] == b["bytes"], f"P-098 arms saw different text for {name}"
        per[name] = {"bpt_truncated": a["bpt"], "bpt_separate": b["bpt"],
                     "tokens_truncated": a["tokens"], "tokens_separate": b["tokens"],
                     "rel_diff_pct": rel_diff_pct(a["bytes"], a["tokens"], b["bytes"], b["tokens"])}
    pooled = per[POOLED]["rel_diff_pct"]
    identical = identity["vocab_identical"] and identity["merges_identical"]
    holds = identical or (pooled is not None and abs(pooled) <= tol)
    return {"per_source": per, "pooled_rel_diff_pct": pooled, "identity": identity, "tolerance_pct": tol,
            "default_holds": bool(holds),
            "verdict": ("identical files" if identical else
                        f"pooled difference {pooled}% {'within' if holds else 'outside'} +-{tol}%")}
