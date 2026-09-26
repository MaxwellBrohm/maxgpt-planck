"""Stage-0 tokenizer health checks (PLAN Phase 1): run on every member of the family.

  python health.py TOK.json [TOK.json ...] --heldout SAMPLE/heldout [--max-docs 5000] [--json OUT.json]

Per tokenizer:
  ids          ids 0..V-1 contiguous, every id has a non-empty string, specials at their fixed ids
  roundtrip    decode(encode(s)) == s byte for byte, on every held-out document and on random strings
               (random bytes decoded with replacement, random code points from every plane)
  specials     each special encodes to exactly its one id, alone and glued to text on both sides;
               near-miss spellings and plain text never produce a special id; a default decode drops
               the control tokens and keeps the tags
  bytes/token  per held-out file (UTF-8 bytes / tokens)
  unseen       merge-made tokens never produced on the held-out sample (under-trained candidates)
  longest      the longest tokens by byte length
Exit status 1 when any hard check (ids, roundtrip, specials) fails.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sample_io  # noqa: E402
import spec  # noqa: E402

NEAR_MISSES = ["<|end|", "<|end", "|end|>", "<| end |>", "<|END|>", "< note>", "<note >", "<NOTE>",
               "</ note>", "<lookup", "lookup>", "<|endoftext", "endoftext|>", "<|assistant||>",
               "<|user |>", "<think/>", "<results>", "<lookups>", "<|tool|"]
GLUE = [("", ""), ("a", "b"), (" x ", " y"), ("<", ">"), ("<|", "|>"), ("12", "34"), ("\n", "\n"),
        ("é", "中"), ("<note>", "</note>")]


def byte_decoder() -> dict[str, int]:
    """GPT-2 bytes_to_unicode, inverted: byte-level character -> byte value."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    dec = {chr(c): b for b, c in zip(bs, cs)}
    assert sorted(dec) == spec.byte_alphabet()
    return dec


def token_bytes(tok, i: int, dec=None) -> bytes:
    s = tok.id_to_token(i)
    if s is None:                  # a gap (reported by check_ids); keep the other checks running
        return b""
    if i < spec.N_SPECIAL:
        return s.encode("utf-8")
    dec = dec or byte_decoder()
    return bytes(dec[c] for c in s)


def enc(tok, s: str) -> list[int]:
    return tok.encode(s, add_special_tokens=False).ids


def random_strings(seed: int = 0, n: int = 300) -> list[str]:
    rng = random.Random(seed)
    out = []
    for k in range(n):
        if k % 2:
            out.append(bytes(rng.randrange(256) for _ in range(rng.randrange(1, 80))).decode("utf-8", "replace"))
        else:
            cps = [rng.choice([rng.randrange(0x20, 0x7F), rng.randrange(0x80, 0x800), rng.randrange(0x800, 0xD800),
                               rng.randrange(0xE000, 0x10000), rng.randrange(0x10000, 0x110000), 0x0A, 0x20, 0x09])
                   for _ in range(rng.randrange(1, 60))]
            out.append("".join(map(chr, cps)))
    return out


def check_ids(tok) -> list[str]:
    """Every id 0..V-1 has exactly one non-empty string (both directions), specials at their fixed ids."""
    errs, v = [], tok.get_vocab_size()
    vocab = tok.get_vocab(with_added_tokens=True)
    with_string = [i for i in range(v) if tok.id_to_token(i)]
    if sorted(vocab.values()) != list(range(v)) or len(with_string) != v:
        missing = sorted(set(range(v)) - set(with_string))[:5]
        errs.append(f"ids are not 0..{v - 1} contiguous with a string each (ids without a string: {missing})")
    for i, s in enumerate(spec.SPECIALS):
        if tok.token_to_id(s) != i:
            errs.append(f"{s} has id {tok.token_to_id(s)}, want {i}")
    return errs


def check_roundtrip(tok, texts) -> list[str]:
    texts = list(texts)
    errs = []
    encs = tok.encode_batch(texts, add_special_tokens=False)
    for t, e in zip(texts, encs):
        back = tok.decode(e.ids, skip_special_tokens=False)
        if back.encode("utf-8") != t.encode("utf-8"):
            errs.append(f"roundtrip differs for {t[:60]!r}")
    return errs


def check_specials(tok, plain_texts) -> list[str]:
    errs, special_ids = [], set(range(spec.N_SPECIAL))
    for i, s in enumerate(spec.SPECIALS):
        for pre, post in GLUE:
            text = pre + s + post
            ids = enc(tok, text)
            others = [x for x in enc(tok, pre) + enc(tok, post)]
            if ids.count(i) != 1 + others.count(i):
                errs.append(f"{s} in {text!r} is not exactly one id {i}: {ids}")
            if tok.decode(ids, skip_special_tokens=False) != text:
                errs.append(f"{text!r} does not round-trip")
        kept = tok.decode([i], skip_special_tokens=True)
        if kept != ("" if s in spec.CONTROL_TOKENS else s):
            errs.append(f"default decode of {s} gives {kept!r}")
    for s in NEAR_MISSES:
        bad = special_ids & set(enc(tok, s))
        if bad and not any(sp in s for sp in spec.SPECIALS):
            errs.append(f"near miss {s!r} produced special ids {sorted(bad)}")
    plain = [t for t in plain_texts if not any(sp in t for sp in spec.SPECIALS)]
    for t, e in zip(plain, tok.encode_batch(plain, add_special_tokens=False)):
        bad = special_ids & set(e.ids)
        if bad:
            errs.append(f"plain text produced special ids {sorted(bad)}: {t[:60]!r}")
            break
    return errs


def compression(tok, texts) -> tuple[int, int, collections.Counter]:
    texts = list(texts)
    counts = collections.Counter()
    n_tok = 0
    for e in tok.encode_batch(texts, add_special_tokens=False):
        counts.update(e.ids)
        n_tok += len(e.ids)
    return sum(len(t.encode("utf-8")) for t in texts), n_tok, counts


def longest(tok, k: int = 20) -> list[dict]:
    dec = byte_decoder()
    rows = [(len(token_bytes(tok, i, dec)), i) for i in range(spec.N_BASE, tok.get_vocab_size())]
    rows.sort(key=lambda r: (-r[0], r[1]))
    return [{"id": i, "bytes": n, "text": token_bytes(tok, i, dec).decode("utf-8", "replace")} for n, i in rows[:k]]


def check_tokenizer(path: str, heldout: dict[str, list[str]], n_random: int = 300) -> dict:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(path)
    all_texts = [t for ts in heldout.values() for t in ts]
    rep = {"path": path, "vocab": tok.get_vocab_size()}
    rep["ids_errors"] = check_ids(tok)
    rep["roundtrip_errors"] = check_roundtrip(tok, all_texts + random_strings(0, n_random))[:10]
    rep["special_errors"] = check_specials(tok, all_texts + random_strings(1, n_random))[:10]
    total = collections.Counter()
    rep["bytes_per_token"] = {}
    for name, ts in sorted(heldout.items()):
        nb, nt, c = compression(tok, ts)
        total.update(c)
        rep["bytes_per_token"][name] = round(nb / nt, 4) if nt else None
    dec = byte_decoder()
    unseen = [i for i in range(spec.N_BASE, tok.get_vocab_size()) if total[i] == 0]
    rep["unseen"] = {"n": len(unseen), "of": tok.get_vocab_size() - spec.N_BASE,
                     "examples": [token_bytes(tok, i, dec).decode("utf-8", "replace") for i in unseen[-15:]]}
    rep["longest"] = longest(tok)
    rep["ok"] = not (rep["ids_errors"] or rep["roundtrip_errors"] or rep["special_errors"])
    return rep


def load_heldout(d: str, max_docs: int | None = None) -> dict[str, list[str]]:
    out = {}
    for name, p in sample_io.source_files(os.path.dirname(d.rstrip("/")), os.path.basename(d.rstrip("/"))).items():
        docs = []
        for t in sample_io.iter_file(p):
            docs.append(t)
            if max_docs and len(docs) >= max_docs:
                break
        out[name] = docs
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tokenizers", nargs="+")
    ap.add_argument("--heldout", required=True, help="a directory of <source>.jsonl files")
    ap.add_argument("--max-docs", type=int, default=5000)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    held = load_heldout(a.heldout, a.max_docs)
    reps = [check_tokenizer(p, held) for p in a.tokenizers]
    for r in reps:
        print(f"{os.path.basename(r['path'])}: V={r['vocab']} ok={r['ok']} unseen={r['unseen']['n']}/{r['unseen']['of']}")
        for k in ("ids_errors", "roundtrip_errors", "special_errors"):
            for e in r[k]:
                print(f"  {k}: {e}")
        print("  bytes/token " + " ".join(f"{k}={v}" for k, v in r["bytes_per_token"].items()))
        print("  longest " + " | ".join(repr(x["text"]) for x in r["longest"][:8]))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(reps, f, indent=1, ensure_ascii=False)
    return 0 if all(r["ok"] for r in reps) else 1


if __name__ == "__main__":
    sys.exit(main())
