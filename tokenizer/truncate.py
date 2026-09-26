"""Nested truncation (P-098): derive a smaller tokenizer from the top one by keeping a prefix of its merges.

The HF BPE trainer gives every new token the next free id, in merge order, so in the trained JSON
  ids 0..N_BASE-1          special tokens and the 256 byte characters (not created by merges)
  id N_BASE + k            the token created by the k-th token-creating merge
(a later merge can re-create an existing string, e.g. "a"+"bc" after "ab"+"c"; it adds a merge but no id).
The size-V member keeps ids < V and the merges up to and including the one that creates id V-1, which
is exactly what the trainer would have stopped at had it been asked for V. Every kept merge's parts
were created earlier in merge order, so they are kept too; check_merge_order asserts it.
"""
from __future__ import annotations

import copy
import json

from spec import N_BASE, N_SPECIAL, SPECIALS


def merge_pair(m) -> tuple[str, str]:
    """A merge as stored in tokenizer JSON: [a, b] (tokenizers >= 0.20) or "a b" (older)."""
    if isinstance(m, str):
        a, b = m.split(" ")
        return a, b
    a, b = m
    return a, b


def creation_index(model: dict) -> dict[int, int]:
    """token id -> index of the merge that first creates it. Asserts ids are created in merge order."""
    vocab, out, nxt = model["vocab"], {}, N_BASE
    for j, m in enumerate(model["merges"]):
        a, b = merge_pair(m)
        t = vocab.get(a + b)
        assert t is not None, f"merge {j} {a!r}+{b!r} makes a string that is not in the vocab"
        if t not in out:
            assert t == nxt, f"merge {j} creates id {t}, expected {nxt}: ids are not in merge order"
            out[t] = j
            nxt += 1
    return out


def check_merge_order(model: dict) -> None:
    """Every merge's parts exist before it (base tokens or created by an earlier merge); ids are
    0..V-1 contiguous; every id >= N_BASE is created by exactly one first merge."""
    vocab = model["vocab"]
    ids = sorted(vocab.values())
    assert ids == list(range(len(ids))), "vocab ids are not contiguous from 0"
    created = creation_index(model)
    for j, m in enumerate(model["merges"]):
        a, b = merge_pair(m)
        for part in (a, b):
            pid = vocab.get(part)
            assert pid is not None, f"merge {j}: part {part!r} is not in the vocab"
            assert pid >= N_SPECIAL, f"merge {j}: part {part!r} is a special token"
            if pid >= N_BASE:
                assert created[pid] < j, f"merge {j}: part {part!r} (id {pid}) is created later, at merge {created[pid]}"
    assert set(created) == set(range(N_BASE, len(ids))), "some merge-made ids are never created by a merge"


def truncate_json(full: dict, v: int) -> dict:
    """The size-v member of the family, as a tokenizer JSON dict (full is not modified)."""
    model = full["model"]
    n_full = len(model["vocab"])
    if not N_BASE < v <= n_full:
        raise ValueError(f"size {v} outside ({N_BASE}, {n_full}]")
    created = creation_index(model)
    last = created[v - 1]
    out = copy.deepcopy(full)
    m = out["model"]
    m["vocab"] = {t: i for t, i in model["vocab"].items() if i < v}
    m["merges"] = copy.deepcopy(model["merges"][: last + 1])
    check_merge_order(m)
    assert len(m["vocab"]) == v
    added = {a["content"]: a["id"] for a in out["added_tokens"]}
    assert added == {s: i for i, s in enumerate(SPECIALS)}, "special tokens moved"
    return out


def truncate_file(full_path: str, v: int, out_path: str) -> str:
    with open(full_path, encoding="utf-8") as f:
        full = json.load(f)
    write_json(truncate_json(full, v), out_path)
    return out_path


def write_json(d: dict, path: str) -> None:
    """Write, then prove the file loads as a Tokenizer with exactly len(vocab) ids."""
    from tokenizers import Tokenizer
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=None, separators=(",", ":"))
        f.write("\n")
    tok = Tokenizer.from_file(path)
    assert tok.get_vocab_size() == len(d["model"]["vocab"]), (path, tok.get_vocab_size())


def is_prefix(small: dict, big: dict) -> bool:
    """small's vocab is a subset of big's with the same ids, and its merges are a prefix of big's."""
    sv, bv = small["model"]["vocab"], big["model"]["vocab"]
    sm, bm = small["model"]["merges"], big["model"]["merges"]
    same_ids = all(bv.get(t) == i for t, i in sv.items())
    return same_ids and [merge_pair(x) for x in bm[: len(sm)]] == [merge_pair(x) for x in sm]
