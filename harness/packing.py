"""Turn a window of items into training units.

pack    First-fit-decreasing packing of whole items into rows of exactly seq_len tokens,
        padded at the end. Each row carries a document id per token, so the model's
        document mask stops attention crossing items (model.document_causal_mask), and
        positions restart per item. Unit = one row. PLAN.md: the CUDA path, if E1 agrees.
bucket  One item per row, rows padded to the smallest bucket length that holds them,
        rows of one bucket batched together (tokens per micro-batch roughly constant).
        No document mask needed (padding is only at the end, causal attention never
        reads it). Unit = one whole micro-batch. PLAN.md: the MPS path (P-141).

Targets: inside an item, target[t] = ids[t+1] when flags[t+1], else -100; the last token
of an item and all padding get -100, so nothing is predicted across items.
"""
from __future__ import annotations

import numpy as np
import torch

IGNORE = -100


def item_targets(ids: np.ndarray, flags: np.ndarray) -> np.ndarray:
    tgt = np.full(len(ids), IGNORE, dtype=np.int64)
    if len(ids) > 1:
        tgt[:-1] = np.where(flags[1:], ids[1:], IGNORE)
    return tgt


def pack_rows(items, L: int, pad_id: int, rng: np.random.Generator) -> list[dict]:
    """FFD over the window. Every item must satisfy len <= L. Returns rows as numpy dicts
    {idx, tgt, doc, pos}, shuffled by rng."""
    order = sorted(range(len(items)), key=lambda i: -len(items[i][0]))
    bins: list[list[int]] = []
    free: list[int] = []
    for i in order:
        n = len(items[i][0])
        assert n <= L, f"item of {n} tokens does not fit seq_len {L}"
        for b in range(len(bins)):
            if free[b] >= n:
                bins[b].append(i)
                free[b] -= n
                break
        else:
            bins.append([i])
            free.append(L - n)
    rows = []
    for members in bins:
        idx = np.full(L, pad_id, dtype=np.int64)
        tgt = np.full(L, IGNORE, dtype=np.int64)
        doc = np.zeros(L, dtype=np.int64)
        pos = np.zeros(L, dtype=np.int64)
        at = 0
        for d, i in enumerate(members):
            ids, flags = items[i]
            n = len(ids)
            idx[at:at + n] = ids
            tgt[at:at + n] = item_targets(ids, flags)
            doc[at:at + n] = d
            pos[at:at + n] = np.arange(n)
            at += n
        if at < L:                                   # padding is its own document
            doc[at:] = len(members)
            pos[at:] = np.arange(L - at)
        rows.append({"idx": idx, "tgt": tgt, "doc": doc, "pos": pos})
    perm = rng.permutation(len(rows))
    return [rows[int(p)] for p in perm]


def bucket_batches(items, buckets: list[int], tokens_per_micro: int, pad_id: int,
                   rng: np.random.Generator) -> list[dict]:
    """Group items by bucket length; each batch holds tokens_per_micro // bucket rows
    (the last batch of a bucket may hold fewer). Returns shuffled batch dicts of numpy
    arrays {idx (B, Lb), tgt (B, Lb)}."""
    buckets = sorted(buckets)
    by_b: dict[int, list[int]] = {b: [] for b in buckets}
    for i, (ids, _) in enumerate(items):
        b = next((b for b in buckets if len(ids) <= b), None)
        assert b is not None, f"item of {len(ids)} tokens exceeds the largest bucket {buckets[-1]}"
        by_b[b].append(i)
    out = []
    for b, members in by_b.items():
        rows = max(1, tokens_per_micro // b)
        members = [members[int(p)] for p in rng.permutation(len(members))]
        for s in range(0, len(members), rows):
            chunk = members[s:s + rows]
            idx = np.full((len(chunk), b), pad_id, dtype=np.int64)
            tgt = np.full((len(chunk), b), IGNORE, dtype=np.int64)
            for r, i in enumerate(chunk):
                ids, flags = items[i]
                idx[r, :len(ids)] = ids
                tgt[r, :len(ids)] = item_targets(ids, flags)
            out.append({"idx": idx, "tgt": tgt})
    perm = rng.permutation(len(out))
    return [out[int(p)] for p in perm]


def collate_rows(rows: list[dict]) -> dict:
    """Stack packed rows into a torch batch with doc and pos."""
    return {k: torch.from_numpy(np.stack([r[k] for r in rows])) for k in ("idx", "tgt", "doc", "pos")}


def to_torch_batch(b: dict) -> dict:
    return {"idx": torch.from_numpy(b["idx"]), "tgt": torch.from_numpy(b["tgt"]),
            "doc": None, "pos": None}
