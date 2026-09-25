"""(a) No leakage between packed conversations, through the REAL data path (chat template ->
pack_rows / bucket_batches -> collate -> model), for a spread of attention and looping arms.

  pack    every item in a packed row gives the same logits as that item run alone
          (float tolerance: different shapes), and replacing one item's tokens leaves every
          other item's logits BITWISE unchanged.
  bucket  every row's real tokens equal the item alone, and neither the padding nor the
          other rows of the batch change them (bitwise).
  loader  the same through Loader.next_batch (window draw, mixing, collate) in pack mode.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from model import build_model
from packing import bucket_batches, collate_rows, pack_rows, to_torch_batch
from testutil import ARMS, random_chat, scramble, template, tiny

PAD = 0
ARMS_PIPE = ["base_gqa", "attn_only", "kv_tie", "qk_share2", "loop2_cyclic",
             "loop3_immediate", "prelude_loop_coda", "loop_share_tie"]


def items(seed: int, n: int = 14):
    rng = np.random.default_rng(seed)
    tm = template()
    out = []
    for k in range(n):
        if k % 3 == 2:                                  # a plain text document
            ids = rng.integers(8, 97, int(rng.integers(2, 12))).astype(np.int64)
            fl = np.ones(len(ids), dtype=bool)
            fl[0] = False
            out.append((ids, fl))
        else:
            out.append(tm.render(random_chat(rng, turns=int(rng.integers(1, 3)),
                                             system=k % 4 == 0, tool=k % 5 == 0)))
    return out


def spans(doc_row: np.ndarray, pad_start: int):
    """(start, end) of each item in a packed row, from the doc ids, ignoring padding."""
    out, s = [], 0
    for t in range(1, pad_start + 1):
        if t == pad_start or doc_row[t] != doc_row[t - 1]:
            out.append((s, t))
            s = t
    return out


def model_for(arm):
    torch.manual_seed(0)
    return scramble(build_model(tiny(**ARMS[arm])), 1)


@pytest.mark.parametrize("arm", ARMS_PIPE)
def test_packed_items_isolated(arm):
    m = model_for(arm)
    its = items(0)
    rows = pack_rows(its, 64, PAD, np.random.default_rng(0))
    assert len(rows) >= 2 and any(len(spans(r["doc"], 64)) >= 3 for r in rows)
    b = collate_rows(rows)
    with torch.no_grad():
        base, _ = m(b["idx"], doc=b["doc"], pos=b["pos"])
        checked = 0
        for r, row in enumerate(rows):
            pad = np.nonzero(row["idx"] == PAD)[0]          # items never contain PAD
            pad_start = int(pad[0]) if len(pad) else 64
            sp = spans(row["doc"], pad_start)
            for (s, e) in sp:
                alone, _ = m(torch.from_numpy(row["idx"][s:e])[None])
                assert torch.allclose(base[r, s:e], alone[0], atol=2e-5, rtol=1e-4), (arm, r, s, e)
                idx2 = b["idx"].clone()
                idx2[r, s:e] = (idx2[r, s:e] + 1 + torch.arange(e - s)) % 97
                out, _ = m(idx2, doc=b["doc"], pos=b["pos"])
                for (s2, e2) in sp:
                    if (s2, e2) != (s, e):
                        assert torch.equal(out[r, s2:e2], base[r, s2:e2]), (arm, r, (s, e), (s2, e2))
                assert not torch.equal(out[r, s:e], base[r, s:e])
                other = [k for k in range(len(rows)) if k != r]
                assert torch.equal(out[other], base[other])
                checked += 1
    assert checked == len(its)


@pytest.mark.parametrize("arm", ARMS_PIPE)
def test_bucket_rows_isolated_from_padding_and_neighbours(arm):
    m = model_for(arm)
    its = items(1, n=18)
    batches = bucket_batches(its, [16, 32, 64], 64, PAD, np.random.default_rng(0))
    lens = {len(ids): 0 for ids, _ in its}
    seen = 0
    with torch.no_grad():
        for bb in batches:
            tb = to_torch_batch(bb)
            idx = tb["idx"]
            base, _ = m(idx)
            for r in range(idx.size(0)):
                real = int(np.nonzero(bb["idx"][r] != PAD)[0].max()) + 1
                assert (tb["tgt"][r, real - 1:] == -100).all()   # nothing predicted past the item
                alone, _ = m(idx[r:r + 1, :real])
                assert torch.allclose(base[r, :real], alone[0], atol=2e-5, rtol=1e-4), (arm, r)
                idx2 = idx.clone()
                idx2[r, real:] = 50                        # scribble over this row's padding
                others = [k for k in range(idx.size(0)) if k != r]
                idx2[others] = (idx2[others] + 3) % 97     # and over every other row
                out, _ = m(idx2)
                assert torch.equal(out[r, :real], base[r, :real]), (arm, r)
                seen += 1
    assert seen == len(its) and len(lens) > 3


def test_loader_batches_isolated(tmp_path):
    """Loader.next_batch in pack mode: each document span equals its tokens run alone."""
    from data import build_loader
    from make_fake_data import SPECIAL, make
    make(str(tmp_path), docs=60, chats=60)
    dcfg = {"mode": "pack", "pad_id": 0, "window_tokens": 512,
            "chat": {"role_ids": SPECIAL["role_ids"], "end_id": SPECIAL["end_id"]},
            "sources": [{"kind": "tokens", "paths": [f"{tmp_path}/text_*.bin"], "eot_id": 1, "share": 0.5},
                        {"kind": "chat", "paths": [f"{tmp_path}/chat_*.jsonl"], "share": 0.5}]}
    ld = build_loader(dcfg, 64, 4, str(tmp_path), 0)
    torch.manual_seed(0)
    m = scramble(build_model(tiny(vocab_size=256, n_loops=2)), 2)
    n_docs = 0
    with torch.no_grad():
        for _ in range(3):
            b = ld.next_batch()
            base, _ = m(b["idx"], doc=b["doc"], pos=b["pos"])
            for r in range(b["idx"].size(0)):
                d = b["doc"][r].numpy()
                for (s, e) in spans(d, 64):
                    alone, _ = m(b["idx"][r:r + 1, s:e])
                    assert torch.allclose(base[r, s:e], alone[0], atol=2e-5, rtol=1e-4)
                    n_docs += 1
    assert n_docs > 12
