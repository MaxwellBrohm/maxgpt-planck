"""Chat template, loss flags, packing, bucketing, token-share mixing and exact resume."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest
import torch

from chat_template import ChatTemplate
from data import build_loader
from make_fake_data import SPECIAL, make
from packing import IGNORE, bucket_batches, item_targets, pack_rows
from sources import ChatJsonlSource, TokenShardSource

T = ChatTemplate(SPECIAL["role_ids"], SPECIAL["end_id"])
U, A, E, SY, TL = 3, 4, 5, 2, 6


def test_render_assistant_only():
    rec = {"system": None, "turns": [{"role": "user", "ids": [10, 11]},
                                     {"role": "assistant", "ids": [12]},
                                     {"role": "tool", "ids": [13]},
                                     {"role": "assistant", "ids": [14, 15]}]}
    ids, fl = T.render(rec)
    assert ids.tolist() == [U, 10, 11, E, A, 12, E, TL, 13, E, A, 14, 15, E]
    sup = [int(i) for i, f in zip(ids, fl) if f]
    assert sup == [12, E, 14, 15, E]                  # content + <|end|>, never the role token


def test_render_all_and_loss_false():
    t = ChatTemplate(SPECIAL["role_ids"], SPECIAL["end_id"], loss="all")
    rec = {"turns": [{"role": "user", "ids": [10]}, {"role": "assistant", "ids": [12], "loss": False}]}
    ids, fl = t.render(rec)
    assert ids.tolist() == [U, 10, E, A, 12, E]
    assert fl.tolist() == [False, True, True, False, False, False]


def test_render_text_needs_encoder_and_system():
    rec = {"system": "ab", "turns": [{"role": "user", "text": "c"}, {"role": "assistant", "text": "d"}]}
    with pytest.raises(AssertionError):
        T.render(rec)
    ids, fl = T.render(rec, encode=lambda s: [100 + ord(ch) - 97 for ch in s])
    assert ids.tolist() == [SY, 100, 101, E, U, 102, E, A, 103, E]
    assert [int(i) for i, f in zip(ids, fl) if f] == [103, E]


def test_item_targets():
    ids = np.array([1, 2, 3, 4])
    fl = np.array([False, True, False, True])
    assert item_targets(ids, fl).tolist() == [2, IGNORE, 4, IGNORE]


def _items(rng, n=60, lo=2, hi=40):
    out = []
    for _ in range(n):
        k = int(rng.integers(lo, hi))
        out.append((rng.integers(8, 200, k).astype(np.int64), rng.random(k) < 0.7))
    return out


def test_pack_rows_keep_items_whole_and_masked():
    rng = np.random.default_rng(0)
    items = _items(rng)
    rows = pack_rows(items, 64, 0, np.random.default_rng(1))
    assert all(len(r["idx"]) == 64 for r in rows)
    found = []
    for r in rows:
        for d in np.unique(r["doc"]):
            sel = r["doc"] == d
            seg = r["idx"][sel]
            assert (r["pos"][sel] == np.arange(sel.sum())).all()
            assert np.flatnonzero(sel).tolist() == list(range(np.flatnonzero(sel)[0], np.flatnonzero(sel)[-1] + 1))
            if (seg == 0).all():
                assert (r["tgt"][sel] == IGNORE).all()   # padding supervises nothing
                continue
            assert r["tgt"][sel][-1] == IGNORE           # no target across the item boundary
            found.append(tuple(seg.tolist()))
    assert Counter(found) == Counter(tuple(i.tolist()) for i, _ in items)
    fill = sum(len(i) for i, _ in items) / (64 * len(rows))
    assert fill > 0.9


def test_bucket_batches_fit_and_budget():
    rng = np.random.default_rng(0)
    items = _items(rng, 80, 2, 64)
    bs = bucket_batches(items, [16, 32, 64], 256, 0, np.random.default_rng(2))
    seen = []
    for b in bs:
        B, L = b["idx"].shape
        assert L in (16, 32, 64) and B <= 256 // L
        for r in range(B):
            n = int((b["idx"][r] != 0).sum())
            assert n > L // 2 or L == 16                 # smallest bucket that holds it
            seen.append(tuple(b["idx"][r][:n].tolist()))
    assert Counter(seen) == Counter(tuple(i.tolist()) for i, _ in items)


def test_token_source_chunks_and_wraps(tmp_path):
    p = tmp_path / "a.bin"
    np.array([9, 9, 9, 1, 8, 8, 8, 8, 8, 8, 8, 1], dtype=np.uint16).tofile(p)
    s = TokenShardSource([str(p)], eot_id=1, max_len=5, shuffle=False)
    got = [s.next_item()[0].tolist() for _ in range(4)]
    assert got == [[9, 9, 9, 1], [8, 8, 8, 8, 8], [8, 8, 1], [9, 9, 9, 1]]
    assert s.epoch == 1


def test_chat_source_drops_long_and_skips_blank(tmp_path):
    p = tmp_path / "c.jsonl"
    long = {"turns": [{"role": "user", "ids": list(range(8, 60))}, {"role": "assistant", "ids": [9]}]}
    ok = {"turns": [{"role": "user", "ids": [8]}, {"role": "assistant", "ids": [9]}]}
    p.write_text(json.dumps(long) + "\n\n" + json.dumps(ok) + "\n")
    s = ChatJsonlSource([str(p)], T, max_len=20, shuffle=False)
    assert s.next_item()[0].tolist() == [U, 8, E, A, 9, E]
    assert s.dropped_long == 1


def _cfg(fake, mode="pack", share_chat=0.25, window=2048):
    return {"mode": mode, "pad_id": 0, "window_tokens": window, "buckets": [32, 64, 128],
            "chat": {"role_ids": SPECIAL["role_ids"], "end_id": SPECIAL["end_id"]},
            "sources": [{"kind": "tokens", "name": "text", "paths": [f"{fake}/text_*.bin"],
                         "eot_id": 1, "share": 1 - share_chat},
                        {"kind": "chat", "name": "chat", "paths": [f"{fake}/chat_*.jsonl"],
                         "share": share_chat}]}


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    d = tmp_path_factory.mktemp("fake")
    make(str(d), docs=200, chats=150)
    return str(d)


@pytest.mark.parametrize("share", [0.1, 0.25, 0.5])
def test_mixing_is_by_tokens(fake, share):
    L = build_loader(_cfg(fake, share_chat=share), 128, 4)
    seen = {}                                   # count tokens independently of the loader
    for name, src in zip(L.names, L.sources):
        def counted(f=src.next_item, name=name):
            ids, fl = f()
            seen[name] = seen.get(name, 0) + len(ids)
            return ids, fl
        src.next_item = counted
    for _ in range(40):
        L.next_batch()
    frac = seen["chat"] / (seen["chat"] + seen["text"])
    assert abs(frac - share) < 0.02
    st = L.stats()
    assert (st["drawn_chat"], st["drawn_text"]) == (seen["chat"], seen["text"])


@pytest.mark.parametrize("mode", ["pack", "bucket"])
def test_resume_is_exact(fake, mode):
    """State after k batches -> fresh loader -> identical next batches (across windows)."""
    a = build_loader(_cfg(fake, mode, window=700), 128, 4, seed=3)
    for _ in range(7):
        a.next_batch()
    st = json.loads(json.dumps(a.state_dict()))       # must survive a JSON round trip
    b = build_loader(_cfg(fake, mode, window=700), 128, 4, seed=3)
    b.load_state_dict(st)
    for _ in range(25):                               # several windows past the resume point
        x, y = a.next_batch(), b.next_batch()
        for k in ("idx", "tgt", "doc", "pos"):
            if x[k] is None:
                assert y[k] is None
            else:
                assert torch.equal(x[k], y[k]), k


def test_seeds_differ(fake):
    a = build_loader(_cfg(fake), 128, 4, seed=0).next_batch()["idx"]
    b = build_loader(_cfg(fake), 128, 4, seed=1).next_batch()["idx"]
    assert not torch.equal(a, b)
