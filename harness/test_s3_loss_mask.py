"""(b) Assistant-only loss counts exactly the assistant tokens, at every stage.

The reference (expected_targets) is computed from the raw records with no template code:
in "assistant" mode the supervised targets are, per assistant turn with loss != False,
its content ids then <|end|>; the role token is prompt. System, user and tool turns and
"loss": false turns contribute nothing. "all" mode supervises every token except the
first of each item and the tokens of loss:false turns.

Stages: render -> item_targets, pack_rows, bucket_batches, Loader (pack and bucket, from
jsonl files), the model's summed loss, and Trainer.train_step (n_sup and the loss).
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import schedule as S
from data import build_loader
from make_fake_data import SPECIAL
from model import build_model
from optim import make_optimizer
from packing import bucket_batches, item_targets, pack_rows
from testutil import random_chat, scramble, template, tiny
from trainer import Trainer

END, ASST = SPECIAL["end_id"], SPECIAL["role_ids"]["assistant"]


def expected_targets(rec: dict, mode: str = "assistant") -> list[int]:
    """Supervised target ids of one record, in order, straight from the turns."""
    out: list[int] = []
    first = True
    for t in rec["turns"]:
        on = t.get("loss", True)
        role_tok = SPECIAL["role_ids"][t["role"]]
        if mode == "all":
            if on and not first:
                out.append(role_tok)
            if on:
                out.extend(t["ids"] + [END])
        elif t["role"] == "assistant" and on:
            out.extend(t["ids"] + [END])
        first = False
    return out


def records(seed: int, n: int = 40) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [random_chat(rng, turns=int(rng.integers(1, 4)), system=k % 3 == 0,
                        tool=k % 4 == 1, off_turn=k % 5 == 2) for k in range(n)]


def supervised(tgt: np.ndarray) -> list[int]:
    return [int(x) for x in tgt[tgt != -100]]


@pytest.mark.parametrize("mode", ["assistant", "all"])
def test_render_and_item_targets(mode):
    tm = template(mode)
    recs = records(0)
    assert any(t.get("loss") is False for r in recs for t in r["turns"] if t["role"] == "assistant")
    assert any(t["role"] == "tool" for r in recs for t in r["turns"])
    for rec in recs:
        ids, flags = tm.render(rec)
        assert supervised(item_targets(ids, flags)) == expected_targets(rec, mode)
        if mode == "assistant":                    # never a user, system or tool id as target
            n_asst = sum(len(t["ids"]) + 1 for t in rec["turns"]
                         if t["role"] == "assistant" and t.get("loss", True))
            assert int(flags.sum()) == n_asst


def test_assistant_role_token_predicts_first_content_token():
    """Position of <|assistant|> carries the first assistant content id as its target, and
    the <|end|> that closes a user turn predicts nothing."""
    rec = {"turns": [{"role": "user", "ids": [10, 11]}, {"role": "assistant", "ids": [20, 21]}]}
    ids, flags = template().render(rec)
    tgt = item_targets(ids, flags)
    assert ids.tolist() == [3, 10, 11, END, ASST, 20, 21, END]
    assert tgt.tolist() == [-100, -100, -100, -100, 20, 21, END, -100]


@pytest.mark.parametrize("mode", ["assistant", "all"])
def test_pack_and_bucket_keep_exact_counts(mode):
    tm = template(mode)
    recs = records(1)
    its = [tm.render(r) for r in recs]
    want = sum(len(expected_targets(r, mode)) for r in recs)
    rows = pack_rows(its, 64, 0, np.random.default_rng(0))
    assert sum(len(supervised(r["tgt"])) for r in rows) == want
    bb = bucket_batches(its, [16, 32, 64], 128, 0, np.random.default_rng(0))
    assert sum(int((b["tgt"] != -100).sum()) for b in bb) == want
    # multiset of target ids survives packing (nothing moved onto the wrong token)
    got = sorted(x for r in rows for x in supervised(r["tgt"]))
    assert got == sorted(x for r in recs for x in expected_targets(r, mode))


@pytest.mark.parametrize("dmode", ["pack", "bucket"])
def test_loader_from_jsonl_counts_exactly(tmp_path, dmode):
    recs = records(2, n=30)
    with open(tmp_path / "c.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    lens = [len(template().render(r)[0]) for r in recs]
    dcfg = {"mode": dmode, "pad_id": 0, "window_tokens": int(1.3 * sum(lens)), "buckets": [16, 32, 64],
            "chat": {"role_ids": SPECIAL["role_ids"], "end_id": END},
            "sources": [{"kind": "chat", "paths": [str(tmp_path / "c.jsonl")], "shuffle": False}]}
    ld = build_loader(dcfg, 64, 4, str(tmp_path), 0)
    ld.next_batch()
    drawn = ld.counts[0]                              # tokens drawn into window 0 (wraps once)
    k, acc = 0, 0                                     # records consumed = prefix summing to drawn
    while acc < drawn:
        acc += lens[k % len(recs)]
        k += 1
    assert acc == drawn
    want = sum(len(expected_targets(recs[i % len(recs)])) for i in range(k))
    units = ld._units
    got = sum(int((u["tgt"] != -100).sum()) for u in units)
    assert got == want and k >= len(recs)


class ListLoader:
    """Stands in for Loader: yields fixed batches, no state."""
    pad = 0

    def __init__(self, batches):
        self.batches, self.i = batches, 0

    def next_batch(self):
        b = self.batches[self.i % len(self.batches)]
        self.i += 1
        return b

    def state_dict(self):
        return {}

    def stats(self):
        return {}


def test_trainer_step_normalizes_by_assistant_tokens():
    """train_step's n_sup is the reference count, and its loss equals the sum of each
    conversation's assistant-token loss computed ALONE, divided by that count."""
    from packing import collate_rows
    tm = template()
    recs = records(3, n=16)
    rows = pack_rows([tm.render(r) for r in recs], 64, 0, np.random.default_rng(0))
    b1, b2 = collate_rows(rows[:len(rows) // 2]), collate_rows(rows[len(rows) // 2:])
    torch.manual_seed(0)
    m = scramble(build_model(tiny(n_loops=2)), 3).train()
    want_n = sum(len(expected_targets(r)) for r in recs)
    with torch.no_grad():
        ref = 0.0
        for r in recs:
            ids, fl = tm.render(r)
            lg, _ = m(torch.from_numpy(ids)[None])
            tgt = torch.from_numpy(item_targets(ids, fl))
            ref += float(F.cross_entropy(lg[0], tgt, ignore_index=-100, reduction="sum"))
    opt = make_optimizer(m, {"lr": 1e-3})
    tr = Trainer(model=m, optimizer=opt, loader=ListLoader([b1, b2]), sched=S.full(10),
                 device="cpu", amp=None, cfg={}, out_dir=".", grad_accum=2, grad_clip=1.0,
                 log_every=1, ckpt_every=0, keep_last=1, stable_points=set(), meta={})
    out = tr.train_step()
    assert out["n_sup"] == want_n
    assert out["loss"] == pytest.approx(ref / want_n, rel=1e-5)


def test_targets_are_next_tokens_inside_one_item():
    """Every supervised target is the NEXT input token of the SAME item: pack rows never
    predict across a document boundary, and bucket rows are left-aligned copies of items."""
    tm = template()
    recs = records(4)
    its = [tm.render(r) for r in recs]
    for row in pack_rows(its, 64, 0, np.random.default_rng(0)):
        for t in np.nonzero(row["tgt"] != -100)[0]:
            assert t + 1 < 64 and row["idx"][t + 1] == row["tgt"][t] and row["doc"][t + 1] == row["doc"][t]
    rows = []
    for b in bucket_batches(its, [16, 32, 64], 128, 0, np.random.default_rng(0)):
        for idx, tgt in zip(b["idx"], b["tgt"]):
            n = int((idx != 0).sum())
            assert (idx[:n] != 0).all() and (idx[n:] == 0).all()           # left-aligned
            for t in np.nonzero(tgt != -100)[0]:
                assert t + 1 < n and idx[t + 1] == tgt[t]
            rows.append(tuple(idx[:n].tolist()))
    assert sorted(rows) == sorted(tuple(ids.tolist()) for ids, _ in its)
