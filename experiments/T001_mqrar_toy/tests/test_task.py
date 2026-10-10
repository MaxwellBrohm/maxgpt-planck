"""T001 generator tests: an independent replay of every token stream checks what make_sample records."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import task  # noqa: E402
from task import ASK, BOS, COPY, IGNORE, KEY0, SET, V, VAL0, make_batch, make_sample, seq_len  # noqa: E402

CASES = [(fam, n, seed) for fam in task.FAMILIES for n in (2, 8, 33, 96) for seed in (0, 1, 7)]


def replay(toks):
    """Walks the events; returns per ASK: (pos, current, history before, latest-write-was-copy, dist)."""
    assert toks[0] == BOS and (len(toks) - 1) % 3 == 0
    cur, hist, copied, last, out = {}, {}, {}, {}, []
    for e, i in enumerate(range(1, len(toks), 3)):
        op, a, b = toks[i:i + 3]
        if op == SET:
            k, v = a - KEY0, b - VAL0
            cur[k] = v
            hist.setdefault(k, []).append(v)
            copied[k], last[k] = False, e
        elif op == COPY:
            ka, kb = a - KEY0, b - KEY0
            cur[kb] = cur[ka]
            hist.setdefault(kb, []).append(cur[ka])
            copied[kb], last[kb] = True, e
        else:
            assert op == ASK
            k = a - KEY0
            out.append((i + 1, k, cur[k], list(hist[k]), copied[k], e - last[k], b - VAL0))
    return out


@pytest.mark.parametrize("fam,n,seed", CASES)
def test_answers_and_records_match_replay(fam, n, seed):
    toks, qs = make_sample(n, np.random.default_rng(seed), fam)
    assert len(toks) == seq_len(n) == 1 + 9 * n
    rep = replay(toks)
    assert len(rep) == len(qs)
    for q, (pos, k, cur, hist, cp, dist, written) in zip(qs, rep):
        assert q["pos"] == pos and toks[pos] == KEY0 + k and toks[pos - 1] == ASK
        assert written == cur and q["ans"] == VAL0 + cur          # the token after ASK k is the LATEST value
        assert q["updated"] == (len(hist) >= 2)
        assert q["stale"] == sorted({VAL0 + x for x in hist if x != cur})
        assert q["first"] == VAL0 + hist[0]
        assert q["tracked"] == cp and q["dist"] == dist


@pytest.mark.parametrize("fam,n,seed", CASES)
def test_initial_block_sets_every_key_once(fam, n, seed):
    toks, _ = make_sample(n, np.random.default_rng(seed), fam)
    first = [toks[1 + 3 * e:4 + 3 * e] for e in range(n)]
    assert all(ev[0] == SET for ev in first)
    keys = {ev[1] for ev in first}
    assert len(keys) == n
    for i in range(1 + 3 * n, len(toks), 3):           # later events only touch the sample's own keys
        assert toks[i + 1] in keys and (toks[i] != COPY or toks[i + 2] in keys)


@pytest.mark.parametrize("seed", range(6))
def test_updates_always_change_the_value(seed):
    for fam in ("mqrar", "track"):
        toks, _ = make_sample(40, np.random.default_rng(seed), fam)
        cur = {}
        for i in range(1, len(toks), 3):
            op, a, b = toks[i:i + 3]
            if op == SET:
                if i > 3 * 40:
                    assert cur[a] != b, "an update re-wrote the current value"
                cur[a] = b
            elif op == COPY:
                cur[b] = cur[a]


def test_update_values_uniform_over_the_other_63():
    rng = np.random.default_rng(3)
    seen_new, hits_cur = set(), 0
    for _ in range(300):
        toks, _ = make_sample(4, rng, "mqrar")
        cur = {}
        for i in range(1, len(toks), 3):
            op, a, b = toks[i:i + 3]
            if op == SET:
                if a in cur:
                    hits_cur += cur[a] == b
                    seen_new.add((b - cur[a]) % V)
                cur[a] = b
    assert hits_cur == 0 and seen_new == set(range(1, V))


def test_event_mix():
    rng = np.random.default_rng(5)
    counts = {"mqrar": [0, 0, 0], "ar": [0, 0, 0], "track": [0, 0, 0]}
    for fam in counts:
        for _ in range(200):
            toks, _ = make_sample(32, rng, fam)
            for i in range(1 + 3 * 32, len(toks), 3):
                counts[fam][[SET, ASK, COPY].index(toks[i])] += 1
    tot = 200 * 64
    assert counts["ar"] == [0, tot, 0]
    assert abs(counts["mqrar"][1] / tot - 0.5) < 0.02 and counts["mqrar"][2] == 0
    s, a, c = (x / tot for x in counts["track"])
    assert abs(a - 0.5) < 0.02 and abs(c - 0.25) < 0.02 and abs(s - 0.25) < 0.02


@pytest.mark.parametrize("fam", task.FAMILIES)
def test_loss_mask_only_on_answers(fam):
    x, y, meta = make_batch(24, 5, np.random.default_rng(11), fam)
    for b in range(5):
        pos = np.nonzero(y[b] != IGNORE)[0].tolist()
        assert pos == [q["pos"] for q in meta[b]]
        for p in pos:
            assert x[b, p - 1] == ASK and y[b, p] == x[b, p + 1] and VAL0 <= y[b, p] < VAL0 + V


def test_determinism_and_disjoint_streams():
    a = make_sample(16, np.random.default_rng(2), "mqrar")[0]
    assert a == make_sample(16, np.random.default_rng(2), "mqrar")[0]
    t = make_sample(16, task.train_rng(0), "mqrar")[0]
    e = make_sample(16, task.eval_rng("test", "mqrar", 16), "mqrar")[0]
    d = make_sample(16, task.eval_rng("dev", "mqrar", 16), "mqrar")[0]
    assert len({tuple(t), tuple(e), tuple(d)}) == 3
    assert make_sample(16, task.eval_rng("test", "mqrar", 16), "mqrar")[0] == e


def test_refusals():
    with pytest.raises(ValueError):
        make_sample(1, np.random.default_rng(0))
    with pytest.raises(ValueError):
        make_sample(task.K + 1, np.random.default_rng(0))
    with pytest.raises(ValueError):
        make_sample(8, np.random.default_rng(0), "boxes")
