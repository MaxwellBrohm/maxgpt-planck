"""decode.KVDecoder == PlanckLM.forward: prefill + one token at a time gives the full forward's logits at every
position, on every attention / looping arm (scrambled weights, so gates and value residual are live)."""
from __future__ import annotations

import pytest
import torch

from decode import KVDecoder
from model import build_model
from testutil import ARMS, scramble, tiny


@pytest.mark.parametrize("arm", sorted(ARMS))
@pytest.mark.parametrize("prefill", [1, 5])
def test_cached_logits_equal_full_forward(arm, prefill):
    m = scramble(build_model(tiny(**ARMS[arm])), seed=11)
    g = torch.Generator().manual_seed(5)
    ids = torch.randint(0, m.cfg.vocab_size, (20,), generator=g).tolist()
    with torch.no_grad():
        full = m(torch.tensor([ids]))[0][0]
    dec = KVDecoder(m)
    got = [dec.forward(ids[:prefill], "cpu")]
    for t in ids[prefill:]:
        got.append(dec.forward([t], "cpu"))
    got = torch.cat(got)
    assert got.shape == full.shape
    torch.testing.assert_close(got, full, atol=2e-5, rtol=1e-4)
    assert torch.equal(got.argmax(-1), full.argmax(-1))


def test_limits():
    m = build_model(tiny())
    dec = KVDecoder(m)
    dec.forward([1, 2, 3], "cpu")
    with pytest.raises(AssertionError):
        dec.forward([4, 5], "cpu")                    # two tokens after the prefill
    dec = KVDecoder(m)
    with pytest.raises(AssertionError):
        dec.forward([1] * (m.cfg.seq_len + 1), "cpu")
