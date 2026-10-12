"""bpb.py on tiny models (CPU): a uniform model against a closed form, a random model against an
unbatched loop that rebuilds every input with its own code, per-role sums, truncation, the CLI."""
from __future__ import annotations

import json
import math
import os

import pytest

torch = pytest.importorskip("torch")

import bpb  # noqa: E402
import prep_common as C  # noqa: E402
from conftest import TOK2, TOK8  # noqa: E402
from config import PlanckConfig  # noqa: E402
from model import build_model  # noqa: E402

torch.set_num_threads(4)


def tiny(vocab=2048, seq_len=4096, seed=0, zero=False):
    torch.manual_seed(seed)
    cfg = PlanckConfig(vocab_size=vocab, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, head_dim=16,
                       mlp_hidden=64, seq_len=seq_len)
    m = build_model(cfg, "cpu").eval()
    with torch.no_grad():
        if zero:
            m.tok_emb.weight.zero_()                      # tied head: every logit is exactly 0
        else:
            for p in m.parameters():                      # scramble so no block is inert
                p.copy_(torch.randn_like(p) * (0.5 / math.sqrt(p.shape[-1]) if p.dim() > 1 else 0.3) +
                        (1.0 if p.dim() == 1 else 0.0))
    return m, cfg


def windows_by_hand(ev, name, enc, info):
    """Rebuild each (context, target) from the docs file with this test's own code (not evalwin)."""
    man = C.read_json(os.path.join(ev, "manifest.json"))
    docs = [json.loads(x) for x in open(os.path.join(ev, f"{name}.docs.jsonl"), encoding="utf-8")]
    out = []
    for x in open(os.path.join(ev, f"{name}.windows.jsonl"), encoding="utf-8"):
        w = json.loads(x)
        d = docs[w["d"]]
        if man["sets"][name]["kind"] == "text":
            b = d["text"].encode()
            ctx = enc(b[w["c"]:w["s"]].decode())
            tgt_text = b[w["s"]:w["e"]].decode()
        else:
            tb = [t["text"].encode() for t in d["turns"]]
            ctx = []
            for j in range(w["ct"], w["t"] + 1):
                ctx.append(info["role_ids"][d["turns"][j]["role"]])
                lo = w["cc"] if j == w["ct"] else 0
                hi = w["s"] if j == w["t"] else len(tb[j])
                ctx += enc(tb[j][lo:hi].decode())
                if j < w["t"]:
                    ctx.append(info["end_id"])
            tgt_text = tb[w["t"]][w["s"]:w["e"]].decode()
        out.append((ctx, enc(tgt_text), len(tgt_text.encode()), w.get("role")))
    return out


@pytest.mark.parametrize("tok_path", [TOK2, TOK8])
def test_uniform_model_matches_closed_form(evalsets, tok_path):
    info = C.tokenizer_info(tok_path)
    m, cfg = tiny(vocab=info["vocab"], zero=True)
    res = bpb.score(m, cfg, evalsets, tok_path)
    enc = C.encoder(C.load_tokenizer(tok_path))
    for name, r in res["sets"].items():
        hand = windows_by_hand(evalsets, name, enc, info)
        n_tok, n_bytes = sum(len(h[1]) for h in hand), sum(h[2] for h in hand)
        assert r["tokens"] == n_tok and r["bytes"] == n_bytes and r["truncated"] == 0
        assert r["bpb"] == pytest.approx(n_tok * math.log2(info["vocab"]) / n_bytes, rel=1e-6)


def test_bytes_do_not_depend_on_the_tokenizer(evalsets):
    r2 = bpb.score(*tiny(vocab=2048, zero=True), evalsets, TOK2)
    r8 = bpb.score(*tiny(vocab=8192, zero=True), evalsets, TOK8)
    for name in r2["sets"]:
        assert r2["sets"][name]["bytes"] == r8["sets"][name]["bytes"]
        assert r2["sets"][name]["tokens"] > r8["sets"][name]["tokens"]


def test_random_model_matches_unbatched_loop(evalsets):
    info = C.tokenizer_info(TOK2)
    m, cfg = tiny(seed=3)
    res = bpb.score(m, cfg, evalsets, TOK2, batch_tokens=3000)
    enc = C.encoder(C.load_tokenizer(TOK2))
    for name, r in res["sets"].items():
        bits = nbytes = 0.0
        role_bits = {}
        for ctx, tgt, nb, role in windows_by_hand(evalsets, name, enc, info):
            ids = torch.tensor([ctx + tgt])
            with torch.no_grad():
                logp = torch.log_softmax(m(ids)[0][0].double(), dim=-1)
            nll = -sum(float(logp[len(ctx) - 1 + i, t]) for i, t in enumerate(tgt))
            bits += nll / math.log(2)
            nbytes += nb
            if role:
                rb = role_bits.setdefault(role, [0.0, 0])
                rb[0] += nll / math.log(2)
                rb[1] += nb
        assert r["bytes"] == nbytes and r["bpb"] == pytest.approx(bits / nbytes, rel=2e-5)
        assert 1.0 < r["bpb"] < 30
        for role, (b, n) in role_bits.items():
            assert r["by_role"][role]["bytes"] == n and r["by_role"][role]["bpb"] == pytest.approx(b / n, rel=2e-5)


def test_batch_size_does_not_change_the_result(evalsets):
    m, cfg = tiny(seed=5)
    a = bpb.score(m, cfg, evalsets, TOK2, batch_tokens=1)
    b = bpb.score(m, cfg, evalsets, TOK2, batch_tokens=10**6)
    for name in a["sets"]:
        assert a["sets"][name]["bpb"] == pytest.approx(b["sets"][name]["bpb"], rel=2e-5)


def test_truncation_keeps_the_target(evalsets):
    m, cfg = tiny(seq_len=1536, seed=1)
    res = bpb.score(m, cfg, evalsets, TOK2)
    full = bpb.score(*tiny(seq_len=4096, seed=1), evalsets, TOK2)
    assert sum(r["truncated"] for r in res["sets"].values()) > 0
    for name, r in res["sets"].items():
        assert r["bytes"] == full["sets"][name]["bytes"] and r["tokens"] == full["sets"][name]["tokens"]


def test_max_windows_keeps_the_lowest_ranks(evalsets):
    m, cfg = tiny(seed=2)
    a = bpb.score(m, cfg, evalsets, TOK2, max_windows=5)
    for name, r in a["sets"].items():
        wins = [json.loads(x) for x in open(os.path.join(evalsets, f"{name}.windows.jsonl"), encoding="utf-8")]
        keep = sorted(wins, key=lambda w: w["h"])[:5]
        assert r["windows"] == len(keep) == 5 and r["bytes"] == sum(w["b"] for w in keep)


def test_cli_on_a_checkpoint(evalsets, tmp_path):
    m, cfg = tiny(seed=4)
    ck = str(tmp_path / "ckpt_00000010.pt")
    torch.save({"model": {f"_orig_mod.{k}": v for k, v in m.state_dict().items()}, "model_cfg": cfg.to_dict(),
                "step": 10, "tokens": 1234}, ck)
    out = str(tmp_path / "res.json")
    assert bpb.main([ck, evalsets, "--tokenizer", TOK2, "--sets", "web,oasst2", "--out", out]) == 0
    res = json.load(open(out))
    direct = bpb.score(m, cfg, evalsets, TOK2, ["web", "oasst2"])
    assert set(res["sets"]) == {"web", "oasst2"} and res["step"] == 10
    for name in res["sets"]:
        assert res["sets"][name]["bpb"] == pytest.approx(direct["sets"][name]["bpb"], rel=1e-9)


def test_tokenizer_larger_than_model_refused(evalsets):
    m, cfg = tiny(vocab=2048)
    with pytest.raises(AssertionError):
        bpb.score(m, cfg, evalsets, TOK8)
