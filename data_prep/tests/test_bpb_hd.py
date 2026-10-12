"""bpb_hd.py (SCREENS C6 HD) and bpb.py --hd, on tiny CPU models and the real tok_v0_8k.

A uniform model (every logit 0) gives HD 0 and HD_none 0 on every turn. A context-bag model (+8 on every token
already in its input) on conversations whose turns repeat the previous turn's words in another order: the own
context lowers the loss, so HD > 0 and HD_none > 0 on every kept turn (this kills "foreign context = own" and
"no-context keeps the context"). The foreign rule (next conversation with enough bytes, wrapping, never itself),
the exclusions (turn index 0, no prior-turn bytes), a window whose context enters its first turn mid-turn (cc > 0:
only the bytes the own window holds count, which kills history_bytes without its cc term) and bpb.py's default
output unchanged by --hd are checked.
"""
from __future__ import annotations

import json
import math
import os

import pytest

torch = pytest.importorskip("torch")

import bpb  # noqa: E402
import bpb_hd as H  # noqa: E402
import evalwin as EW  # noqa: E402
import prep_common as C  # noqa: E402
from conftest import TOK2, TOK8  # noqa: E402
from test_bpb import tiny  # noqa: E402

TOPICS = ["dog cat horse sheep goat mouse", "red green blue yellow purple orange",
          "apple pear plum cherry grape melon", "hammer saw drill wrench chisel file",
          "Paris Rome Berlin Madrid Vienna Prague"]


def convs(n_turns=4):
    docs = []
    for i, topic in enumerate(TOPICS):
        w = topic.split()
        turns = [{"role": "user" if j % 2 == 0 else "assistant",
                  "text": " ".join(w[(j + k) % len(w)] for k in range(len(w)))} for j in range(n_turns)]
        docs.append({"id": f"c{i}", "turns": turns})
    wins = [{**w, "d": i, "k": k, "h": EW.window_hash("t", d["id"], k)}
            for i, d in enumerate(docs) for k, w in enumerate(EW.chat_windows(d["turns"], 2048, 3072))]
    return docs, wins


class Bag(torch.nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab

    def forward(self, idx):
        seen = torch.nn.functional.one_hot(idx, self.vocab).float().cumsum(1).clamp(max=1.0)
        return 8.0 * seen, None


def hd_of(model, max_len=4096):
    info = C.tokenizer_info(TOK8)
    docs, wins = convs()
    return H.score_hd(model, docs, wins, C.encoder(C.load_tokenizer(TOK8)), info, max_len, "cpu", 2048, "fp32")


def test_uniform_model_gives_hd_zero():
    m, _ = tiny(vocab=8192, zero=True)
    r = hd_of(m)
    assert r["kept"] == 15 and r["first_turn"] == 5 and r["no_history"] == 0 and r["no_foreign"] == 0
    rows = list(r["by_role"].values()) + [x for t in r["by_turn"].values() for x in t.values()]
    assert rows and all(abs(x["HD"]) < 1e-12 and abs(x["HD_none"]) < 1e-12 for x in rows)


def test_own_context_lowers_loss_so_hd_is_positive():
    r = hd_of(Bag(8192))
    assert set(r["by_turn"]) == {"1", "2", "3"} and set(r["by_role"]) == {"user", "assistant"}
    for t, per in r["by_turn"].items():
        for role, x in per.items():
            assert x["HD"] > 0.5 and x["HD_none"] > 0.5 and x["own"] < x["foreign"], (t, role, x)


def test_same_target_tokens_and_bytes_in_every_mode():
    docs, wins = convs()
    items, keys, cnt = H.build_hd_items(docs, wins, C.encoder(C.load_tokenizer(TOK8)), C.tokenizer_info(TOK8), 4096)
    assert len(keys) == cnt["kept"] == 15 and all(t > 0 for t, _ in keys)
    for i in range(len(keys)):
        tg = [items[m][i][0][items[m][i][1]:] for m in H.MODES]
        assert tg[0] == tg[1] == tg[2] and len({items[m][i][2] for m in H.MODES}) == 1
    n = [len(items[m][i][0]) - items[m][i][1] for m in H.MODES for i in range(len(keys))]
    assert min(n) > 0


def test_foreign_rule():
    tbs = [[b"x" * 10], [b"y" * 3], [b"z" * 25, b"w" * 25], [b"v" * 5]]
    assert H.foreign_source(tbs, 0, 20) == 2            # 1 is too short
    assert H.foreign_source(tbs, 2, 8) == 0             # 3 is too short; after the last comes the first
    assert H.foreign_source(tbs, 2, 40) is None         # only the conversation itself is that long
    tb = [b"one two three", b"four five six seven"]
    assert H.tail_context(tb, len(tb[1])) == (1, 0)     # a whole turn fits
    ct, cc = H.tail_context(tb, len(tb[1]) + 7)          # 7 more bytes from turn 0, cut at a boundary
    assert ct == 0 and tb[0][cc:] == b" three" and len(tb[0]) - cc <= 7


def test_turn_zero_and_no_history_windows_are_left_out():
    docs = [{"id": "a", "turns": [{"role": "user", "text": "hello there " * 30},
                                  {"role": "assistant", "text": "fine thanks " * 600}]},
            {"id": "b", "turns": [{"role": "user", "text": "other words " * 40}]}]
    wins = [{**w, "d": 0, "k": k, "h": k} for k, w in enumerate(EW.chat_windows(docs[0]["turns"], 2048, 3072))]
    assert any(w["t"] == 1 and w["ct"] == 1 for w in wins)          # a long turn's later windows: no history
    items, keys, cnt = H.build_hd_items(docs, wins, C.encoder(C.load_tokenizer(TOK8)), C.tokenizer_info(TOK8), 4096)
    assert cnt["first_turn"] == sum(w["t"] == 0 for w in wins) > 0
    assert cnt["no_history"] == sum(w["t"] == 1 and w["ct"] == 1 for w in wins) > 0
    assert cnt["kept"] == len(keys) == sum(w["t"] == 1 and w["ct"] == 0 for w in wins)


def words(stem: str, n: int) -> str:
    return " ".join(f"{stem}{i % 10}" for i in range(n))


def test_mid_turn_context_counts_only_the_bytes_the_own_window_holds():
    # Turn 1's 3,072-byte context budget runs out inside turn 0, so the own window enters turn 0 mid-turn
    # (ct 0 < t 1, cc > 0). A's turn 0 is longer than everything B holds, while A's in-window context is not:
    # the foreign context must match the in-window bytes, not the whole first turn (else A finds no source).
    docs = [{"id": "a", "turns": [{"role": "user", "text": words("alpha", 600)},
                                  {"role": "assistant", "text": words("omega", 15)}]},
            {"id": "b", "turns": [{"role": "user", "text": words("beta", 560)},
                                  {"role": "assistant", "text": words("gamma", 15)}]}]
    wins = [{**w, "d": i, "k": k, "h": k} for i, d in enumerate(docs)
            for k, w in enumerate(EW.chat_windows(d["turns"], 2048, 3072))]
    tbs = [[t["text"].encode("utf-8") for t in d["turns"]] for d in docs]
    mid = [w for w in wins if w["t"] == 1]
    assert [w["d"] for w in mid] == [0, 1] and all(w["ct"] == 0 < w["cc"] for w in mid)
    wa = mid[0]
    own = len(tbs[0][0]) - wa["cc"]                    # prior-turn bytes in A's own window
    assert own <= 3072 < sum(len(b) for b in tbs[1]) < len(tbs[0][0])
    assert H.history_bytes(tbs[0], wa) == own
    assert H.foreign_source(tbs, 0, own) == 1
    fct, fcc = H.tail_context(tbs[1], own)
    got = sum(len(tbs[1][j]) - (fcc if j == fct else 0) for j in range(fct, len(tbs[1])))
    assert fct == 0 and own - 16 <= got <= own         # B's tail, cut at whitespace
    items, keys, cnt = H.build_hd_items(docs, wins, C.encoder(C.load_tokenizer(TOK8)), C.tokenizer_info(TOK8), 4096)
    assert cnt["no_foreign"] == 0 and cnt["no_history"] == 0 and cnt["kept"] == len(keys) == 2
    assert keys == [(1, "assistant"), (1, "assistant")]


def test_cli_hd_adds_a_key_and_changes_nothing_else(evalsets, tmp_path):
    m, cfg = tiny(vocab=2048, seed=2)
    ck = str(tmp_path / "final_00000010.pt")
    torch.save({"model": m.state_dict(), "model_cfg": cfg.to_dict(), "step": 10, "tokens": 1}, ck)
    outs = {}
    for flag in ([], ["--hd"]):
        out = str(tmp_path / f"r{len(flag)}.json")
        assert bpb.main([ck, evalsets, "--tokenizer", TOK2, "--sets", "web,oasst2", "--out", out, *flag]) == 0
        outs[len(flag)] = json.load(open(out))
    a, b = outs[0], outs[1]
    hd = b["sets"]["oasst2"].pop("hd")
    a.pop("seconds"), b.pop("seconds")
    assert a == b and "hd" not in b["sets"]["web"]
    assert hd["kept"] > 0 and "0" not in hd["by_turn"] and math.isfinite(hd["by_role"]["assistant"]["HD"])
    assert hd["kept"] + hd["first_turn"] + hd["no_history"] + hd["no_foreign"] == hd["windows"]
    assert os.path.exists(ck)
