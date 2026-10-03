"""History dependence (SCREENS C6 HD): every CHAT target window scored three ways on the SAME target bytes and the
same target tokens (asserted), with bpb.py's scorer (nll_items: one causal forward per window, doc=None).

  own      the standard window (evalwin.chat_item): the conversation so far (turns ct..t-1, the first from cc),
           then the scored turn's role token, its own bytes before s, and the target.
  foreign  the own window's prior-turn context replaced by the same number of context bytes, cut from the left
           at whitespace (evalwin's rule: whole turns, newest first, while they fit; the first that does not fit
           enters from its first boundary >= its length minus the bytes left), taken from the end of the next
           conversation in eval-set order whose turns hold that many bytes (after the last comes the first; never
           the conversation itself). Rendered as own context is (role token, text, <|end|> per turn), then the
           scored turn's role token, its own bytes before s, and the target. A fixed rule, no RNG.
  none     the scored turn's role token, its own bytes before s, and the target.
Left out (HD is 0 there by construction): windows of turn index 0 (no prior conversation) and windows whose own
window has no prior-turn bytes (ct == t: the turn's own bytes before s fill the context budget). A window with no
foreign conversation long enough is left out and counted. Each input is fit to max_len by evalwin.fit.
HD = bpb(foreign) - bpb(own) and HD_none = bpb(none) - bpb(own), per turn index (0-based, as in the docs file) and
role, and per role over all turns; bpb = sum of target NLL / ln 2 / target bytes, as bpb.py.
    python data_prep/bpb.py CKPT EVALSET_DIR --tokenizer TOK --sets oasst2 --hd ...   (adds "hd" to chat sets)
"""
from __future__ import annotations

import math

import numpy as np

import evalwin as EW

MODES = ("own", "foreign", "none")


def history_bytes(tb: list[bytes], w: dict) -> int:
    """Bytes of prior-turn text in the own window (turns ct..t-1, the first from cc)."""
    return sum(len(tb[j]) - (w["cc"] if j == w["ct"] else 0) for j in range(w["ct"], w["t"]))


def foreign_source(tbs: list[list[bytes]], d: int, need: int) -> int | None:
    """The next conversation after d in eval-set order (wrapping) whose turns hold >= need bytes; None if none."""
    n = len(tbs)
    for step in range(1, n):
        e = (d + step) % n
        if sum(len(b) for b in tbs[e]) >= need:
            return e
    return None


def tail_context(tb: list[bytes], need: int) -> tuple[int, int]:
    """(ct, cc): the last `need` bytes of a conversation by evalwin's newest-first rule."""
    ct, cc, rem = len(tb), 0, need
    for j in range(len(tb) - 1, -1, -1):
        L = len(tb[j])
        if L <= rem:
            ct, cc, rem = j, 0, rem - L
            continue
        bnd = EW.boundaries(tb[j])
        k = int(np.searchsorted(bnd, L - rem, side="left"))
        if k < len(bnd):
            ct, cc = j, int(bnd[k])
        break
    return ct, cc


def render(ctx_tb, ctx_roles, ct, cc, tb_t: bytes, role_t: str, s: int, e: int, encode, role_ids: dict,
           end_id: int) -> tuple[list[int], int]:
    ids: list[int] = []
    for j in range(ct, len(ctx_tb)):
        ids.append(role_ids[ctx_roles[j]])
        ids += encode(ctx_tb[j][(cc if j == ct else 0):].decode("utf-8"))
        ids.append(end_id)
    ids.append(role_ids[role_t])
    ids += encode(tb_t[:s].decode("utf-8"))
    n_ctx = len(ids)
    return ids + encode(tb_t[s:e].decode("utf-8")), n_ctx


def build_hd_items(docs: list[dict], wins: list[dict], encode, info: dict, max_len: int,
                   max_windows: int | None = None) -> tuple[dict, list[tuple], dict]:
    """-> ({mode: [(ids, n_ctx, bytes, role)]}, [(t, role)] per kept window, counts)."""
    if max_windows:
        wins = sorted(wins, key=lambda w: (w["h"], w["d"], w["k"]))[:max_windows]
    tbs = [[t["text"].encode("utf-8") for t in d["turns"]] for d in docs]
    roles = [[t["role"] for t in d["turns"]] for d in docs]
    rid, end = info["role_ids"], info["end_id"]
    items = {m: [] for m in MODES}
    keys, cnt = [], {"windows": len(wins), "kept": 0, "first_turn": 0, "no_history": 0, "no_foreign": 0,
                     "truncated": {m: 0 for m in MODES}}
    for w in wins:
        d, t = w["d"], w["t"]
        tb, rl = tbs[d], roles[d]
        if t == 0:
            cnt["first_turn"] += 1
            continue
        if w["ct"] == t:
            cnt["no_history"] += 1
            continue
        need = history_bytes(tb, w)
        src = foreign_source(tbs, d, need)
        if src is None:
            cnt["no_foreign"] += 1
            continue
        fct, fcc = tail_context(tbs[src], need)
        built = {"own": EW.chat_item(tb, rl, w, encode, rid, end),
                 "foreign": render(tbs[src], roles[src], fct, fcc, tb[t], rl[t], w["s"], w["e"], encode, rid, end),
                 "none": render([], [], 0, 0, tb[t], rl[t], w["s"], w["e"], encode, rid, end)}
        tgt = None
        for m in MODES:
            ids, n_ctx, cut = EW.fit(*built[m], max_len)
            cnt["truncated"][m] += cut
            assert tgt is None or ids[n_ctx:] == tgt, "HD modes must score the same target tokens"
            tgt = ids[n_ctx:]
            items[m].append((ids, n_ctx, w["b"], w["role"]))
        keys.append((t, w["role"]))
        cnt["kept"] += 1
    return items, keys, cnt


def summarize_hd(items: dict, keys: list[tuple], nll: dict, cnt: dict) -> dict:
    def agg(sel):
        nb = int(sum(items["own"][i][2] for i in sel))
        out = {"windows": len(sel), "bytes": nb}
        for m in MODES:
            out[m] = float(nll[m][sel].sum() / math.log(2)) / nb if nb else None
        out["HD"] = out["foreign"] - out["own"] if nb else None
        out["HD_none"] = out["none"] - out["own"] if nb else None
        return out
    res = {**cnt, "by_role": {}, "by_turn": {}}
    for r in sorted({k[1] for k in keys}):
        res["by_role"][r] = agg(np.array([i for i, k in enumerate(keys) if k[1] == r], dtype=np.int64))
    for t in sorted({k[0] for k in keys}):
        res["by_turn"][str(t)] = {r: agg(np.array([i for i, k in enumerate(keys) if k == (t, r)], dtype=np.int64))
                                  for r in sorted({k[1] for k in keys if k[0] == t})}
    return res


def score_hd(model, docs, wins, encode, info, max_len, device, batch_tokens, precision, max_windows=None) -> dict:
    from bpb import nll_items
    items, keys, cnt = build_hd_items(docs, wins, encode, info, max_len, max_windows)
    nll = {m: nll_items(model, items[m], device, info["pad_id"], batch_tokens, precision)[0] for m in MODES}
    return summarize_hd(items, keys, nll, cnt)
