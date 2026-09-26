"""Bits per byte of a harness checkpoint on the fixed-window eval sets (eval_sets.py, evalwin.py).

    python data_prep/bpb.py CKPT EVALSET_DIR --tokenizer tok.json [--sets a,b] [--device cuda]
        [--precision fp32|bf16] [--batch-tokens 16384] [--max-windows N] [--out result.json]

What is computed, exactly:
  For each window (evalwin.py): input = context tokens + target tokens (text: tok(doc[c:s]) +
  tok(doc[s:e]); chat: the rendered turns before, the turn's own role token and prefix, then the target),
  at most the model's seq_len tokens (context dropped from the left if needed, counted as truncated).
  Each target token's NLL (nats) is -log softmax(logits at the previous position)[token], from one causal
  forward pass over the window alone (no other window, no cache: windows are scored independently).
  Set bits per byte = sum of target NLLs / ln 2 / sum of target bytes. It is a micro-average over bytes,
  not a mean of per-window ratios. Chat sets also report it per role of the scored turn.
  Batching: windows sorted by length, right-padded with pad_id; causal attention never reads the pad,
  so a window's result does not depend on its batch (tests compare against an unbatched loop).
  --max-windows N keeps the N windows with the lowest stored rank h, the same subsample for every
  checkpoint and tokenizer.
  Precision: fp32 by default (logits and log-softmax in fp32 either way); bf16 autocast on request.
The tokenizer must match the checkpoint's vocab and have the harness control ids; the eval set is
tokenizer-free, so any tokenizer family member can be scored on it (P-158).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import evalwin as EW
import prep_common as C


def load_set(evaldir: str, name: str) -> tuple[dict, list[dict], list[dict]]:
    man = C.read_json(os.path.join(evaldir, "manifest.json"))
    st = man["sets"][name]
    out = []
    for k in ("docs", "windows"):
        p = os.path.join(evaldir, st["files"][k]["path"])
        assert C.sha256_file(p) == st["files"][k]["sha256"], f"{p} changed since the manifest"
        with open(p, encoding="utf-8") as f:
            out.append([json.loads(line) for line in f])
    return st, out[0], out[1]


def build_items(kind: str, docs: list[dict], wins: list[dict], encode, info: dict, max_len: int,
                max_windows: int | None = None) -> tuple[list[tuple], int]:
    """-> ([(ids, n_ctx, bytes, role)], n_truncated)."""
    if max_windows:
        wins = sorted(wins, key=lambda w: (w["h"], w["d"], w["k"]))[:max_windows]
    cache: dict = {}
    items, trunc = [], 0
    for w in wins:
        d = docs[w["d"]]
        if w["d"] not in cache:
            cache.clear()
            cache[w["d"]] = (d["text"].encode("utf-8") if kind == "text"
                             else [t["text"].encode("utf-8") for t in d["turns"]])
        b = cache[w["d"]]
        if kind == "text":
            ids, n_ctx = EW.text_item(b, w, encode)
            role = None
        else:
            ids, n_ctx = EW.chat_item(b, [t["role"] for t in d["turns"]], w, encode, info["role_ids"],
                                      info["end_id"])
            role = w["role"]
        ids, n_ctx, cut = EW.fit(ids, n_ctx, max_len)
        trunc += cut
        items.append((ids, n_ctx, w["b"], role))
    return items, trunc


@torch.no_grad()
def nll_items(model, items: list[tuple], device: str, pad_id: int, batch_tokens: int,
              precision: str = "fp32") -> tuple[np.ndarray, np.ndarray]:
    """-> (sum of target NLL in nats per item, float64; target token count per item)."""
    order = sorted(range(len(items)), key=lambda i: -len(items[i][0]))
    nll = np.zeros(len(items), dtype=np.float64)
    ntok = np.array([len(it[0]) - it[1] for it in items], dtype=np.int64)
    at = 0
    while at < len(order):
        T = len(items[order[at]][0])
        B = max(1, batch_tokens // T)
        rows = order[at:at + B]
        at += B
        idx = torch.full((len(rows), T), pad_id, dtype=torch.long)
        tgt = torch.full((len(rows), T), -100, dtype=torch.long)
        for r, i in enumerate(rows):
            ids, n_ctx = items[i][0], items[i][1]
            idx[r, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            tgt[r, n_ctx - 1:len(ids) - 1] = torch.tensor(ids[n_ctx:], dtype=torch.long)
        idx, tgt = idx.to(device), tgt.to(device)
        if precision == "bf16":
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                logits, _ = model(idx)
        else:
            logits, _ = model(idx)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1),
                               ignore_index=-100, reduction="none").view(len(rows), T)
        per = loss.double().sum(dim=1).cpu().numpy()
        for r, i in enumerate(rows):
            nll[i] = per[r]
    return nll, ntok


def load_model(ckpt: str, device: str):
    from config import PlanckConfig
    from model import build_model
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = PlanckConfig.from_dict(ck["model_cfg"])
    model = build_model(cfg, "cpu")
    sd = {k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()}
    model.load_state_dict(sd)
    return model.to(device).eval(), cfg, ck


def summarize(items, nll, ntok, trunc) -> dict:
    def agg(sel):
        bits = float(nll[sel].sum() / math.log(2))
        nb = int(sum(items[i][2] for i in sel))
        nt = int(ntok[sel].sum())
        return {"bpb": bits / nb if nb else None, "bits": bits, "bytes": nb, "tokens": nt,
                "windows": len(sel), "bytes_per_token": nb / nt if nt else None}
    allsel = np.arange(len(items))
    out = {**agg(allsel), "truncated": int(trunc)}
    roles = sorted({it[3] for it in items if it[3]})
    if roles:
        out["by_role"] = {r: agg(np.array([i for i, it in enumerate(items) if it[3] == r])) for r in roles}
    return out


def score(model, cfg, evaldir: str, tok_path: str, sets=None, device: str = "cpu", precision: str = "fp32",
          batch_tokens: int = 16384, max_windows: int | None = None) -> dict:
    info = C.tokenizer_info(tok_path)
    assert info["vocab"] <= cfg.vocab_size, f"tokenizer vocab {info['vocab']} > model vocab {cfg.vocab_size}"
    encode = C.encoder(C.load_tokenizer(tok_path))
    man = C.read_json(os.path.join(evaldir, "manifest.json"))
    res = {"evalset_sha256": man["evalset_sha256"], "evalset_config": man["config"],
           "tokenizer": {"file": info["file"], "sha256": info["sha256"]}, "precision": precision,
           "max_windows": max_windows, "seq_len": cfg.seq_len, "sets": {}}
    for name in sets or sorted(man["sets"]):
        st, docs, wins = load_set(evaldir, name)
        items, trunc = build_items(st["kind"], docs, wins, encode, info, cfg.seq_len, max_windows)
        nll, ntok = nll_items(model, items, device, info["pad_id"], batch_tokens, precision)
        res["sets"][name] = summarize(items, nll, ntok, trunc)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("evalset_dir")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--sets", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--batch-tokens", type=int, default=16384)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    model, cfg, ck = load_model(a.ckpt, a.device)
    res = score(model, cfg, a.evalset_dir, a.tokenizer, a.sets.split(",") if a.sets else None, a.device,
                a.precision, a.batch_tokens, a.max_windows)
    res.update({"checkpoint": os.path.basename(a.ckpt), "step": ck.get("step"), "tokens_trained": ck.get("tokens"),
                "seconds": round(time.time() - t0, 2)})
    txt = json.dumps(res, indent=1, sort_keys=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(txt + "\n")
    print(txt, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
