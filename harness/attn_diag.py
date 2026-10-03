"""Attention diagnostic, SCREENS C6 SINK and GATE, on the doc=None path data_prep/bpb.py scores (one document per
row, plain causal). Reads a model; never trains. CPU or CUDA, fp32 (no autocast).

How it reads each arm's REAL attention: a forward pre-hook on every blocks.Attention call takes the arguments the
block passed (the attention input x, which is already the normed, norm-scaled and, for S004, Canon-A'd input, plus
cos, sin, v1, fg, smear), and a forward hook takes the module's own output. From x it rebuilds the attention with
the module's own layers and every transform the arm applies: S007's smear on the raw key, QK-norm when on, RoPE,
the logits q k^T / sqrt(head_dim) with the causal mask, S005's forget bias (the module's forget_bias with the fg
the model built), the softmax, the values (value residual mix when on), the gate when on and o_proj.
check: the rebuilt output equals the module's own output (max abs error <= atol x max(1, |out|)) on every call,
so the probabilities read are the ones the model used. An effective layer is one call (looped blocks count once
per call).
  SINK  probability on key 0 (the row's first token), per effective layer and head, averaged over the real
        (non-pad) query positions >= 64.
  GATE  2*sigmoid(attn_gate_proj(x)) per effective layer and head over the real tokens: mean, p10, p90; None when
        the arm has no gate.
    python attn_diag.py CKPT EVALSET_DIR --tokenizer TOK [--set oasst2] [--windows 200] [--device cuda]
        [--batch-tokens 8192] [--out result.json]
Windows: data_prep/bpb.py's --max-windows rule (the N with the lowest stored rank h), built by bpb.build_items.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

from blocks import Attention, apply_rope, repeat_kv

ARGS = ("x", "cos", "sin", "v1", "mask", "fg", "smear")
SINK_FROM = 64


def rebuild(attn: Attention, a: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """-> (output as the module computes it, probabilities fp32 (B, H, T, T), gate (B, T, H) or None)."""
    x = a["x"]
    if a.get("mask") is not None:
        raise ValueError("attn_diag reads the doc=None path only (one document per row)")
    B, T, _ = x.shape
    H, Hkv, hd = attn.n_heads, attn.n_kv_heads, attn.head_dim
    q = attn.q_proj(x).view(B, T, H, hd)
    k_raw = attn.k_proj(x).view(B, T, Hkv, hd)
    v = k_raw if attn.kv_tie else attn.v_proj(x).view(B, T, Hkv, hd)
    k = attn.smear(k_raw, a.get("smear")) if attn.smear_key else k_raw
    if attn.qk_norm:
        q, k = attn.q_norm(q), attn.k_norm(k)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    v1 = a.get("v1")
    if attn.has_vr and v1 is not None:
        a1, a2 = attn.vr_alpha[0], attn.vr_alpha[1]
        v = attn.vr_scale * (a1 * v + a2 * v1) * torch.rsqrt(a1 * a1 + a2 * a2)
    q, k = apply_rope(q, a["cos"], a["sin"]), apply_rope(k, a["cos"], a["sin"])
    k_, v_ = repeat_kv(k, attn.n_rep), repeat_kv(v, attn.n_rep)
    s = torch.matmul(q.float(), k_.float().transpose(-1, -2)) / math.sqrt(hd)
    causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
    s = s.masked_fill(~causal, float("-inf"))
    if attn.forget_gate:
        s = s + attn.forget_bias(x, a["fg"]).float()
    p = torch.softmax(s, dim=-1)
    o = torch.matmul(p, v_.float()).to(x.dtype)
    gate = None
    if attn.attn_gate:
        gate = 2.0 * torch.sigmoid(attn.attn_gate_proj(x))
        o = o * gate.transpose(1, 2).unsqueeze(-1).to(o.dtype)
    o = o.transpose(1, 2).reshape(B, T, H * hd)
    return attn.o_proj(o), p, gate


class Reader:
    """Hooks on every Attention module of a model; collects SINK, GATE and the check per effective layer."""

    def __init__(self, model, lens: torch.Tensor, sink: bool = True, gate_values: bool = True,
                 check: bool = True, atol: float = 1e-4):
        self.model, self.lens, self.sink, self.gate_values, self.check, self.atol = \
            model, lens, sink, gate_values, check, atol
        self.calls, self.max_err, self.sink_sum, self.sink_n, self.gate_sum, self.gate_n, self.gate_all = \
            0, 0.0, [], [], [], [], []
        self._args, self._handles = None, []

    def __enter__(self):
        for m in self.model.modules():
            if isinstance(m, Attention):
                self._handles.append(m.register_forward_pre_hook(self._pre, with_kwargs=True))
                self._handles.append(m.register_forward_hook(self._post, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _pre(self, mod, args, kwargs):
        self._args = {**dict(zip(ARGS, args)), **kwargs}

    def _post(self, mod, args, kwargs, out):
        i = self.calls
        self.calls += 1
        x = self._args["x"]
        if self.sink or self.check:
            y, p, gate = rebuild(mod, self._args)
        else:                        # GATE only (the in-training curve): no logits rebuilt
            y, p, gate = None, None, (2.0 * torch.sigmoid(mod.attn_gate_proj(x)) if mod.attn_gate else None)
        T = x.size(1)
        real = torch.arange(T, device=x.device)[None, :] < self.lens.to(x.device)[:, None]      # (B, T)
        if self.check:
            ref = out[0]
            err = float(((y - ref).abs() * real[..., None]).max())
            scale = max(1.0, float((ref.abs() * real[..., None]).max()))
            self.max_err = max(self.max_err, err / scale)
        while len(self.sink_sum) <= i:
            for lst in (self.sink_sum, self.gate_sum, self.gate_all):
                lst.append(None)
            self.sink_n.append(0)
            self.gate_n.append(0)
        if self.sink:
            qsel = real & (torch.arange(T, device=x.device)[None, :] >= SINK_FROM)       # (B, T)
            s = (p[..., 0] * qsel[:, None, :]).sum(dim=(0, 2)).double().cpu()             # (H,)
            self.sink_sum[i] = s if self.sink_sum[i] is None else self.sink_sum[i] + s
            self.sink_n[i] += int(qsel.sum())
        if gate is not None:
            g = gate.float()[real]                                                        # (n_real, H)
            gs = g.sum(dim=0).double().cpu()
            self.gate_sum[i] = gs if self.gate_sum[i] is None else self.gate_sum[i] + gs
            self.gate_n[i] += g.size(0)
            if self.gate_values:
                self.gate_all[i] = g.cpu() if self.gate_all[i] is None else torch.cat([self.gate_all[i], g.cpu()])

    def result(self) -> dict:
        L = len(self.sink_n)
        out = {"layers": L, "check_max_rel_err": self.max_err if self.check else None,
               "check_ok": (self.max_err <= self.atol) if self.check else None}
        if self.sink:
            out["sink"] = [None if not self.sink_n[i] else (self.sink_sum[i] / self.sink_n[i]).tolist()
                           for i in range(L)]
            out["sink_queries"] = self.sink_n[:]
        if any(x is not None for x in self.gate_sum):
            out["gate_mean"] = [(self.gate_sum[i] / self.gate_n[i]).tolist() for i in range(L)]
            if self.gate_values:
                out["gate_p10"] = [torch.quantile(self.gate_all[i], 0.1, dim=0).tolist() for i in range(L)]
                out["gate_p90"] = [torch.quantile(self.gate_all[i], 0.9, dim=0).tolist() for i in range(L)]
        else:
            out["gate_mean"] = None
        return out


@torch.no_grad()
def read(model, rows: list[list[int]], device: str, pad_id: int = 0, batch_tokens: int = 8192, sink: bool = True,
         gate_values: bool = True, check: bool = True) -> dict:
    """rows: token id lists (one document each). Batched by length, right-padded with pad_id (causal attention
    never reads the pad; pads are left out of every average). Returns Reader.result()."""
    order = sorted(range(len(rows)), key=lambda i: -len(rows[i]))
    acc = None
    at = 0
    while at < len(order):
        T = len(rows[order[at]])
        sel = order[at:at + max(1, batch_tokens // T)]
        at += len(sel)
        idx = torch.full((len(sel), T), pad_id, dtype=torch.long)
        for r, i in enumerate(sel):
            idx[r, :len(rows[i])] = torch.tensor(rows[i], dtype=torch.long)
        lens = torch.tensor([len(rows[i]) for i in sel])
        rd = Reader(model, lens, sink, gate_values, check)
        if acc is not None:      # carry the running sums across batches
            for k in ("sink_sum", "sink_n", "gate_sum", "gate_n", "gate_all"):
                setattr(rd, k, getattr(acc, k))
            rd.max_err = acc.max_err
        with rd:
            model(idx.to(device))
        rd.calls, acc = 0, rd
    return acc.result()


def eval_rows(evaldir: str, set_name: str, tok_path: str, n: int, max_len: int) -> tuple[list[list[int]], dict]:
    """The n lowest-h windows of a set (bpb.py's --max-windows rule) as full model inputs (context + target)."""
    dp = os.environ.get("PLANCK_DATA_PREP") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                             "data_prep")   # the env: mutation_check's scratch copies
    if dp not in sys.path:
        sys.path.insert(0, dp)
    import bpb
    import prep_common as C
    info = C.tokenizer_info(tok_path)
    st, docs, wins = bpb.load_set(evaldir, set_name)
    items, trunc = bpb.build_items(st["kind"], docs, wins, C.encoder(C.load_tokenizer(tok_path)), info, max_len, n)
    return [it[0] for it in items], {"pad_id": info["pad_id"], "windows": len(items), "truncated": int(trunc),
                                     "tokenizer_sha256": info["sha256"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("evalset_dir")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--set", default="oasst2")
    ap.add_argument("--windows", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-tokens", type=int, default=8192)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from config import PlanckConfig
    from model import build_model
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = PlanckConfig.from_dict(ck["model_cfg"])
    model = build_model(cfg, "cpu")
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()})
    model = model.to(a.device).eval()
    rows, meta = eval_rows(a.evalset_dir, a.set, a.tokenizer, a.windows, cfg.seq_len)
    res = read(model, rows, a.device, meta["pad_id"], a.batch_tokens)
    res.update(meta, checkpoint=os.path.basename(a.ckpt), step=ck.get("step"), set=a.set, sink_from=SINK_FROM)
    txt = json.dumps(res, sort_keys=True)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt + "\n")
    print(txt)
    return 0 if res["check_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
