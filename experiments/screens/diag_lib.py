"""Readings for the C6 diagnostics runner (diag_runner.py; SCREENS C6, S004/S005/S007 DIAGNOSTICS, E3 Part 4d), its
pins (the IND file, tok_v0_8k, eval sets v0) and compute(job, ...), which returns one job's JSON-able result from a
loaded model (or its state dict). Nothing here trains or reads a result file.

  modules(sd)          CPU, from the checkpoint's state dict: S004's Canon kernels (mean |w| per unique layer and
                       site, per tap, and the max), S007's smear alpha per unique layer and KV head, S005's forget
                       bias b and the norm of its weight row w per unique layer and head. {} when the arm has none.
  halflife(model, ..)  S005: per effective layer and head, -log f over every real token of the rows (attn_diag's 200
                       CHAT windows): its mean, the half-life ln 2 / mean(-log f) in tokens, the 10th and 90th
                       percentiles over tokens of the per-token half-life ln 2 / (-log f_t), and those of -log f itself
                       (S005's "its 10th and 90th percentiles" read either way). f is the module's own
                       forget_logf on the attention input the block passed (the same pre-hook attn_diag uses).
  compute(job, ..)     modules and halflife as above; "ind": induction.ind_score on the pinned IND file (C6 IND on
                       the final checkpoint); "hd": data_prep/bpb.py score(..., sets=["oasst2"], hd=True), the E2
                       METRIC invocation (fp32, batch tokens 16,384, every window) plus bpb_hd's own / foreign /
                       no-context reading; "attn": attn_diag.read on the 200 lowest-h oasst2 windows (SINK, GATE mean /
                       p10 / p90, the rebuilt-output check, which must pass).
"""
from __future__ import annotations

import json
import math
import os
import re

import torch

from blocks import Attention

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
IND_FILE = os.path.join(HERE, "ind_v0_8k.json")
IND_SHA = "0eafbca2eb050958f48add5a6271d053a16c5f57d211ae894ccc0dec85fdc338"
TOK_FILE = os.path.join(ROOT, "tokenizer", "v0", "tok_v0_8k.json")
TOK_SHA = "078b24c4b0755d81985ebc122e912d7c70ff204d335519721235a405756fcfb5"
EVALSET_SHA = "fe1b55bbe3612d0c968934cbad831a17fd3d73d84aa8f1b7cb2f4523cf4d505b"
WINDOWS, BPB_BATCH, ATTN_BATCH = 200, 16384, 8192
ARGS = ("x", "cos", "sin", "v1", "mask", "fg", "smear")
LN2 = math.log(2.0)
CANON = re.compile(r"^blocks\.(\d+)\.canon_([ac])\.weight$")
SMEAR = re.compile(r"^blocks\.(\d+)\.attn\.smear_alpha$")
FGB = re.compile(r"^blocks\.(\d+)\.attn\.forget_b$")


def modules(sd: dict) -> dict:
    out: dict = {}
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    canon = sorted((int(m.group(1)), m.group(2).upper(), k) for k in sd if (m := CANON.match(k)))
    if canon:
        out["canon"] = []
        for layer, site, k in canon:
            w = sd[k].detach().float().abs()                      # (d, K)
            out["canon"].append({"layer": layer, "site": site, "mean_abs": float(w.mean()),
                                 "mean_abs_by_tap": w.mean(dim=0).tolist(), "max_abs": float(w.max())})
    smear = sorted((int(m.group(1)), k) for k in sd if (m := SMEAR.match(k)))
    if smear:
        out["smear_alpha"] = [{"layer": layer, "alpha": sd[k].detach().float().tolist()} for layer, k in smear]
    fgb = sorted((int(m.group(1)), k) for k in sd if (m := FGB.match(k)))
    if fgb:
        out["forget"] = [{"layer": layer, "b": sd[k].detach().float().tolist(),
                          "w_norm": sd[k[:-1] + "w"].detach().float().norm(dim=1).tolist()} for layer, k in fgb]
    return out


class ForgetReader:
    """Pre-hooks on every Attention module: per effective layer (call order), -log f of the real tokens."""

    def __init__(self, model, lens: torch.Tensor):
        self.model, self.lens, self.calls, self.vals, self._h = model, lens, 0, [], []

    def __enter__(self):
        for m in self.model.modules():
            if isinstance(m, Attention):
                self._h.append(m.register_forward_pre_hook(self._pre, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self._h:
            h.remove()
        self._h = []

    def _pre(self, mod, args, kwargs):
        i = self.calls
        self.calls += 1
        while len(self.vals) <= i:
            self.vals.append([])
        if not mod.forget_gate:
            return
        x = {**dict(zip(ARGS, args)), **kwargs}["x"]
        nlf = -mod.forget_logf(x)                                                   # (B, H, T) fp32
        real = torch.arange(x.size(1), device=x.device)[None, :] < self.lens.to(x.device)[:, None]
        self.vals[i].append(nlf.transpose(1, 2)[real].double().cpu())              # (n_real, H)


@torch.no_grad()
def halflife(model, rows: list[list[int]], device: str, pad_id: int = 0, batch_tokens: int = 8192) -> dict:
    """rows: token id lists (one document each), batched and right-padded as attn_diag.read does."""
    order = sorted(range(len(rows)), key=lambda i: -len(rows[i]))
    acc: list[list[torch.Tensor]] = []
    at = 0
    while at < len(order):
        T = len(rows[order[at]])
        sel = order[at:at + max(1, batch_tokens // T)]
        at += len(sel)
        idx = torch.full((len(sel), T), pad_id, dtype=torch.long)
        for r, i in enumerate(sel):
            idx[r, :len(rows[i])] = torch.tensor(rows[i], dtype=torch.long)
        rd = ForgetReader(model, torch.tensor([len(rows[i]) for i in sel]))
        with rd:
            model(idx.to(device))
        while len(acc) < len(rd.vals):
            acc.append([])
        for i, v in enumerate(rd.vals):
            acc[i] += v
    layers = []
    for i, v in enumerate(acc):
        if not v:
            layers.append(None)
            continue
        nlf = torch.cat(v)                                                          # (n_tokens, H)
        mean = nlf.mean(dim=0)
        hl = LN2 / nlf.clamp(min=1e-30)
        layers.append({"tokens": int(nlf.size(0)), "mean_neg_log_f": mean.tolist(),
                       "half_life": (LN2 / mean).tolist(),
                       "half_life_p10": torch.quantile(hl, 0.1, dim=0).tolist(),
                       "half_life_p90": torch.quantile(hl, 0.9, dim=0).tolist(),
                       "neg_log_f_p10": torch.quantile(nlf, 0.1, dim=0).tolist(),
                       "neg_log_f_p90": torch.quantile(nlf, 0.9, dim=0).tolist(),
                       "f_exactly_1_tokens": int((nlf == 0).sum())})
    return {"layers": len(acc), "by_layer": layers}


def compute(job: str, model, cfg, sd, device: str, evalset: str, cache: dict) -> dict:
    """One job's result. sd: the checkpoint's state dict (modules); cache: the attention rows per seq_len."""
    if job == "modules":
        return modules(sd)
    if job == "ind":
        import induction
        return {**induction.ind_score(model, induction.load_seqs(IND_FILE, IND_SHA), device),
                "ind_file_sha256": IND_SHA}
    if job == "hd":
        import bpb
        return bpb.score(model, cfg, evalset, TOK_FILE, ["oasst2"], device, "fp32", BPB_BATCH, None, hd=True)
    import attn_diag as D
    key = cfg.seq_len
    if key not in cache:
        cache[key] = D.eval_rows(evalset, "oasst2", TOK_FILE, WINDOWS, cfg.seq_len)
    rows, meta = cache[key]
    if job == "halflife":
        return {**halflife(model, rows, device, meta["pad_id"], ATTN_BATCH), **meta, "set": "oasst2"}
    r = D.read(model, rows, device, meta["pad_id"], ATTN_BATCH)
    if not r["check_ok"]:
        raise RuntimeError(f"attn_diag check failed (max rel err {r['check_max_rel_err']})")
    return {**r, **meta, "set": "oasst2", "sink_from": D.SINK_FROM}


def write_json(path: str, obj: dict) -> None:
    """Atomic: tmp written, fsynced, renamed onto path (harness durable.py), so a cut job leaves no partial JSON."""
    import durable
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with durable.atomic_write(path) as f:
        f.write(json.dumps(obj, indent=1, sort_keys=True) + "\n")
