"""Log-probability scorer shared by every E001 likelihood run.

score_item() is the old capacity_probe scorer (summed log-prob of each candidate after the
prompt, all candidates in one right-padded batch), with one change: add_special_tokens is a
parameter, because a chat-template render already contains its BOS/role tokens.

render() turns a structured item (turns, question, prefix) into a prompt string:
  plain : the old battery's "User: ... / Assistant: ..." transcript (identical text)
  chat  : the model's own chat template + the answer prefix (format control)
          (for MaxGPT-3, whose "template" is "USER: ... / ASSISTANT: ...", see maxgpt3.py)
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch

import items as I


def cand_ids(tok, prompt, cont, add_special=True):
    pre = tok(prompt, add_special_tokens=add_special).input_ids
    full = tok(prompt + cont, add_special_tokens=add_special).input_ids
    eos = getattr(tok, "eos_token_id", None)
    if eos is not None:
        while pre and pre[-1] == eos and full and full[-1] == eos:
            pre, full = pre[:-1], full[:-1]
    if full[: len(pre)] == pre and len(full) > len(pre):
        return pre, full[len(pre):], "joint"
    return pre, tok(cont, add_special_tokens=False).input_ids, "split"


@torch.no_grad()
def score_item(model, tok, prompt, cands, device, add_special=True):
    """Return {label: (sum_logprob, n_tokens, how)} for every candidate continuation."""
    seqs, meta = [], []
    if getattr(tok, "canonical_scoring", False):
        # Tokenizers that merge across spaces (MaxGPT-3's byte BPE has no pre-tokenization, so
        # "on Monday" is "on " + "Mon" + "day" but " Monday" alone is " M" + "on" + "day"):
        # score each candidate's CANONICAL full-text tokenization after the longest prefix all
        # candidates share, so sums compare log P(prompt + candidate) up to a shared constant.
        fulls = {lab: tok(prompt + c, add_special_tokens=add_special).input_ids for lab, c in cands.items()}
        vals = list(fulls.values())
        lcp = 0
        while all(len(f) > lcp for f in vals) and all(f[lcp] == vals[0][lcp] for f in vals):
            lcp += 1
        for lab, f in fulls.items():
            seqs.append(f)
            meta.append((lab, lcp, len(f) - lcp, "canonical"))
    else:
        for lab, c in cands.items():
            pre, cid, how = cand_ids(tok, prompt, c, add_special)
            seqs.append(pre + cid)
            meta.append((lab, len(pre), len(cid), how))
    L = max(len(s) for s in seqs)
    pad = tok.pad_token_id if getattr(tok, "pad_token_id", None) is not None else 0
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        att[i, : len(s)] = 1
    keep = L - min(m[1] for m in meta) + 1
    try:
        logits = model(input_ids=ids.to(device), attention_mask=att.to(device), logits_to_keep=keep).logits
        off = L - logits.shape[1]
    except TypeError:
        logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits
        off = 0
    out = {}
    for i, (lab, npre, nc, how) in enumerate(meta):
        tgt = ids[i, npre: npre + nc].to(device)
        pos = torch.arange(npre - 1 - off, npre + nc - 1 - off, device=device)
        lp = torch.log_softmax(logits[i, pos].float(), -1)
        out[lab] = (float(lp[torch.arange(nc, device=device), tgt].sum().item()), nc, how)
    return out, L


@torch.no_grad()
def score_item_shared(model, tok, prompt, cands, device, add_special=True):
    """Same quantity as score_item, computed with one batch-1 forward over the shared prompt
    (single-token candidates read from its last position) plus one batch-1 forward per
    multi-token candidate. No padding. Used where compute is slow (Falcon on CPU); falls back
    to score_item when candidates do not share the prompt tokens. check_shared.py verifies
    it matches score_item."""
    parts = {lab: cand_ids(tok, prompt, c, add_special) for lab, c in cands.items()}
    if getattr(tok, "canonical_scoring", False) or len({tuple(p[0]) for p in parts.values()}) != 1:
        return score_item(model, tok, prompt, cands, device, add_special)
    pre = list(next(iter(parts.values()))[0])
    out, L = {}, len(pre)
    last = None
    for lab, (p, cid, how) in parts.items():
        if len(cid) == 1:
            if last is None:
                lg = model(input_ids=torch.tensor([pre], device=device), logits_to_keep=1).logits[0, -1].float()
                last = torch.log_softmax(lg, -1)
            out[lab] = (float(last[cid[0]].item()), 1, how)
        else:
            seq = pre + list(cid)
            L = max(L, len(seq))
            lg = model(input_ids=torch.tensor([seq], device=device), logits_to_keep=len(cid) + 1).logits[0].float()
            lp = torch.log_softmax(lg[:-1], -1)  # positions predicting cid[0..n-1]
            out[lab] = (float(lp[torch.arange(len(cid), device=device), torch.tensor(cid, device=device)].sum().item()), len(cid), how)
    return out, L


def chat_prompt(tok, model_id, turns, question, prefix):
    msgs = []
    for u, a in turns:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": question})
    kw = {"enable_thinking": False} if "Qwen3" in model_id else {}
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw) + prefix


def render(item, mode, tok, model_id):
    """-> (prompt, add_special_tokens)"""
    if mode == "plain":
        return I.transcript(item["turns"], item["question"], item["prefix"]), True
    if mode == "chat":
        if hasattr(tok, "own_format"):
            return tok.own_format(item["turns"], item["question"], item["prefix"]), False
        return chat_prompt(tok, model_id, item["turns"], item["question"], item["prefix"]), False
    raise ValueError(mode)


def load(model_id, dtype="fp32", device="mps"):
    """-> (model, tok, info). model_id 'maxgpt3:<ckpt name>' loads Max's MaxGPT-3 via the adapter."""
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    if model_id.startswith("maxgpt3:"):
        import maxgpt3
        model, tok, info = maxgpt3.load(model_id.split(":", 1)[1], dt, device)
        return model, tok, info
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dt, low_cpu_mem_usage=True).to(device).eval()
    n = sum(p.numel() for p in model.parameters())
    emb = model.get_input_embeddings().weight.numel()
    return model, tok, {"params": n, "emb": emb, "dtype": dtype, "device": device}
