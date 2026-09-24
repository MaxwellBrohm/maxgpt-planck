"""Score every probe item for ONE model (likelihood only, no generation), then exit.

usage: python run_capacity.py <hf_model_id> [--dtype fp32|bf16] [--limit-scen N]

For each item and each candidate continuation, computes the summed log-probability
of the candidate tokens given the prompt (plain "User:/Assistant:" transcript).
Writes out/<slug>.jsonl. One model per process; never run two at once on this Mac.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, json, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import items as I


def cand_ids(tok, prompt, cont):
    pre = tok(prompt, add_special_tokens=True).input_ids
    full = tok(prompt + cont, add_special_tokens=True).input_ids
    # some tokenizers append EOS with add_special_tokens; strip a trailing EOS if present in both
    if tok.eos_token_id is not None:
        while pre and pre[-1] == tok.eos_token_id and full and full[-1] == tok.eos_token_id:
            pre, full = pre[:-1], full[:-1]
    if full[: len(pre)] == pre and len(full) > len(pre):
        return pre, full[len(pre):], "joint"
    return pre, tok(cont, add_special_tokens=False).input_ids, "split"


@torch.no_grad()
def score_item(model, tok, prompt, cands, device):
    """Return {label: (sum_logprob, n_tokens)} scoring all candidates in one padded batch."""
    seqs, meta = [], []
    for lab, c in cands.items():
        pre, cid, how = cand_ids(tok, prompt, c)
        seqs.append(pre + cid)
        meta.append((lab, len(pre), len(cid), how))
    L = max(len(s) for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        att[i, : len(s)] = 1
    # only the last `keep` positions are needed (all candidates share the prompt), which
    # avoids materializing full-vocab logits for every prompt position
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
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit-scen", type=int, default=32)
    a = ap.parse_args()
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[a.dtype]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt).to(a.device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    emb = model.get_input_embeddings().weight.numel()
    print(f"loaded {a.model} params={n_params/1e6:.1f}M emb={emb/1e6:.1f}M in {time.time()-t0:.1f}s", flush=True)
    its = I.build(n_scen=a.limit_scen)
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    path = os.path.join(HERE, "out", a.model.replace("/", "__") + ".jsonl")
    with open(path, "w") as f:
        f.write(json.dumps({"meta": True, "model": a.model, "params": n_params, "emb": emb,
                            "dtype": a.dtype, "n_items": len(its)}) + "\n")
        for k, it in enumerate(its):
            sc = score_item(model, tok, it["prompt"], it["cands"], a.device)
            rec = {kk: it[kk] for kk in ("task", "cond", "d")}
            rec["scores"] = {lab: v[0] for lab, v in sc.items()}
            rec["ntok"] = {lab: v[1] for lab, v in sc.items()}
            rec["split"] = any(v[2] == "split" for v in sc.values())
            rec["prompt_tokens"] = len(tok(it["prompt"]).input_ids)
            f.write(json.dumps(rec) + "\n")
            if k % 200 == 0:
                print(f"{k}/{len(its)} {time.time()-t0:.0f}s", flush=True)
    print(f"done {a.model} in {time.time()-t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
