"""Parameter counts from config.json alone (E003). No model is built or loaded.

total      = every stored parameter (tied matrices counted once)
embedding  = token embedding + an UNTIED output head + learned position embeddings
body       = total - embedding (attention, MLP, norms, biases): the axis E003 cares about
Rotary (Pythia) has no parameters. GPT-Neo (TinyStories) ties the head to the token embedding and
has a learned 2048 x d position table; its q/k/v projections have no bias.

usage: python params.py            prints the table for the E003 models (+ the E002 reference)
       python params.py --verify   also checks every count against the tensor shapes stored in the
                                   downloaded weight files (safetensors header, or the .bin read to the
                                   meta device: shapes only, no tensor data, no model)
"""
import argparse, json, os, struct, sys

E003_MODELS = [  # queue order: smallest body first
    "roneneldan/TinyStories-1M", "EleutherAI/pythia-14m", "roneneldan/TinyStories-3M", "EleutherAI/pythia-31m",
    "roneneldan/TinyStories-8M", "EleutherAI/pythia-70m", "roneneldan/TinyStories-33M", "EleutherAI/pythia-160m",
]
REFERENCE = ["HuggingFaceTB/SmolLM2-135M-Instruct"]


def snapshot_dir(model_id):
    # (snapshot_download(local_files_only) refuses: the unused .pt/.bin duplicates were never downloaded)
    from huggingface_hub import hf_hub_download
    return os.path.dirname(hf_hub_download(model_id, "config.json", local_files_only=True))


def load_config(model_id):
    return json.load(open(os.path.join(snapshot_dir(model_id), "config.json")))


def count(cfg):
    """-> dict(total, embedding, body, tok_emb, out_head, pos_emb, layers, d, vocab, n_ctx)."""
    mt = cfg["model_type"]
    if mt == "gpt_neox":
        h, i, L, V = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"], cfg["vocab_size"]
        bias = cfg.get("attention_bias", True)
        per = (2 * h + 2 * h                      # input_layernorm, post_attention_layernorm
               + h * 3 * h + (3 * h if bias else 0)   # query_key_value
               + h * h + (h if bias else 0)       # attention dense
               + h * i + i + i * h + h)           # mlp (biases always on)
        body = L * per + 2 * h                    # + final_layer_norm
        tok = V * h
        head = 0 if cfg.get("tie_word_embeddings", False) else V * h
        pos = 0
        n_ctx = cfg["max_position_embeddings"]
    elif mt == "gpt_neo":
        h, L, V = cfg["hidden_size"], cfg["num_layers"], cfg["vocab_size"]
        i = cfg.get("intermediate_size") or 4 * h
        per = (2 * h                              # ln_1
               + 3 * h * h                        # q, k, v (no bias)
               + h * h + h                        # out_proj
               + 2 * h                            # ln_2
               + h * i + i + i * h + h)           # c_fc, c_proj
        body = L * per + 2 * h                    # + ln_f
        tok = V * h
        head = 0 if cfg.get("tie_word_embeddings", True) else V * h
        pos = cfg["max_position_embeddings"] * h
        n_ctx = cfg["max_position_embeddings"]
    elif mt == "llama":
        h, i, L, V = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"], cfg["vocab_size"]
        nh, nkv = cfg["num_attention_heads"], cfg.get("num_key_value_heads", cfg["num_attention_heads"])
        hd = cfg.get("head_dim") or h // nh
        per = h + h + h * nh * hd + 2 * h * nkv * hd + nh * hd * h + 3 * h * i
        body = L * per + h
        tok = V * h
        head = 0 if cfg.get("tie_word_embeddings", False) else V * h
        pos = 0
        n_ctx = cfg["max_position_embeddings"]
    else:
        raise ValueError(f"no parameter formula for model_type {mt}")
    emb = tok + head + pos
    return dict(total=body + emb, embedding=emb, body=body, tok_emb=tok, out_head=head, pos_emb=pos,
                layers=L, d=h, vocab=V, n_ctx=n_ctx)


def stored_shapes(model_id):
    """{tensor name: numel} of floating tensors in the downloaded weights, without building a model.
    safetensors: parse the JSON header. .bin: torch.load onto the meta device (shapes, no data)."""
    d = snapshot_dir(model_id)
    st = os.path.join(d, "model.safetensors")
    if os.path.exists(st):
        with open(st, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        out = {}
        for k, v in hdr.items():
            if k == "__metadata__" or not v["dtype"].startswith(("F", "BF")):
                continue
            m = 1
            for s in v["shape"]:
                m *= s
            out[k] = m
        return out
    import torch
    sd = torch.load(os.path.join(d, "pytorch_model.bin"), map_location="meta", weights_only=True, mmap=True)
    return {k: t.numel() for k, t in sd.items() if t.is_floating_point()}


def verify(model_id, c):
    """Compare the formula with the stored tensors. Buffers (GPT-Neo causal-mask 'bias'/'masked_bias',
    rotary inv_freq) are skipped; a stored tied head (lm_head == wte) is counted once."""
    sh = stored_shapes(model_id)
    skip = lambda k: k.endswith(("attn.attention.bias", "attn.attention.masked_bias", "attention.bias",
                                 "attention.masked_bias", "rotary_emb.inv_freq"))
    params = {k: v for k, v in sh.items() if not skip(k)}
    tied_head = [k for k in params if k in ("lm_head.weight",)] if c["out_head"] == 0 else []
    total = sum(v for k, v in params.items() if k not in tied_head)
    emb_names = [k for k in params if k.endswith(("embed_in.weight", "embed_out.weight", "wte.weight", "wpe.weight",
                                                  "embed_tokens.weight")) or (k == "lm_head.weight" and c["out_head"])]
    emb = sum(params[k] for k in emb_names)
    return {"stored_total": total, "stored_embedding": emb, "stored_body": total - emb,
            "match": total == c["total"] and emb == c["embedding"], "emb_tensors": sorted(emb_names),
            "skipped_buffers": sorted(k for k in sh if skip(k))[:3], "tied_head_in_file": tied_head}


def table(models, do_verify=False):
    rows = []
    for m in models:
        cfg = load_config(m)
        c = count(cfg)
        r = dict(model=m, **c)
        if do_verify:
            r["verify"] = verify(m, c)
        rows.append(r)
    return rows


def fmt_rows(rows):
    out = [f"{'model':38s} {'total':>12s} {'embedding':>12s} {'body':>12s}  body/total  layers  d_model  ctx"]
    for r in rows:
        out.append(f"{r['model']:38s} {r['total']:12,d} {r['embedding']:12,d} {r['body']:12,d}  {r['body']/r['total']:9.3f}  "
                   f"{r['layers']:6d}  {r['d']:7d}  {r['n_ctx']}"
                   + ("" if "verify" not in r else ("  file-check OK" if r["verify"]["match"] else f"  FILE MISMATCH {r['verify']}")))
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = table(E003_MODELS + REFERENCE, a.verify)
    print(fmt_rows(rows))
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
    if a.verify and not all(r["verify"]["match"] for r in rows):
        sys.exit(1)
