"""Adapter: Max's MaxGPT-3 (235M, from scratch) -> the lik.py log-prob scorer.

Reads the model code, tokenizer and checkpoints from
  /Users/brohm/Documents/Projects/Max's AI Model/maxgpt-3/
without modifying that repo. The checkpoint is opened with mmap so the optimizer state it
carries (~2.9 GB file) is never read into RAM; only model_state is copied.

Model facts (maxgpt-3/config.py, model.py): GPT-2 style pre-norm decoder, 16 blocks,
hidden 1024, 16 heads, learned absolute positions, context 1024, byte-level BPE with 16,000
tokens and NO pre-tokenization and no special tokens, untied lm_head. Chat format in its SFT
data: "USER: ...\\nASSISTANT: ..." (prepare_chat_data.py), which own_format() reproduces.

Attention: use_flash=False (manual masked softmax), because the checkpoint's sdpa_kernel
list (FLASH, EFFICIENT) has no MPS/CPU kernel; the math is the same.
"""
import os, sys
from types import SimpleNamespace
import torch

MG3 = "/Users/brohm/Documents/Projects/Max's AI Model/maxgpt-3"


class MG3Tok:
    eos_token_id = None
    pad_token_id = 0
    canonical_scoring = True  # see lik.score_item: its BPE merges across spaces

    def __init__(self, bpe):
        self.bpe = bpe
        self._cache = {}

    def encode(self, text):
        ids = self._cache.get(text)
        if ids is None:
            ids = self.bpe.encode(text)
            if len(self._cache) > 20000:
                self._cache.clear()
            self._cache[text] = ids
        return list(ids)

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        ids = self.encode(text)
        if return_tensors == "pt":
            return SimpleNamespace(input_ids=torch.tensor([ids]))
        return SimpleNamespace(input_ids=ids)

    def decode(self, ids, skip_special_tokens=True):
        return self.bpe.decode(list(ids))

    @staticmethod
    def own_format(turns, question, prefix):
        lines = []
        for u, a in turns:
            lines += [f"USER: {u}", f"ASSISTANT: {a}"]
        lines += [f"USER: {question}", f"ASSISTANT: {prefix}"]
        return "\n".join(lines)


class MG3Wrap(torch.nn.Module):
    """forward(input_ids, attention_mask=None, logits_to_keep=None) -> .logits, like HF.
    Right padding only (the scorer right-pads), so the mask is not needed: a causal model's
    outputs at real positions never see the padding after them."""

    def __init__(self, m):
        super().__init__()
        self.m = m
        self.context_window = m.context_window

    def forward(self, input_ids, attention_mask=None, logits_to_keep=None):
        m = self.m
        B, T = input_ids.shape
        if T > m.context_window:
            raise ValueError(f"sequence of {T} tokens exceeds MaxGPT-3 context {m.context_window}")
        pos = torch.arange(0, T, device=input_ids.device)
        x = m.token_embed(input_ids) + m.pos_embed(pos)
        for blk in m.blocks:
            x = blk(x)
        if logits_to_keep:
            x = x[:, -logits_to_keep:]
        return SimpleNamespace(logits=m.lm_head(m.ln_f(x)))


def load(name, dt=torch.float32, device="mps"):
    if MG3 not in sys.path:
        sys.path.append(MG3)  # appended, so it cannot shadow E001 modules
    from model import Transformer
    from tokenizer import BPETokenizer
    import config as mg3config  # the checkpoint pickles a config.Config dataclass
    path = os.path.join(MG3, "checkpoints", name if name.endswith(".pt") else name + ".pt")
    ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model_state"].items()}
    cfg = ck.get("config") if isinstance(ck, dict) else None
    c = cfg if cfg is not None else mg3config.Config()
    m = Transformer(vocab_size=c.vocab_size, context_window=c.context_window, hidden_dim=c.hidden_dim,
                    num_heads=c.num_heads, num_blocks=c.num_blocks, use_flash=False)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.endswith("causal_mask")]
    if bad or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={bad[:5]} unexpected={list(unexpected)[:5]}")
    step = ck.get("step")
    del ck, sd
    m = m.to(dtype=dt).to(device).eval()
    bpe = BPETokenizer()
    bpe.load(os.path.join(MG3, "data", "tokenizer.json"))
    n = sum(p.numel() for p in m.parameters())
    emb = m.token_embed.weight.numel()
    info = {"params": n, "emb": emb, "lm_head": m.lm_head.weight.numel(), "pos_emb": m.pos_embed.weight.numel(),
            "dtype": str(dt).replace("torch.", ""), "device": device, "step": step, "checkpoint": path,
            "context_window": c.context_window}
    return MG3Wrap(m), MG3Tok(bpe), info
