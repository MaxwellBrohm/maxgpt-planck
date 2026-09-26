"""Model adapters for the likelihood rows (lik_score.py): Hugging Face causal LMs and Planck checkpoints.

UNTESTED ON A MODEL: LIK_MODEL_TESTED = False. No model has been loaded through this file; lik_run.py refuses hf: and
planck: without --untested-ok. Importing it imports nothing heavy: torch, transformers and the harness load in
HFLik.load / PlanckLik.load only. test_lik_adapters.py (run with a Python that has torch) checks the log-prob
position arithmetic of both adapters against a fixed random bigram table on synthetic ids, not on RC-12 rows.
A scorer is logprob(prefix_ids, continuation_ids) -> summed natural-log probability of the continuation, plus
many(prefix_ids, conts) for one row's candidates (HF: one right-padded batch, as E004 lik.score_item; Planck: one
forward per candidate, since the harness model takes no attention mask). Sequences longer than ctx raise
(lik_score.score_row skips them first and counts them as over_ctx).
Tokenizer wrappers give lik_score its protocol: HFTok (encode with or without special tokens, the chat template
through render.template, enable_thinking=False for Qwen3-family models, eos) and PlanckTok (encode_text, which never
adds special tokens: Planck's template has no BOS; template_ids = planck_responder's template render, ending with
<|assistant|>). Hard rule (prereg draft s0): no Planck model is scored on any RC-12 split, dev included, until the
pre-registration is pushed; lik_run.py asks for --prereg-pushed before it loads a planck: checkpoint.
Attention (STEP 11 FIX ROUND): HFLik.load reads the model_type from the config first and loads Gemma 3 with eager
attention (attn_for): engines.json records that transformers' default sdpa path puts p(<end_of_turn>) about 1 at
token 0 on long gemma-3-270m-it prompts, where eager and vLLM agree; asking for another attention on Gemma 3 is
refused. Other models keep the library default (the attention their HF parity ran with). dtype defaults to float32
(E004 lik.load's default; bf16 logits tie and quantize margins); scorer.info records model_type, dtype, attention
and device, and lik_run.py writes it into summary.json."""
import render as RD

LIK_MODEL_TESTED = False


def attn_for(model_type, requested=None):
    """the attn_implementation to load with (None = the library default): eager for Gemma 3, else as requested."""
    if str(model_type or "").startswith("gemma3"):
        if requested not in (None, "eager"):
            raise ValueError(f"{model_type}: attention {requested!r} refused; Gemma 3 is scored with eager attention "
                             "(engines.json: HF's default sdpa path is wrong on long prompts)")
        return "eager"
    return requested


def seq_logprob(torch, logits, ids, npre, nc):
    """sum over k < nc of log p(ids[npre + k] | ids[:npre + k]); logits[t] predicts ids[t + 1]."""
    if npre < 1 or nc < 1:
        raise ValueError("need a non-empty prefix and continuation")
    lp = torch.log_softmax(logits[npre - 1:npre + nc - 1].float(), -1)
    tgt = torch.tensor(ids[npre:npre + nc], device=lp.device)
    return float(lp[torch.arange(nc, device=lp.device), tgt].sum().item())


class HFTok:
    def __init__(self, tok, qwen3=False):
        self.tok, self.qwen3 = tok, qwen3
        self.eos = getattr(tok, "eos_token_id", None)
        self.canonical = False

    def encode(self, text, special):
        return list(self.tok(text, add_special_tokens=special).input_ids)

    def decode(self, ids):
        return self.tok.decode(ids)

    def template(self, messages):
        return RD.template(self.tok, messages, self.qwen3)


class HFLik:
    def __init__(self, torch, model, device="cpu", ctx=None, pad=0, info=None):
        self.torch, self.model, self.device, self.ctx, self.pad = torch, model, device, ctx, pad
        self.info = info or {}

    @classmethod
    def load(cls, model_id, dtype="float32", device="cuda", attn=None):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        model_type = getattr(AutoConfig.from_pretrained(model_id), "model_type", "")
        attn = attn_for(model_type, attn)
        kw = dict(torch_dtype=getattr(torch, dtype))
        if attn is not None:
            kw["attn_implementation"] = attn
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw).to(device).eval()
        qwen3 = str(getattr(model.config, "model_type", "")).startswith("qwen3")
        pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
        ctx = getattr(model.config, "max_position_embeddings", None)
        info = dict(model_type=model_type, dtype=dtype, device=device, attn_requested=attn,
                    attn=getattr(model.config, "_attn_implementation", None))
        return cls(torch, model, device, ctx, pad, info), HFTok(tok, qwen3)

    def check(self, n):
        if self.ctx is not None and n > self.ctx:
            raise ValueError(f"sequence of {n} tokens exceeds ctx {self.ctx}")

    def __call__(self, pre, cont):
        return self.many(pre, [cont])[0]

    def many(self, pre, conts):
        torch = self.torch
        seqs = [list(pre) + list(c) for c in conts]
        n = max(len(s) for s in seqs)
        self.check(n)
        ids = torch.full((len(seqs), n), self.pad, dtype=torch.long)
        att = torch.zeros((len(seqs), n), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
        with torch.no_grad():
            logits = self.model(input_ids=ids.to(self.device), attention_mask=att.to(self.device)).logits
        return [seq_logprob(torch, logits[i], seqs[i], len(pre), len(c)) for i, c in enumerate(conts)]


class PlanckTok:
    def __init__(self, responder):
        self.r = responder
        self.eos, self.canonical = None, False

    def encode(self, text, special):
        return list(self.r.encode_text(text))

    def decode(self, ids):
        return self.r.tok.decode(ids, skip_special_tokens=False)

    def template_ids(self, messages):
        return list(self.r.encode(messages))


class PlanckLik:
    def __init__(self, torch, model, device="cpu", ctx=None, amp=None, info=None):
        self.torch, self.model, self.device, self.ctx, self.amp = torch, model, device, ctx, amp
        self.info = info or {}

    @classmethod
    def load(cls, ckpt, config=None, tokenizer=None, device="cpu", precision="fp32"):
        import torch
        import planck_responder as PR
        r = PR.from_checkpoint(ckpt, config, tokenizer, device, precision, "template")
        info = dict(model_type="planck", precision=precision, amp=bool(r.amp), device=device)
        return cls(torch, r.model, device, r.ctx, r.amp, info), PlanckTok(r)

    def __call__(self, pre, cont):
        torch = self.torch
        seq = list(pre) + list(cont)
        if self.ctx is not None and len(seq) > self.ctx:
            raise ValueError(f"sequence of {len(seq)} tokens exceeds ctx {self.ctx}")
        x = torch.tensor([seq], dtype=torch.long, device=self.device)
        with torch.no_grad():
            if self.amp:
                with self.amp():
                    logits = self.model(x)[0][0]
            else:
                logits = self.model(x)[0][0]
        return seq_logprob(torch, logits, seq, len(pre), len(cont))

    def many(self, pre, conts):
        return [self(pre, c) for c in conts]
