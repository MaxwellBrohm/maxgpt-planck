"""RC-12 Hugging Face causal-LM responder for runner.py (SPEC s2).

HF_TESTED = True since the logged step notes STEP 9b (2026-09-26): greedy on 20 conversations per family through
parity_hf_vllm.py (Qwen2.5, Qwen3, Qwen3.5, LFM2.5, LFM2, Falcon-H1, SmolLM2, Gemma 3; 240 turns each), every prompt
equal to vLLM's, stop reasons and transcripts read by eye. KNOWN DEFECT: gemma-3-270m-it under transformers 5.17's
default (sdpa) attention ends the reply at once (p of <end_of_turn> about 1) on 6 long-prompt parity turns where
eager attention and vLLM agree on a real reply; never run Gemma 3 through this file as it stands (engines.json
names vLLM). Doge-160M-Instruct does not load under transformers 5.17 (notes STEP 9b); it runs through hf_batched.py
with trust_remote_code=True (its own modeling_doge.py) in ~/planck/venv-doge (transformers 4.55.0) and with EAGER
attention: under sdpa its dynamic mask turns SDPA's causal flag off whenever a batch has no padding, so a prompt
token sees later tokens (notes STEP 9c). engines.json names all three per model; trust_remote_code is False and
attn_implementation the library default unless a caller passes them. runner.py still asks for --hf-untested-ok
(its gate never read this flag).

Importing this module imports nothing heavy; torch and transformers load in HFResponder.__init__ only.
Decoding: greedy (seed None) or sampling with T = 0.6, top-p 1.0, top-k OFF, repetition penalty 1.0, max 256 new
tokens (SPEC s2). Every knob is passed explicitly because a model's generation_config.json can carry its own
defaults (Qwen ships top_p 0.8, top_k 20, repetition_penalty 1.05), which would silently change the protocol.
Seeding: torch.manual_seed(md5(seed, conversation id, turn)) before each sampled reply, so a reply does not depend
on batch order or on how many conversations ran before it.
Render: template = tokenizer.apply_chat_template (no system prompt; enable_thinking=False for Qwen3-family
config.model_type), tokenized without extra special tokens; plain = render.plain(), tokenized WITH the tokenizer's
BOS if it adds one. Stops: eos_token_id plus the end-of-turn tokens below that exist in the vocabulary.
Stop reason: "eot" (an end-of-turn token), "eos" (the eos token, or generation ended early), "cap" (256 tokens)."""
import hashlib

import render as RD

HF_TESTED = True
DECODE = dict(temperature=0.6, top_p=1.0, top_k=0, repetition_penalty=1.0, max_new_tokens=RD.MAX_NEW_TOKENS)
EOT_TOKENS = ["<|im_end|>", "<|eot_id|>", "<|end|>", "<end_of_turn>", "<|endoftext|>", "<|end_of_text|>"]


def conv_seed(seed, rid, turn):
    return int(hashlib.md5(f"{seed}:{rid}:{turn}".encode()).hexdigest()[:8], 16)


def set_template(tok, chat_template):
    """engines.json "chat_template" (notes STEP 11, item 14): the model's documented chat format for a checkpoint
    that ships none (Loom-Spark-3.2, its card's <user> / <loom> format). None leaves the tokenizer's own template;
    a string replaces it before any prompt is built. Returns what was set (None = the tokenizer's own)."""
    if chat_template is None:
        return None
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError(f"chat_template must be a non-empty string, not {chat_template!r}")
    tok.chat_template = chat_template
    return chat_template


def _zero_nans(module, inputs, out):
    t = out[0] if isinstance(out, tuple) else out
    t.masked_fill_(t.isnan(), 0.0)


def guard_nans(model, cls_name):
    """engines.json "nan_guard" (hf / hfb; notes STEP 11, ITEM 14): a forward hook on every module of class cls_name
    sets NaN in its output to 0. For remote code whose attention fills masked scores with -inf (Swen-28M): under left
    padding a pad query sees only masked keys, its softmax is NaN, and the NaN reaches real rows through the next
    layer's value product (0 x NaN). With the hook a pad row's attention output is 0 and real rows never see a NaN;
    a real row's own scores are never all masked, so its numbers are unchanged. None: no hook. Returns the count."""
    if cls_name is None:
        return 0
    if not isinstance(cls_name, str) or not cls_name:
        raise ValueError(f"nan_guard must be a module class name, not {cls_name!r}")
    mods = [m for _, m in model.named_modules() if type(m).__name__ == cls_name]
    if not mods:
        raise ValueError(f"nan_guard: no module of class {cls_name} in {type(model).__name__}")
    for m in mods:
        m.register_forward_hook(_zero_nans)
    return len(mods)


def set_use_cache(model, use_cache):
    """engines.json "use_cache" (hf / hfb; notes STEP 11, ITEM 14): a bool sets model.config.use_cache, which remote
    code may read in prepare_inputs_for_generation. cRia-LM-75M-Instruct ships config.json use_cache false, so its
    generate recomputes the whole prefix at every step although its own KV cache is wired in (CPU probe 2026-10-04:
    3 of 3 greedy replies equal with and without the cache, 0.4 s vs 1.1-2.1 s each). None leaves the config."""
    if use_cache is None:
        return None
    if type(use_cache) is not bool:
        raise ValueError(f"use_cache must be a bool, not {use_cache!r}")
    model.config.use_cache = use_cache
    return use_cache


class HFResponder:
    def __init__(self, model_id, render="template", dtype="bfloat16", device="cuda", trust_remote_code=False,
                 attn_implementation=None, chat_template=None, nan_guard=None, use_cache=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.device, self.render = torch, device, render
        self.trust_remote_code = bool(trust_remote_code)
        attn = {} if attn_implementation is None else {"attn_implementation": attn_implementation}
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.trust_remote_code)
        self.chat_template = set_template(self.tok, chat_template)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=getattr(torch, dtype),
                                                          trust_remote_code=self.trust_remote_code, **attn)
        self.attn = getattr(self.model.config, "_attn_implementation", None)       # as loaded
        self.nan_guard, self.nan_guarded = nan_guard, guard_nans(self.model, nan_guard)
        self.use_cache = set_use_cache(self.model, use_cache)
        self.model.to(device).eval()
        self.has_template = bool(getattr(self.tok, "chat_template", None))
        if render == "template" and not self.has_template:
            raise ValueError(f"{model_id} ships no chat template: use --render plain")
        self.qwen3 = str(getattr(self.model.config, "model_type", "")).startswith("qwen3")
        vocab = self.tok.get_vocab()
        self.eos = self.tok.eos_token_id
        self.eot = sorted({vocab[t] for t in EOT_TOKENS if t in vocab} - {self.eos})
        self.stop_ids = ([self.eos] if self.eos is not None else []) + self.eot
        self.ctx = getattr(self.model.config, "max_position_embeddings", None)
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos

    def start(self, rec, seed=None, render=None):
        self.rid, self.seed = rec["id"], seed
        if render and render != self.render:
            raise ValueError("render changed after loading")

    def prompt(self, messages):
        return RD.template(self.tok, messages, self.qwen3) if self.render == "template" else RD.plain(messages)

    def encode(self, messages):
        return self.tok(self.prompt(messages), add_special_tokens=self.render == "plain").input_ids

    def count(self, messages, render=None):
        return len(self.encode(messages))

    def reply(self, history, i):
        torch = self.torch
        ids = torch.tensor([self.encode(history)], device=self.device)
        kw = dict(max_new_tokens=DECODE["max_new_tokens"], eos_token_id=self.stop_ids, pad_token_id=self.pad,
                  repetition_penalty=DECODE["repetition_penalty"])
        if self.seed is None:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        else:
            torch.manual_seed(conv_seed(self.seed, self.rid, i))
            kw.update(do_sample=True, temperature=DECODE["temperature"], top_p=DECODE["top_p"],
                      top_k=DECODE["top_k"])
        with torch.no_grad():
            out = self.model.generate(ids, attention_mask=torch.ones_like(ids), **kw)[0, ids.shape[1]:].tolist()
        if out and out[-1] in self.eot:
            stop = "eot"
        elif len(out) >= DECODE["max_new_tokens"] and (not out or out[-1] not in self.stop_ids):
            stop = "cap"
        else:
            stop = "eos"
        return self.tok.decode(out, skip_special_tokens=True), stop
