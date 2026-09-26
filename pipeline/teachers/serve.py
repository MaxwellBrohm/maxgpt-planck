"""Teacher wrapper for vLLM's offline Python API on the PC (RTX 5070). DRY PREP: nothing it produces is training data.

    import serve
    t = serve.load("ministral-3-8b")                       # take ~/planck/locks/gpu.lock BEFORE this call
    outs = serve.generate(t, ["plain instruction prompt", ...], {"temperature": 0.8, "max_tokens": 400},
                          seeds=[11, 12, ...])
    serve.unload(t)                                        # then release the lock

Callers pass the plain instruction prompt (render_prompt.build(skel)["prompt"]). serve renders each teacher's wire
format to token ids itself, with the teacher's own tokenizer, and refuses a render that breaks its invariants:
  gemma-4-12b     raw Gemma 4 turn format of tools/gemma_chat.py, thinking OFF (empty thought channel pre-filled),
                  plus the <bos> the Gemma 4 tokenizer does not add on its own: ids[0] == 2, exactly one 2;
  ministral-3-8b  mistral_common chat encoding from the OFFICIAL tekken.json: '<s>[INST]...[/INST]', no system
                  prompt ('Le Chat' default of the HF template never enters), no [THINK] ids;
  qwen3.5-9b      HF chat template with enable_thinking=False: the prompt must end '<think>\\n\\n</think>\\n\\n'.
Outputs are decoded with skip_special_tokens=False, so a thought or channel token stays visible; every output
carries "thought" (the markers or ids found) and the checker's THOUGHT_TAG sees the raw text too.
Pins (repo, revision, license) live in hf_pins.json next to this file; TEACHERS below repeats what load() needs.
Environment: ~/planck/venv-teach, an overlay of ~/planck/venv-vllm (vLLM 0.30.0, torch 2.13.0+cu132) through a .pth
file, with its own transformers 5.16.0: vLLM 0.30.0's pixtral.py imports PixtralRotaryEmbedding, which transformers
5.17.0 (venv-vllm) renamed, so Ministral 3 fails model inspection there. venv-vllm itself is left unchanged."""
import gc
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.environ.get("PLANCK_MODELS", os.path.expanduser("~/planck/models"))
STATUS = "dry"

ENGINE = {"max_model_len": 2048, "gpu_memory_utilization": 0.88, "language_model_only": True, "seed": 0}
# per-teacher max_num_seqs / max_num_batched_tokens: vLLM's warm-up profiles the sampler at max_num_seqs rows of fp32
# logits, and Gemma (262k vocab) at 256 seqs left -0.94 GiB for KV at util 0.88 (8.28 GiB weights). Gemma's KV room
# is a few sequences anyway, so a smaller cap costs nothing. Qwen's DeltaNet state shares one block pool with the
# attention KV (528-token blocks); vLLM refuses max_num_seqs above the state block count (64 > 32 on 2026-09-26).

TEACHERS = {
    "gemma-4-12b": {
        "dir": "gemma-4-12B-it-qat-w4a16-ct", "repo": "google/gemma-4-12B-it-qat-w4a16-ct",
        "revision": "1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee", "license": "Apache-2.0", "quant": "W4A16 QAT int4 g32",
        "wire": "gemma_raw", "engine": {"max_num_seqs": 32, "max_num_batched_tokens": 2048},
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.05},
        "stop": ["<turn|>", "<|turn>"], "bos_id": 2,
        "thought_strings": ["<|channel>", "<channel|>", "<|think|>"], "thought_tokens": ["<|channel>", "<channel|>",
                                                                                         "<|think|>"]},
    "ministral-3-8b": {
        "dir": "Ministral-3-8B-Instruct-2512-AWQ-4bit", "repo": "cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-4bit",
        "revision": "e3a799844ba08f27fddbf118e154057eb0105889", "license": "Apache-2.0", "quant": "AWQ int4 g32",
        "tokenizer_dir": "Ministral-3-8B-Instruct-2512-official-tok",
        "wire": "mistral_chat",
        "engine": {"tokenizer_mode": "mistral", "config_format": "hf", "load_format": "safetensors",
                   "max_num_seqs": 256},
        "sampling": {"temperature": 0.8, "top_p": 0.95},
        "stop": [], "thought_strings": ["[THINK]", "[/THINK]"], "thought_ids": [34, 35]},
    "qwen3.5-9b": {
        "dir": "Qwen3.5-9B-AWQ-4bit", "repo": "cyankiwi/Qwen3.5-9B-AWQ-4bit",
        "revision": "156edc4bbeb8d1910ee7be9196bafaf1bc052156", "license": "Apache-2.0", "quant": "AWQ int4 g32",
        "wire": "hf_chat", "chat_kwargs": {"enable_thinking": False},
        "engine": {"max_num_seqs": 32, "max_num_batched_tokens": 4096},
        "sampling": {"temperature": 0.8, "top_p": 0.95},
        "stop": ["<|im_end|>"], "must_end": "<think>\n\n</think>\n\n",
        "thought_strings": ["<think>", "</think>"], "thought_tokens": ["<think>", "</think>"]},
}
END_MARKS = ("<turn|>", "<|turn>", "</s>", "<|im_end|>", "<|endoftext|>", "<eos>")


class RenderError(ValueError):
    pass


class Teacher:
    """a loaded (or tokenizer-only) teacher: name, cfg, tok, llm, engine kwargs, load seconds."""
    def __init__(self, name, cfg, tok, llm=None, engine=None, load_s=None):
        self.name, self.cfg, self.tok, self.llm, self.engine, self.load_s = name, cfg, tok, llm, engine, load_s
        self.thought_ids = set(cfg.get("thought_ids", []))


def model_path(cfg, models=None):
    return os.path.join(models or MODELS, cfg["dir"])


def tokenizer(name, models=None):
    """the teacher's own tokenizer on CPU (no model): mistral_common for Ministral, transformers otherwise."""
    cfg = TEACHERS[name]
    if cfg["wire"] == "mistral_chat":
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        return MistralTokenizer.from_file(os.path.join(models or MODELS, cfg["tokenizer_dir"], "tekken.json"))
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path(cfg, models))


def tokenizer_only(name, models=None):
    t = Teacher(name, TEACHERS[name], tokenizer(name, models))
    _resolve_thought_ids(t)
    return t


def _resolve_thought_ids(t):
    for w in t.cfg.get("thought_tokens", []):
        i = t.tok.convert_tokens_to_ids(w)
        if i is not None and i != getattr(t.tok, "unk_token_id", None):
            t.thought_ids.add(i)


def render(t, prompt):
    """plain prompt -> (token ids, rendered text). Raises RenderError when a wire invariant does not hold."""
    cfg, w = t.cfg, t.cfg["wire"]
    if w == "gemma_raw":
        text = "<bos><|turn>user\n" + prompt + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
        ids = t.tok.encode(text, add_special_tokens=False)
        if not ids or ids[0] != cfg["bos_id"] or ids.count(cfg["bos_id"]) != 1:
            raise RenderError(f"gemma: bos {cfg['bos_id']} must be first and only once, got {ids[:4]}")
        if ids[-3:] != t.tok.encode("<|channel>thought\n<channel|>", add_special_tokens=False)[-3:]:
            raise RenderError("gemma: prompt does not end in the empty thought channel")
    elif w == "mistral_chat":
        from mistral_common.protocol.instruct.messages import UserMessage
        from mistral_common.protocol.instruct.request import ChatCompletionRequest
        r = t.tok.encode_chat_completion(ChatCompletionRequest(messages=[UserMessage(content=prompt)]))
        ids, text = list(r.tokens), r.text
        if not text.startswith("<s>[INST]") or not text.endswith("[/INST]") or "[SYSTEM_PROMPT]" in text:
            raise RenderError(f"ministral: unexpected wrapper {text[:40]!r} ... {text[-20:]!r}")
        if t.thought_ids & set(ids):
            raise RenderError("ministral: a [THINK] id is in the prompt")
    elif w == "hf_chat":
        text = t.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True, **cfg.get("chat_kwargs", {}))
        if not text.endswith(cfg["must_end"]):
            raise RenderError(f"qwen: thinking is not off, prompt ends {text[-40:]!r}")
        ids = t.tok.encode(text, add_special_tokens=False)
    else:
        raise RenderError(f"unknown wire {w}")
    return ids, text


def thought_in(t, text, ids):
    """markers of a thought block in one output: strings in the raw text, and thought token ids."""
    found = [s for s in t.cfg["thought_strings"] if s in text]
    found += [f"id{i}" for i in sorted(t.thought_ids & set(ids))]
    return found


def load(name, models=None, **engine_overrides):
    """load one teacher under vLLM. The caller holds the GPU lock from before this call until unload()."""
    # FlashInfer's top-k/top-p sampler JIT-builds with nvcc, which this WSL has not got (warm-up died with "Could
    # not find nvcc"); vLLM's own sampler needs no build. Set before vllm is imported; the engine core inherits it.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM
    cfg = TEACHERS[name]
    t = tokenizer_only(name, models)
    render(t, "Say hello.")                     # fail on a broken wire format before spending GPU time
    kw = {**ENGINE, **cfg["engine"], **engine_overrides}
    if cfg.get("tokenizer_dir"):
        kw["tokenizer"] = os.path.join(models or MODELS, cfg["tokenizer_dir"])
    t0 = time.time()
    t.llm = LLM(model=model_path(cfg, models), **kw)
    t.load_s, t.engine = round(time.time() - t0, 1), kw
    return t


def _params(t, sampling, n_prompt, seed):
    from vllm import SamplingParams
    s = {**t.cfg["sampling"], **(sampling or {})}
    s.setdefault("stop", t.cfg["stop"] or None)
    s["skip_special_tokens"] = False
    room = t.engine["max_model_len"] - n_prompt
    clamped = s.get("max_tokens", 256) > room
    s["max_tokens"] = max(1, min(s.get("max_tokens", 256), room))
    if s.get("min_tokens", 0) > s["max_tokens"]:
        s["min_tokens"] = s["max_tokens"]
    if seed is not None:
        s["seed"] = seed
    return SamplingParams(**s), clamped


def strip_end(text):
    text = text.rstrip()
    changed = True
    while changed:
        changed = False
        for m in END_MARKS:
            if text.endswith(m):
                text, changed = text[: -len(m)].rstrip(), True
    return text


def generate(t, prompts, sampling=None, seeds=None, max_tokens=None, use_tqdm=False):
    """-> one dict per prompt: text (end markers stripped), raw, finish, n_prompt, n_out, thought, clamped.
    seeds and max_tokens are optional per-prompt lists (the pipeline's render seed and max_tokens per skeleton).
    A prompt whose render breaks an invariant gets finish 'render_error' and no teacher call."""
    from vllm.inputs import TokensPrompt
    todo, out = [], [None] * len(prompts)
    for k, p in enumerate(prompts):
        try:
            ids, _ = render(t, p)
        except RenderError as e:
            out[k] = {"text": "", "raw": "", "finish": "render_error", "error": str(e), "n_prompt": 0, "n_out": 0,
                      "thought": [], "clamped": False}
            continue
        s = sampling if max_tokens is None else {**(sampling or {}), "max_tokens": max_tokens[k]}
        sp, clamped = _params(t, s, len(ids), None if seeds is None else seeds[k])
        todo.append((k, TokensPrompt(prompt_token_ids=ids), sp, clamped, len(ids)))
    if todo:
        res = t.llm.generate([x[1] for x in todo], [x[2] for x in todo], use_tqdm=use_tqdm)
        for (k, _, _, clamped, n_prompt), r in zip(todo, res):
            c = r.outputs[0]
            ids = list(c.token_ids)
            out[k] = {"text": strip_end(c.text), "raw": c.text, "finish": c.finish_reason, "n_prompt": n_prompt,
                      "n_out": len(ids), "thought": thought_in(t, c.text, ids), "clamped": clamped}
    return out


def kv_info(t):
    """what the frontend knows about the KV cache (the engine core logs the token capacity line)."""
    cc, sc = t.llm.llm_engine.vllm_config.cache_config, t.llm.llm_engine.vllm_config.scheduler_config
    return {"num_gpu_blocks": cc.num_gpu_blocks, "block_size": cc.block_size, "kv_cache_dtype": str(cc.cache_dtype),
            "max_num_seqs": sc.max_num_seqs, "max_num_batched_tokens": sc.max_num_batched_tokens}


def kv_from_log(text):
    """the last 'KV cache size: N tokens, Maximum concurrency for M tokens per request: X' line of a vLLM log."""
    import re
    hits = re.findall(r"KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) tokens per request: "
                      r"([\d.]+)x", text)
    if not hits:
        return None
    n, m, x = hits[-1]
    return {"kv_tokens": int(n.replace(",", "")), "per_request_tokens": int(m.replace(",", "")),
            "max_concurrency": float(x)}


def unload(t):
    """shut the engine core down and drop every reference, so the GPU memory goes back before the lock is released."""
    if t.llm is None:
        return
    core = getattr(t.llm.llm_engine, "engine_core", None)
    if core is not None and hasattr(core, "shutdown"):
        try:
            core.shutdown()
        except Exception:  # an engine that already died still has to be dropped
            pass
    t.llm = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except ImportError:
        pass
