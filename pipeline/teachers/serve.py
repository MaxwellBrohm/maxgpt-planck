"""Teacher wrapper for vLLM's offline Python API on the PC (RTX 5070). DRY PREP: nothing it produces is training data.

    import serve
    with serve.loaded("ministral-3-8b") as t:              # run under flock ~/planck/locks/gpu.lock (checked)
        outs = serve.generate(t, ["plain instruction prompt", ...], {"temperature": 0.8, "max_tokens": 400},
                              seeds=[11, 12, ...])
    # unloaded here, even on an exception; the lock is released when the flock'd process exits

load() refuses unless gpu.lock is held by this process or an ancestor (gpulock.py: /proc/locks, then the ancestors'
/proc/<pid>/fdinfo), so nothing loads on the Mac or outside the shared lock. A prompt whose render breaks an invariant, or that leaves no room for one
output token under max_model_len, comes back as finish 'render_error' without a teacher call (vLLM would otherwise
fail the whole batch on it).

Callers pass the plain instruction prompt (render_prompt.build(skel)["prompt"]). serve renders each teacher's wire
format to token ids itself, with the teacher's own tokenizer, and refuses a render that breaks its invariants:
  gemma-4-12b     raw Gemma 4 turn format of tools/gemma_chat.py, thinking OFF (empty thought channel pre-filled),
                  plus the <bos> the Gemma 4 tokenizer does not add on its own: ids[0] == 2, exactly one 2;
  ministral-3-8b  mistral_common chat encoding from the OFFICIAL tekken.json: '<s>[INST]...[/INST]', no system
                  prompt ('Le Chat' default of the HF template never enters), no [THINK] ids;
  qwen3.5-9b      HF chat template with enable_thinking=False: the prompt must end '<think>\\n\\n</think>\\n\\n'.
Outputs are decoded with skip_special_tokens=False, so a thought or channel token stays visible; every output
carries "thought" (the markers or ids found) and the checker sees the raw text too: THOUGHT_TAG for Gemma and Qwen
tags; Ministral's think ids decode as '<SPECIAL_34>'/'<SPECIAL_35>' under the official tekken, which THOUGHT_TAG_RE
does not match but DIGIT (every turn) rejects (test_drive.MinistralThoughtText, 2026-09-27).
Decoding controls (2026-09-27, D3/D4; off unless a request asks): a request's sampling may carry "preset" (a named
D4 preset from TEACHERS, which replaces the default sampling), "regex" (structured output through xgrammar; load
with structured_backend="xgrammar") and "ban_ids" (logit_bias -100 on each id: dash_ban(), cached per tokenizer
hash). decode.py holds the regex builder and the ban rule. gpu_memory_utilization is 0.86 by default and never above.
Round 2 (2026-09-28): "bad_words" (the AI-ism phrase ban, decode.PHRASES minus the chat's literals) goes to vLLM's
own SamplingParams.bad_words; phrase_ban() runs vLLM's update_from_tokenizer on the full list once per server, so
the sequence and token caps are checked before the first request and the token form of every phrase is recorded.
Pins (repo, revision, license) live in hf_pins.json next to this file; TEACHERS below repeats what load() needs.
Environment: ~/planck/venv-teach, an overlay of ~/planck/venv-vllm (vLLM 0.30.0, torch 2.13.0+cu132) through a .pth
file, with its own transformers 5.16.0: vLLM 0.30.0's pixtral.py imports PixtralRotaryEmbedding, which transformers
5.17.0 (venv-vllm) renamed, so Ministral 3 fails model inspection there. venv-vllm itself is left unchanged."""
import contextlib
import gc
import hashlib
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import decode  # noqa: E402
import gpulock  # noqa: E402
MODELS = os.environ.get("PLANCK_MODELS", os.path.expanduser("~/planck/models"))
STATUS = "dry"

# 0.86 since 2026-09-27: at 0.88 the peaks (11,830 / 11,723 / 11,599 MiB device-wide) met gpuguard's 11,600 MiB
# threshold; at 0.86 the decode probe still saw 11,725 MiB with Ministral loaded (the Windows desktop's share varies)
ENGINE = {"max_model_len": 2048, "gpu_memory_utilization": 0.86, "language_model_only": True, "seed": 0}
MAX_UTIL = 0.86
DEFAULT_PRESET = "yld0926"      # TEACHERS[...]["sampling"]: what the 2026-09-26 yield runs used
BAN_CACHE = os.environ.get("PLANCK_TEACH_CACHE", os.path.expanduser("~/planck/teach/cache"))
# mistral_common: Tokenized.text (read by render until 2026-09-27) warns in 1.12 and is removed in 1.13; render now
# decodes the ids with SpecialTokenPolicy.KEEP (equal to .text on 30 of 30 FAKE renders, probe 09-27), a call 1.11.7
# also has. Pinned to the range that was read (1.11.7 source) or run (1.12.0): >=1.11.7,<1.13.
MISTRAL_COMMON_OK = ((1, 11, 7), (1, 13, 0))
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
        # D4 presets; a preset REPLACES "sampling" (no key of it leaks in). Card, google/gemma-4-12B-it at the pin,
        # "1. Sampling Parameters": "Use the following standardized sampling configuration across all use cases:
        # temperature=1.0, top_p=0.95, top_k=64". It names no min_p, so the card preset has vLLM's 0.0; the
        # 0.05 of SPEC 5 (every Gemma run so far) is not the card's. No "shared" preset: D4 runs Gemma at its card only.
        "presets": {"card": {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.0}},
        "vocab": 262144, "stop": ["<turn|>", "<|turn>"], "bos_id": 2,
        "thought_strings": ["<|channel>", "<channel|>", "<|think|>"], "thought_tokens": ["<|channel>", "<channel|>",
                                                                                         "<|think|>"]},
    "ministral-3-8b": {
        "dir": "Ministral-3-8B-Instruct-2512-AWQ-4bit", "repo": "cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-4bit",
        "revision": "e3a799844ba08f27fddbf118e154057eb0105889", "license": "Apache-2.0", "quant": "AWQ int4 g32",
        "tokenizer_dir": "Ministral-3-8B-Instruct-2512-official-tok",
        "wire": "mistral_chat",
        "engine": {"tokenizer_mode": "mistral", "config_format": "hf", "load_format": "safetensors",
                   "max_num_seqs": 256},
        # "sampling" (preset yld0926) is NOT the card's setting: the 2026-09-26 yield run (0/399) used this 0.8, so
        # that result says nothing about Ministral at its own setting
        "sampling": {"temperature": 0.8, "top_p": 0.95},
        # Card, mistralai/Ministral-3-8B-Instruct-2512 at the pin, "Recommended Settings": "Use a **temperature below
        # 0.1** for daily-driver and production environments" (it adds that higher temperatures may be explored for
        # creative use). No single value and no top_p: 0.05 is what the 09-27 probe ran (dashes still 26 of 30,
        # distinct-2 0.71 against 0.79); top_p 1.0 is off. "shared" is 0.8 / 0.95, the 09-26 yield run's setting.
        "presets": {"card": {"temperature": 0.05, "top_p": 1.0}, "shared": {"temperature": 0.8, "top_p": 0.95}},
        "vocab": 131072, "line_sep": "\\n+",   # blank lines between lines: see decode.py
        "stop": [], "thought_ids": [34, 35],   # the official tekken decodes 34/35 as <SPECIAL_34>/<SPECIAL_35>
        "thought_strings": ["[THINK]", "[/THINK]", "<SPECIAL_34>", "<SPECIAL_35>"]},
    "qwen3.5-9b": {
        "dir": "Qwen3.5-9B-AWQ-4bit", "repo": "cyankiwi/Qwen3.5-9B-AWQ-4bit",
        "revision": "156edc4bbeb8d1910ee7be9196bafaf1bc052156", "license": "Apache-2.0", "quant": "AWQ int4 g32",
        "wire": "hf_chat", "chat_kwargs": {"enable_thinking": False},
        # at most 32 run at once (28 seen at the KV limit), whatever the caller's concurrency (the yield run's 64)
        "engine": {"max_num_seqs": 32, "max_num_batched_tokens": 4096},
        # "sampling" (preset yld0926) is NOT the card's non-thinking setting
        "sampling": {"temperature": 0.8, "top_p": 0.95},
        # Card, Qwen/Qwen3.5-9B at the pin, sampling tip, the line for thinking off: "Instruct (or non-thinking) mode
        # for general tasks: temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5,
        # repetition_penalty=1.0" (vLLM took all of these in the 09-27 probe). "shared" is 0.8 / 0.95.
        "presets": {"card": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
                             "repetition_penalty": 1.0},
                    "shared": {"temperature": 0.8, "top_p": 0.95}},
        "vocab": 248320,
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
        check_mistral_common()
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        return MistralTokenizer.from_file(os.path.join(models or MODELS, cfg["tokenizer_dir"], "tekken.json"))
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path(cfg, models))


def check_mistral_common(version=None):
    """refuse a mistral_common outside MISTRAL_COMMON_OK (a loud failure before any GPU time)."""
    if version is None:
        from importlib.metadata import version as _v
        version = _v("mistral_common")
    got = tuple(int(x) for x in re.findall(r"\d+", version)[:3])
    lo, hi = MISTRAL_COMMON_OK
    if not lo <= got < hi:
        raise RuntimeError(f"mistral_common {version} is outside the pin >={'.'.join(map(str, lo))},"
                           f"<{'.'.join(map(str, hi[:2]))} (serve.MISTRAL_COMMON_OK)")
    return version


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
        from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy
        r = t.tok.encode_chat_completion(ChatCompletionRequest(messages=[UserMessage(content=prompt)]))
        ids = list(r.tokens)       # the text is decoded from the ids: Tokenized.text is deprecated (MISTRAL_COMMON_OK)
        text = t.tok.decode(ids, special_token_policy=SpecialTokenPolicy.KEEP)
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


def load(name, models=None, require_lock=True, structured_backend=None, **engine_overrides):
    """load one teacher under vLLM. Refuses unless gpu.lock is held by this process or an ancestor (tests on a stub
    pass require_lock=False); hold it until unload(). structured_backend ("xgrammar", the backend the 09-27 probe
    ran) lets requests carry a regex (resolve: "regex"); gpu_memory_utilization above MAX_UTIL is refused."""
    if (engine_overrides.get("gpu_memory_utilization") or 0) > MAX_UTIL:
        raise ValueError(f"gpu_memory_utilization {engine_overrides['gpu_memory_utilization']} is above {MAX_UTIL} "
                         "(gpuguard stops a job past 11,600 MiB)")
    if require_lock:
        gpulock.check()
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
    if structured_backend:
        from vllm.config import StructuredOutputsConfig
        kw["structured_outputs_config"] = StructuredOutputsConfig(backend=structured_backend)
    t0 = time.time()
    t.llm = LLM(model=model_path(cfg, models), **kw)
    t.load_s, t.engine = round(time.time() - t0, 1), kw
    return t


@contextlib.contextmanager
def loaded(name, models=None, **kw):
    """load(...) for a with-block; unload() runs on the way out, exceptions included."""
    t = load(name, models, **kw)
    try:
        yield t
    finally:
        unload(t)


def preset_names(name):
    return sorted({DEFAULT_PRESET, *TEACHERS[name].get("presets", {})})


def preset(name, p):
    """a named sampling preset (D4): DEFAULT_PRESET is TEACHERS sampling, the others TEACHERS presets."""
    cfg = TEACHERS[name]
    if p == DEFAULT_PRESET:
        return dict(cfg["sampling"])
    if p not in cfg.get("presets", {}):
        raise ValueError(f"{name} has no sampling preset {p!r}: it has {preset_names(name)} (D4: Gemma runs at its "
                         "card setting only)")
    return dict(cfg["presets"][p])


EXTRAS = ("preset", "regex", "ban_ids", "bad_words")


def resolve(t, sampling):
    """per-request sampling -> (SamplingParams kwargs, extras). Extras: preset (the named preset REPLACES the
    teacher's default sampling, so no default key leaks in), regex (structured output), ban_ids (logit_bias),
    bad_words (vLLM's phrase ban)."""
    s = dict(sampling or {})
    ex = {k: s.pop(k) for k in EXTRAS if s.get(k) is not None}
    base = preset(t.name, ex["preset"]) if "preset" in ex else dict(t.cfg["sampling"])
    return {**base, **s}, ex


def sampling_record(t, sampling):
    """what one request runs with, for its result record: preset, sampling values, regex hash, ban size."""
    s, ex = resolve(t, sampling)
    return {"preset": ex.get("preset", DEFAULT_PRESET),
            "sampling": {k: v for k, v in s.items() if k not in ("max_tokens", "min_tokens", "seed", "stop")},
            "regex_sha256": decode.sha(ex["regex"]) if "regex" in ex else None, "ban_n": len(ex.get("ban_ids", ())),
            "bad_words_n": len(ex.get("bad_words", ())),
            "bad_words_sha256": decode.sha("\n".join(ex["bad_words"])) if "bad_words" in ex else None}


def _params(t, sampling, n_prompt, seed):
    from vllm import SamplingParams
    s, ex = resolve(t, sampling)
    if "regex" in ex:
        from vllm.sampling_params import StructuredOutputsParams
        s["structured_outputs"] = StructuredOutputsParams(regex=ex["regex"])
    if "ban_ids" in ex:
        s["logit_bias"] = decode.logit_bias(ex["ban_ids"])
    if ex.get("bad_words"):
        s["bad_words"] = list(ex["bad_words"])
    s.setdefault("stop", t.cfg["stop"] or None)
    s["skip_special_tokens"] = False
    room = t.engine["max_model_len"] - n_prompt
    want = 256 if s.get("max_tokens") is None else s["max_tokens"]
    clamped = want > room
    s["max_tokens"] = max(1, min(want, room))
    if (s.get("min_tokens") or 0) > s["max_tokens"]:
        s["min_tokens"] = s["max_tokens"]
    if seed is not None:
        s["seed"] = seed
    return SamplingParams(**s), clamped


def prepare(t, prompt, sampling=None, seed=None):
    """plain prompt -> (ids, SamplingParams, clamped) for one request; RenderError when a wire invariant breaks or
    the prompt leaves no room for one output token under max_model_len."""
    ids, _ = render(t, prompt)
    if len(ids) >= t.engine["max_model_len"]:
        raise RenderError(f"prompt of {len(ids)} tokens leaves no room under max_model_len "
                          f"{t.engine['max_model_len']}")
    sp, clamped = _params(t, sampling, len(ids), seed)
    return ids, sp, clamped


def tokenizer_files(name, models=None):
    cfg = TEACHERS[name]
    if cfg["wire"] == "mistral_chat":
        d, names = os.path.join(models or MODELS, cfg["tokenizer_dir"]), ["tekken.json"]
    else:
        d, names = model_path(cfg, models), ["tokenizer.json", "tokenizer_config.json"]
    paths = [os.path.join(d, n) for n in names if os.path.exists(os.path.join(d, n))]
    if not paths:
        raise FileNotFoundError(f"no tokenizer file for {name} in {d}")
    return paths


def tokenizer_sha(name, models=None):
    """sha256 over the teacher's tokenizer files and its model vocab size: the dash ban's cache key."""
    h = hashlib.sha256(f"{name}|vocab={TEACHERS[name]['vocab']}".encode())
    for p in tokenizer_files(name, models):
        h.update(b"\0" + os.path.basename(p).encode() + b"\0")
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def token_bytes(t):
    """[bytes] per token id, the tables the 09-27 probe built its ban from: mistral_common byte pieces (tekken,
    special tokens kept) or xgrammar's decoded vocabulary at the model's vocab size (HF tokenizers)."""
    if t.cfg["wire"] == "mistral_chat":
        from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy
        tk = t.tok.instruct_tokenizer.tokenizer
        out = []
        for i in range(tk.n_words):
            try:
                out.append(tk.id_to_byte_piece(i, SpecialTokenPolicy.KEEP))
            except Exception:  # an id with no byte piece is never sampled as text
                out.append(b"")
        return out
    import xgrammar as xgr
    info = xgr.TokenizerInfo.from_huggingface(t.tok, vocab_size=t.cfg["vocab"])
    return [b if isinstance(b, bytes) else str(b).encode() for b in info.decoded_vocab]


def dash_ban(t, models=None, cache_dir=None):
    """t's dash-token ban (decode.dash_ban_ids), computed once and cached as JSON under the tokenizer files' sha256
    and the rule name; an entry whose hash, rule, teacher or id checksum disagrees is computed again."""
    tsha = tokenizer_sha(t.name, models)
    d = cache_dir or BAN_CACHE
    path = os.path.join(d, f"dashban_{t.name}_{decode.DASH_RULE}_{tsha[:16]}.dry.json")
    try:
        with open(path) as f:
            c = json.load(f)
        if (c.get("teacher"), c.get("rule"), c.get("tokenizer_sha256"), c.get("ids_sha256")) == \
                (t.name, decode.DASH_RULE, tsha, decode.ids_sha(c.get("ids") or [])):
            return {**c, "cached": True, "path": path}
    except (OSError, ValueError):
        pass
    vocab = token_bytes(t)
    ids, counts = decode.dash_ban_ids(vocab)
    decode.logit_bias(ids)                      # refuses a set over the per-request cap
    c = {"status": "dry", "teacher": t.name, "rule": decode.DASH_RULE, "tokenizer_sha256": tsha,
         "vocab_n": len(vocab), "n": len(ids), "ids": ids, "ids_sha256": decode.ids_sha(ids), "counts": counts}
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(c, f)
    os.replace(tmp, path)
    return {**c, "cached": False, "path": path}


def phrase_ban(vtok, words=decode.PHRASES):
    """the phrase ban as vLLM will apply it: SamplingParams(bad_words).update_from_tokenizer on vLLM's own
    tokenizer (t.llm.get_tokenizer(), or vllm.tokenizers.registry.get_tokenizer on CPU). Raises when the token
    sequences exceed vLLM's caps; returns the rule, the counts and the phrases whose space-prefixed form vLLM drops
    (it keeps that form only when it has as many tokens as the bare one: those are not banned after a space)."""
    from vllm import SamplingParams
    sp = SamplingParams(bad_words=list(words))
    sp.update_from_tokenizer(vtok)
    seqs = sp.bad_words_token_ids or []
    n_tok = sum(len(x) for x in seqs)
    if len(seqs) > decode.BAD_WORDS_CAP or n_tok > decode.BAD_TOKENS_CAP:
        raise ValueError(f"phrase ban: {len(seqs)} sequences / {n_tok} tokens over vLLM's caps")
    no_space = []
    for w in words:
        a = vtok.encode(text=w.lstrip(), add_special_tokens=False)
        b = vtok.encode(text=" " + w.lstrip(), add_special_tokens=False)
        if not (b and a and b[0] != a[0] and len(a) == len(b)):
            no_space.append(w)
    return {"rule": decode.PHRASE_RULE, "n_phrases": len(words), "n_seqs": len(seqs), "n_tokens": n_tok,
            "sha256": decode.sha("\n".join(words)), "no_space_form": no_space, "seqs": [list(x) for x in seqs]}


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
    for name, v in (("seeds", seeds), ("max_tokens", max_tokens)):
        if v is not None and len(v) != len(prompts):
            raise ValueError(f"{name} has {len(v)} entries for {len(prompts)} prompts")
    todo, out = [], [None] * len(prompts)
    for k, p in enumerate(prompts):
        s = sampling if max_tokens is None else {**(sampling or {}), "max_tokens": max_tokens[k]}
        try:
            ids, sp, clamped = prepare(t, p, s, None if seeds is None else seeds[k])
        except RenderError as e:
            out[k] = {"text": "", "raw": "", "finish": "render_error", "error": str(e), "n_prompt": 0, "n_out": 0,
                      "thought": [], "clamped": False}
            continue
        todo.append((k, TokensPrompt(prompt_token_ids=ids), sp, clamped, len(ids)))
    if todo:
        res = t.llm.generate([x[1] for x in todo], [x[2] for x in todo], use_tqdm=use_tqdm)
        if len(res) != len(todo):
            raise RuntimeError(f"vLLM returned {len(res)} outputs for {len(todo)} requests")
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
    hits = re.findall(r"KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) tokens per request: "
                      r"([\d.]+)x", text)
    if not hits:
        return None
    n, m, x = hits[-1]
    return {"kv_tokens": int(n.replace(",", "")), "per_request_tokens": int(m.replace(",", "")),
            "max_concurrency": float(x)}


LOG_FACTS = {"model_load_gib": r"Model loading took ([\d.]+) GiB",
             "kv_cache_gib": r"Available KV cache memory: (-?[\d.]+) GiB",      # negative: no room for KV at all
             "cuda_graph_gib": r"CUDA graph pool memory: ([\d.]+) GiB \(actual\)"}


def log_facts(text):
    """vLLM's own load facts from the part of a log written by ONE load: the KV line above, the last model, KV and
    CUDA graph GiB figures, the attention backends chosen and any line that names fp8."""
    out = kv_from_log(text) or {}
    for k, rx in LOG_FACTS.items():
        hits = re.findall(rx, text)
        if hits:
            out[k] = float(hits[-1])
    out["attention_backends"] = sorted(set(re.findall(r"Using (\w+) attention backend out of", text))
                                       | set(re.findall(r"Using AttentionBackendEnum\.(\w+) backend\.", text)))
    out["fp8_lines"] = sorted({ln.split("]", 1)[-1].strip()[:160] for ln in text.splitlines() if "fp8" in ln.lower()
                               and ("WARN" in ln or "INFO" in ln) and "non-default args" not in ln
                               and "Initializing a V1 LLM engine" not in ln})[:8]
    return out


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
