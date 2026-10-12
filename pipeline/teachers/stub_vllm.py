"""A stand-in for vLLM, torch and a teacher tokenizer: CPU only, no model anywhere, for the stub tests of serve.py,
thinking_control.py and capacity.py on any machine (the Mac included).

FakeTok   the Gemma 4 / Qwen special strings get their real ids; every word, space run or punctuation mark is one
          id (so a FAKE render is a few hundred ids, as with the real tokenizers). apply_chat_template renders the Gemma 4 or Qwen template, thinking on or off, like the real ones
          end (Gemma off: '<|turn>model\\n<|channel>thought\\n<channel|>'; Qwen off: '<think>\\n\\n</think>\\n\\n').
FakeLLM   records its kwargs and every generate call; answers like a model whose thinking follows the prompt:
          a Gemma prompt ending '<|turn>model\\n' opens a thought ('<|channel>', id 100), a Qwen prompt ending
          'assistant\\n' opens '<think>', one ending '<think>\\n' closes it ('</think>'); anything else is plain text.
          ignore_eos requests get exactly max_tokens ids. With LOG set, construction appends vLLM-style load lines
          to that file, as the engine core does. FAIL_FP8 makes a kv_cache_dtype='fp8' load raise; NVCC_FP8
          makes it raise with vLLM's missing-nvcc line unless attention_backend is TRITON_ATTN.
install() patch.dict of sys.modules with the fake `vllm`, `vllm.inputs`, `vllm.sampling_params` (StructuredOutputsParams),
          `vllm.config` (StructuredOutputsConfig) and `torch` modules.
install_mistral() the same for a fake `mistral_common` (FakeMistralTok: '<s>[INST]' + prompt + '[/INST]' as ids, a
          .text that raises when read, decode(ids, special_token_policy=KEEP), byte pieces for the dash ban)."""
import re
import sys
import types
import zlib
from unittest import mock

SPECIAL = {"<bos>": 2, "<|turn>": 105, "<turn|>": 106, "<|channel>": 100, "<channel|>": 101, "<|think|>": 98,
           "<|im_start|>": 248045, "<|im_end|>": 248046, "<think>": 248068, "</think>": 248069}
REV = {v: k for k, v in SPECIAL.items()}
SPLIT = re.compile("(" + "|".join(re.escape(s) for s in sorted(SPECIAL, key=len, reverse=True)) + ")")


WORDS = {}                      # word / space run / punctuation piece -> id, assigned on first sight
PIECE = re.compile(r"\s+|\w+|[^\w\s]")


def encode(text):
    ids = []
    for part in SPLIT.split(text):
        if part in SPECIAL:
            ids.append(SPECIAL[part])
            continue
        for w in PIECE.findall(part):
            if w not in WORDS:
                WORDS[w] = 300000 + len(WORDS)
                REV[WORDS[w]] = w
            ids.append(WORDS[w])
    return ids


def decode(ids):
    return "".join(REV[i] for i in ids)


class FakeTok:
    unk_token_id = 3

    def __init__(self, family):
        self.family = family

    def encode(self, text, add_special_tokens=False):
        return encode(text)

    def convert_tokens_to_ids(self, tok):
        return SPECIAL.get(tok, self.unk_token_id)

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        p = msgs[-1]["content"]
        if self.family == "qwen":
            return ("<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n"
                    + ("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"))
        return ("<bos>" + ("<|turn>system\n<|think|><turn|>\n" if enable_thinking else "") + "<|turn>user\n" + p
                + "<turn|>\n<|turn>model\n" + ("" if enable_thinking else "<|channel>thought\n<channel|>"))


def tokenizer(name, models=None):
    return FakeTok("qwen" if name.startswith("qwen") else "gemma")


class SamplingParams:
    def __init__(self, **kw):
        self.kw = kw
        for k, v in kw.items():
            setattr(self, k, v)


class StructuredOutputsParams:
    def __init__(self, regex=None, **kw):
        self.regex, self.kw = regex, kw


class StructuredOutputsConfig:
    def __init__(self, backend="auto", **kw):
        self.backend, self.kw = backend, kw


def TokensPrompt(prompt_token_ids):
    return {"prompt_token_ids": list(prompt_token_ids)}


def answer(prompt_text, sp):
    if prompt_text.endswith("<|turn>model\n"):
        text = "<|channel>thought\nLet me see.<channel|>391 = 17 x 23."
    elif prompt_text.endswith("assistant\n"):
        text = "<think>\nHmm"
    elif prompt_text.endswith("<think>\n"):
        text = "It is 391.\n</think>\n\n391."
    else:
        text = f"U1: fine words here\nA1: more fine words {zlib.crc32(prompt_text.encode()) % 997}\nEND<turn|>"
    ids = encode(text)
    if sp.kw.get("ignore_eos"):
        ids = (ids * (sp.max_tokens // len(ids) + 1))[:sp.max_tokens]
        return decode(ids), ids, "length"
    ids = ids[:sp.max_tokens]
    return decode(ids), ids, "stop"


class FakeLLM:
    LOG, FAIL_FP8, NVCC_FP8, instances = None, False, False, []

    def __init__(self, model, **kw):
        fp8 = kw.get("kv_cache_dtype") == "fp8"
        if self.LOG:
            with open(self.LOG, "a") as f:
                f.write(f"(EngineCore pid=1) INFO [model_runner.py:428] Model loading took 5.99 GiB memory\n"
                        f"(EngineCore pid=1) INFO [cuda.py:538] Using {'TRITON_ATTN' if fp8 else 'FLASH_ATTN'} "
                        f"attention backend out of potential backends: ['FLASH_ATTN', 'TRITON_ATTN']\n")
                if fp8:
                    f.write("(EngineCore pid=1) WARNING [kv_cache.py:1] Using fp8 KV cache without scales\n")
        if fp8 and self.FAIL_FP8:
            raise RuntimeError("Engine core initialization failed (stub fp8 refusal)")
        if fp8 and self.NVCC_FP8 and kw.get("attention_backend") != "TRITON_ATTN":
            with open(self.LOG, "a") as f:
                f.write("(EngineCore pid=1) ERROR [core.py:1366] RuntimeError: Could not find nvcc\n")
            raise RuntimeError("Engine core initialization failed. See root cause above.")
        tokens = 43616 if fp8 else 21808
        if self.LOG:
            with open(self.LOG, "a") as f:
                f.write(f"(EngineCore pid=1) INFO [gpu_worker.py:640] Available KV cache memory: 2.83 GiB\n"
                        f"(EngineCore pid=1) INFO [kv_cache_utils.py:2395] GPU KV cache size: {tokens:,} tokens, "
                        f"Maximum concurrency for 2,048 tokens per request: {tokens / 2048:.2f}x\n")
        self.model, self.kw, self.calls, self.shutdowns, self.running = model, kw, [], 0, 0
        cache = types.SimpleNamespace(num_gpu_blocks=tokens // 16, block_size=16,
                                      cache_dtype=kw.get("kv_cache_dtype", "auto"))
        sched = types.SimpleNamespace(max_num_seqs=kw.get("max_num_seqs", 256),
                                      max_num_batched_tokens=kw.get("max_num_batched_tokens", 8192))
        self.llm_engine = types.SimpleNamespace(engine_core=types.SimpleNamespace(shutdown=self._shutdown),
                                                vllm_config=types.SimpleNamespace(cache_config=cache,
                                                                                  scheduler_config=sched))
        FakeLLM.instances.append(self)

    def _shutdown(self):
        self.shutdowns += 1

    def generate(self, prompts, params, use_tqdm=False):
        self.calls.append((prompts, params))
        self.running = len(prompts)
        out = []
        for i, (p, sp) in enumerate(zip(prompts, params)):
            text, ids, fin = answer(decode(p["prompt_token_ids"]), sp)
            c = types.SimpleNamespace(text=text, token_ids=ids, finish_reason=fin)
            out.append(types.SimpleNamespace(request_id=str(i), finished=True, outputs=[c]))
        return out

    def get_metrics(self):
        return [types.SimpleNamespace(name="vllm:num_requests_running", value=self.running),
                types.SimpleNamespace(name="vllm:kv_cache_usage_perc", value=0.5)]


def install():
    v, vi, tm = types.ModuleType("vllm"), types.ModuleType("vllm.inputs"), types.ModuleType("torch")
    vsp, vc = types.ModuleType("vllm.sampling_params"), types.ModuleType("vllm.config")
    v.LLM, v.SamplingParams, v.__version__, v.inputs = FakeLLM, SamplingParams, "stub", vi
    vi.TokensPrompt = TokensPrompt
    vsp.SamplingParams, vsp.StructuredOutputsParams, vc.StructuredOutputsConfig = (SamplingParams,
                                                                                   StructuredOutputsParams,
                                                                                   StructuredOutputsConfig)
    tm.__version__, tm.cuda = "stub", types.SimpleNamespace(is_initialized=lambda: False, empty_cache=lambda: None)
    FakeLLM.instances, FakeLLM.LOG, FakeLLM.FAIL_FP8, FakeLLM.NVCC_FP8 = [], None, False, False
    return mock.patch.dict(sys.modules, {"vllm": v, "vllm.inputs": vi, "vllm.sampling_params": vsp, "vllm.config": vc,
                                         "torch": tm})


MISTRAL_PIECES = [b"<unk>", b"<s>", b"</s>", b"[INST]", b"[/INST]", b" the", b"\xe2\x80\x94", b"well", b"-", b" -",
                  b"--", b" \xe2\x80\x93 x", b"\xe2\x80", b"\xe2\x80\x99", b"a-b", b" -\xe2\x88\x92"]


class FakeTokenized:
    def __init__(self, ids):
        self.tokens = ids

    @property
    def text(self):
        raise AssertionError("Tokenized.text read (deprecated in mistral_common 1.12, gone in 1.13)")


class FakeMistralTok:
    """encodes '<s>[INST]' + content + '[/INST]' (content words get stub ids) and decodes it back."""
    def __init__(self):
        inner = types.SimpleNamespace(n_words=len(MISTRAL_PIECES), id_to_byte_piece=self.piece)
        self.instruct_tokenizer = types.SimpleNamespace(tokenizer=inner)

    def piece(self, i, policy):
        return MISTRAL_PIECES[i]

    def encode_chat_completion(self, req):
        return FakeTokenized([1, 3] + encode(req.messages[-1].content) + [4])

    def decode(self, ids, special_token_policy=None):
        assert special_token_policy == "KEEP", special_token_policy
        return "".join({1: "<s>", 3: "[INST]", 4: "[/INST]"}.get(i) or REV[i] for i in ids)


def install_mistral():
    names = ["mistral_common", "mistral_common.protocol", "mistral_common.protocol.instruct",
             "mistral_common.protocol.instruct.messages", "mistral_common.protocol.instruct.request",
             "mistral_common.tokens", "mistral_common.tokens.tokenizers", "mistral_common.tokens.tokenizers.base",
             "mistral_common.tokens.tokenizers.mistral"]
    mods = {n: types.ModuleType(n) for n in names}
    mods["mistral_common.protocol.instruct.messages"].UserMessage = lambda content: types.SimpleNamespace(
        content=content)
    mods["mistral_common.protocol.instruct.request"].ChatCompletionRequest = lambda messages: types.SimpleNamespace(
        messages=messages)
    mods["mistral_common.tokens.tokenizers.base"].SpecialTokenPolicy = types.SimpleNamespace(KEEP="KEEP")
    mods["mistral_common.tokens.tokenizers.mistral"].MistralTokenizer = types.SimpleNamespace(
        from_file=lambda path: FakeMistralTok())
    return mock.patch.dict(sys.modules, mods)
