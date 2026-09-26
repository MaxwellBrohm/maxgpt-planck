"""RC-12 vLLM responder (draft s4 engines; notes STEP 9): the same interface as hf_responder.HFResponder (start,
count, reply) plus reply_batch for lockstep.py, backed by vLLM offline (LLM.generate with token-id prompts).

UNTESTED ON A REAL MODEL. VLLM_TESTED = False: runner.py refuses it without --vllm-untested-ok. It is set True only
in a logged step after parity_hf_vllm.py has passed on a real model. Tests without a model: test_vllm_responder.py
(a stub LLM, a stub SamplingParams and a toy tokenizer).
Importing this module imports nothing heavy; transformers and vllm load in VLLMResponder.__init__ only.

Prompts: the HF path's own code. prompt / encode / count ARE HFResponder's functions (the template render via
render.template with add_generation_prompt, no system prompt; the plain render via render.plain, tokenized WITH
the tokenizer's BOS), and the ids go to vLLM as {"prompt_token_ids": ids}, so vLLM never re-tokenizes a prompt.
enable_thinking=False: HF's rule (config model_type starts with "qwen3") OR the chat template mentions
enable_thinking; self.thinking_rule_differs is True when only the second part fired (then HF renders the prompt
differently and parity_hf_vllm.py reports a prompt mismatch).
Decoding (hf_responder.DECODE): greedy = temperature 0; sampling = temperature 0.6, top-p 1.0, top-k off (-1),
repetition penalty 1.0, max_tokens 256, seed = hf_responder.conv_seed(seed, conversation id, turn) per request.
LLM(generation_config="vllm") so the model's generation_config.json sets neither defaults nor extra stop ids.
Stops: the tokenizer's eos (vLLM's own eos check) plus stop_token_ids = HF's stop list ([eos] + the end-of-turn
tokens of hf_responder.EOT_TOKENS in the vocabulary, minus eos). Stop reason as in HF: "cap" when vLLM reports
finish "length"; on finish "stop" the stop token (vLLM stop_reason, or eos when that is None) is "eot" if it is an
end-of-turn token, else "eos"; any other finish reason raises. The ids are normalized to HF's convention (the
stop token last; vLLM may or may not include it) and decoded with the SAME tokenizer, skip_special_tokens=True.
The plain render's role cut is the runner's (render.cut_plain), exactly as on the HF path; no stop strings.
Context: ctx = max_model_len = min(native max_position_embeddings, 32768) unless given; the runner fits the history
to ctx - 256 (the HF path uses the native length; the dev set never comes near 32k).
batch_invariant (--batch-invariant): VLLM_BATCH_INVARIANT=1 before vllm loads, so a reply does not depend on which
other requests share its batch (a per-request seed alone does not promise that); off by default, cost unmeasured.
Sampler: VLLM_USE_FLASHINFER_SAMPLER=0 (set unless the caller set it) before vllm loads: vLLM's FlashInfer top-k /
top-p sampler JIT-compiles with nvcc at engine start, and the PC's WSL has no CUDA toolkit (the first parity run died
there, notes STEP 9 parity); the PyTorch sampler is used instead (top-k off and top-p 1.0 skip both anyway).
trace: set to a list to record every reply (rid, turn, seed, history, prompt_ids, ids, stop, finish, stop_reason,
vllm_text); parity_hf_vllm.py reads it."""
import os

import hf_responder as HR

VLLM_TESTED = False
TOP_K_OFF = -1
GPU_MEM = 0.85
MAX_LEN_CAP = 32768
assert HR.DECODE["top_k"] == 0, "HF top-k is no longer off: TOP_K_OFF would not match"


def sampling_kwargs(seed, rid, i, stop_ids):
    kw = dict(top_p=HR.DECODE["top_p"], top_k=TOP_K_OFF, repetition_penalty=HR.DECODE["repetition_penalty"],
              max_tokens=HR.DECODE["max_new_tokens"], stop_token_ids=list(stop_ids), skip_special_tokens=True)
    if seed is None:
        kw["temperature"] = 0.0
    else:
        kw.update(temperature=HR.DECODE["temperature"], seed=HR.conv_seed(seed, rid, i))
    return kw


def native_len(cfg):
    for c in (cfg, getattr(cfg, "text_config", None)):
        n = getattr(c, "max_position_embeddings", None) if c is not None else None
        if n:
            return int(n)
    return None


class VLLMResponder:
    prompt = HR.HFResponder.prompt
    encode = HR.HFResponder.encode
    count = HR.HFResponder.count

    def __init__(self, model_id, render="template", dtype="bfloat16", gpu_memory_utilization=GPU_MEM,
                 max_model_len=None, batch_invariant=False, trust_remote_code=False,
                 tokenizer=None, model_type=None, native=None, llm=None, sampling_params=None):
        """tokenizer / model_type / native / llm / sampling_params: injected by the tests (no transformers, no
        vllm); left None, they are loaded from model_id."""
        if tokenizer is None:
            from transformers import AutoConfig, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
            model_type, native = getattr(cfg, "model_type", ""), native_len(cfg)
        self.model_id, self.tok, self.render = model_id, tokenizer, render
        self.has_template = bool(getattr(tokenizer, "chat_template", None))
        if render == "template" and not self.has_template:
            raise ValueError(f"{model_id} ships no chat template: use --render plain")
        hf_rule = str(model_type or "").startswith("qwen3")
        takes = "enable_thinking" in str(getattr(tokenizer, "chat_template", "") or "")
        self.qwen3, self.thinking_rule_differs = hf_rule or takes, takes and not hf_rule
        vocab = tokenizer.get_vocab()
        self.eos = tokenizer.eos_token_id
        self.eot = sorted({vocab[t] for t in HR.EOT_TOKENS if t in vocab} - {self.eos})
        self.stop_ids = ([self.eos] if self.eos is not None else []) + self.eot
        self.ctx = int(max_model_len) if max_model_len else min(native or MAX_LEN_CAP, MAX_LEN_CAP)
        self.batch_invariant, self.batches, self.trace = batch_invariant, [], None
        self.rec, self.seed = None, None
        if llm is None:
            if batch_invariant:
                os.environ["VLLM_BATCH_INVARIANT"] = "1"
            os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")   # its sampler JIT-builds with nvcc (none in WSL)
            from vllm import LLM, SamplingParams
            llm = LLM(model=model_id, tokenizer=model_id, dtype=dtype, gpu_memory_utilization=gpu_memory_utilization,
                      max_model_len=self.ctx, seed=0, generation_config="vllm", trust_remote_code=trust_remote_code)
            sampling_params = SamplingParams
        self.llm, self.SP = llm, sampling_params

    def audit(self):
        """the run config for meta.json (verifier 2026-09-26): what this responder passes, and what vLLM adds on top.
        vLLM 0.30 still reads generation_config.json with generation_config="vllm": its sampling defaults are off
        (get_diff_sampling_param -> {}), but every request gets the file's eos_token_id list as extra stop ids
        (update_from_generation_config). vllm_stop_ids is that final list; gen_config_stops_not_in_hf are ids vLLM
        stops on that HF's list lacks (stop_of then raises: e.g. gemma-3's <eos>, id 1)."""
        out = dict(stop_ids=self.stop_ids, eos=self.eos, eot=self.eot, qwen3=self.qwen3,
                   thinking_rule_differs=self.thinking_rule_differs, ctx=self.ctx, batch_invariant=self.batch_invariant,
                   greedy=sampling_kwargs(None, "x", 1, self.stop_ids),
                   sampled={k: v for k, v in sampling_kwargs(0, "x", 1, self.stop_ids).items() if k != "seed"})
        try:
            mc = self.llm.llm_engine.model_config
            gc = mc.try_get_generation_config() or {}
            out["vllm_default_sampling"] = mc.get_diff_sampling_param()
            ge = gc.get("eos_token_id")
            ge = sorted({ge} if isinstance(ge, int) else set(ge or []))
            out["gen_config_eos"] = ge
            out["gen_config_stops_not_in_hf"] = [i for i in ge if i not in self.stop_ids]
            sp = self.SP(**sampling_kwargs(None, "x", 1, self.stop_ids))
            sp.update_from_generation_config(gc, self.eos)
            out["vllm_stop_ids"] = sorted(set(sp.stop_token_ids or []) | ({self.eos} if self.eos is not None else set()))
        except Exception as e:  # noqa: BLE001  (a stub or another vLLM version: recorded, never fatal)
            out["vllm_config_error"] = f"{type(e).__name__}: {e}"[:200]
        return out

    # sequential interface (runner.play)
    def start(self, rec, seed=None, render=None):
        self.rec, self.seed = rec, seed
        if render and render != self.render:
            raise ValueError("render changed after loading")

    def reply(self, history, i):
        return self.reply_batch([(0, self.rec, self.seed, history, i)])[0]

    # batch interface (lockstep.py)
    def stop_of(self, ids, finish, stop_reason):
        """(ids in HF's convention, stop reason) for one vLLM completion."""
        ids = list(ids)
        if finish == "length":
            return ids, "cap"
        if finish != "stop":
            raise ValueError(f"vLLM finish reason {finish!r} (expected stop or length)")
        if stop_reason is not None and not isinstance(stop_reason, int):
            raise ValueError(f"vLLM stopped on a string {stop_reason!r}; this responder passes no stop strings")
        tok = self.eos if stop_reason is None else stop_reason
        if tok is None or tok not in self.stop_ids:
            raise ValueError(f"vLLM stopped on {stop_reason!r}, not a stop id of this run")
        if not ids or ids[-1] != tok:
            ids.append(tok)
        return ids, "eot" if tok in self.eot else "eos"

    def reply_batch(self, reqs):
        prompts, params, meta = [], [], []
        for _, rec, seed, history, i in reqs:
            ids = self.encode(history)
            prompts.append({"prompt_token_ids": ids})
            params.append(self.SP(**sampling_kwargs(seed, rec["id"], i, self.stop_ids)))
            meta.append((rec["id"], i, seed, history, ids))
        self.batches.append(len(reqs))
        outs = self.llm.generate(prompts, params, use_tqdm=False)
        if len(outs) != len(reqs):
            raise RuntimeError(f"vLLM returned {len(outs)} outputs for {len(reqs)} prompts")
        res = []
        for (rid, i, seed, history, pids), o in zip(meta, outs):
            c = o.outputs[0]
            ids, stop = self.stop_of(c.token_ids, c.finish_reason, c.stop_reason)
            text = self.tok.decode(ids, skip_special_tokens=True)
            res.append((text, stop))
            if self.trace is not None:
                self.trace.append(dict(rid=rid, turn=i, seed=seed, history=[dict(m) for m in history],
                                       prompt_ids=list(pids), ids=ids, text=text, stop=stop, finish=c.finish_reason,
                                       stop_reason=c.stop_reason, vllm_text=getattr(c, "text", None)))
        return res
