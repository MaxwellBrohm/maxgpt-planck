"""RC-12 vllm_responder tests without a model (notes STEP 9): a toy word tokenizer, a stub SamplingParams that
refuses unknown fields, and a stub LLM whose completion is a deterministic function of (prompt ids, seed).
  light      importing vllm_responder imports no torch, vllm or transformers
  params     one generate call per reply_batch, use_tqdm False, token-id prompts; per request: greedy = temperature
             0 and no seed; sampling = T 0.6 with seed conv_seed(seed, id, turn); top-p 1.0, top-k -1, repetition
             1.0, max_tokens 256, stop ids [eos] + end-of-turn ids, skip_special_tokens
  prompts    prompt ids equal HFResponder.encode's (template: no BOS; plain: with BOS)
  thinking   enable_thinking=False for model_type qwen3*, and for a template that takes it (flagged as differing
             from the HF rule); never passed otherwise
  stops      stop_of: eos / eot / cap, the stop token appended when vLLM leaves it out, HF's stop formula agrees,
             odd finish reasons, string and foreign stop reasons raise; special tokens never reach the text
  ctx        min(native, 32768), or --max-model-len
  lockstep   runner.run with and without --lockstep give identical rows (template seeds greedy + 1, plain with a
             role-tag word, --own-cf on OWN); one generate call per turn position
  refuse     runner refuses vllm: without --vllm-untested-ok; a template-less model refuses the template render
Run: python3 -B test_vllm_responder.py   (exit 1 on any failure)"""
import argparse
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import hf_responder as HR
import lockstep as LS
import runner as R
import vllm_responder as VR

SPECIAL = {"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2, "<s>": 3}
WORDS = ["alpha", "bravo", "Tuesday", "green", "Okay,", "noted.", "the", "list:", "1.", "2.", "3.", "\nUser:", "\n"]


def wid(w):
    return SPECIAL.get(w, 10 + int(hashlib.md5(w.encode()).hexdigest()[:6], 16))


class ToyTok:
    def __init__(self, template="chatml", eos="<|endoftext|>"):
        self.chat_template, self.eos_token_id, self.pad_token_id = template, SPECIAL[eos], None
        self.inv, self.calls = {v: k for k, v in SPECIAL.items()}, []
        for w in WORDS:
            self.inv[wid(w)] = w

    def get_vocab(self):
        return dict(SPECIAL)

    def apply_chat_template(self, messages, **kw):
        self.calls.append(kw)
        assert kw.get("add_generation_prompt") is True and kw.get("tokenize") is False, kw
        s = "".join(f"<|im_start|> {m['role']} {m['content']} <|im_end|> " for m in messages)
        return s + "<|im_start|> assistant" + (" <think> </think>" if kw.get("enable_thinking") is False else "")

    def __call__(self, text, add_special_tokens=True):
        ids = [wid(w) for w in text.split()]
        for i, w in zip(ids, text.split()):
            self.inv[i] = w
        return SimpleNamespace(input_ids=([3] if add_special_tokens else []) + ids)

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(self.inv[i] for i in ids if not (skip_special_tokens and i in SPECIAL.values()))


class StubSP:
    FIELDS = {"temperature", "top_p", "top_k", "repetition_penalty", "max_tokens", "seed", "stop_token_ids",
              "skip_special_tokens"}

    def __init__(self, **kw):
        if set(kw) - self.FIELDS:
            raise TypeError(f"unknown SamplingParams fields {set(kw) - self.FIELDS}")
        self.seed = None
        self.__dict__.update(kw)


class StubLLM:
    def __init__(self, eot_mode="include"):
        self.calls, self.eot_mode = [], eot_mode

    def generate(self, prompts, params, use_tqdm=True):
        self.calls.append(dict(n=len(prompts), use_tqdm=use_tqdm, params=params, prompts=prompts))
        return [SimpleNamespace(outputs=[self.one(p["prompt_token_ids"], sp)]) for p, sp in zip(prompts, params)]

    def one(self, ids, sp):
        h = int(hashlib.md5(json.dumps([ids, sp.seed, sp.temperature]).encode()).hexdigest(), 16)
        toks = [wid(WORDS[(h >> (3 * j)) % len(WORDS)]) for j in range(2 + h % 9)]
        kind = (h >> 60) % 4
        if kind == 0:
            return SimpleNamespace(token_ids=toks[:sp.max_tokens - 1] + [sp.stop_token_ids[0]], finish_reason="stop",
                                   stop_reason=None, text="x")
        if kind == 3:
            return SimpleNamespace(token_ids=(toks * 64)[:sp.max_tokens], finish_reason="length", stop_reason=None)
        eot = sp.stop_token_ids[-1]
        return SimpleNamespace(token_ids=toks + ([eot] if kind == 1 else []), finish_reason="stop", stop_reason=eot)


def make(template="chatml", eos="<|endoftext|>", model_type="qwen2", native=32768, render="template", **kw):
    tok = ToyTok(template, eos)
    return VR.VLLMResponder("toy", render=render, tokenizer=tok, model_type=model_type, native=native,
                            llm=StubLLM(), sampling_params=StubSP, **kw)


def hf_stop(r, ids):                        # hf_responder.HFResponder.reply's stop formula, restated
    if ids and ids[-1] in r.eot:
        return "eot"
    if len(ids) >= HR.DECODE["max_new_tokens"] and (not ids or ids[-1] not in r.stop_ids):
        return "cap"
    return "eos"


def c_light():
    code = "import sys, vllm_responder; print(sorted(m for m in ('torch','vllm','transformers') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, cwd=R.HERE).stdout
    return [] if out.strip() == "[]" else [f"light: import pulled in {out.strip()}"]


def c_params(recs):
    fails, v = [], make()
    hist = [{"role": "user", "content": recs[0]["turns"][0]["text"]}]
    reqs = [(0, recs[0], None, hist, 1), (1, recs[1], 1, hist, 1), (2, recs[1], 2, hist, 3)]
    v.reply_batch(reqs)
    call = v.llm.calls[-1]
    if len(v.llm.calls) != 1 or call["n"] != 3 or call["use_tqdm"] is not False:
        fails.append("params: not one generate call with use_tqdm False")
    if any(set(p) != {"prompt_token_ids"} for p in call["prompts"]):
        fails.append("params: prompts are not token-id prompts")
    for (_, rec, seed, _, i), sp in zip(reqs, call["params"]):
        want = dict(top_p=1.0, top_k=-1, repetition_penalty=1.0, max_tokens=256, stop_token_ids=[0, 2],
                    skip_special_tokens=True, temperature=0.0 if seed is None else 0.6,
                    seed=None if seed is None else HR.conv_seed(seed, rec["id"], i))
        got = {k: getattr(sp, k) for k in want}
        if got != want:
            fails.append(f"params: seed {seed} turn {i}: {got} != {want}")
    return fails


def c_prompts(recs):
    fails = []
    msgs = [{"role": "user", "content": recs[0]["turns"][0]["text"]}, {"role": "assistant", "content": "Okay, noted."},
            {"role": "user", "content": recs[0]["turns"][1]["text"]}]
    for render in ("template", "plain"):
        v = make(render=render)
        h = object.__new__(HR.HFResponder)
        h.tok, h.render, h.qwen3 = v.tok, render, False
        if v.encode(msgs) != h.encode(msgs) or v.count(msgs, render) != len(h.encode(msgs)):
            fails.append(f"prompts: {render} ids differ from HFResponder.encode")
        if (v.encode(msgs)[0] == 3) != (render == "plain"):
            fails.append(f"prompts: {render} BOS rule broken")
    return fails


def c_thinking(recs):
    fails, msgs = [], [{"role": "user", "content": "hi"}]
    for mt, tmpl, want, differs in (("qwen3_5", "chatml", False, False), ("qwen3", "x enable_thinking y", False, False),
                                    ("lfm2", "x enable_thinking y", False, True), ("qwen2", "chatml", None, False)):
        v = make(template=tmpl, model_type=mt)
        v.encode(msgs)
        if v.tok.calls[-1].get("enable_thinking", None) is not want or v.thinking_rule_differs != differs:
            fails.append(f"thinking: {mt} / {tmpl!r}: {v.tok.calls[-1]} differs={v.thinking_rule_differs}")
    return fails


def c_stops():
    fails = []
    for eos in ("<|endoftext|>", "<|im_end|>"):
        v = make(eos=eos)
        e, t = v.eos, (v.eot or [None])[0]
        cases = [([5, 6, e], "stop", None, [5, 6, e], "eos"), ([5, 6], "stop", None, [5, 6, e], "eos"),
                 ([5, 6, t], "stop", t, [5, 6, t], "eot"), ([5, 6], "stop", t, [5, 6, t], "eot"),
                 ([5] * 256, "length", None, [5] * 256, "cap"), ([], "stop", None, [e], "eos")]
        for ids, fin, why, want_ids, want in cases:
            got = v.stop_of(ids, fin, why)
            if got != (want_ids, want) or hf_stop(v, got[0]) != want:
                fails.append(f"stops: eos {eos} {ids[:3]} {fin} {why}: {got} (HF {hf_stop(v, got[0])}, want {want})")
        for fin, why in (("abort", None), ("stop", "\nUser:"), ("stop", 77)):
            try:
                v.stop_of([5], fin, why)
                fails.append(f"stops: {fin} / {why!r} did not raise")
            except ValueError:
                pass
    v = make()
    text, _ = v.reply_batch([(0, {"id": "x"}, None, [{"role": "user", "content": "hi"}], 1)])[0]
    if any(s in text for s in SPECIAL):
        fails.append(f"stops: special token in the decoded text {text!r}")
    return fails


def c_ctx():
    got = [make(native=n, **kw).ctx for n, kw in ((2048, {}), (131072, {}), (None, {}), (2048, {"max_model_len": 4096}))]
    return [] if got == [2048, 32768, 32768, 4096] else [f"ctx: {got}"]


def c_lockstep(recs):
    fails, few = [], recs[:60] + [r for r in recs if r["family"] == "OWN"][:40]
    for render, seeds, own_cf, data in (("template", [None, 1], False, few), ("plain", [None, 2], False, few),
                                        ("template", [3], True, [r for r in few if r["family"] == "OWN"])):
        seq = R.run(data, make(render=render), render, seeds, 32768, None, "toy", 0, own_cf)
        v = make(render=render)
        lock = R.run(data, v, render, seeds, 32768, None, "toy", 0, own_cf, lockstep=True)
        if [json.dumps(x, sort_keys=True) for x in seq] != [json.dumps(x, sort_keys=True) for x in lock]:
            fails.append(f"lockstep: {render} seeds {seeds} own_cf {own_cf}: rows differ")
        if v.batches[0] != len(data) * len(seeds) or len(v.batches) != max(len(r["turns"]) for r in data):
            fails.append(f"lockstep: batches {v.batches[:3]}... for {len(data)} x {len(seeds)} conversations")
        stops = {t["stop"] for x in lock for t in x["turns"]}
        if not {"eos", "eot", "cap"} | ({"role"} if render == "plain" else set()) <= stops:
            fails.append(f"lockstep: stub stops not all exercised: {stops}")
    return fails


def c_refuse():
    fails = []
    ns = argparse.Namespace(vllm_untested_ok=False, render="template", dtype="bfloat16", gpu_mem=0.85,
                            max_model_len=None, batch_invariant=False)
    try:
        R.make_responder("vllm:none", ns)
        fails.append("refuse: runner built a vllm responder without --vllm-untested-ok")
    except SystemExit:
        pass
    except Exception as e:  # noqa: BLE001
        fails.append(f"refuse: no refusal, runner tried to build it ({type(e).__name__})")
    try:
        make(template=None)
        fails.append("refuse: a template-less model accepted the template render")
    except ValueError:
        pass
    if VR.VLLM_TESTED:
        fails.append("refuse: VLLM_TESTED is True but no parity check has passed")
    return fails


def checks(recs, light=True):
    fails = c_light() if light else []
    for fn, a in ((c_params, (recs,)), (c_prompts, (recs,)), (c_thinking, (recs,)), (c_stops, ()), (c_ctx, ()),
                  (c_lockstep, (recs,)), (c_refuse, ())):
        try:
            fails += fn(*a)
        except Exception as e:  # noqa: BLE001  (a raising check is a failure line, not a crash)
            fails.append(f"{fn.__name__}: raised {type(e).__name__}: {str(e)[:120]}")
    return fails


def main():
    fails = checks(R.load())
    print("\n".join(f"FAIL {f}" for f in fails) or "ALL VLLM RESPONDER CHECKS PASS (stub LLM, no model)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
