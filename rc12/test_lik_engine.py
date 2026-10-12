"""Likelihood rows: engines.json options (rc12/notes.txt STEP 14S; the self-check moved to test_lik_selfcheck.py at the
STEP 14S review, D5). python3, no torch: fake transformers / torch modules and stub scorers only, no model and no RC-12
row text beyond the dev records.
  python3 -B test_lik_engine.py      (mutants: mutation_lik.py, the STEP 14S block)"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import types

import lik_adapters as LA
import lik_run as LU
import lik_stubs as ST
import lik_tokcheck as LT

FAILS = []
ENG = {"models": {
    "d/Doge": {"engine": "hfb", "trust_remote_code": True, "attn_implementation": "eager", "max_batch": 32},
    "l/Loom": {"engine": "vllm", "chat_template": "{% for m in messages %}<{{ m['role'] }}>{% endfor %}<loom>"},
    "q/Qwen": {"engine": "vllm", "dtype": "bfloat16"}}}


def check(ok, msg):
    if not ok:
        FAILS.append(msg)
        print("FAIL", msg)


class FakeTok:
    """word-level: one id per word (stable hash), a chat template only if one is set."""

    def __init__(self, template=None):
        self.chat_template, self.pad_token_id, self.eos_token_id = template, 0, None

    def __call__(self, text, add_special_tokens=False):
        ids = [sum(map(ord, w)) % 997 + 2 for w in text.split()]
        return types.SimpleNamespace(input_ids=([1] if add_special_tokens else []) + ids)

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, enable_thinking=True):
        if not self.chat_template:
            raise ValueError("Cannot use chat template functions because tokenizer.chat_template is not set")
        return "".join(f"<{m['role']}> {m['content']} " for m in messages) + "<a> "

    def decode(self, ids):
        return " ".join(map(str, ids))


def fake_modules(calls, shipped_template=None):
    ns = types.SimpleNamespace

    class Model:
        def __init__(self, kw):
            self.config = ns(model_type="doge", max_position_embeddings=64, vocab_size=500,
                             _attn_implementation=kw.get("attn_implementation", "sdpa"))
            self.to, self.eval = (lambda device: self), (lambda: self)

    def config(m, **kw):
        calls.append(("config", m, kw))
        return ns(model_type="doge")

    def tok(m, **kw):
        calls.append(("tok", m, kw))
        return FakeTok(shipped_template)

    def model(m, **kw):
        calls.append(("model", m, kw))
        return Model(kw)
    tf, th = types.ModuleType("transformers"), types.ModuleType("torch")
    tf.AutoConfig, tf.AutoTokenizer = ns(from_pretrained=config), ns(from_pretrained=tok)
    tf.AutoModelForCausalLM = ns(from_pretrained=model)
    th.float32, th.bfloat16 = "F32", "BF16"
    return dict(transformers=tf, torch=th)


@contextlib.contextmanager
def faked(calls, shipped_template=None):
    saved = {k: sys.modules.get(k) for k in ("transformers", "torch")}
    sys.modules.update(fake_modules(calls, shipped_template))
    try:
        yield
    finally:
        for k, v in saved.items():
            sys.modules.pop(k, None) if v is None else sys.modules.__setitem__(k, v)


def test_engine_opts():
    check(LA.engine_opts("d/Doge", ENG) == dict(trust_remote_code=True, attn_implementation="eager"),
          f"Doge opts {LA.engine_opts('d/Doge', ENG)}")
    check(set(LA.engine_opts("l/Loom", ENG)) == {"chat_template"}, "Loom opts")
    check(LA.engine_opts("q/Qwen", ENG) == {} and LA.engine_opts("x/none", ENG) == {}, "plain models get no opts")
    real = LA.engine_opts("SmallDoge/Doge-160M-Instruct", os.path.join(LU.HERE, "engines.json"))
    check(real == dict(trust_remote_code=True, attn_implementation="eager"), f"engines.json Doge {real}")


def test_hflik_load():
    calls = []
    with faked(calls):
        sc, tok = LA.HFLik.load("d/Doge", "float32", "cpu", "eager", True, ENG["models"]["l/Loom"]["chat_template"])
        check([c[2].get("trust_remote_code") for c in calls] == [True, True, True], f"trust not everywhere {calls}")
        check(calls[-1][2].get("attn_implementation") == "eager", f"attention {calls[-1]}")
        check(tok.tok.chat_template == ENG["models"]["l/Loom"]["chat_template"], "chat_template not set")
        check(sc.info["trust_remote_code"] is True and sc.info["chat_template_set"] is True, f"info {sc.info}")
        check(sc.vocab == 500 and sc.budget == LA.LOGIT_BUDGET, f"logit budget not wired: {sc.vocab} {sc.budget}")
        del calls[:]
        sc, tok = LA.HFLik.load("q/Qwen", "float32", "cpu")
        check(all("trust_remote_code" not in c[2] for c in calls), f"trust passed for a plain model {calls}")
        check(tok.tok.chat_template is None and sc.info["trust_remote_code"] is False, "plain model changed")


def run_hf(engines, extra=()):
    seen, out = {}, tempfile.mkdtemp(prefix="rc12_like_")

    class Scorer:
        ctx, info = None, dict(model_type="fake")

        def __call__(self, pre, cont):
            return 0.0

    def load(*a):
        seen["args"] = a
        return Scorer(), ST.make_tok("word")
    old = LA.HFLik.load
    LA.HFLik.load = load
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            code = LU.main(["--responder", "hf:d/Doge", "--untested-ok", "--render", "plain", "--families", "BIND",
                            "--engines", engines, "--out", out] + list(extra))
    finally:
        LA.HFLik.load = old
        shutil.rmtree(out, ignore_errors=True)
    return code, seen.get("args")


def test_lik_run_opts():
    d = tempfile.mkdtemp(prefix="rc12_lik_eng_")
    path = os.path.join(d, "engines.json")
    json.dump(ENG, open(path, "w"))
    try:
        code, a = run_hf(path)
        check(code == 0 and a[3] == "eager" and a[4] is True and a[5] is None, f"lik_run Doge args {a}")
        code, a = run_hf(path, ["--attn", "eager"])
        check(code == 0 and a[3] == "eager", f"an --attn equal to the entry's {a}")
        try:
            run_hf(path, ["--attn", "sdpa"])
            check(False, "--attn sdpa against the entry's eager not refused")
        except SystemExit as e:
            check("refused" in str(e), f"refusal text {e}")
        json.dump({"models": {}}, open(path, "w"))
        code, a = run_hf(path)
        check(code == 0 and a[3] is None and a[4] is False and a[5] is None, f"no entry {a}")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_tokcheck_opts():
    rows = [r for r in LT.LR.build(LT.LR.load())[0] if r["key"] == "RECALL"][:6]
    calls = []
    with faked(calls):
        mt, sc, skipped, used = LT.score("d/Doge", rows, ["template", "plain"], ENG)
        check(set(sc) == {"plain"} and skipped == ["template"], f"no template: {list(sc)} {skipped}")
        check(all(c[2].get("trust_remote_code") is True for c in calls) and len(calls) == 2, f"tokcheck trust {calls}")
        check(used == ["attn_implementation", "trust_remote_code"], f"opts used {used}")
        del calls[:]
        mt, sc, skipped, used = LT.score("l/Loom", rows, ["template", "plain"], ENG)
        check(set(sc) == {"template", "plain"} and skipped == [] and used == ["chat_template"], f"Loom {skipped}")
        check(all("trust_remote_code" not in c[2] for c in calls), f"trust for Loom {calls}")
        try:
            LT.score("q/Qwen", rows, ["template"], ENG)
            check(False, "a template-only check of a model without a template not refused")
        except ValueError:
            pass


def main():
    for t in (test_engine_opts, test_hflik_load, test_lik_run_opts, test_tokcheck_opts):
        try:
            t()
        except Exception as e:                      # reported as the test's FAIL, never a silent pass
            check(False, f"{t.__name__} raised {type(e).__name__}: {str(e)[:200]}")
    check("torch" not in sys.modules and "transformers" not in sys.modules, "torch or transformers left imported")
    print(f"test_lik_engine: {len(FAILS)} failures")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
