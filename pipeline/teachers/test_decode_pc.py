"""Decoding controls against the real libraries, CPU only (the PC; every class skips where its library or the
teacher's tokenizer files are missing, so the Mac skips them all). No model is loaded.
  XgrammarRegex    each teacher's label regex (labels_exact, its own separator) compiles in xgrammar with the
                   TokenizerInfo vLLM builds; the canonical FAKE render (literals as parse.exact_text) is accepted,
                   form breaks and an altered literal are refused; Ministral's blank-line form is accepted; under
                   labels-v2 (2026-09-28) END inside a line is refused and no line is capped.
  RealPhraseBan    serve.phrase_ban on vLLM's own tokenizer: under vLLM's bad_words caps, every phrase registered.
  RealBan          serve.dash_ban from the real tokenizer (fresh cache dir): the probe's ban set when its file exists
                   (~/planck/runs/teachers/dry/probe_0927/ban_<teacher>.dry.json), under the 1024 cap, then cached.
  RealParams       serve._params builds vLLM's own SamplingParams with structured_outputs, logit_bias and a preset.
  MistralPin       the installed mistral_common is inside the pin; render never reads Tokenized.text (the property
                   is replaced by one that raises: mistral_common's warn_once prints once per process, so a warnings
                   filter alone would not see a second read) and its decoded text equals Tokenized.text on 30 FAKE
                   prompts.
    cd pipeline/teachers && python3 -B -m unittest test_decode_pc -v"""
import json
import os
import shutil
import sys
import tempfile
import unittest
import warnings
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import decode  # noqa: E402
import driver  # noqa: E402
import fake_engine as FE  # noqa: E402
import parse  # noqa: E402
import render_prompt as R  # noqa: E402
import serve  # noqa: E402
import skeleton  # noqa: E402

PROBE = os.path.expanduser("~/planck/runs/teachers/dry/probe_0927")
SKELS = [k for k in skeleton.shard("test-decode-pc", 34) if not R.feasible(k)][:30]


def have(mod):
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def have_tok(name):
    try:
        return bool(serve.tokenizer_files(name))
    except FileNotFoundError:
        return False


def xgr_info(name):
    """TokenizerInfo as vllm/v1/structured_output/backend_xgrammar.py builds it (the probe's construction)."""
    import xgrammar as xgr
    from vllm.tokenizers.registry import get_tokenizer
    cfg = serve.TEACHERS[name]
    if cfg.get("tokenizer_dir"):
        v = get_tokenizer(os.path.join(serve.MODELS, cfg["tokenizer_dir"]), tokenizer_mode="mistral")
        return xgr.TokenizerInfo(encoded_vocab=v.vocab, vocab_type=xgr.VocabType.RAW if v.is_tekken
                                 else xgr.VocabType.BYTE_FALLBACK, vocab_size=len(v.vocab),
                                 stop_token_ids=[v.eos_token_id], add_prefix_space=True)
    return xgr.TokenizerInfo.from_huggingface(get_tokenizer(serve.model_path(cfg)), vocab_size=cfg["vocab"])


LIBS = have("xgrammar") and have("vllm")


class XgrammarRegex(unittest.TestCase):
    def check(self, name, blank_lines_ok):
        import xgrammar as xgr
        sep = serve.TEACHERS[name].get("line_sep", "\\n")
        comp = xgr.GrammarCompiler(xgr_info(name), max_threads=4)
        n_lit = 0
        for sk in SKELS:
            ctx = comp.compile_regex(decode.label_regex(driver.constraint(sk, "labels_exact")["lines"], sep))
            raw = FE.canon(sk)
            lines = raw.split("\n")
            ex = next((t for t in sk["turns"] if t["mode"] == "exact"), None)
            last = "\n".join(lines[:-2])
            cases = {raw: True, "\n".join(lines[:-1]): False, "\n".join(lines[:1] + lines[2:]): False,
                     raw + "\nThanks!": False, raw.replace(lines[0].split(":")[0] + ":", "User:", 1): False,
                     raw.replace("\n", "\n\n"): blank_lines_ok,
                     f"{last}\n{lines[-2]} END. END.\nEND": False,                       # END inside the last line
                     f"{last}\n{lines[-2]} ENDING\nEND": False,
                     f"{last}\n{lines[-2].split(':')[0]}: {'Bye now. ' * 50}\nEND": True}             # no cap
            if ex:
                lab = ("U" if ex["role"] == "user" else "T") + str(ex["i"] + 1)
                lit = parse.exact_text(sk, ex)
                cases[raw.replace(f"{lab}: {lit}", f"{lab}: {lit} too", 1)] = False
                n_lit += 1
            for text, want in cases.items():
                m = xgr.GrammarMatcher(ctx)
                got = m.accept_string(text) and m.is_completed()
                self.assertEqual(bool(got), want, f"{name} {sk['skel_id']}: {text[-60:]!r}")
        self.assertGreater(n_lit, 5)

    @unittest.skipUnless(LIBS and have_tok("qwen3.5-9b"), "xgrammar + vllm + Qwen tokenizer")
    def test_qwen(self):
        self.check("qwen3.5-9b", False)

    @unittest.skipUnless(LIBS and have_tok("gemma-4-12b"), "xgrammar + vllm + Gemma tokenizer")
    def test_gemma(self):
        self.check("gemma-4-12b", False)

    @unittest.skipUnless(LIBS and have_tok("ministral-3-8b"), "xgrammar + vllm + Ministral tokenizer")
    def test_ministral(self):
        self.check("ministral-3-8b", True)          # it writes a blank line between lines (probe 09-27)


class RealBan(unittest.TestCase):
    def check(self, name):
        d = tempfile.mkdtemp(prefix="test_ban_")
        try:
            t = serve.tokenizer_only(name)
            a = serve.dash_ban(t, cache_dir=d)
            b = serve.dash_ban(t, cache_dir=d)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        self.assertEqual((a["cached"], b["cached"], a["ids"]), (False, True, b["ids"]))
        self.assertLessEqual(a["n"], decode.LOGIT_BIAS_CAP)
        self.assertGreater(a["n"], 100)
        p = os.path.join(PROBE, f"ban_{name}.dry.json")
        if os.path.exists(p):
            with open(p) as f:
                self.assertEqual(a["ids"], json.load(f)["ban_ids"], "not the probe's ban set")
        return a

    @unittest.skipUnless(have("xgrammar") and have_tok("qwen3.5-9b"), "xgrammar + Qwen tokenizer")
    def test_qwen(self):
        self.assertEqual(self.check("qwen3.5-9b")["n"], 185)

    @unittest.skipUnless(have("xgrammar") and have_tok("gemma-4-12b"), "xgrammar + Gemma tokenizer")
    def test_gemma(self):
        self.assertEqual(self.check("gemma-4-12b")["n"], 149)

    @unittest.skipUnless(have("mistral_common") and have_tok("ministral-3-8b"), "mistral_common + tekken")
    def test_ministral(self):
        self.assertEqual(self.check("ministral-3-8b")["n"], 140)


def vllm_tok(name):
    from vllm.tokenizers.registry import get_tokenizer
    cfg = serve.TEACHERS[name]
    if cfg.get("tokenizer_dir"):
        return get_tokenizer(os.path.join(serve.MODELS, cfg["tokenizer_dir"]), tokenizer_mode="mistral")
    return get_tokenizer(serve.model_path(cfg))


# dp2 sentences with the AI-isms the teachers wrote (records of 2026-09-27), as they follow a label ("A2:")
DP2_AI_ISMS = ["I'd be happy to help with that.", "I am happy to help.", "I'm happy to help.", "I’m happy to help.",
               "We would be happy to help.", "Happy to help!", "I am here to help.", "I'm here to help.",
               "I’m here to help.", "Hello, how can I help you today?", "How can I help?", "Is there anything else?",
               "Is there anything else I can do?", "Have a good day, and feel free to ask.", "Feel free to ask.",
               "Please feel free to ask.", "I hope this helps."]


class RealPhraseBan(unittest.TestCase):
    def check(self, name):
        vt = vllm_tok(name)
        pb = serve.phrase_ban(vt)
        self.assertLessEqual(pb["n_seqs"], decode.BAD_WORDS_CAP)
        self.assertLessEqual(pb["n_tokens"], decode.BAD_TOKENS_CAP)
        self.assertGreaterEqual(pb["n_seqs"], len(decode.PHRASES))       # every phrase at least in its bare form
        for sent in DP2_AI_ISMS:                                          # canonical tokens after "A2:"
            self.assertTrue(decode.covered(pb["seqs"], vt.encode(text=" " + sent, add_special_tokens=False)), sent)
        self.assertFalse(decode.covered(pb["seqs"], vt.encode(text=" I would be glad to see you then.",
                                                               add_special_tokens=False)))
        from vllm import SamplingParams
        t = serve.tokenizer_only(name)
        t.engine = {"max_model_len": 2048}
        sp, _ = serve._params(t, {"max_tokens": 64, "bad_words": decode.phrase_words(["Happy to help!"])}, 100, 7)
        self.assertIsInstance(sp, SamplingParams)
        self.assertEqual(len(sp.bad_words), len(decode.PHRASES) - 2)     # "happy to help" in both cases
        return pb

    @unittest.skipUnless(have("vllm") and have_tok("qwen3.5-9b"), "vllm + Qwen tokenizer")
    def test_qwen(self):
        self.check("qwen3.5-9b")

    @unittest.skipUnless(have("vllm") and have_tok("gemma-4-12b"), "vllm + Gemma tokenizer")
    def test_gemma(self):
        self.check("gemma-4-12b")

    @unittest.skipUnless(have("vllm") and have("mistral_common") and have_tok("ministral-3-8b"), "vllm + tekken")
    def test_ministral(self):
        self.check("ministral-3-8b")


@unittest.skipUnless(have("vllm") and have_tok("qwen3.5-9b"), "vllm + Qwen tokenizer")
class RealParams(unittest.TestCase):
    def test_vllm_takes_regex_ban_and_preset(self):
        from vllm.sampling_params import StructuredOutputsParams
        t = serve.tokenizer_only("qwen3.5-9b")
        t.engine = {"max_model_len": 2048}
        rx = decode.label_regex(driver.constraint(SKELS[0], "labels")["lines"])
        sp, _ = serve._params(t, {"max_tokens": 64, "preset": "card", "regex": rx, "ban_ids": [5, 9]}, 100, 7)
        self.assertIsInstance(sp.structured_outputs, StructuredOutputsParams)
        self.assertEqual(sp.structured_outputs.regex, rx)
        self.assertEqual(dict(sp.logit_bias), {5: -100.0, 9: -100.0})
        self.assertEqual((sp.temperature, sp.top_p, sp.top_k, sp.presence_penalty, sp.seed), (0.7, 0.8, 20, 1.5, 7))


@unittest.skipUnless(have("mistral_common") and have_tok("ministral-3-8b"), "mistral_common + tekken")
class MistralPin(unittest.TestCase):
    def test_pin_and_no_deprecated_read(self):
        from mistral_common.protocol.instruct.messages import UserMessage
        from mistral_common.protocol.instruct.request import ChatCompletionRequest
        from mistral_common.tokens.tokenizers.base import Tokenized
        serve.check_mistral_common()
        t = serve.tokenizer_only("ministral-3-8b")

        def no_read(self):
            raise AssertionError("serve.render read Tokenized.text")
        same = 0
        for sk in SKELS:
            p = R.build(sk)["prompt"]
            with warnings.catch_warnings(), mock.patch.object(Tokenized, "text", property(no_read)):
                warnings.simplefilter("error")
                ids, text = serve.render(t, p)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                old = t.tok.encode_chat_completion(ChatCompletionRequest(messages=[UserMessage(content=p)]))
            same += (old.text, list(old.tokens)) == (text, ids)
        self.assertEqual(same, len(SKELS))


if __name__ == "__main__":
    unittest.main()
