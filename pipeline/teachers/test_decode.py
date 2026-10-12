"""Tests of the decoding controls (2026-09-27): decode.py (label regex, dash ban rule), serve.py presets (D4),
per-request resolution into SamplingParams, the cached dash ban, the util cap and the mistral_common pin. CPU only,
anywhere: vLLM, torch, mistral_common and the tokenizers are stub_vllm stand-ins; Python's re stands in for
xgrammar (test_decode_pc.py checks the same regexes with xgrammar and the real tokenizers on the PC).
    cd pipeline/teachers && python3 -B -m unittest test_decode -v"""
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import decode  # noqa: E402
import driver  # noqa: E402
import fake_engine as FE  # noqa: E402
import fake_teacher  # noqa: E402
import serve  # noqa: E402
import skeleton  # noqa: E402
import stub_vllm as S  # noqa: E402

SKELS = skeleton.shard("test-decode", 30)
REAL_TOKENIZER = serve.tokenizer
VOCAB = [b"a", b" the", "\u2014".encode(), b"-", b"well-", b"-known", b" -", b"--", b" - ", b"x--y",
         "\u2013".encode(), " \u2212".encode(), b"\xe2\x80", "\u2019".encode(), b"", "\u2012\u2015".encode()]
BANNED = [2, 6, 7, 8, 9, 10, 11, 15]


class LabelRegex(unittest.TestCase):
    def full(self, sk, text, mode="labels", sep="\\n"):
        return re.fullmatch(decode.label_regex(driver.constraint(sk, mode)["lines"], sep), text) is not None

    def test_canonical_render_matches_both_modes(self):
        self.assertTrue(all(self.full(sk, FE.canon(sk), m) for sk in SKELS for m in ("labels", "labels_exact")))

    def test_form_breaks_do_not_match(self):
        for sk in SKELS[:10]:
            raw = fake_teacher.raw(sk)
            lines = raw.split("\n")
            first = lines[0].split(":")[0]
            for bad in ("\n".join(lines[:-1]),                                    # no END
                        raw.replace(first + ":", "User:", 1),                    # renamed label
                        "\n".join(lines[:1] + lines[2:]),                        # a label missing
                        raw + "\nThanks!",                                       # text after END
                        "Sure.\n" + raw,                                         # text before the first label
                        lines[0] + " END\n" + "\n".join(lines[1:]).replace("\nEND", "")):
                self.assertFalse(self.full(sk, bad), bad[:80])

    def test_blank_lines_only_with_the_ministral_separator(self):
        sk = SKELS[0]
        spaced = fake_teacher.raw(sk).replace("\n", "\n\n")
        self.assertFalse(self.full(sk, spaced))
        self.assertTrue(self.full(sk, spaced, sep="\\n+"))
        self.assertEqual(serve.TEACHERS["ministral-3-8b"]["line_sep"], "\\n+")
        self.assertNotIn("line_sep", serve.TEACHERS["qwen3.5-9b"])

    def test_exact_lines_are_literal_only_in_labels_exact(self):
        hit = 0
        for sk in SKELS:
            ex = [t for t in sk["turns"] if t["mode"] == "exact"]
            if not ex:
                continue
            raw = fake_teacher.raw(sk)
            lab = ("U" if ex[0]["role"] == "user" else "T") + str(ex[0]["i"] + 1)
            bad = raw.replace(f"{lab}: {ex[0]['text']}", f"{lab}: {ex[0]['text'].lower()} indeed", 1)
            self.assertNotEqual(bad, raw)
            self.assertTrue(self.full(sk, bad, "labels"))
            self.assertFalse(self.full(sk, bad, "labels_exact"))
            hit += 1
        self.assertGreater(hit, 5)

    def test_literals_escape_every_metacharacter(self):
        for s in ("What's (this)? [a] {b} 1.5*2+3 ^$|\\ end.", "<result>Stikloum Theatre location: Graz</result>",
                  "a.b", "Sure!? Really..."):
            self.assertIsNotNone(re.fullmatch(decode.esc(s), s), s)
        for lit, other in (("a.b", "axb"), ("ok?", "o"), ("a+", "aa"), ("b*", ""), ("(c)", "c"), ("[de]", "d"),
                           ("f{2}", "ff"), ("g|h", "g"), ("^i$", "i"), ("j\\k", "j\\k\\")):
            self.assertIsNotNone(re.fullmatch(decode.esc(lit), lit), lit)
            self.assertIsNone(re.fullmatch(decode.esc(lit), other), lit)

    def test_bad_plans_are_refused(self):
        for lines, sep in (([["U1", None], ["U1", None]], "\\n"), ([["X1", None]], "\\n"), ([["U0", None]], "\\n"),
                           ([["U1", "two\nlines"]], "\\n"), ([["U1", ""]], "\\n"), ([], "\\n"),
                           ([["U1", None]], "\\n*")):
            with self.assertRaises(ValueError, msg=(lines, sep)):
                decode.label_regex(lines, sep)
        self.assertFalse(decode.constraint_ok({"mode": "labels", "lines": [["U1", "lit"]]}))
        self.assertFalse(decode.constraint_ok({"mode": "off", "lines": []}))
        self.assertTrue(decode.constraint_ok(driver.constraint(SKELS[0], "labels_exact")))


class DashBan(unittest.TestCase):
    def test_rule(self):
        ids, counts = decode.dash_ban_ids(VOCAB)
        self.assertEqual(ids, BANNED)
        self.assertEqual(counts, {"contains": 7, "end_space_hyphen": 1})

    def test_logit_bias_and_its_cap(self):
        self.assertEqual(decode.logit_bias([3, 5]), {3: -100.0, 5: -100.0})
        with self.assertRaises(ValueError):
            decode.logit_bias(range(decode.LOGIT_BIAS_CAP + 1))


class Presets(unittest.TestCase):
    def test_card_values(self):
        self.assertEqual(serve.preset("qwen3.5-9b", "card"), {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                                                             "min_p": 0.0, "presence_penalty": 1.5,
                                                             "repetition_penalty": 1.0})
        self.assertEqual(serve.preset("gemma-4-12b", "card"), {"temperature": 1.0, "top_p": 0.95, "top_k": 64,
                                                              "min_p": 0.0})
        self.assertLess(serve.preset("ministral-3-8b", "card")["temperature"], 0.1)

    def test_shared_is_the_0926_setting_and_gemma_has_none(self):
        for name in ("ministral-3-8b", "qwen3.5-9b"):
            self.assertEqual(serve.preset(name, "shared"), {"temperature": 0.8, "top_p": 0.95})
            self.assertEqual(serve.preset(name, "shared"), serve.preset(name, serve.DEFAULT_PRESET))
        for name, p in (("gemma-4-12b", "shared"), ("qwen3.5-9b", "hot")):
            with self.assertRaises(ValueError):
                serve.preset(name, p)

    def test_a_preset_replaces_the_default_and_is_a_copy(self):
        t = serve.Teacher("gemma-4-12b", serve.TEACHERS["gemma-4-12b"], tok=None)
        self.assertEqual(serve.resolve(t, {"max_tokens": 9})[0]["min_p"], 0.05)
        s, ex = serve.resolve(t, {"max_tokens": 9, "preset": "card"})
        self.assertEqual((s["min_p"], s["max_tokens"], ex), (0.0, 9, {"preset": "card"}))
        serve.preset("gemma-4-12b", "card")["top_k"] = 1
        self.assertEqual(serve.preset("gemma-4-12b", "card")["top_k"], 64)
        bare = {**serve.TEACHERS["gemma-4-12b"]["presets"], "bare": {"temperature": 0.5}}
        with mock.patch.dict(serve.TEACHERS["gemma-4-12b"], {"presets": bare}):
            self.assertEqual(serve.resolve(t, {"preset": "bare"})[0], {"temperature": 0.5})   # nothing leaks in
        rec = serve.sampling_record(t, {"max_tokens": 9, "preset": "card", "regex": "U1: x", "ban_ids": [1, 2]})
        self.assertEqual((rec["preset"], rec["regex_sha256"], rec["ban_n"]), ("card", decode.sha("U1: x"), 2))
        self.assertNotIn("max_tokens", rec["sampling"])


class ServeStub(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="test_decode_")
        for p in (S.install(), S.install_mistral(), mock.patch.object(serve, "tokenizer", S.tokenizer)):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(shutil.rmtree, self.d, True)

    def test_params_carry_regex_ban_and_preset(self):
        t = serve.load("qwen3.5-9b", require_lock=False, structured_backend="xgrammar")
        self.assertEqual(S.FakeLLM.instances[0].kw["structured_outputs_config"].backend, "xgrammar")
        sp, _ = serve._params(t, {"max_tokens": 50, "preset": "card", "regex": "U1: [^\\n]+\\nEND",
                                  "ban_ids": [7, 9]}, 10, 3)
        self.assertEqual(sp.structured_outputs.regex, "U1: [^\\n]+\\nEND")
        self.assertEqual(sp.logit_bias, {7: -100.0, 9: -100.0})
        self.assertEqual((sp.temperature, sp.top_k, sp.presence_penalty, sp.seed), (0.7, 20, 1.5, 3))
        plain, _ = serve._params(t, {"max_tokens": 50}, 10, 3)
        self.assertFalse(hasattr(plain, "structured_outputs") or hasattr(plain, "logit_bias"))
        self.assertEqual((plain.temperature, plain.top_p), (0.8, 0.95))
        with self.assertRaises(ValueError):
            serve._params(t, {"max_tokens": 5, "ban_ids": list(range(1025))}, 10, 3)

    def test_util_default_and_cap(self):
        self.assertEqual(serve.ENGINE["gpu_memory_utilization"], 0.86)
        with self.assertRaises(ValueError):
            serve.load("qwen3.5-9b", require_lock=False, gpu_memory_utilization=0.88)
        self.assertEqual(S.FakeLLM.instances, [])

    def models(self, name, content):
        cfg = serve.TEACHERS[name]
        d = os.path.join(self.d, "models", cfg.get("tokenizer_dir", cfg["dir"]))
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "tekken.json" if cfg.get("tokenizer_dir") else "tokenizer.json"), "w") as f:
            f.write(content)
        return os.path.join(self.d, "models")

    def test_ban_is_computed_once_and_cached_under_the_tokenizer_hash(self):
        m, cache = self.models("gemma-4-12b", "v1"), os.path.join(self.d, "cache")
        t = serve.tokenizer_only("gemma-4-12b")
        calls = []
        with mock.patch.object(serve, "token_bytes", lambda t: calls.append(1) or VOCAB):
            a = serve.dash_ban(t, m, cache)
            b = serve.dash_ban(t, m, cache)
            self.assertEqual((a["ids"], a["cached"], b["ids"], b["cached"], len(calls)), (BANNED, False, BANNED, True, 1))
            self.assertEqual(a["tokenizer_sha256"], serve.tokenizer_sha("gemma-4-12b", m))
            with open(b["path"]) as f:
                c = json.load(f)
            c["ids"] = c["ids"][:-1]                                # a damaged entry is computed again
            with open(b["path"], "w") as f:
                json.dump(c, f)
            self.assertFalse(serve.dash_ban(t, m, cache)["cached"])
            self.models("gemma-4-12b", "v2")                       # a new tokenizer file: a new key
            d = serve.dash_ban(t, m, cache)
            self.assertFalse(d["cached"])
            self.assertNotEqual(d["tokenizer_sha256"], a["tokenizer_sha256"])
        self.assertEqual(len(calls), 3)

    def test_ministral_ban_render_and_pin(self):
        m = self.models("ministral-3-8b", "tekken")
        self.enterContext(mock.patch.object(serve, "tokenizer", REAL_TOKENIZER))
        with mock.patch.object(serve, "check_mistral_common", lambda version=None: "1.12.0"):
            t = serve.tokenizer_only("ministral-3-8b", m)
        self.assertEqual(serve.dash_ban(t, m, os.path.join(self.d, "c"))["ids"], [6, 9, 10, 11, 15])
        ids, text = serve.render(t, "Say hi.")                 # FakeTokenized.text raises if render reads it
        self.assertEqual((text, ids[:2]), ("<s>[INST]Say hi.[/INST]", [1, 3]))
        for v, ok in (("1.11.6", False), ("1.11.7", True), ("1.12.0", True), ("1.12.9", True), ("1.13.0", False)):
            if ok:
                self.assertEqual(serve.check_mistral_common(v), v)
            else:
                self.assertRaises(RuntimeError, serve.check_mistral_common, v)
        with mock.patch.object(serve, "check_mistral_common", mock.Mock(side_effect=RuntimeError("pin"))):
            with self.assertRaisesRegex(RuntimeError, "pin"):
                serve.tokenizer_only("ministral-3-8b", m)


if __name__ == "__main__":
    unittest.main()
