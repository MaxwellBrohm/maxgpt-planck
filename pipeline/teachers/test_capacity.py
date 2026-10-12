"""Stub run of capacity.py end to end: CPU only, anywhere; stub_vllm stands in for vLLM, torch and the tokenizers,
and the gpu.lock check is patched out (the stub loads nothing). The prompts phase is the real one (skeleton.shard +
render_prompt.build on the FAKE banks), so its dry/FAKE markers are what capacity.py reads back.
    cd pipeline/teachers && python3 -B -m unittest test_capacity -v"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import capacity  # noqa: E402
import capmeter  # noqa: E402
import serve  # noqa: E402
import stub_vllm as S  # noqa: E402
import thinking_control  # noqa: E402


class CapacityStub(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp(prefix="capstub_")
        cls.prompts = os.path.join(cls.d, "p.dry.jsonl")
        capacity.main(["prompts", "--out", cls.prompts, "--n", "140"])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def setUp(self):
        for p in (S.install(), mock.patch.object(serve, "tokenizer", S.tokenizer),
                  mock.patch.object(serve.gpulock, "check", lambda *a, **k: 1),
                  mock.patch.object(capmeter, "gpu", lambda: {"used_mib": 900, "temp_c": 40})):
            p.start()
            self.addCleanup(p.stop)
        self.out = tempfile.mkdtemp(dir=self.d)
        self.log = os.path.join(self.out, "run.log")
        with open(self.log, "w") as f:
            f.write("an older load: GPU KV cache size: 5 tokens, Maximum concurrency for 2,048 tokens per request: "
                    "0.01x\n")
        S.FakeLLM.LOG = self.log

    def run_phase(self, phase, teacher="gemma-4-12b"):
        rc = capacity.main([phase, "--teacher", teacher, "--prompts", self.prompts, "--out-dir", self.out,
                            "--log", self.log])
        with open(os.path.join(self.out, f"{teacher}.{phase}.dry.json")) as f:
            return rc, json.load(f)

    def test_prompts_are_dry_fake_and_refused_otherwise(self):
        rows = capmeter.read_prompts(self.prompts)
        self.assertEqual(len(rows), 140)
        bad = os.path.join(self.out, "bad.jsonl")
        old = {k: v for k, v in rows[0].items() if k != "provenance"}           # the older PC file's row shape
        self.assertTrue(rows[0]["variant"].startswith("instr.fake.") and capmeter.is_fake(old))
        for row in ({**rows[0], "provenance": "teacher"}, {**old, "variant": "instr.real.0"}, {**rows[0], "status": "x"}):
            with open(bad, "w") as f:
                f.write(json.dumps(rows[1]) + "\n" + json.dumps(row) + "\n")
            with self.assertRaises(SystemExit, msg=row):
                capmeter.read_prompts(bad)

    def test_auto_then_fp8(self):
        rc, a = self.run_phase("auto")
        self.assertEqual(rc, 0)
        self.assertTrue(a["complete"] and a["works"] and a["status"] == "dry")
        self.assertEqual(a["load"]["vllm_log"]["kv_tokens"], 21808, "read a KV line from before this load")
        self.assertTrue(a["load"]["engine"]["disable_log_stats"] is False)
        self.assertEqual(a["thinking"]["n"], 10)
        self.assertTrue(a["thinking"]["pass"])
        self.assertTrue(all(r["raw"] for r in a["thinking"]["rows"]))
        self.assertTrue(a["control"]["pass"])
        self.assertEqual(a["control"]["fired"], {"on": 3})
        self.assertTrue(all(w["count_id2"] == 1 and w["first_ids"][0] == 2 and not w["system_turn"] for w in a["wire"]))
        lim = a["kv_limit"]["seqs"]
        self.assertEqual(lim, min(21808 // int(a["kv_limit"]["seq_tokens"] + 1), 32))
        self.assertEqual(a["kv_limit"]["batches"], sorted({1, 8, 32, 64, lim}))
        done = [r for r in a["sweep"] if "skipped" not in r]
        self.assertEqual([r["batch"] for r in done], [b for b in a["kv_limit"]["batches"] if b <= 140 - 17])
        for r in done:
            self.assertTrue(r["all_exact_len"])
            self.assertEqual(r["out_tokens"], 200 * r["batch"])
        self.assertEqual(a["greedy"]["self_agreement"], {"n": 8, "identical": 8, "mean_prefix_frac": 1.0})
        self.assertEqual(S.FakeLLM.instances[0].shutdowns, 1)
        rc, f8 = self.run_phase("fp8")
        self.assertEqual(rc, 0)
        self.assertEqual(S.FakeLLM.instances[1].kw["kv_cache_dtype"], "fp8")
        self.assertEqual(f8["load"]["vllm_log"]["kv_tokens"], 43616)
        self.assertEqual(f8["load"]["vllm_log"]["attention_backends"], ["TRITON_ATTN"])
        self.assertEqual(f8["greedy"]["vs_auto"]["identical"], 8)
        self.assertNotIn("thinking", f8)
        self.assertEqual(f8["kv_limit"]["batches"], sorted({f8["kv_limit"]["seqs"], 64}))

    def test_qwen_control_scores_the_opening_tag(self):
        rc, a = self.run_phase("auto", "qwen3.5-9b")
        self.assertEqual(rc, 0)
        c = a["control"]
        self.assertEqual((c["scored_probe"], c["fired"], c["pass"]), ("on_open", {"on": 3, "on_open": 3}, True))
        self.assertTrue(all(r["on_open_first_ids"][0] == 248068 for r in c["rows"]))
        self.assertTrue(all(w["tail"].endswith("<think>\n\n</think>\n\n") for w in a["wire"]))

    def test_control_fails_when_the_detector_is_blind(self):
        with mock.patch.object(serve, "thought_in", lambda t, text, ids: []):
            _, a = self.run_phase("auto")
        self.assertFalse(a["control"]["pass"])

    def test_a_thought_on_the_off_path_fails_both_checks(self):
        real = S.answer

        def leaky(text, sp):     # thinking "off" render, but the model opens a thought anyway
            if text.endswith("<|channel>thought\n<channel|>"):
                out = "<|channel>thought\nsneaky<channel|>answer"
                return out, S.encode(out), "stop"
            return real(text, sp)
        with mock.patch.object(S, "answer", leaky):
            _, a = self.run_phase("auto")
        self.assertEqual((a["thinking"]["pass"], a["thinking"]["with_marker"]), (False, 10))
        self.assertEqual((a["control"]["pass"], a["control"]["off_with_marker"]), (False, 3))

    def test_a_failed_auto_load_is_not_complete(self):
        with mock.patch.object(S.FakeLLM, "__init__", mock.Mock(side_effect=RuntimeError("stub: no free memory"))):
            rc, a = self.run_phase("auto")
        self.assertEqual((rc, a["works"], a["complete"]), (3, False, False))

    def test_fp8_retries_on_triton_only_after_a_missing_nvcc(self):
        self.run_phase("auto")
        S.FakeLLM.NVCC_FP8 = True
        rc, f8 = self.run_phase("fp8")
        self.assertEqual(rc, 0)
        self.assertEqual([x["works"] for x in f8["attempts"]], [False, True])
        self.assertEqual(f8["attempts"][1]["engine_extra"], {"attention_backend": "TRITON_ATTN"})
        self.assertEqual(S.FakeLLM.instances[-1].kw["attention_backend"], "TRITON_ATTN")
        self.assertEqual(f8["load"]["vllm_log"]["kv_tokens"], 43616)
        self.assertTrue(f8["complete"] and f8["works"])

    def test_fp8_load_failure_is_recorded(self):
        S.FakeLLM.FAIL_FP8 = True
        rc, f8 = self.run_phase("fp8")
        self.assertEqual(rc, 3)
        self.assertFalse(f8["works"])
        self.assertTrue(f8["complete"], "fp8 is tried once: a failed fp8 load is final")
        self.assertIn("stub fp8 refusal", f8["error"])
        self.assertEqual(len(f8["attempts"]), 1, "a failure without the nvcc line is not retried")
        self.assertEqual(f8["attempts"][0]["vllm_log"]["fp8_lines"], ["Using fp8 KV cache without scales"])

    def test_agree(self):
        a = [{"raw": "abcd"}, {"raw": "abcd"}, {"raw": ""}]
        b = [{"raw": "abcd"}, {"raw": "abXY"}, {"raw": ""}]
        self.assertEqual(capmeter.agree(a, b), {"n": 3, "identical": 2, "mean_prefix_frac": round((1 + 0.5 + 0) / 3, 3)})

    def test_batch_list(self):
        self.assertEqual(capacity.batch_list(5), [1, 5, 8, 32, 64])
        self.assertEqual(capacity.batch_list(300), [1, 8, 32, 64, 128, 256, 300])
        self.assertEqual(thinking_control.SCORED, {"gemma-4-12b": "on", "qwen3.5-9b": "on_open"})


if __name__ == "__main__":
    unittest.main()
