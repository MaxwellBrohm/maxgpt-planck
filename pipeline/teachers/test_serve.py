"""Tests for serve.py. Three layers, each skipped where it cannot run:
  pure       anywhere (the Mac included): log parsing, end-marker stripping, thought detection;
  tokenizer  where transformers / mistral_common and ~/planck/models exist (the PC, CPU only, no model):
             each teacher's render against its invariants, and renders built to break them are refused;
  gpu        only with PLANCK_GPU_TEST=1 and the GPU lock held by the caller: load Ministral, generate 2 prompts,
             unload, and check the GPU memory came back.
    cd pipeline/teachers && python3 -B -m unittest test_serve -v"""
import os
import subprocess
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import serve  # noqa: E402

LOG = ("INFO 09-26 kv_cache_utils.py:2397] GPU KV cache size: 1,234 tokens, Maximum concurrency for 2,048 tokens "
       "per request: 0.60x\nlater\nINFO GPU KV cache size: 52,480 tokens, Maximum concurrency for 2,048 tokens per "
       "request: 25.62x\n")


def have(mod):
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def have_model(name):
    cfg = serve.TEACHERS[name]
    return os.path.isdir(os.path.join(serve.MODELS, cfg.get("tokenizer_dir", cfg["dir"])))


class Pure(unittest.TestCase):
    def test_kv_from_log_takes_the_last_line(self):
        self.assertEqual(serve.kv_from_log(LOG), {"kv_tokens": 52480, "per_request_tokens": 2048,
                                                  "max_concurrency": 25.62})
        self.assertIsNone(serve.kv_from_log("no such line"))

    def test_strip_end(self):
        self.assertEqual(serve.strip_end("U1: hi\nEND<turn|>\n"), "U1: hi\nEND")
        self.assertEqual(serve.strip_end("x</s><|im_end|>"), "x")
        self.assertEqual(serve.strip_end("keep <turn|> inside"), "keep <turn|> inside")

    def test_thought_in(self):
        t = serve.Teacher("qwen3.5-9b", serve.TEACHERS["qwen3.5-9b"], tok=None)
        t.thought_ids = {248068}
        self.assertEqual(serve.thought_in(t, "fine text", [1, 2]), [])
        self.assertEqual(serve.thought_in(t, "<think>hm</think>ok", [248068]), ["<think>", "</think>", "id248068"])

    def test_every_teacher_is_apache_and_pinned(self):
        for name, cfg in serve.TEACHERS.items():
            self.assertEqual(cfg["license"], "Apache-2.0", name)
            self.assertRegex(cfg["revision"], r"^[0-9a-f]{40}$", name)


@unittest.skipUnless(have("transformers") and have_model("gemma-4-12b"), "needs transformers + the Gemma files")
class GemmaRender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = serve.tokenizer_only("gemma-4-12b")

    def test_matches_official_template_thinking_off(self):
        ids, text = serve.render(self.t, "Say hi.")
        official = self.t.tok.apply_chat_template([{"role": "user", "content": "Say hi."}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        self.assertEqual(text, official)
        self.assertEqual(ids[0], 2)
        self.assertEqual(ids.count(2), 1)
        self.assertEqual(self.t.thought_ids, {98, 100, 101})

    def test_refuses_a_prompt_that_carries_a_second_bos(self):
        with self.assertRaises(serve.RenderError):
            serve.render(self.t, "<bos>hello")


@unittest.skipUnless(have("transformers") and have_model("qwen3.5-9b"), "needs transformers + the Qwen files")
class QwenRender(unittest.TestCase):
    def test_thinking_off_and_refused_when_on(self):
        t = serve.tokenizer_only("qwen3.5-9b")
        ids, text = serve.render(t, "Say hi.")
        self.assertTrue(text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        self.assertIn(248068, t.thought_ids)
        t.cfg = {**t.cfg, "chat_kwargs": {"enable_thinking": True}}
        with self.assertRaises(serve.RenderError):
            serve.render(t, "Say hi.")


@unittest.skipUnless(have("mistral_common") and have_model("ministral-3-8b"), "needs mistral_common + tekken.json")
class MinistralRender(unittest.TestCase):
    def test_no_system_prompt_no_think_ids(self):
        t = serve.tokenizer_only("ministral-3-8b")
        ids, text = serve.render(t, "Say hi.")
        self.assertEqual(text, "<s>[INST]Say hi.[/INST]")
        self.assertEqual(ids[:2], [1, 3])
        self.assertFalse({34, 35} & set(ids))


def gpu_used():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=20).stdout
    return int(out.strip().splitlines()[0])


@unittest.skipUnless(os.environ.get("PLANCK_GPU_TEST") == "1", "GPU test: set PLANCK_GPU_TEST=1 and hold gpu.lock")
class GpuSmoke(unittest.TestCase):
    def test_load_generate_unload(self):
        before = gpu_used()
        t = serve.load("ministral-3-8b", max_num_seqs=8)
        try:
            self.assertGreater(gpu_used() - before, 6000)
            out = serve.generate(t, ["Say hello in five words.", "Name one colour."], {"max_tokens": 32},
                                 seeds=[1, 2])
            self.assertEqual(len(out), 2)
            for o in out:
                self.assertTrue(o["text"].strip(), o)
                self.assertEqual(o["thought"], [], o)
                self.assertGreater(o["n_prompt"], 5)
        finally:
            serve.unload(t)
        time.sleep(5)
        self.assertLess(gpu_used() - before, 800, "GPU memory did not come back after unload")


if __name__ == "__main__":
    unittest.main()
