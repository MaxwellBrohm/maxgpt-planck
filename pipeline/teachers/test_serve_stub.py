"""Stub tests of serve.py load / generate / unload and of gpulock.py: CPU only, anywhere (the Mac included). vLLM,
torch and the teacher tokenizers are stub_vllm's stand-ins, so what is tested is serve's own logic: the lock
refusal, engine kwargs, the render gate before any GPU time, per-request refusals, sampling params, output order,
thought detection, and that unload always drops the engine. Linux only: gpulock against the real /proc/locks.
    cd pipeline/teachers && python3 -B -m unittest test_serve_stub -v"""
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gpulock  # noqa: E402
import serve  # noqa: E402
import stub_vllm as S  # noqa: E402


class Stub(unittest.TestCase):
    def setUp(self):
        for p in (S.install(), mock.patch.object(serve, "tokenizer", S.tokenizer),
                  mock.patch.dict(os.environ, {}, clear=False)):
            p.start()
            self.addCleanup(p.stop)
        os.environ.pop("VLLM_USE_FLASHINFER_SAMPLER", None)

    def test_load_refuses_without_the_gpu_lock(self):
        with self.assertRaises(gpulock.GpuLockError):
            serve.load("qwen3.5-9b")
        self.assertEqual(S.FakeLLM.instances, [])

    def test_load_merges_engine_kwargs(self):
        t = serve.load("qwen3.5-9b", require_lock=False, max_num_seqs=8)
        kw = S.FakeLLM.instances[0].kw
        self.assertEqual({k: kw[k] for k in ("max_model_len", "gpu_memory_utilization", "language_model_only",
                                               "max_num_seqs", "max_num_batched_tokens")},
                         {"max_model_len": 2048, "gpu_memory_utilization": 0.86, "language_model_only": True,
                          "max_num_seqs": 8, "max_num_batched_tokens": 4096})
        self.assertTrue(S.FakeLLM.instances[0].model.endswith("Qwen3.5-9B-AWQ-4bit"))
        self.assertEqual(os.environ["VLLM_USE_FLASHINFER_SAMPLER"], "0")
        self.assertEqual(t.thought_ids, {248068, 248069})
        self.assertIsInstance(t.load_s, float)

    def test_load_refuses_a_broken_wire_before_the_gpu(self):
        bad = {**serve.TEACHERS["qwen3.5-9b"], "chat_kwargs": {"enable_thinking": True}}
        with mock.patch.dict(serve.TEACHERS, {"qwen3.5-9b": bad}):
            with self.assertRaisesRegex(serve.RenderError, "thinking is not off"):
                serve.load("qwen3.5-9b", require_lock=False)
        self.assertEqual(S.FakeLLM.instances, [])

    def test_generate_refuses_per_request_and_keeps_order(self):
        t = serve.load("gemma-4-12b", require_lock=False)
        prompts = ["first", "<bos>a second bos", "third", "x " * 1100]
        out = serve.generate(t, prompts, {"max_tokens": 5000}, seeds=[1, 2, 3, 4])
        self.assertEqual([o["finish"] for o in out], ["stop", "render_error", "stop", "render_error"])
        self.assertIn("bos", out[1]["error"])
        self.assertIn("no room", out[3]["error"])
        sent, params = S.FakeLLM.instances[0].calls[-1]
        self.assertEqual([p["prompt_token_ids"] for p in sent], [serve.render(t, p)[0] for p in ("first", "third")])
        for sp, seed, p in zip(params, (1, 3), sent):
            self.assertFalse(sp.skip_special_tokens)
            self.assertEqual(sp.stop, ["<turn|>", "<|turn>"])
            self.assertEqual(sp.seed, seed)
            self.assertEqual(sp.max_tokens, 2048 - len(p["prompt_token_ids"]))
            self.assertEqual((sp.temperature, sp.top_k), (1.0, 64))
        self.assertTrue(out[0]["clamped"] and out[2]["clamped"])
        self.assertRegex(out[0]["text"], r"\AU1: fine words here\nA1: more fine words \d+\nEND\Z")
        self.assertTrue(out[0]["raw"].endswith("END<turn|>"))
        self.assertEqual(out[0]["thought"], [])

    def test_generate_sees_thought_markers_and_ids(self):
        t = serve.load("gemma-4-12b", require_lock=False)
        with mock.patch.object(S, "answer", lambda text, sp: ("<|channel>thought\nhm", S.encode("<|channel>thought\nhm"),
                                                              "stop")):
            out = serve.generate(t, ["q"], {"max_tokens": 20})
        self.assertEqual(out[0]["thought"], ["<|channel>", "id100"])

    def test_generate_min_tokens_and_list_lengths(self):
        t = serve.load("qwen3.5-9b", require_lock=False)
        serve.generate(t, ["q"], {"max_tokens": 50, "min_tokens": 400})
        sp = S.FakeLLM.instances[0].calls[-1][1][0]
        self.assertEqual((sp.max_tokens, sp.min_tokens), (50, 50))
        with self.assertRaises(ValueError):
            serve.generate(t, ["a", "b"], seeds=[1])
        calls = len(S.FakeLLM.instances[0].calls)
        self.assertEqual(serve.generate(t, ["x " * 1100])[0]["finish"], "render_error")
        self.assertEqual(len(S.FakeLLM.instances[0].calls), calls, "an all-refused batch still called the engine")

    def test_unload_always_drops_the_engine(self):
        t = serve.load("qwen3.5-9b", require_lock=False)
        llm = t.llm
        serve.unload(t)
        serve.unload(t)
        self.assertIsNone(t.llm)
        self.assertEqual(llm.shutdowns, 1)
        t = serve.load("qwen3.5-9b", require_lock=False)
        t.llm.llm_engine.engine_core.shutdown = mock.Mock(side_effect=RuntimeError("engine already dead"))
        serve.unload(t)
        self.assertIsNone(t.llm)

    def test_loaded_unloads_on_an_exception(self):
        with self.assertRaises(KeyError):
            with serve.loaded("gemma-4-12b", require_lock=False) as t:
                raise KeyError("boom")
        self.assertIsNone(t.llm)
        self.assertEqual(S.FakeLLM.instances[0].shutdowns, 1)

    def test_log_facts(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".log") as f:
            S.FakeLLM.LOG = f.name
            S.FakeLLM("m", kv_cache_dtype="fp8")
            with open(f.name) as g:
                got = serve.log_facts(g.read())
        self.assertEqual((got["kv_tokens"], got["max_concurrency"], got["model_load_gib"], got["kv_cache_gib"]),
                         (43616, 21.3, 5.99, 2.83))
        self.assertEqual(got["attention_backends"], ["TRITON_ATTN"])
        self.assertEqual(len(got["fp8_lines"]), 1)
        more = serve.log_facts("(EngineCore pid=9) INFO [cuda.py:478] Using AttentionBackendEnum.TRITON_ATTN backend.\n"
                               "INFO [core.py:123] Initializing a V1 LLM engine (v0.30.0) with config: kv_cache_dtype=fp8\n"
                               "(EngineCore pid=9) INFO [gpu_worker.py:640] Available KV cache memory: -2.11 GiB\n")
        self.assertEqual((more["attention_backends"], more["fp8_lines"], more["kv_cache_gib"]), (["TRITON_ATTN"], [], -2.11))


class GpuLock(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="gpulock_")
        self.lock = os.path.join(self.d, "gpu.lock")
        open(self.lock, "w").close()
        for pid, ppid in ((500, 400), (400, 1), (1, 0)):
            os.makedirs(os.path.join(self.d, "proc", str(pid)))
            with open(os.path.join(self.d, "proc", str(pid), "stat"), "w") as f:
                f.write(f"{pid} (odd) name) S {ppid} 1 1 0\n")
        st = os.stat(self.lock)
        self.key = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"

    def check(self, lines):
        p = os.path.join(self.d, "locks")
        with open(p, "w") as f:
            f.write("\n".join(lines) + "\n")
        return gpulock.check(self.lock, locks_path=p, proc=os.path.join(self.d, "proc"), pid=500)

    def test_ancestor_holder_passes(self):
        self.assertEqual(self.check(["1: POSIX  ADVISORY  WRITE 7 00:01:5 0 EOF",
                                     f"2: FLOCK  ADVISORY  WRITE 400 {self.key} 0 EOF"]), 400)

    def fdinfo(self, pid, *locks):
        d = os.path.join(self.d, "proc", str(pid), "fdinfo")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "3"), "w") as f:
            f.write("pos:\t0\nflags:\t0100000\n" + "".join(f"lock:\t{x}\n" for x in locks))

    def test_ancestor_fdinfo_holder_passes_when_proc_locks_omits_it(self):
        self.fdinfo(400, f"1: FLOCK  ADVISORY  WRITE 400 {self.key} 0 EOF")
        self.assertEqual(self.check(["1: POSIX  ADVISORY  WRITE 7 00:01:5 0 EOF"]), 400)

    def test_fdinfo_shared_other_file_or_non_ancestor_is_refused(self):
        other = self.key.rsplit(":", 1)[0] + ":999999999"
        self.fdinfo(400, f"1: FLOCK  ADVISORY  READ 400 {self.key} 0 EOF",
                    f"2: FLOCK  ADVISORY  WRITE 400 {other} 0 EOF")
        os.makedirs(os.path.join(self.d, "proc", "999"))
        with open(os.path.join(self.d, "proc", "999", "stat"), "w") as f:
            f.write("999 (x) S 1 1 1 0\n")
        self.fdinfo(999, f"1: FLOCK  ADVISORY  WRITE 999 {self.key} 0 EOF")
        with self.assertRaises(gpulock.GpuLockError):
            self.check([])

    def test_everything_else_is_refused(self):
        other = self.key.rsplit(":", 1)[0] + ":999999999"
        for lines in ([f"1: FLOCK  ADVISORY  WRITE 999 {self.key} 0 EOF",
                       f"1: -> FLOCK  ADVISORY  WRITE 500 {self.key} 0 EOF"],      # held elsewhere, we wait
                      [f"1: FLOCK  ADVISORY  READ 400 {self.key} 0 EOF"],         # shared, not exclusive
                      [f"1: FLOCK  ADVISORY  WRITE 400 {other} 0 EOF"],           # another file
                      []):
            with self.assertRaises(gpulock.GpuLockError, msg=lines):
                self.check(lines)
        with self.assertRaises(gpulock.GpuLockError):
            gpulock.check(self.lock, locks_path=os.path.join(self.d, "missing"))


@unittest.skipUnless(os.path.exists("/proc/locks") and subprocess.run(["which", "flock"], capture_output=True)
                     .returncode == 0, "Linux with flock(1) only")
class RealProcLocks(unittest.TestCase):
    def test_flock_wrapper_counts_and_bare_does_not(self):
        with tempfile.TemporaryDirectory() as d:
            lock = os.path.join(d, "x.lock")
            open(lock, "w").close()
            code = f"import sys; sys.path.insert(0, {HERE!r}); import gpulock; print(gpulock.check({lock!r}))"
            held = subprocess.run(["flock", "-w", "10", lock, sys.executable, "-c", code], capture_output=True,
                                  text=True, timeout=60)
            self.assertEqual(held.returncode, 0, held.stderr)
            bare = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
            self.assertNotEqual(bare.returncode, 0)
            self.assertIn("GpuLockError", bare.stderr)


if __name__ == "__main__":
    unittest.main()
