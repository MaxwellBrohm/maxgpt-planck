"""One gpu.lock hold per teacher on the PC: load facts, thinking off, throughput, fp8 KV (DRY: FAKE skeleton renders;
nothing here is training data; outputs stay on the PC under ~/planck/runs/teachers/dry/).

    python capacity.py prompts --out P.dry.jsonl [--shard-seed capdry --n 400]      # CPU only: no lock, no model
    flock -w 7200 ~/planck/locks/gpu.lock bash -c '
      timeout -k 60 1500 python capacity.py auto --teacher T --prompts P --out-dir D --log LOG_A
      timeout -k 60 900  python capacity.py fp8  --teacher T --prompts P --out-dir D --log LOG_B'

auto (kv_cache_dtype auto) writes D/<T>.auto.dry.json after every step, complete: true at the end:
  load     vLLM's own log facts (KV tokens, concurrency at 2048, KV / model / CUDA-graph GiB, attention backend),
           the frontend's block count, nvidia-smi before and after load;
  wire     the render facts serve.render enforces (Gemma ids[0] == 2 once; Ministral '<s>[INST]..[/INST]', no
           system prompt; Qwen ending in the empty think block);
  thinking 10 prompts (5 that invite reasoning, 5 FAKE renders), each raw output kept whole with special tokens
           visible, plus thinking_control.control (the positive control; Gemma and Qwen);
  sweep    200 forced output tokens per prompt (min_tokens = max_tokens, ignore_eos, no stop), a disjoint slice of
           FAKE renders per batch size: 1, 8, 32, 64, the KV limit, and doubling past 64 up to it. Per batch: wall
           time, out tok/s, and peaks sampled every 0.5 s of running / waiting requests and KV usage (vLLM's own
           metrics) and device MiB (nvidia-smi);
  greedy   8 FAKE renders at temperature 0, natural stops, run twice (run-to-run agreement is the noise floor).
fp8 is a fresh process with kv_cache_dtype fp8: loads or not (the error is kept), the same load facts, the greedy
batch again (agreement with auto's), and the sweep at the fp8 KV limit and at 64. The KV limit is how many whole
sequences of (mean prompt + 200) tokens vLLM's KV line holds, capped by max_num_seqs."""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import serve  # noqa: E402
import thinking_control  # noqa: E402
from capmeter import Sampler, agree, gpu, log_since, make_prompts, read_prompts  # noqa: E402

OUT_TOKENS = 200
TEMPTERS = thinking_control.PROMPTS + [
    "Is 391 a prime number? Show your reasoning before you answer.",
    "Which is larger, 9.11 or 9.9? Think carefully first, then answer.",
]
PLAIN_THINKING = ("thinking process", "thought:", "let me think", "<think", "<|channel")


def wire(t, prompts):
    out = []
    for p in prompts:
        ids, text = serve.render(t, p)
        out.append({"n_ids": len(ids), "first_ids": ids[:3], "count_id2": ids.count(2), "head": text[:24],
                    "tail": text[-48:], "system_turn": any(s in text for s in ("[SYSTEM_PROMPT]", "<|turn>system",
                                                                                  "<|im_start|>system"))})
    return out


def thinking(t, fake):
    prompts = TEMPTERS + [r["prompt"] for r in fake]
    res = serve.generate(t, prompts, {"max_tokens": 400}, seeds=list(range(101, 101 + len(prompts))))
    rows = [{"prompt_head": p[:70], "finish": r["finish"], "n_out": r["n_out"], "thought": r["thought"],
             "plain_thinking_start": r["raw"].lstrip().lower().startswith(PLAIN_THINKING), "raw": r["raw"]}
            for p, r in zip(prompts, res)]
    return {"n": len(rows), "with_marker": sum(1 for r in rows if r["thought"]),
            "plain_thinking_start": sum(r["plain_thinking_start"] for r in rows),
            "pass": all(not r["thought"] and r["finish"] != "render_error" for r in rows), "rows": rows}


def kv_limit(t, log, prompts):
    seq = sum(len(serve.render(t, r["prompt"])[0]) for r in prompts) / len(prompts) + OUT_TOKENS
    cap = serve.kv_info(t)["max_num_seqs"]
    lim = max(1, min(int(log.get("kv_tokens", 0) // seq), cap))
    return lim, round(seq, 1)


def batch_list(lim, base=(1, 8, 32, 64)):
    out, b = set(base) | {lim}, 128
    while b < lim:
        out.add(b)
        b *= 2
    return sorted(out)


def sweep(t, smp, rows, batches, k):
    out = []
    for b in batches:
        if k + b > len(rows):
            out.append({"batch": b, "skipped": f"only {len(rows) - k} unused prompts left"})
            continue
        ps, k = [r["prompt"] for r in rows[k:k + b]], k + b
        smp.take()
        t0 = time.time()
        res = serve.generate(t, ps, {"max_tokens": OUT_TOKENS, "min_tokens": OUT_TOKENS, "ignore_eos": True,
                                     "stop": None})
        dt = time.time() - t0
        n_out, n_in = sum(r["n_out"] for r in res), sum(r["n_prompt"] for r in res)
        out.append({"batch": b, "wall_s": round(dt, 2), "out_tokens": n_out, "out_tok_per_s": round(n_out / dt, 1),
                    "per_seq_out_tok_per_s": round(n_out / dt / b, 2), "mean_prompt_tokens": round(n_in / b, 1),
                    "all_exact_len": all(r["n_out"] == OUT_TOKENS for r in res),
                    "render_errors": sum(r["finish"] == "render_error" for r in res), "peak": smp.take()})
        print(f"[capacity] batch {b}: {out[-1]['out_tok_per_s']} out tok/s peak {out[-1]['peak']}", flush=True)
    return out


def greedy(t, rows):
    res = serve.generate(t, [r["prompt"] for r in rows], {"temperature": 0.0, "max_tokens": OUT_TOKENS})
    return [{"skel_id": r["skel_id"], "finish": o["finish"], "n_out": o["n_out"], "thought": o["thought"],
             "raw": o["raw"]} for r, o in zip(rows, res)]


def load_block(t, g0, log_text):
    return {"load_s": t.load_s, "gpu_before": g0, "gpu_after": gpu(), "vllm_log": serve.log_facts(log_text),
            "frontend": serve.kv_info(t), "engine": {k: v for k, v in t.engine.items() if k != "tokenizer"}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prompts", "auto", "fp8"])
    ap.add_argument("--teacher", choices=sorted(serve.TEACHERS))
    ap.add_argument("--prompts")
    ap.add_argument("--out", help="prompts phase: the file to write")
    ap.add_argument("--shard-seed", default="capdry")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out-dir")
    ap.add_argument("--log", help="this process's own log (vLLM's load lines are read from it)")
    a = ap.parse_args(argv)
    if a.phase == "prompts":
        return make_prompts(a.out, a.shard_seed, a.n) or 0
    rows = read_prompts(a.prompts)
    cfg = serve.TEACHERS[a.teacher]
    path = os.path.join(a.out_dir, f"{a.teacher}.{a.phase}.dry.json")
    res = {"status": "dry", "what": f"teacher capacity run ({a.phase}) on FAKE renders; not training data",
           "teacher": a.teacher, "phase": a.phase, "repo": cfg["repo"], "revision": cfg["revision"],
           "license": cfg["license"], "quant": cfg["quant"], "wire": cfg["wire"], "complete": False,
           "started": time.strftime("%Y-%m-%dT%H:%M:%S")}

    def save():
        with open(path + ".tmp", "w") as f:
            json.dump(res, f, indent=1)
        os.replace(path + ".tmp", path)

    import torch
    import vllm
    res["versions"] = {"vllm": vllm.__version__, "torch": torch.__version__}
    over = {"disable_log_stats": False, **({"kv_cache_dtype": "fp8"} if a.phase == "fp8" else {})}
    # fp8 KV drops FlashAttention 2 from vLLM's candidates and it picks FLASHINFER, whose prefill kernel JIT-builds
    # with nvcc (absent here). A load that died on that alone gets one more try on TRITON_ATTN (no nvcc build).
    tries, t, res["attempts"] = [{}] + ([{"attention_backend": "TRITON_ATTN"}] if a.phase == "fp8" else []), None, []
    for extra in tries:
        pos = os.path.getsize(a.log) if a.log and os.path.exists(a.log) else 0
        g0 = gpu()
        try:
            t = serve.load(a.teacher, **over, **extra)
            res["attempts"].append({"engine_extra": extra, "works": True})
            break
        except Exception as e:
            text = log_since(a.log, pos)
            res["attempts"].append({"engine_extra": extra, "works": False, "error": f"{type(e).__name__}: {e}"[:800],
                                    "gpu_before": g0, "gpu_after": gpu(), "vllm_log": serve.log_facts(text),
                                    "error_lines": [ln[:300] for ln in text.splitlines() if "Error" in ln][-6:]})
            save()
            print(f"[capacity] {a.phase} load {extra or 'default'} failed: {res['attempts'][-1]['error'][:200]}",
                  flush=True)
            if "nvcc" not in text:
                break
            time.sleep(5)
    if t is None:
        # fp8 is final after its tries; a failed auto load stays incomplete, so the orchestrator retries it
        res.update({"works": False, "error": res["attempts"][-1]["error"], "complete": a.phase == "fp8"})
        save()
        return 3
    smp = Sampler(t.llm)
    try:
        res["works"], res["load"] = True, load_block(t, g0, log_since(a.log, pos))
        print(f"[capacity] loaded {a.teacher} ({a.phase}) in {t.load_s}s: {res['load']['vllm_log']}", flush=True)
        save()
        greedy_rows, warm, start = rows[5:13], rows[13:17], 17
        if a.phase == "auto":
            res["wire"] = wire(t, [TEMPTERS[0]] + [r["prompt"] for r in rows[:5]])
            res["thinking"] = thinking(t, rows[:5])
            print(f"[capacity] thinking off: {res['thinking']['pass']} markers {res['thinking']['with_marker']}",
                  flush=True)
            save()
            if a.teacher in thinking_control.SCORED:
                res["control"] = thinking_control.control(t)
                print(f"[capacity] control: pass {res['control']['pass']} fired {res['control']['fired']}", flush=True)
                save()
        serve.generate(t, [r["prompt"] for r in warm], {"max_tokens": 16})          # warm-up, not timed
        lim, seq = kv_limit(t, res["load"]["vllm_log"], rows[start:start + 64])
        batches = batch_list(lim) if a.phase == "auto" else sorted({lim, 64})
        res["kv_limit"] = {"seqs": lim, "seq_tokens": seq, "batches": batches}
        g = greedy(t, greedy_rows)
        res["greedy"] = {"runs": [g]}
        if a.phase == "auto":
            res["greedy"]["runs"].append(greedy(t, greedy_rows))
            res["greedy"]["self_agreement"] = agree(*res["greedy"]["runs"])
        else:
            auto_path = os.path.join(a.out_dir, f"{a.teacher}.auto.dry.json")
            if os.path.exists(auto_path):
                with open(auto_path) as f:
                    base = json.load(f).get("greedy", {}).get("runs")
                res["greedy"]["vs_auto"] = agree(base[0], g) if base else None
        res["greedy"]["thought_hits"] = sum(1 for r in g if r["thought"])
        save()
        res["sweep"] = sweep(t, smp, rows, batches, start)
        res["complete"] = True
    finally:
        res["peak_loaded"] = smp.stop()
        serve.unload(t)
        time.sleep(5)
        res["gpu_after_unload"] = gpu()
        res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        save()
    print(f"[capacity] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
