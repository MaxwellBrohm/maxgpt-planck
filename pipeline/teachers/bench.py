"""Teacher prep bench on the PC (DRY: FAKE skeleton renders, nothing here is training data). One teacher per process:

    flock -w 7200 ~/planck/locks/gpu.lock python bench.py --teacher qwen3.5-9b --prompts prompts.jsonl \
        --out-dir ~/planck/runs/teachers --log ~/planck/logs/THIS.log

Steps, all with one load: (1) load through serve.load and read vLLM's own KV capacity line from --log;
(2) thinking check: 5 prompts (3 that invite step-by-step reasoning, 2 FAKE renders), outputs decoded with special
tokens kept, pass only if no thought marker or thought id appears; (3) throughput: for each batch size a DISJOINT
slice of FAKE render prompts (no prefix-cache reuse between sizes), exactly --out-tokens output tokens per prompt
(ignore_eos, no stop); (4) yield run: --yield-n FAKE renders with the pipeline's max_tokens and sampling and natural
stops, all submitted at once; raw outputs go to <teacher>.dry.outputs.jsonl for the checker on the Mac.
Writes <out-dir>/<teacher>.dry.json. The prompts file comes from the Mac (render_prompt.build on skeleton.shard)."""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import serve  # noqa: E402

TEMPTERS = [
    "What is 17 times 23? Think it through step by step before you answer.",
    "A bat and a ball cost 1.10 dollars in total. The bat costs one dollar more than the ball. How much is the ball? "
    "Reason carefully.",
    "Plan a three day trip to Lisbon. First think about what matters, then give the plan.",
]


def gpu_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        used, total, temp = [int(x) for x in out.strip().split(",")]
        return {"used_mib": used, "total_mib": total, "temp_c": temp}
    except Exception as e:  # a blind nvidia-smi must not end the bench
        return {"error": f"{type(e).__name__}: {e}"[:120]}


def read_prompts(path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(ln) for ln in f if ln.strip()]
    return rows


def thinking_check(t, fake):
    prompts = TEMPTERS + [r["prompt"] for r in fake[:2]]
    res = serve.generate(t, prompts, {"max_tokens": 400}, seeds=[101, 102, 103, 104, 105])
    rows = [{"prompt_head": p[:80], "finish": r["finish"], "n_out": r["n_out"], "thought": r["thought"],
             "text_head": r["raw"][:300]} for p, r in zip(prompts, res)]
    return {"pass": all(not r["thought"] and r["finish"] != "render_error" for r in res), "rows": rows}


def throughput(t, fake, batches, out_tokens, start):
    rows, k = [], start
    for b in batches:
        if k + b > len(fake):
            rows.append({"batch": b, "skipped": f"only {len(fake) - k} unused prompts left"})
            continue
        prompts = [r["prompt"] for r in fake[k:k + b]]
        k += b
        g0 = gpu_used_mib()
        t0 = time.time()
        res = serve.generate(t, prompts, {"max_tokens": out_tokens, "min_tokens": out_tokens, "ignore_eos": True,
                                          "stop": None})
        dt = time.time() - t0
        n_out = sum(r["n_out"] for r in res)
        n_in = sum(r["n_prompt"] for r in res)
        rows.append({"batch": b, "wall_s": round(dt, 2), "out_tokens": n_out, "prompt_tokens": n_in,
                     "out_tok_per_s": round(n_out / dt, 1), "total_tok_per_s": round((n_out + n_in) / dt, 1),
                     "per_seq_out_tok_per_s": round(n_out / dt / b, 2), "mean_prompt_tokens": round(n_in / b, 1),
                     "all_exact_len": all(r["n_out"] == out_tokens for r in res), "gpu_before": g0,
                     "gpu_after": gpu_used_mib()})
        print(f"[bench] batch {b}: {rows[-1]['out_tok_per_s']} out tok/s ({dt:.1f}s)", flush=True)
    return rows, k


def yield_run(t, rows, sampling, out_path):
    t0 = time.time()
    res = serve.generate(t, [r["prompt"] for r in rows], sampling, seeds=[r["seed"] for r in rows],
                         max_tokens=[r["max_tokens"] for r in rows])
    dt = time.time() - t0
    with open(out_path, "w", encoding="utf-8") as f:
        for r, o in zip(rows, res):
            f.write(json.dumps({"status": "dry", "skel_id": r["skel_id"], "teacher": t.name, **o}) + "\n")
    fin = {}
    for o in res:
        fin[o["finish"]] = fin.get(o["finish"], 0) + 1
    n_out = sum(o["n_out"] for o in res)
    return {"n": len(rows), "wall_s": round(dt, 1), "out_tokens": n_out, "out_tok_per_s": round(n_out / dt, 1),
            "finish": fin, "clamped": sum(o["clamped"] for o in res),
            "thought_hits": sum(1 for o in res if o["thought"]), "sampling": sampling, "outputs": out_path}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, choices=sorted(serve.TEACHERS))
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--log", default=None, help="this process's log file (vLLM's KV capacity line is read from it)")
    ap.add_argument("--batches", default="1,16,64,128,256")
    ap.add_argument("--out-tokens", type=int, default=200)
    ap.add_argument("--yield-n", type=int, default=200)
    ap.add_argument("--util", type=float, default=serve.ENGINE["gpu_memory_utilization"])
    ap.add_argument("--max-model-len", type=int, default=serve.ENGINE["max_model_len"])
    ap.add_argument("--max-num-seqs", type=int, default=None, help="default: the teacher's serve.TEACHERS engine value")
    ap.add_argument("--max-num-batched-tokens", type=int, default=None)
    ap.add_argument("--kv-bytes", type=int, default=None, help="vLLM kv_cache_memory_bytes: a fixed KV size instead of "
                    "the util-based profile (used when device-wide profiling leaves no KV room)")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    rows = read_prompts(a.prompts)
    yrows, fake = rows[:a.yield_n], rows[a.yield_n:]
    cfg = serve.TEACHERS[a.teacher]
    res = {"status": "dry", "what": "teacher prep bench on FAKE skeleton renders; not training data",
           "teacher": a.teacher, "repo": cfg["repo"], "revision": cfg["revision"], "license": cfg["license"],
           "quant": cfg["quant"], "wire": cfg["wire"], "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "gpu_before_load": gpu_used_mib()}
    import vllm
    import torch
    res["versions"] = {"vllm": vllm.__version__, "torch": torch.__version__}
    out_json = os.path.join(a.out_dir, f"{a.teacher}.dry.json")

    def save():
        with open(out_json + ".tmp", "w") as f:
            json.dump(res, f, indent=1)
        os.replace(out_json + ".tmp", out_json)

    over = {k: v for k, v in (("max_num_seqs", a.max_num_seqs), ("max_num_batched_tokens",
                                                                  a.max_num_batched_tokens),
                                          ("kv_cache_memory_bytes", a.kv_bytes)) if v is not None}
    t = serve.load(a.teacher, gpu_memory_utilization=a.util, max_model_len=a.max_model_len, **over)
    try:
        res["engine"] = {k: v for k, v in t.engine.items() if k != "tokenizer"}
        res["load_s"] = t.load_s
        res["gpu_after_load"] = gpu_used_mib()
        res["kv_frontend"] = serve.kv_info(t)
        if a.log and os.path.exists(a.log):
            with open(a.log, encoding="utf-8", errors="replace") as f:
                res["kv_vllm_log"] = serve.kv_from_log(f.read())
        print(f"[bench] loaded in {t.load_s}s kv={res.get('kv_vllm_log')}", flush=True)
        save()
        res["thinking"] = thinking_check(t, fake)
        print(f"[bench] thinking off: {res['thinking']['pass']}", flush=True)
        save()
        serve.generate(t, [r["prompt"] for r in fake[:4]], {"max_tokens": 16})      # warm-up, not timed
        res["throughput"], _ = throughput(t, fake, [int(x) for x in a.batches.split(",")], a.out_tokens, start=2)
        save()
        res["yield_run"] = yield_run(t, yrows, dict(cfg["sampling"]),
                                     os.path.join(a.out_dir, f"{a.teacher}.dry.outputs.jsonl"))
        print(f"[bench] yield run: {res['yield_run']['out_tok_per_s']} out tok/s", flush=True)
        res["gpu_end_loaded"] = gpu_used_mib()
    finally:
        serve.unload(t)
        time.sleep(5)
        res["gpu_after_unload"] = gpu_used_mib()
        res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        save()
    print(f"[bench] wrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
