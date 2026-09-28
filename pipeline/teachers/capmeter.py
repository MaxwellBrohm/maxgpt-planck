"""Measurement helpers for capacity.py (DRY; nothing here is training data): nvidia-smi and vLLM-metrics peak
sampling, the FAKE prompt file (written and read back only as dry FAKE renders), per-load log slices, and the
text agreement of two greedy runs. No vLLM import here; the Sampler only calls the llm object it is given."""
import json
import os
import subprocess
import threading


GAUGES = {"vllm:num_requests_running": "running", "vllm:num_requests_waiting": "waiting",
          "vllm:kv_cache_usage_perc": "kv_usage"}


def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        used, temp = [int(x) for x in out.strip().splitlines()[0].split(",")]
        return {"used_mib": used, "temp_c": temp}
    except Exception as e:  # a blind nvidia-smi must not end the run
        return {"error": f"{type(e).__name__}: {e}"[:120]}


class Sampler:
    """peaks of device MiB / temperature (nvidia-smi) and vLLM's running, waiting and KV-usage gauges."""
    def __init__(self, llm, period=0.5):
        self.llm, self.period, self.win, self.whole, self.err = llm, period, {}, {}, None
        self.lock, self.halt = threading.Lock(), threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        while not self.halt.is_set():
            g, vals = gpu(), {}
            if "used_mib" in g:
                vals = {"mib": g["used_mib"], "temp_c": g["temp_c"]}
            try:
                for m in self.llm.get_metrics():
                    k = GAUGES.get(m.name)
                    if k and getattr(m, "value", None) is not None:
                        vals[k] = max(vals.get(k, 0), round(float(m.value), 4))
            except Exception as e:  # metrics are a bonus; nvidia-smi peaks still count
                self.err = f"{type(e).__name__}: {e}"[:200]
            with self.lock:
                for d in (self.win, self.whole):
                    for k, v in vals.items():
                        d[k] = max(d.get(k, v), v)
            self.halt.wait(self.period)

    def take(self):
        with self.lock:
            w, self.win = self.win, {}
        return w

    def stop(self):
        self.halt.set()
        self.th.join(10)
        self.llm = None                 # no reference left to keep the engine alive past serve.unload
        return {**self.whole, "metrics_error": self.err}


def is_fake(r):
    """a dry render of the FAKE banks: provenance 'FAKE' when the row says, else render_prompt's FAKE-only variant
    name (the 700-row file of 2026-09-26 01:06 on the PC carries status and variant 'instr.fake.N' only)."""
    if r.get("status") != "dry":
        return False
    if "provenance" in r:
        return r["provenance"] == "FAKE"
    return str(r.get("variant", "")).startswith("instr.fake.")


def read_prompts(path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(ln) for ln in f if ln.strip()]
    if not rows or not all(is_fake(r) for r in rows):
        raise SystemExit(f"{path}: every row must be a dry FAKE render")
    return rows


def make_prompts(out, seed, n):
    import render_prompt as R
    import skeleton
    with open(out + ".tmp", "w", encoding="utf-8") as f:
        for k in skeleton.shard(seed, n):
            b = R.build(k)
            if b["provenance"] != "FAKE":
                raise SystemExit(f"{k['skel_id']}: provenance {b['provenance']!r}, not FAKE")
            f.write(json.dumps({"status": "dry", "provenance": "FAKE", "variant": b["variant"], "skel_id": k["skel_id"],
                                "prompt": b["prompt"], "max_tokens": b["max_tokens"]}) + "\n")
    os.replace(out + ".tmp", out)
    print(f"[capacity] wrote {n} FAKE renders to {out}", flush=True)


def log_since(path, pos):
    if not path or not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        f.seek(pos)
        return f.read()


def agree(a, b):
    """text agreement of two greedy runs: identical outputs, mean shared-prefix fraction (chars)."""
    pre = []
    for x, y in zip(a, b):
        n = next((i for i, (c, d) in enumerate(zip(x["raw"], y["raw"])) if c != d), min(len(x["raw"]), len(y["raw"])))
        pre.append(n / max(1, len(x["raw"]), len(y["raw"])))
    return {"n": len(pre), "identical": sum(x["raw"] == y["raw"] for x, y in zip(a, b)),
            "mean_prefix_frac": round(sum(pre) / len(pre), 3) if pre else None}
