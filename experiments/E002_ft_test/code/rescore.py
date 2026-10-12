"""Score extra eval sets for ONE model (an HF id or a saved fine-tuned weights dir) in ONE process.
Run only through guard.py. Used for the post-hoc crossed control on runs that predate it.
usage: rescore.py <model_id> <weights_dir or 'hf'> --tag TAG --sets cross [--device mps]
Writes ../out/<slug of model_id>__<TAG>__<set>__<render>.jsonl (same format as ft_test.py).
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, json, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ft_test as FT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("weights")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sets", default="cross")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    if a.device == "mps" and (os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO") != "0.7" or
                              os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO") != "0.6"):
        sys.exit("refusing: MPS watermarks must be 0.7/0.6; run through guard.py")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    src = a.model if a.weights == "hf" else a.weights
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32, low_cpu_mem_usage=True).to(a.device).eval()
    mem = FT.Mem(a.device)
    want = set(a.sets.split(","))
    stem = os.path.join(FT.OUT, f"{FT.slug(a.model)}__{a.tag}")
    for set_name, render, its in FT.eval_sets(False):
        if set_name not in want:
            continue
        with open(f"{stem}__{set_name}__{render}.jsonl", "w") as f:
            recs, dt = FT.score_set(model, tok, a.model, set_name, render, its, a.device, f, mem)
        print(f"scored {set_name}/{render}: {len(recs)} items in {dt:.0f}s {mem.s()}", flush=True)
    print("DONE", a.model, a.tag, flush=True)


if __name__ == "__main__":
    main()
