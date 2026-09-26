"""E006 chat probe of ONE model on CUDA (notes.txt; run_chat.py and battery.py as copied from E005, unedited: greedy,
the same 53 conversations and caps). Run on the PC only, through guard_e006_pc.py under gpu.lock.
  numerics_e006.apply() first; run_chat.py runs unchanged (runpy) into <out>.partial; then every conversation
  record is written once to <out> with weights_sha256 and tag added, and the partial file is removed.
  The weights hash: <dir>/model.safetensors for a weights directory, the pinned snapshot's file for the HF id.
usage: chat_e006.py <weights dir (relative) | HuggingFaceTB/SmolLM2-135M-Instruct> --tag T [--only IDS] [--tf32]
writes ../transcripts/<slug>__<tag>__greedy.jsonl (refuses if it exists)."""
import argparse
import json
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
SMOL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="chat_e006.py")
    ap.add_argument("model")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tf32", action="store_true")
    a = ap.parse_args(argv)
    out = os.path.join(EXP, "transcripts", f"{SMOL.replace('/', '__')}__{a.tag}__greedy.jsonl")
    part = out + ".partial"
    if os.path.exists(out) or os.path.exists(part):
        sys.exit(f"refusing: {out} (or its .partial) exists; transcripts are never overwritten")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    import numerics_e006 as NU
    import e006_score as SC
    NU.apply(a.tf32)
    if os.path.isdir(a.model):
        wfile = os.path.join(a.model, "model.safetensors")
    else:
        import params as PR
        wfile = os.path.join(PR.snapshot_dir(a.model), "model.safetensors")
    wsha = SC.sha_file(wfile)
    args = [os.path.join(HERE, "run_chat.py"), a.model, "--mode", "greedy", "--device", a.device, "--out", part]
    if a.only:
        args += ["--only", a.only]
    sys.argv = args
    runpy.run_path(args[0], run_name="__main__")
    n = 0
    with open(out, "x") as f:
        for line in open(part):
            if line.strip():
                f.write(json.dumps(dict(json.loads(line), weights_sha256=wsha, tag=a.tag)) + "\n")
                n += 1
    os.remove(part)
    print(f"chat_e006 DONE {a.tag}: {n} conversations, weights_sha256={wsha[:16]} -> {os.path.relpath(out, EXP)}",
          flush=True)


if __name__ == "__main__":
    main()
