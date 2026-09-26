"""RC-12 PC job scripts (notes STEP 9). Prints a bash script for the WSL side of the PC; send it with the session's
detach helper. The script logs to ~/planck/logs/<job>.log, runs from the synced code dir ~/planck/dev/<code>/
(the rc12 directory, synced without runs/), holds the GPU lock (flock -w 7200 ~/planck/locks/gpu.lock) only for
its own model load and run, and touches ~/planck/logs/<job>.DONE at the end. No machine names or paths from
pc/local.env are written.
  python3 pc_jobs.py probe                                  vLLM / HF API checks, no model, no lock
  python3 pc_jobs.py parity <model id> [--n 20]            parity_hf_vllm.py --stage all -> <runs>/<model>/parity
  python3 pc_jobs.py dev <vllm:id | hf:id> --render template [--seeds greedy,1,2,3] [--limit N] [--extra "..."]
Common: --code rc12_s9 (the synced dir name), --runs ~/planck/runs/rc12_dev, --venv ~/planck/venv-vllm."""
import argparse
import shlex

import dev_batch as DB

HEAD = """#!/bin/bash
exec > ~/planck/logs/{job}.log 2>&1
export PATH="/usr/lib/wsl/lib:$PATH" HF_HOME=~/planck/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
PY={venv}/bin/python
cd ~/planck/dev/{code} || {{ echo "no code dir"; touch ~/planck/logs/{job}.DONE; exit 1; }}
mkdir -p ~/planck/locks {runs}
echo "$(date -Is) {job}: start"
"""
LOCKED = """echo "$(date -Is) {job}: waiting for the GPU lock"
flock -w 7200 ~/planck/locks/gpu.lock $PY -B {cmd}
rc=$?
echo "$(date -Is) {job}: exit $rc"
touch ~/planck/logs/{job}.DONE
"""
PROBE = r'''$PY -B - <<'PY'
import inspect, json
import vllm, transformers, torch
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
import hf_responder as HR, vllm_responder as VR
print("versions", vllm.__version__, transformers.__version__, torch.__version__)
for seed in (None, 3):
    kw = VR.sampling_kwargs(seed, "rc12-dev-x", 4, [2, 7])
    sp = SamplingParams(**kw)
    print("SamplingParams ok", {k: getattr(sp, k) for k in kw})
fields = set(inspect.signature(EngineArgs).parameters)
for k in ("generation_config", "seed", "max_model_len", "gpu_memory_utilization", "dtype", "trust_remote_code"):
    print("EngineArgs", k, k in fields)
print("LLM.generate", "use_tqdm" in inspect.signature(LLM.generate).parameters)
print("from_pretrained torch_dtype kw", "torch_dtype" in inspect.signature(transformers.PreTrainedModel.from_pretrained).parameters)
PY
echo "$(date -Is) {job}: exit $?"
touch ~/planck/logs/{job}.DONE
'''


def script(job, body, a):
    return HEAD.format(job=job, venv=a.venv, code=a.code, runs=a.runs) + body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["probe", "parity", "dev"])
    ap.add_argument("target", nargs="?")
    ap.add_argument("--code", default="rc12_s9")
    ap.add_argument("--runs", default="~/planck/runs/rc12_dev")
    ap.add_argument("--venv", default="~/planck/venv-vllm")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--render", choices=["template", "plain"], default=None)
    ap.add_argument("--seeds", default="greedy,1,2,3")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--extra", default="", help="more dev_batch.py / parity flags, e.g. \"--gpu-mem 0.7\"")
    a = ap.parse_args()
    if a.kind == "probe":
        print(script("rc12_probe", PROBE.replace("{job}", "rc12_probe"), a), end="")
        return 0
    if not a.target:
        ap.error("a model id (parity) or responder spec (dev) is needed")
    extra = shlex.split(a.extra)
    if a.kind == "parity":
        model = a.target
        job = f"rc12_parity_{DB.slug(model)}"
        args = ["parity_hf_vllm.py", "--model", model, "--out", f"{a.runs}/{DB.slug(model)}/parity", "--n", str(a.n)]
    else:
        if not a.render:
            ap.error("dev needs --render")
        model = a.target.partition(":")[2]
        job = f"rc12_dev_{DB.slug(model)}_{a.render}"
        args = ["dev_batch.py", "--responder", a.target, "--render", a.render, "--seeds", a.seeds, "--root", a.runs]
        args += ["--limit", str(a.limit)] if a.limit else []
    cmd = " ".join(x if x.startswith("~/") else shlex.quote(x) for x in args + extra)
    print(script(job, LOCKED.format(job=job, cmd=cmd), a), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
