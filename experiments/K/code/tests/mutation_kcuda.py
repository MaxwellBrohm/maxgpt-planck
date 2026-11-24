"""Mutants for tests/test_k_cuda.py (K3 round 16, 2026-11-22), on mutation_k.py's runner. PC ONLY for the first two:
each breaks a path only CUDA exercises, so where CUDA is missing the kit's cuda cases skip and both survive by design;
the last two are controls the kit's cpu cases kill as well. The PC runs this list after the kit passes (code/notes.txt
K3 ROUND 16, PC PLAN step 2) and expects all four killed.
  python tests/mutation_kcuda.py [--id NAME ...] [--list] [--baseline] [--why] [--log FILE]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mutation_k as MK  # noqa: E402

KC = "test_k_cuda.py"
M = [
    ("kc_route_mask_on_cpu", "kroute.py", "self.ctx.mask = torch.isin(src, self.fact).to(idx.device)",
     "self.ctx.mask = torch.isin(src, self.fact)", [KC],
     "the routed fact mask stays on the CPU while the run is on CUDA"),
    ("kc_route_grad_dtype", "kroute.py", "return gh.to(h.dtype), gw.to(w.dtype), None, None",
     "return gh, gw, None, None", [KC], "the routed backward returns autocast-dtype gradients for fp32 weights"),
    ("kc_route_grad_unmasked_down", "kroute.py", "hm = (h * allowed).to(gy.dtype)", "hm = h.to(gy.dtype)", [KC],
     "control: W_down gets every position's gradient (the CPU lists kill it too: s14_down_grad_unmasked)"),
    ("kc_probe_draws_before_its_guard", "kprobe.py", "        cpu = torch.get_rng_state()\n",
     "        torch.rand(1)\n        cpu = torch.get_rng_state()\n", [KC],
     "control: the hook draws from the torch RNG where its own guard cannot see it"),
]


def main(argv=None) -> int:
    MK.M = M
    return MK.main(argv)


if __name__ == "__main__":
    sys.exit(main())
