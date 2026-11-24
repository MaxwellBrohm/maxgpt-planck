"""K3's reset on a checkpoint (K3 notes RESET AND REGROW, K3-R; SPEC 14 prune.py): writes the weights-only file a K3
continuation starts from (ktrain.py k.init_weights). The reset itself is ablate.reset_units.

  python prune.py --ckpt K1_FH_s201/final.pt --units pf_s201.json --seed 7901 --out pf_s201.pt   (P-F, P-R, P-S)
  python prune.py --ckpt K3R_s201/final.pt --block-last 122 --seed 7901 --out rcut_s201.pt        (K3-R's block)
  python prune.py --ckpt K1_F0_s201/final.pt --out f0cont_s201.pt                                 (P-0, F0-cont)

--units: a JSON {layer: [unit, ...]} (K2 HANDOFF's set for that seed); --block-last n: the last n units of every
layer; neither: no reset (the weights as they are). --seed is K3's generator seed 7700 + s (required with a reset).
The file holds {"model": state_dict, "model_cfg", "k_prune": {source checkpoint sha256, step, units per layer, seed,
the weights' sha256 before and after}}; nothing of the optimizer, the loader or the RNG goes in, so the continuation
starts with fresh optimizer state. Prints the file's sha256 (the config pins it).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import kcommon as K

if K.HARNESS not in sys.path:
    sys.path.insert(0, K.HARNESS)


def units_of(model, units_json: str | None, block_last: int | None) -> dict:
    if units_json and block_last:
        raise ValueError("--units and --block-last are exclusive")
    H = {i: b.mlp.gate_proj.out_features for i, b in enumerate(model.blocks) if getattr(b, "mlp", None) is not None}
    if block_last:
        import kroute
        return {i: kroute.reserved_units(h, block_last) for i, h in H.items()}
    if not units_json:
        return {}
    raw = json.load(open(units_json))
    out = {}
    for k, v in raw.items():
        li, us = int(k), sorted(set(int(u) for u in v))
        if li not in H or (us and not 0 <= us[0] <= us[-1] < H[li]) or len(us) != len(v):
            raise ValueError(f"--units: layer {k} units {v} do not fit the model (or repeat)")
        if us:
            out[li] = us
    return out


def prune(ckpt: str, out: str, units_json: str | None = None, block_last: int | None = None,
          seed: int | None = None) -> dict:
    import torch
    import ablate
    import probes as PR
    import runio
    from config import PlanckConfig
    from model import build_model
    ck = runio.load_checkpoint(ckpt)
    model = build_model(PlanckConfig.from_dict(ck["model_cfg"]))
    model.load_state_dict(ck["model"])
    units = units_of(model, units_json, block_last)
    if units and seed is None:
        raise ValueError("a reset needs --seed (K3: 7700 + s)")
    before = PR.weights_sha(model)
    if units:
        ablate.reset_units(model, units, int(seed))
    info = {"source_sha256": K.sha256_file(ckpt), "source_step": int(ck["step"]),
            "units": {str(k): v for k, v in sorted(units.items())}, "n_units": sum(len(v) for v in units.values()),
            "seed": None if seed is None else int(seed), "weights_before": before, "weights_after": PR.weights_sha(model)}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.save({"model": model.state_dict(), "model_cfg": ck["model_cfg"], "k_prune": info}, out)
    return {**info, "sha256": hashlib.sha256(open(out, "rb").read()).hexdigest(), "out": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--units")
    ap.add_argument("--block-last", type=int)
    ap.add_argument("--seed", type=int)
    a = ap.parse_args(argv)
    r = prune(a.ckpt, a.out, a.units, a.block_last, a.seed)
    print(json.dumps({k: r[k] for k in ("sha256", "n_units", "seed", "source_step", "weights_after")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
