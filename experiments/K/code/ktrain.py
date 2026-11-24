"""Track K's training entry (SPEC 14's harness pieces, built K-side: no harness file is edited).

  python ktrain.py path/to/run_dir/config.yaml [harness train.py arguments: --device, --max-steps, ...]

Runs harness/train.py's main on the config. The config's top-level k: block (harness train.py ignores it) selects:
  probe: {every, files}         K-PROBE in-training hook (kprobe.py; SPEC 14 eval.k_probe)
  init_weights: {file, sha256}  weights-only start (K3 continuations): the model's weights from FILE (prune.py's
                                output), a fresh optimizer and a fresh loader at the config's seed / data_seed; a
                                resume from out_dir's own checkpoint still wins (it holds the run's state)
  grad_route: {sources, reserved}  K3-R (kroute.py; SPEC 14 train.grad_route)
  wd_mult: {mlp: x}             K4b (kopt.py; SPEC 14's per-group weight-decay multiplier)
Every key is off when absent. With no k: block (or an empty one) nothing is patched and the run is harness train.py's
bit for bit (tests/test_ktrain.py). Otherwise train.main runs with train.build_loader, train.make_optimizer and
train.Trainer wrapped for that call only, and one record goes to out_dir/k_start.jsonl per start: the k block, its
sha256, the config's sha256 (train.py's rule), what was installed and the init file's sha256. The checkpoints'
payload gains "k_sha256" (and "k_init_weights_sha256"). An unknown key is refused before anything runs, as are
any train.compile but eager, with or without a k: block (Track K runs eager: code/notes.txt K COMPILE DECISION),
grad_route outside data.mode pack or with two data sources of one name, and init_weights with schedule.init_from.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time

import kcommon as K

if K.HARNESS not in sys.path:
    sys.path.insert(0, K.HARNESS)

KEYS = ("probe", "init_weights", "grad_route", "wd_mult")


def _abs(p: str, base: str) -> str:
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))


def k_sha(kc: dict) -> str:
    return hashlib.sha256(json.dumps(kc, sort_keys=True, default=str).encode()).hexdigest()


def check(cfg: dict) -> dict:
    """The config's k: block, validated; {} when absent or empty."""
    kc = cfg.get("k") or {}
    if not isinstance(kc, dict):
        raise ValueError("k: must be a mapping")
    bad = sorted(set(kc) - set(KEYS))
    if bad:
        raise ValueError(f"k: unknown keys {bad} (known: {list(KEYS)})")
    kc = {k: v for k, v in kc.items() if v not in (None, {}, [])}
    tc, sc, dc = cfg.get("train") or {}, cfg.get("schedule") or {}, cfg.get("data") or {}
    from model import compile_mode     # harness: false, null, "off", "none", "eager" are eager
    if compile_mode(tc.get("compile", False)) is not None:     # every K run, k: block or not (REVIEW 14c D3)
        raise ValueError("Track K runs eager (code/notes.txt K COMPILE DECISION): set train.compile false")
    if "grad_route" in kc:
        gr = kc["grad_route"]
        if sorted(gr) != ["reserved", "sources"] or not gr["sources"]:
            raise ValueError("k.grad_route needs exactly sources (a non-empty list) and reserved")
        if dc.get("mode", "pack") != "pack":
            raise ValueError("k.grad_route needs data.mode pack")
        names = [s.get("name", s.get("kind")) for s in dc.get("sources", [])]
        miss = [s for s in gr["sources"] if s not in names]
        if miss or len(set(names)) != len(names):
            raise ValueError(f"k.grad_route.sources {miss} not among the data sources {names} (names must be unique)")
    if "init_weights" in kc:
        iw = kc["init_weights"]
        if sorted(iw) != ["file", "sha256"]:
            raise ValueError("k.init_weights needs exactly file and sha256")
        if sc.get("init_from"):
            raise ValueError("k.init_weights and schedule.init_from both set: a weights-only start or a branch")
    if "wd_mult" in kc:
        import kopt
        kopt.check(kc["wd_mult"])
    return kc


def load_init(path: str, sha256: str) -> dict:
    import torch
    raw = open(path, "rb").read()
    got = hashlib.sha256(raw).hexdigest()
    if got != sha256:
        raise ValueError(f"k.init_weights {path} has sha256 {got}, the config pins {sha256}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if set(ck) - {"model", "model_cfg", "k_prune"} or "model" not in ck or "model_cfg" not in ck:
        raise ValueError(f"k.init_weights {path}: expected a prune.py weights file")
    return {**ck, "sha256": got}


@contextlib.contextmanager
def installed(cfg: dict, kc: dict, base: str):
    """Patch train.build_loader, train.make_optimizer and train.Trainer for one train.main call."""
    import train
    import kprobe
    import kroute
    import kopt
    saved = (train.build_loader, train.make_optimizer, train.Trainer)
    names = [s.get("name", s.get("kind")) for s in cfg["data"].get("sources", [])]
    info: dict = {}

    def build_loader(*a, **kw):
        ld = saved[0](*a, **kw)
        return kroute.as_routed(ld) if "grad_route" in kc else ld

    def make_optimizer(model, ocfg, device_type="cpu", extra=None):
        opt = saved[1](model, ocfg, device_type, extra)
        if "wd_mult" in kc:
            info["wd_mult"] = kopt.split_mlp_wd(opt, model, kopt.check(kc["wd_mult"])["mlp"])
        return opt

    class KTrainer(saved[2]):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.meta["k_sha256"] = k_sha(kc)
            if "init_weights" in kc:
                iw = load_init(_abs(kc["init_weights"]["file"], base), kc["init_weights"]["sha256"])
                if iw["model_cfg"] != self.meta["model_cfg"]:
                    raise ValueError("k.init_weights holds another model shape")
                self.model.load_state_dict(iw["model"])
                self.meta["k_init_weights_sha256"] = info["init_weights_sha256"] = iw["sha256"]
            if "grad_route" in kc:
                gr = kc["grad_route"]
                ctx = kroute.RouteCtx()
                info["grad_route"] = {"reserved": {str(k): [v[0], v[-1]] for k, v in
                                                   kroute.install(self.model, int(gr["reserved"]), ctx).items()},
                                      "fact_sources": [names.index(s) for s in gr["sources"]]}
                self.fwd = kroute.RoutedForward(self.fwd, self.loader, ctx, info["grad_route"]["fact_sources"])
            if "probe" in kc:
                hook = kprobe.KProbe(kc["probe"], base, self.out_dir, self.device)
                self.hooks.append(hook)
                info["probe"] = {"every": hook.every, "n": len(hook.recs), "file_sha256": hook.file_sha256}
            import runio
            runio.append_jsonl(os.path.join(self.out_dir, "k_start.jsonl"), {
                "event": "k_start", "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "k": kc, "k_sha256": k_sha(kc),
                "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest(),
                "installed": info})

    train.build_loader, train.make_optimizer, train.Trainer = build_loader, make_optimizer, KTrainer
    try:
        yield info
    finally:
        train.build_loader, train.make_optimizer, train.Trainer = saved


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    a, _ = ap.parse_known_args(argv)
    import runio
    import train
    cfg_path = os.path.abspath(a.config)
    cfg = runio.load_yaml(cfg_path)
    try:
        kc = check(cfg)
    except ValueError as e:
        print(f"[ktrain] {e}", file=sys.stderr, flush=True)
        return 2
    if not kc:
        return train.main(argv)
    with installed(cfg, kc, os.path.dirname(cfg_path)):
        return train.main(argv)


if __name__ == "__main__":
    sys.exit(main())
