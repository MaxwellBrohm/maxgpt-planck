"""RC-12 item 14 (notes STEP 11, ITEM 14): two engines.json HF options for remote code, "nan_guard"
(hf_responder.guard_nans; Swen-28M, whose attention fills masked scores with -inf, so a left-padded batch turns NaN)
and "use_cache" (hf_responder.set_use_cache; cRia-LM-75M-Instruct, whose config.json switches its own KV cache off).
  toy        a two-layer toy attention stack written like Swen's (causal and padding masks filled with -inf, eager
             softmax; a few random weights, no checkpoint): left-padded, the padded row's output is NaN without the
             guard (the fixture really produces the defect), and with guard_nans("Attention") equals the row run alone
             (max abs diff <= 1e-5) while the unpadded row is unchanged; guard_nans counts the hooked modules, None
             hooks nothing, an unknown class or a non-string raises
  use_cache  set_use_cache sets config.use_cache for True / False, leaves it for None, refuses a non-bool
  plumbing   (test_chat_template's stubs) for each option: engines.json's value reaches its function through dev_batch
             -> runner for hfb and hf and is recorded in meta.json; it is refused on vllm and when malformed (nan_guard
             empty or not a string, use_cache not a bool); the parity tool applies nan_guard to the hfb stage only
             (the serial reference has no padding) and use_cache to both stages (a load option, like trust_remote_code
             and attn); verify C1 fails a run whose value is not engines.json's;
             rerun_tie's replay passes the run's value
  files      engines.json: Swen-28M hfb with nan_guard Attention in venv-doge; cRia hfb with use_cache true; both
             trust_remote_code; no other model carries either option
Run: python3 -B test_engine_patches.py   (exit 1 on any failure; needs torch on CPU, no GPU)"""
import contextlib
import io
import json
import os
import sys
import tempfile
import types

import test_chat_template as TC        # installs the stub transformers / torch / vllm (needed by the plumbing part)

fails = []


def check(cond, what):
    if not cond:
        fails.append(what)


def c_toy():
    stub = sys.modules.pop("torch")
    try:
        import torch
        from torch import nn
        import hf_responder as HR

        class Attention(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.q, self.k, self.v = (nn.Linear(d, d, bias=False) for _ in range(3))

            def forward(self, x, mask):
                s = self.q(x) @ self.k(x).transpose(-2, -1) / x.shape[-1] ** 0.5
                n = x.shape[1]
                s = s.masked_fill(torch.ones(n, n, dtype=torch.bool).triu(1), float("-inf"))
                s = s.masked_fill((mask == 0)[:, None, :], float("-inf"))
                return torch.softmax(s, dim=-1) @ self.v(x)

        class Toy(nn.Module):
            def __init__(self, d=8, vocab=11):
                super().__init__()
                self.emb, self.layers, self.out = nn.Embedding(vocab, d), nn.ModuleList([Attention(d), Attention(d)]), \
                    nn.Linear(d, vocab)

            def forward(self, ids, mask):
                h = self.emb(ids)
                for a in self.layers:
                    h = h + a(h, mask)
                return self.out(h)

        torch.manual_seed(0)
        with torch.no_grad():
            alone = Toy()
            row_b = torch.tensor([[3, 4, 5]])
            want_b = alone(row_b, torch.ones(1, 3))[0, -1]
            want_a = alone(torch.tensor([[1, 2, 3, 4, 5]]), torch.ones(1, 5))[0, -1]
            batch = torch.tensor([[1, 2, 3, 4, 5], [0, 0, 3, 4, 5]])
            mask = torch.tensor([[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]])
            bare = alone(batch, mask)
            check(bool(torch.isnan(bare[1, -1]).any()), "toy: the padded row is not NaN without the guard")
            check(HR.guard_nans(alone, "Attention") == 2, "toy: guard_nans did not hook the two Attention modules")
            got = alone(batch, mask)
            check(not bool(torch.isnan(got).any()), "toy: NaN with the guard")
            check(float((got[1, -1] - want_b).abs().max()) <= 1e-5, f"toy: padded row {got[1, -1]} != alone {want_b}")
            check(float((got[0, -1] - want_a).abs().max()) <= 1e-5, "toy: the unpadded row changed")
            fresh = Toy()
            check(HR.guard_nans(fresh, None) == 0 and bool(torch.isnan(fresh(batch, mask)).any()),
                  "toy: guard_nans(None) hooked something")
            for bad in ("Nope", "", 5):
                with contextlib.suppress(ValueError):
                    HR.guard_nans(Toy(), bad)
                    check(False, f"toy: guard_nans accepted {bad!r}")
    finally:
        sys.modules["torch"] = stub


OPTS = [("nan_guard", "guard_nans", "Attention", ("", 1)),     # option, hf_responder function, value, bad values
        ("use_cache", "set_use_cache", True, (1, "true"))]


def spy(fn):
    HR, seen = TC.VD.HR, []
    real = getattr(HR, fn)
    setattr(HR, fn, lambda model, v: seen.append(v) or 0)
    return seen, lambda: setattr(HR, fn, real)


def c_set_use_cache():
    HR = TC.VD.HR
    for v in (None, True, False):
        m = types.SimpleNamespace(config=types.SimpleNamespace(use_cache="as shipped"))
        check(HR.set_use_cache(m, v) == v and m.config.use_cache == ("as shipped" if v is None else v),
              f"set_use_cache {v}: {m.config.use_cache}")
    for bad in (1, "true"):
        with contextlib.suppress(ValueError):
            HR.set_use_cache(types.SimpleNamespace(config=types.SimpleNamespace()), bad)
            check(False, f"set_use_cache accepted {bad!r}")


def c_dev_batch():
    b = tempfile.mkdtemp(prefix="rc12_patch_")
    for opt, fn, val, bads in OPTS:
        seen, undo = spy(fn)
        try:
            for kind in ("hfb", "hf"):
                seen.clear()
                rc, meta = TC.run_dev(os.path.join(b, opt + kind), kind, {"engine": kind, opt: val})
                check(rc == 0 and seen == [val] and (meta or {}).get(opt) == val,
                      f"{opt} {kind}: called {seen} rc {rc} meta {meta and meta.get(opt)}")
            seen.clear()
            rc, meta = TC.run_dev(os.path.join(b, opt + "none"), "hfb", {"engine": "hfb"})
            check(rc == 0 and seen == [None] and meta is not None and meta.get(opt) is None, f"no {opt}: {seen}")
            for n, (kind, v) in enumerate([("vllm", val)] + [("hfb", x) for x in bads]):
                seen.clear()
                rc, meta = TC.run_dev(os.path.join(b, f"{opt}_bad{n}"), kind, {"engine": kind, opt: v})
                check(rc != 0 and (opt in str(rc) or "HF load options" in str(rc)), f"{opt} {kind} {v!r}: rc {rc}")
                check(not seen and not TC.CALLS and meta is None, f"{opt} {kind} {v!r}: loaded {seen} {TC.CALLS}")
        finally:
            undo()


def c_parity():
    PH = TC.PH
    for opt, fn, val, _ in OPTS:
        seen, undo = spy(fn)
        b = tempfile.mkdtemp(prefix="rc12_patch_par_")
        engines = os.path.join(b, "engines.json")
        json.dump({"models": {"org/M": {"engine": "hfb", opt: val}}}, open(engines, "w"))
        out = os.path.join(b, "run")
        os.makedirs(out)
        try:
            for stage in ("vllm", "hf"):                     # one out dir: stage hf replays stage vllm's prompts
                seen.clear()
                ns = types.SimpleNamespace(model="org/M", out=out, n=0, pick="spread", data=TC.R.DEV, dtype="float32",
                                           device="cpu", gpu_mem=0.85, max_model_len=None, engine="hfb",
                                           trust_remote_code=False, attn_implementation=None, engines=engines)
                getattr(PH, "stage_" + stage)(ns)
                both = opt == "use_cache"               # a load option of both stages; nan_guard: hfb only
                check(seen == ([val] if stage == "vllm" or both else [None]), f"parity {stage} stage {opt} {seen}")
            vm = json.load(open(os.path.join(out, "vllm_meta.json")))
            check(vm.get(opt) == val, f"parity meta {opt} {vm.get(opt)}")
        finally:
            undo()


def c_verify_rerun():
    VD = TC.VD
    import rerun_tie as RT
    base = dict(engine="hfb", decode=VD.HR.DECODE, audit={"hf_generation_config": {}}, conversations=0, limit=1,
                seconds=0, own_cf=False)
    for opt, _, val, _ in OPTS:
        b = tempfile.mkdtemp(prefix="rc12_patch_ver_")
        engines = os.path.join(b, "engines.json")
        json.dump({"models": {"org/M": {"engine": "hfb", opt: val}, "org/N": {"engine": "hfb"}}}, open(engines, "w"))
        for model, g, bad in (("org/M", val, False), ("org/M", None, True), ("org/N", None, False),
                              ("org/N", val, True)):
            VD.FAILS.clear()
            with contextlib.redirect_stdout(io.StringIO()):
                VD.c1_config("greedy", {"meta": dict(base, model=model, **{opt: g}), "rows": []}, engines, (0, 0))
            check(bool([f for f in VD.FAILS if opt in f]) == bad, f"C1 {opt} {model} {g}: {VD.FAILS}")
        got, real = [], TC.R.make_responder
        TC.R.make_responder = lambda spec, ns: got.append(getattr(ns, opt, "MISSING")) or (_ for _ in ()).throw(
            RuntimeError("stop"))
        try:
            with contextlib.suppress(RuntimeError):
                RT.replay_runs(types.SimpleNamespace(render="template", model="org/M", rerun=1),
                               {"greedy": {"meta": dict(base, model="org/M", seed=None, **{opt: val}), "rows": []}}, {})
        finally:
            TC.R.make_responder = real
        check(got == [val], f"rerun_tie replay {opt} {got}")


def c_files():
    e = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "engines.json")))["models"]
    sw, cr = e.get("Sensente/Swen-28M", {}), e.get("sz14/cRia-LM-75M-Instruct", {})
    check((sw.get("engine"), sw.get("nan_guard"), sw.get("trust_remote_code"), sw.get("python")) ==
          ("hfb", "Attention", True, "~/planck/venv-doge/bin/python"), f"engines.json Swen-28M {sw}")
    check((cr.get("engine"), cr.get("use_cache"), cr.get("trust_remote_code")) == ("hfb", True, True),
          f"engines.json cRia {cr}")
    other = [m for m, v in e.items() if m not in ("Sensente/Swen-28M", "sz14/cRia-LM-75M-Instruct")
             and ("nan_guard" in v or "use_cache" in v)]
    check(not other, f"nan_guard / use_cache on other models: {other}")


def main():
    for c in (c_toy, c_set_use_cache, c_dev_batch, c_parity, c_verify_rerun, c_files):
        c()
    print("\n".join(f"FAIL {f}" for f in fails) + "\n" * bool(fails) + "test_engine_patches: " +
          ("PASS" if not fails else f"{len(fails)} FAILURES"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
