"""RC-12 HF load options from engines.json (trust_remote_code, attn_implementation) with stub transformers / torch
modules (no model, no GPU; notes STEP 9c).
  dev_batch  engines.json "trust_remote_code": true and "attn_implementation": "eager" for an hfb model reach
             AutoTokenizer.from_pretrained (trust only) and AutoModelForCausalLM.from_pretrained (both) through
             dev_batch -> runner.make_responder -> HFBatched -> HFResponder; meta.json records both and
             sys.executable, engine_audit the attention as loaded; serial hf takes them too; a model without the
             fields loads with trust_remote_code False and NO attn_implementation argument (the library default);
             a non-boolean trust value is False; either field on a vllm model is refused before anything loads;
             "max_batch" reaches HFBatched (absent: hf_batched.MAX_BATCH) and meta.json, and is refused on hf,
             on vllm and when not a positive int
  parity     parity_hf_vllm passes both to both HF stages and forwards them from stage all; --trust-remote-code
             with --engine vllm is refused
  verify     verify_dev_runs --rerun rebuilds the engine with the run's meta.json options
The stubs replace transformers and torch in sys.modules before anything imports them, so no real library loads.
Run: python3 -B test_hf_trust.py   (exit 1 on any failure)"""
import json
import os
import sys
import tempfile
import types

CALLS = []
NO = "ABSENT"


class Tok:
    chat_template, eos_token_id, pad_token_id = "stub", 1, 2

    def get_vocab(self):
        return {"<|end_of_text|>": 1, "<|eot_id|>": 6}

    def decode(self, ids, **kw):
        return ""


class Model:
    generation_config = None

    def __init__(self, attn):
        self.config = types.SimpleNamespace(model_type="doge", max_position_embeddings=2048,
                                            _attn_implementation="sdpa" if attn == NO else attn)

    def to(self, device):
        return self

    def eval(self):
        return self


def loader(kind):
    def from_pretrained(model_id, **kw):
        attn = kw.get("attn_implementation", NO)
        CALLS.append((kind, model_id, kw.get("trust_remote_code", NO), attn))
        return Tok() if kind == "tok" else Model(attn)
    return types.SimpleNamespace(from_pretrained=from_pretrained)


def install_stubs():
    tf = types.ModuleType("transformers")
    tf.AutoTokenizer, tf.AutoModelForCausalLM, tf.__version__ = loader("tok"), loader("model"), "stub"
    torch = types.ModuleType("torch")
    torch.bfloat16, torch.float32, torch.__version__ = "bf16", "fp32", "stub"
    sys.modules["transformers"], sys.modules["torch"] = tf, torch


install_stubs()
import dev_batch as DB  # noqa: E402
import fakes_family as FF  # noqa: E402
import hf_batched as HB  # noqa: E402
import lockstep as LS  # noqa: E402
import parity_hf_vllm as PH  # noqa: E402
import runner as R  # noqa: E402

fails = []
BUILT = []


def check(cond, what):
    if not cond:
        fails.append(what)


def fake_engine(spec, ns):
    """the real make_responder (stub libraries), recorded; then a fake plays so the run completes without torch."""
    eng, name = REAL_MAKE(spec, ns)
    BUILT.append((type(eng).__name__, getattr(eng, "trust_remote_code", NO), getattr(eng, "attn", NO),
                  getattr(eng, "max_batch", NO)))
    return LS.PerConv(lambda: FF.make("IDEAL"), ns.render), name


REAL_MAKE = R.make_responder


def run_dev(base, kind, entry):
    os.makedirs(base)
    engines = os.path.join(base, "engines.json")
    json.dump({"models": {"org/Doge": entry}}, open(engines, "w"))
    root = os.path.join(base, "root")
    CALLS.clear()
    BUILT.clear()
    old = sys.argv
    sys.argv = ["dev_batch.py", "--responder", f"{kind}:org/Doge", "--render", "template", "--seeds", "greedy",
                "--root", root, "--engines", engines, "--hf-untested-ok", "--limit", "2", "--device", "cpu"]
    R.make_responder = fake_engine
    try:
        rc = DB.main()
    except SystemExit as e:
        rc = str(e.code)
    finally:
        sys.argv, R.make_responder = old, REAL_MAKE
    p = os.path.join(root, "Doge", "template", "greedy", "meta.json")
    return rc, (json.load(open(p)) if os.path.exists(p) else None)


def c_dev_batch():
    b = tempfile.mkdtemp(prefix="rc12_trust_")
    doge = {"engine": "hfb", "python": "~/planck/venv-doge/bin/python", "trust_remote_code": True,
            "attn_implementation": "eager", "max_batch": 32}
    hf = {k: v for k, v in doge.items() if k != "max_batch"}
    cases = (("hfb", doge, "doge", True, "eager", 32),
             ("hf", dict(hf, engine="hf"), "serial hf", True, "eager", None),
             ("hfb", {"engine": "hfb", "attn_implementation": "eager"}, "attn only", False, "eager", None),
             ("hf", {"engine": "hf"}, "absent, hf", False, None, None),
             ("hfb", {"engine": "hfb"}, "absent, hfb", False, None, None),
             ("hfb", {"engine": "hfb", "trust_remote_code": "yes"}, "string", False, None, None))
    for kind, entry, what, t, a, mb in cases:
        rc, meta = run_dev(os.path.join(b, what.replace(", ", "_").replace(" ", "_")), kind, entry)
        want = "HFBatched" if kind == "hfb" else "HFResponder"
        check(rc == 0 and CALLS == [("tok", "org/Doge", t, NO), ("model", "org/Doge", t, NO if a is None else a)],
              f"{what}: from_pretrained {rc} {CALLS}")
        check(BUILT == [(want, t, a or "sdpa", (mb or HB.MAX_BATCH) if kind == "hfb" else NO)],
              f"{what}: built {BUILT}")
        check(meta is not None and meta.get("trust_remote_code") is t and meta.get("attn_implementation") == a
              and meta.get("max_batch") == mb and meta.get("executable") == sys.executable, f"{what}: meta")
    for name, kind, entry in (("mb_hf", "hf", dict(hf, engine="hf", max_batch=8)),
                              ("mb_vllm", "vllm", {"engine": "vllm", "max_batch": 8}),
                              ("mb_zero", "hfb", {"engine": "hfb", "max_batch": 0}),
                              ("mb_str", "hfb", {"engine": "hfb", "max_batch": "32"})):
        rc, meta = run_dev(os.path.join(b, name), kind, entry)
        check("max_batch" in str(rc) and not CALLS and not BUILT and meta is None, f"{name} refused: {rc} {CALLS}")
    ns = types.SimpleNamespace(render="template", dtype="bfloat16", device="cpu", hf_untested_ok=True,
                               trust_remote_code=True, attn_implementation="eager")
    a = DB.engine_audit(REAL_MAKE("hfb:org/Doge", ns)[0])
    check(a.get("trust_remote_code") is True and a.get("attn") == "eager" and a.get("stop_ids") == [1, 6],
          f"audit of the HF engine: {a}")
    for name, entry in (("v_trust", {"engine": "vllm", "trust_remote_code": True}),
                        ("v_attn", {"engine": "vllm", "attn_implementation": "eager"})):
        rc, meta = run_dev(os.path.join(b, name), "vllm", entry)
        check("hf / hfb only" in str(rc) and not CALLS and not BUILT and meta is None, f"vllm refused: {rc} {CALLS}")


def c_parity():
    b = tempfile.mkdtemp(prefix="rc12_trust_par_")
    for trust, attn in ((True, "eager"), (False, None)):
        out = os.path.join(b, str(trust))
        os.makedirs(out)
        CALLS.clear()
        ns = types.SimpleNamespace(model="org/Doge", out=out, n=0, pick="spread", data=R.DEV, dtype="bfloat16",
                                   device="cpu", gpu_mem=0.85, max_model_len=None, engine="hfb",
                                   trust_remote_code=trust, attn_implementation=attn)
        PH.stage_vllm(ns)
        PH.stage_hf(ns)
        a = NO if attn is None else attn
        check(CALLS == [("tok", "org/Doge", trust, NO), ("model", "org/Doge", trust, a)] * 2,
              f"parity stages ({trust}, {attn}): {CALLS}")
    seen, old_run, old_argv = [], PH.subprocess.run, sys.argv
    PH.subprocess.run = lambda cmd, **kw: seen.append(cmd) or types.SimpleNamespace(returncode=1)
    try:
        for argv in (["--engine", "hfb", "--trust-remote-code", "--attn-implementation", "eager"],
                     ["--engine", "hfb"], ["--attn-implementation", "eager"], ["--trust-remote-code"]):
            seen.clear()
            sys.argv = ["parity_hf_vllm.py", "--model", "org/Doge", "--out", os.path.join(b, "all")] + argv
            try:
                PH.main()
                rc = 0
            except SystemExit as e:
                rc = str(e.code)
            if argv == ["--trust-remote-code"]:
                check("hfb only" in rc and not seen, f"parity refuses trust on vllm: {rc} {seen}")
                continue
            got = seen[0] if len(seen) == 1 else []
            check(len(seen) == 1 and ("--trust-remote-code" in got) == ("--trust-remote-code" in argv)
                  and ("eager" in argv) == (got[got.index("--attn-implementation") + 1:][:1] == ["eager"]
                                            if "--attn-implementation" in got else False),
                  f"parity all forwards the flags {argv}: {seen}")
    finally:
        PH.subprocess.run, sys.argv = old_run, old_argv


def c_verify_rerun():
    """verify_dev_runs --rerun rebuilds the engine with the run's own HF load options (meta.json)."""
    import verify_dev_runs as VD
    for meta, t, a, mb in (({"engine": "hfb", "dtype": "bfloat16", "trust_remote_code": True, "max_batch": 32,
                             "attn_implementation": "eager"}, True, "eager", 32), ({"engine": "hfb"}, False, NO, 64)):
        BUILT.clear()
        CALLS.clear()
        R.make_responder = fake_engine
        try:
            VD.rerun(types.SimpleNamespace(render="template", rerun=0, model="org/Doge"),
                     {"greedy": {"meta": meta, "rows": []}}, {})
        finally:
            R.make_responder = REAL_MAKE
        check(CALLS[:2] == [("tok", "org/Doge", t, NO), ("model", "org/Doge", t, a)] and BUILT[:1]
              and BUILT[0][3] == mb, f"verify rerun {meta}: {CALLS} {BUILT}")


def main():
    c_dev_batch()
    c_parity()
    c_verify_rerun()
    for f in fails:
        print("FAIL", f)
    print("test_hf_trust:", "PASS" if not fails else f"{len(fails)} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
