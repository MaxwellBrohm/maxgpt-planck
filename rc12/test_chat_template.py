"""RC-12 item 14 engine options (notes STEP 11): engines.json "chat_template" (any engine) and "token_budget" (hfb),
with stub transformers / torch / vllm modules (no model, no GPU).
  dev_batch  chat_template reaches the tokenizer of HFResponder, HFBatched and VLLMResponder through dev_batch ->
             runner.make_responder; absent leaves the tokenizer's own template; meta.json records it; a non-string or
             empty value is refused before anything loads; token_budget reaches HFBatched (absent: TOKEN_BUDGET) and
             meta.json, and is refused on hf, on vllm and when not a positive int
  parity     --engines gives both stages the model's chat_template and the hfb stage its max_batch / token_budget;
             stage all forwards --engines; without it the tokenizer's own template stays
  verify     C1 fails a run whose chat_template is not engines.json's; C5 builds its prompt with the run's template and
             trust_remote_code, and asks a Qwen3-architecture prompt for the empty think block only when the template has
             a thinking mode (--rerun moved to rerun_tie.py, item 2; not covered here)
  (the Loom-Spark-3.2 template itself: test_loom_template.py)
Run: python3 -B test_chat_template.py   (exit 1 on any failure)"""
import contextlib, io, json, os, sys, tempfile, types  # noqa: E401

import jinja2

CALLS, BUILT, fails = [], [], []
OWN = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"     # a model's own template


def render_jinja(template, messages):
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)        # transformers' chat-template settings

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)
    env.globals["raise_exception"] = raise_exception
    return env.from_string(template).render(messages=messages, add_generation_prompt=True)


class Tok:
    eos_token_id, pad_token_id = 1, 2
    def __init__(self): self.chat_template = OWN  # noqa: E301, E704
    def get_vocab(self): return {"<|im_end|>": 1}  # noqa: E704
    def apply_chat_template(self, messages, **kw): return render_jinja(self.chat_template, messages)  # noqa: E704
    def __call__(self, text, **kw): return types.SimpleNamespace(input_ids=[ord(c) for c in text])  # noqa: E704
    def decode(self, ids, **kw): return "".join(chr(i) for i in ids)  # noqa: E704


MTYPE = ["llama"]                       # the stub config's model_type (C5's Qwen3 thinking check)


def _model():
    m = types.SimpleNamespace(config=types.SimpleNamespace(model_type=MTYPE[0], max_position_embeddings=2048,
                                                           _attn_implementation="sdpa"), generation_config=None)
    m.to, m.eval = (lambda d: m), (lambda: m)
    return m


def loader(kind):
    def from_pretrained(model_id, **kw):
        CALLS.append((kind, model_id, kw.get("trust_remote_code")))
        return Tok() if kind == "tok" else _model() if kind == "model" else _model().config
    return types.SimpleNamespace(from_pretrained=from_pretrained)


def install_stubs():
    tf = types.ModuleType("transformers")
    tf.AutoTokenizer, tf.AutoModelForCausalLM, tf.AutoConfig = loader("tok"), loader("model"), loader("config")
    tf.__version__ = "stub"
    torch, vllm = types.ModuleType("torch"), types.ModuleType("vllm")
    torch.bfloat16, torch.float32, torch.__version__ = "bf16", "fp32", "stub"
    vllm.LLM = lambda **kw: CALLS.append(("llm", kw.get("model"), kw.get("trust_remote_code"))) or object()
    vllm.SamplingParams, vllm.__version__ = dict, "stub"
    sys.modules.update(transformers=tf, torch=torch, vllm=vllm)


install_stubs()
import dev_batch as DB  # noqa: E402
import fakes_family as FF  # noqa: E402
import hf_batched as HB  # noqa: E402
import lockstep as LS  # noqa: E402
import parity_hf_vllm as PH  # noqa: E402
import runner as R  # noqa: E402
import verify_dev_runs as VD  # noqa: E402

REAL_MAKE = R.make_responder


def check(cond, what):
    if not cond:
        fails.append(what)


def fake_engine(spec, ns):
    eng, name = REAL_MAKE(spec, ns)
    BUILT.append((type(eng).__name__, eng.tok.chat_template, getattr(eng, "chat_template", "MISSING"),
                  getattr(eng, "token_budget", None)))
    return LS.PerConv(lambda: FF.make("IDEAL"), ns.render), name


def run_dev(base, kind, entry):
    os.makedirs(base)
    engines = os.path.join(base, "engines.json")
    json.dump({"models": {"org/M": entry}}, open(engines, "w"))
    root = os.path.join(base, "root")
    CALLS.clear()
    BUILT.clear()
    old = sys.argv
    sys.argv = ["dev_batch.py", "--responder", f"{kind}:org/M", "--render", "template", "--seeds", "greedy",
                "--root", root, "--engines", engines, "--hf-untested-ok", "--limit", "2", "--device", "cpu"]
    R.make_responder = fake_engine
    try:
        rc = DB.main()
    except SystemExit as e:
        rc = str(e.code)
    finally:
        sys.argv, R.make_responder = old, REAL_MAKE
    p = os.path.join(root, "M", "template", "greedy", "meta.json")
    return rc, (json.load(open(p)) if os.path.exists(p) else None)


def c_dev_batch():
    b = tempfile.mkdtemp(prefix="rc12_tmpl_")
    T = "<x>{% for m in messages %}{{ m['content'] }}|{% endfor %}"
    for kind, entry, want_t, want_tb in (("hfb", {"engine": "hfb", "chat_template": T, "token_budget": 999}, T, 999),
                                         ("hf", {"engine": "hf", "chat_template": T}, T, None),
                                         ("vllm", {"engine": "vllm", "chat_template": T}, T, None),
                                         ("hfb", {"engine": "hfb"}, None, None), ("vllm", {"engine": "vllm"}, None, None)):
        what = f"{kind} {sorted(entry)}"
        rc, meta = run_dev(os.path.join(b, f"{kind}_{len(os.listdir(b))}"), kind, entry)
        cls = {"hfb": "HFBatched", "hf": "HFResponder", "vllm": "VLLMResponder"}[kind]
        tb = (want_tb or HB.TOKEN_BUDGET) if kind == "hfb" else None
        check(rc == 0 and BUILT == [(cls, want_t or OWN, want_t, tb)], f"{what}: built {rc} {BUILT}")
        check(meta is not None and meta.get("chat_template") == want_t and meta.get("token_budget") == want_tb,
              f"{what}: meta {meta and (meta.get('chat_template'), meta.get('token_budget'))}")
    for name, kind, entry, word in (("t_int", "hfb", {"engine": "hfb", "chat_template": 5}, "chat_template"),
                                    ("t_empty", "vllm", {"engine": "vllm", "chat_template": ""}, "chat_template"),
                                    ("tb_hf", "hf", {"engine": "hf", "token_budget": 9}, "token_budget"),
                                    ("tb_vllm", "vllm", {"engine": "vllm", "token_budget": 9}, "token_budget"),
                                    ("tb_zero", "hfb", {"engine": "hfb", "token_budget": 0}, "token_budget"),
                                    ("tb_str", "hfb", {"engine": "hfb", "token_budget": "9"}, "token_budget")):
        rc, meta = run_dev(os.path.join(b, name), kind, entry)
        check(word in str(rc) and not CALLS and not BUILT and meta is None, f"{name} refused: {rc} {CALLS}")


def c_parity():
    b = tempfile.mkdtemp(prefix="rc12_tmpl_par_")
    T = "<p>{% for m in messages %}{{ m['content'] }}{% endfor %}"
    engines = os.path.join(b, "engines.json")
    json.dump({"models": {"org/M": {"engine": "hfb", "chat_template": T, "max_batch": 7, "token_budget": 11}}},
              open(engines, "w"))
    seen = []
    import hf_responder as HR
    import vllm_responder as VR
    orig = (HB.HFBatched.__init__, HR.HFResponder.__init__, VR.VLLMResponder.__init__)

    def spy(cls_name, init):
        def wrapped(self, *a, **kw):
            init(self, *a, **kw)
            seen.append((cls_name, self.tok.chat_template, getattr(self, "max_batch", None),
                         getattr(self, "token_budget", None)))
        return wrapped
    HB.HFBatched.__init__ = spy("HFBatched", orig[0])
    HR.HFResponder.__init__ = spy("HFResponder", orig[1])
    VR.VLLMResponder.__init__ = spy("VLLMResponder", orig[2])
    try:
        for engine, eng_path in (("hfb", engines), ("vllm", engines), ("hfb", None)):
            out = os.path.join(b, f"{engine}_{bool(eng_path)}")
            os.makedirs(out)
            seen.clear()
            ns = types.SimpleNamespace(model="org/M", out=out, n=0, pick="spread", data=R.DEV, dtype="bfloat16",
                                       device="cpu", gpu_mem=0.85, max_model_len=None, engine=engine,
                                       trust_remote_code=False, attn_implementation=None, engines=eng_path)
            PH.stage_vllm(ns)
            PH.stage_hf(ns)
            t = T if eng_path else OWN
            first = ("HFBatched", t, 7, 11) if engine == "hfb" and eng_path else \
                ("HFBatched", OWN, HB.MAX_BATCH, HB.TOKEN_BUDGET) if engine == "hfb" else ("VLLMResponder", t, None, None)
            check(first in seen and seen[-1][:2] == ("HFResponder", t), f"parity {engine} {eng_path}: {seen}")
            vm = json.load(open(os.path.join(out, "vllm_meta.json")))
            check(vm.get("chat_template") == (T if eng_path else None), f"parity meta template {engine} {eng_path}")
    finally:
        HB.HFBatched.__init__, HR.HFResponder.__init__, VR.VLLMResponder.__init__ = orig
    cmds, old_run, old_argv = [], PH.subprocess.run, sys.argv
    PH.subprocess.run = lambda cmd, **kw: cmds.append(cmd) or types.SimpleNamespace(returncode=1)
    try:
        for extra in (["--engines", engines], []):
            cmds.clear()
            sys.argv = ["parity_hf_vllm.py", "--model", "org/M", "--out", os.path.join(b, "all"), "--engine", "hfb"] + extra
            try:
                PH.main()
            except SystemExit:
                pass
            got = cmds[0] if cmds else []
            check(("--engines" in got) == bool(extra) and (not extra or got[got.index("--engines") + 1] == engines),
                  f"parity all forwards --engines {extra}: {cmds}")
    finally:
        PH.subprocess.run, sys.argv = old_run, old_argv


def c_verify():
    b = tempfile.mkdtemp(prefix="rc12_tmpl_ver_")
    T = "<v>{% for m in messages %}[{{ m['role'] }}]{{ m['content'] }}{% endfor %}<a>"
    engines = os.path.join(b, "engines.json")
    json.dump({"models": {"org/M": {"engine": "hf", "chat_template": T}, "org/N": {"engine": "hf"}}}, open(engines, "w"))
    base = dict(engine="hf", decode=VD.HR.DECODE, audit={"hf_generation_config": {}}, conversations=0, limit=1,
                seconds=0, own_cf=False)
    for model, tmpl, bad in (("org/M", T, False), ("org/M", None, True), ("org/M", "other", True),
                             ("org/N", None, False), ("org/N", T, True)):
        VD.FAILS.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            VD.c1_config("greedy", {"meta": dict(base, model=model, chat_template=tmpl), "rows": []}, engines, (0, 0))
        hit = [f for f in VD.FAILS if "chat_template" in f]
        check(bool(hit) == bad, f"C1 template check {model} {tmpl!r}: {VD.FAILS}")
    row = {"id": "r-1", "family": "RECALL", "turns": [{"user": "u1", "reply": "a1"}, {"user": "u2", "reply": "a2"},
                                         {"user": "u3", "reply": "a3"}]}
    for trust, mtype, tmpl, bad in ((True, "llama", T, False), (False, "llama", T, False), (False, "qwen3", T, False),
                                    (False, "qwen3", T + "<think>", True)):   # Qwen3 arch, own template: think only
        CALLS.clear()                                                              # if the template has a think mode
        VD.FAILS.clear()
        MTYPE[0], buf = mtype, io.StringIO()
        with contextlib.redirect_stdout(buf):
            VD.c5_prompt("org/M", "template", {"meta": dict(base, chat_template=tmpl, trust_remote_code=trust),
                                               "rows": [row]})
        check("<v>[user]u1[assistant]a1[user]u2[assistant]a2[user]u3<a>" in buf.getvalue() and bool(VD.FAILS) == bad,
              f"C5 prompt with the run's template ({mtype}): {buf.getvalue()[-160:]!r} {VD.FAILS}")
        check([c for c in CALLS if c[0] in ("tok", "config")] == [("tok", "org/M", trust), ("config", "org/M", trust)],
              f"C5 load options trust {trust}: {CALLS}")
    MTYPE[0] = "llama"


def c_set_template():
    import hf_responder as HR
    t = Tok()
    check(HR.set_template(t, None) is None and t.chat_template == OWN, "set_template(None) changed the template")
    check(HR.set_template(t, "<a>") == "<a>" and t.chat_template == "<a>", "set_template did not set the template")
    for bad in ("", 5, ["<a>"]):
        with contextlib.suppress(ValueError):
            HR.set_template(Tok(), bad)
            check(False, f"set_template accepted {bad!r}")


def main():
    for c in (c_set_template, c_dev_batch, c_parity, c_verify):
        c()
    print("\n".join(f"FAIL {f}" for f in fails) + "\n" * bool(fails) + "test_chat_template: " +
          ("PASS" if not fails else f"{len(fails)} FAILURES"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
