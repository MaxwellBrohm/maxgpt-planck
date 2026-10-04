"""Mutation test of the item 14 engine options (notes STEP 11; Max's rule: every test claim is watched failing):
engines.json "chat_template" and "token_budget" through hf_responder, vllm_responder, hf_batched, runner, dev_batch,
parity_hf_vllm and verify_dev_runs, the Loom-Spark-3.2 template in engines.json, and the card row's engines "decode"
(dev_batch.decode_override / apply_decode, verify_dev_runs, engines_card.json), and the HF options nan_guard and
use_cache (hf_responder, hf_batched, runner, dev_batch, parity_hf_vllm, verify_dev_runs, rerun_tie, engines.json).
Each mutant replaces one text in a scratch copy of rc12 (outside the repo: --scratch, default the system temp dir),
then test_chat_template.py,
test_loom_template.py, test_decode_variant.py and test_engine_patches.py run there; KILLED = any exits non-zero. The
unmutated copy must pass all four; malformed = the text is not found exactly once. No model.
  python3 -B mutation_s14.py [--scratch DIR]   (writes logs/mutation_s14.txt; exit 1 unless all are killed)"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
E004 = os.path.join(HERE, "..", "experiments", "E004_general_updating", "code")   # quick_shortcuts' import, by path
ENV = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (E004, os.environ.get("PYTHONPATH")) if p))
TESTS = ("test_chat_template.py", "test_loom_template.py", "test_decode_variant.py", "test_engine_patches.py")
HRm, VRm, HBm, Rm, DBm, PHm, VDm = ("hf_responder.py", "vllm_responder.py", "hf_batched.py", "runner.py",
                                    "dev_batch.py", "parity_hf_vllm.py", "verify_dev_runs.py")
MUTANTS = [
    (HRm, "    tok.chat_template = chat_template\n", "", "set_template does not set the tokenizer's template"),
    (HRm, "    if chat_template is None:\n        return None", "    if True:\n        return None",
     "set_template ignores every template"),
    (HRm, "if not isinstance(chat_template, str) or not chat_template:", "if False:", "set_template takes anything"),
    (HRm, "self.chat_template = set_template(self.tok, chat_template)", "self.chat_template = set_template(self.tok, None)",
     "HFResponder ignores the template"),
    (VRm, "self.chat_template = HR.set_template(tokenizer, chat_template)",
     "self.chat_template = HR.set_template(tokenizer, None)", "VLLMResponder ignores the template"),
    (HBm, "attn_implementation, chat_template,\n                         nan_guard",
     "attn_implementation, None,\n                         nan_guard", "HFBatched does not pass the template on"),
    (Rm, 'tmpl = getattr(args, "chat_template", None)', "tmpl = None", "runner drops the template"),
    (Rm, "batch_invariant=args.batch_invariant,\n                                chat_template=tmpl), name",
     "batch_invariant=args.batch_invariant), name", "runner: vLLM without the template"),
    (Rm, "token_budget=tb,\n                                chat_template=tmpl, nan_guard",
     "token_budget=tb,\n                                nan_guard", "runner: hfb without the template"),
    (Rm, "attn_implementation=attn, chat_template=tmpl,\n                             nan_guard",
     "attn_implementation=attn,\n                             nan_guard", "runner: serial hf without the template"),
    (Rm, "attn_implementation=attn, token_budget=tb,", "attn_implementation=attn,", "runner: hfb budget dropped"),
    (Rm, 'tb = getattr(args, "token_budget", None) or HB.TOKEN_BUDGET', "tb = HB.TOKEN_BUDGET",
     "runner ignores engines.json's token_budget"),
    (DBm, 'token_budget=e.get("token_budget"), chat_template=e.get("chat_template"),',
     'token_budget=None, chat_template=e.get("chat_template"),', "dev_batch drops token_budget"),
    (DBm, 'token_budget=e.get("token_budget"), chat_template=e.get("chat_template"),',
     'token_budget=e.get("token_budget"), chat_template=None,', "dev_batch drops chat_template"),
    (DBm, 'for k in ("max_batch", "token_budget"):', 'for k in ("max_batch",):', "token_budget not validated"),
    (DBm, "if hfo[k] is not None and (type(hfo[k]) is not str or not hfo[k]):", "if False:",
     "chat_template and nan_guard not validated"),
    (PHm, 'for k in ("chat_template", "max_batch", "token_budget", "nan_guard", "use_cache")}',
     'for k in ("max_batch", "token_budget", "nan_guard", "use_cache")} | dict(chat_template=None)',
     "parity ignores engines.json's template"),
    (PHm, 'args.max_model_len,\n                             chat_template=ek["chat_template"])', "args.max_model_len)",
     "parity: vLLM stage without the template"),
    (PHm, "**hf_kw(args), **sized,", "**hf_kw(args),", "parity: hfb stage without max_batch / token_budget"),
    (PHm, 'chat_template=ek["chat_template"], use_cache=ek["use_cache"])', 'chat_template=None, use_cache=ek["use_cache"])',
     "parity: HF stage without the template"),
    (PHm, 'base += ["--engines", args.engines] if args.engines else []', "pass", "parity all drops --engines"),
    (PHm, 'ids=[r["id"] for r in recs], **ek)', 'ids=[r["id"] for r in recs])', "parity meta lacks the template"),
    (VDm, 'for k in ("chat_template", "nan_guard", "use_cache"):', 'for k in ("nan_guard", "use_cache"):',
     "verify C1 ignores the template"),
    (VDm, 'trust = run["meta"].get("trust_remote_code") is True', "trust = False", "verify C5 without trust"),
    (VDm, 'chat_template=run["meta"].get("chat_template"))', "chat_template=None)", "verify C5 without the template"),
    (VDm, 'thinks = "<think>" in tmpl or "enable_thinking" in tmpl', "thinks = True",
     "verify C5 asks every Qwen3-architecture prompt for a think block"),
    (VDm, 'thinks = "<think>" in tmpl or "enable_thinking" in tmpl', "thinks = False", "verify C5 never checks thinking"),
    ("engines.json", "{% else %}{{ raise_exception(", "{% else %}{{ (", "Loom template accepts a system turn"),
    ("engines.json", "<|eot|>\\n<loom>\\n{% elif", "<|eot|>\\n<loom>\\n\\n{% elif", "Loom template: extra newline"),
    ("engines.json", "{{ message['content'] }}<|eot|>\\n{% else %}", "{{ message['content'] }}\\n<|eot|>\\n{% else %}",
     "Loom template: newline before a reply's <|eot|>"),
    ("engines.json", '"chat_template": "<tools:off>\\n', '"chat_template": "<tools:on>\\n', "Loom with tools on"),
    # the card row (engines "decode"; NOVELTY_2026-10 s6, notes STEP 11, ITEM 14)
    (DBm, "    HR.DECODE.update(d)\n    return d", "    return d", "apply_decode does not set DECODE"),
    (DBm, 'if d and kind != "vllm":', "if False:", "decode allowed on hf / hfb"),
    (DBm, "if not isinstance(d, dict) or set(d) - set(DECODE_KEYS):", "if not isinstance(d, dict):",
     "decode takes any key"),
    (DBm, "if not isinstance(d, dict) or set(d) - set(DECODE_KEYS):", "if set(d) - set(DECODE_KEYS):",
     "decode takes a non-dict"),
    (DBm, "if type(v) not in (int, float) or not (", "if not (", "decode takes bools and strings"),
    (DBm, 'not (0 < v <= 1 if k == "top_p" else v >= 1)', 'not (0 < v <= 1 if k == "top_p" else v > 0)',
     "repetition_penalty below 1 accepted"),
    (DBm, 'not (0 < v <= 1 if k == "top_p" else v >= 1)', 'not (0 <= v <= 1 if k == "top_p" else v >= 1)',
     "top_p 0 accepted"),
    (DBm, 'not (0 < v <= 1 if k == "top_p" else v >= 1)', 'not (0 < v if k == "top_p" else v >= 1)',
     "top_p above 1 accepted"),
    (DBm, 'd = e.get("decode", {})', "d = {}", "decode_override ignores engines.json"),
    (DBm, "        apply_decode(model_id, args.engines, kind)", "        pass", "dev_batch never applies decode"),
    (VDm, "    HR.DECODE.update(dec)\n    if dec:", "    if dec:", "verify does not apply the decode"),
    (VDm, "dec = DB.decode_override(args.model, args.engines)", "dec = {}", "verify ignores the file's decode"),
    ("engines_card.json", '"top_p": 0.9', '"top_p": 1.0', "card row top_p is not the card's"),
    ("engines_card.json", '"repetition_penalty": 1.3', '"repetition_penalty": 1.0', "card row penalty off"),
    ("engines_card.json", '"dtype": "float32"', '"dtype": "bfloat16"', "card row dtype differs from the panel row"),
    ("engines.json", '"VertexResearch/Vertex-0.6-15M-Instruct": {"engine": "vllm",',
     '"VertexResearch/Vertex-0.6-15M-Instruct": {"decode": {"top_p": 0.9}, "engine": "vllm",',
     "the main engines.json carries a decode"),
    # HF options for remote code: nan_guard (Swen-28M) and use_cache (cRia); notes STEP 11, ITEM 14
    (HRm, "        m.register_forward_hook(_zero_nans)", "        pass", "guard_nans hooks nothing"),
    (HRm, "t.masked_fill_(t.isnan(), 0.0)", "t.masked_fill_(t.isinf(), 0.0)", "the guard zeroes inf, not NaN"),
    (HRm, "    if not mods:\n        raise", "    if False:\n        raise", "guard_nans accepts an unknown class"),
    (HRm, "    if cls_name is None:\n        return 0", "    if False:\n        return 0", "guard_nans(None) raises"),
    (HRm, "guard_nans(self.model, nan_guard)", "guard_nans(self.model, None)", "HFResponder never hooks"),
    (HRm, "    model.config.use_cache = use_cache\n", "", "set_use_cache sets nothing"),
    (HRm, "    if type(use_cache) is not bool:", "    if False:", "set_use_cache takes a non-bool"),
    (HRm, "set_use_cache(self.model, use_cache)", "set_use_cache(self.model, None)", "HFResponder ignores use_cache"),
    (HBm, "chat_template,\n                         nan_guard, use_cache)",
     "chat_template,\n                         None, use_cache)", "HFBatched drops nan_guard"),
    (HBm, "chat_template,\n                         nan_guard, use_cache)",
     "chat_template,\n                         nan_guard)", "HFBatched drops use_cache"),
    (Rm, 'guard = getattr(args, "nan_guard", None)', "guard = None", "runner drops nan_guard"),
    (Rm, 'cache = getattr(args, "use_cache", None)', "cache = None", "runner drops use_cache"),
    (Rm, "chat_template=tmpl, nan_guard=guard, use_cache=cache), name", "chat_template=tmpl), name",
     "runner: hfb without the HF options"),
    (Rm, "\n                             nan_guard=guard, use_cache=cache), name", "), name",
     "runner: serial hf without the HF options"),
    (DBm, 'nan_guard=e.get("nan_guard"), use_cache=e.get("use_cache"))',
     'nan_guard=None, use_cache=e.get("use_cache"))', "dev_batch drops nan_guard"),
    (DBm, 'nan_guard=e.get("nan_guard"), use_cache=e.get("use_cache"))',
     'nan_guard=e.get("nan_guard"), use_cache=None)', "dev_batch drops use_cache"),
    (DBm, 'hf_only = ("attn_implementation", "nan_guard", "use_cache")', 'hf_only = ("attn_implementation",)',
     "nan_guard / use_cache allowed on vllm"),
    (DBm, 'for k in ("chat_template", "nan_guard"):', 'for k in ("chat_template",):', "nan_guard not validated"),
    (DBm, 'if hfo["use_cache"] is not None and type(hfo["use_cache"]) is not bool:', "if False:",
     "use_cache not validated"),
    (PHm, '"token_budget", "nan_guard", "use_cache")}', '"token_budget")} | dict(nan_guard=None, use_cache=None)',
     "parity ignores nan_guard and use_cache"),
    (PHm, 'nan_guard=ek["nan_guard"],', "nan_guard=None,", "parity: hfb stage without nan_guard"),
    (PHm, 'use_cache=ek["use_cache"])                     #', "use_cache=None)                     #",
     "parity: hfb stage without use_cache"),
    (PHm, 'chat_template=ek["chat_template"], use_cache=ek["use_cache"])', 'chat_template=ek["chat_template"])',
     "parity: serial HF stage without use_cache"),
    (VDm, 'for k in ("chat_template", "nan_guard", "use_cache"):', 'for k in ("chat_template",):',
     "verify C1 ignores nan_guard and use_cache"),
    ("rerun_tie.py", 'nan_guard=m.get("nan_guard"), use_cache', "nan_guard=None, use_cache", "rerun drops nan_guard"),
    ("rerun_tie.py", 'use_cache=m.get("use_cache"))', "use_cache=None)", "rerun drops use_cache"),
    ("engines.json", '"nan_guard": "Attention", ', "", "Swen without the guard"),
    ("engines.json", '"use_cache": true, ', "", "cRia without the cache"),
]


def run_in(copy):
    for t in TESTS:
        p = subprocess.run([sys.executable, "-B", t], cwd=copy, capture_output=True, text=True, timeout=600, env=ENV)
        if p.returncode:
            return p.returncode, t, [ln for ln in p.stdout.splitlines() if ln.startswith("FAIL")][:1] or [p.stderr[-200:]]
    return 0, None, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default=None)
    args = ap.parse_args()
    t0, lines, bad = time.time(), [], 0
    for k, (f, old, new, what) in enumerate([(None, None, None, "baseline")] + MUTANTS):
        tmp = tempfile.mkdtemp(dir=args.scratch)
        copy = os.path.join(tmp, "rc12")
        shutil.copytree(HERE, copy, ignore=shutil.ignore_patterns("logs", "runs", "__pycache__"))
        if k:
            p = os.path.join(copy, f)
            src = open(p).read()
            if src.count(old) != 1:
                lines.append(f"MALFORMED #{k} {f}: {what}")
                bad += 1
                shutil.rmtree(tmp)
                continue
            open(p, "w").write(src.replace(old, new))
        rc, test, first = run_in(copy)
        shutil.rmtree(tmp)
        if not k:
            lines.append(f"baseline: exit {rc}" + ("" if rc == 0 else f" {test} {first}"))
            bad += rc != 0
            continue
        lines.append(f"{'KILLED  ' if rc else 'SURVIVED'} #{k:02d} {f}: {what}" + (f"  <- {test}: {first[0][:120]}"
                                                                                  if rc else ""))
        bad += rc == 0
    lines.append(f"{len(MUTANTS)} mutants, {bad} problems, {time.time() - t0:.0f} s")
    lines.append("ALL MUTANTS KILLED" if not bad else "NOT ALL KILLED")
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    open(os.path.join(HERE, "logs", "mutation_s14.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
