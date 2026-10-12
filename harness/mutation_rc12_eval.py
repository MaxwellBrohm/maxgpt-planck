"""Mutation check for the RC-12 eval plumbing (planck_responder.py, decode.py, rc12_eval.py, the runner flag and
the trainer hook). Each mutant replaces ONE exact string (it must occur once) in a scratch copy of harness/ and
rc12/ (.py files + rc12/dev; experiments/ is symlinked, read only), then runs the named test file there.
Killed = pytest exit 1; exit 2-5 = INVALID. The unmutated copy runs first and must be green.
  python mutation_rc12_eval.py --part 1/3      (each part stays under 2 minutes with 4 workers)
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EV, HK, DC = "test_rc12_eval.py", "test_rc12_hook.py", "test_decode.py"
PR, RE, DE = "rc12/planck_responder.py", "harness/rc12_eval.py", "harness/decode.py"
M = [
    (PR, "return [int(x) for x in ids] + [self.asst_id]", "return [int(x) for x in ids]", EV, "no <|assistant|> cue"),
    (PR, 'if self.render == "plain":', "if False:", EV, "plain render ignored"),
    (PR, "return int(torch.argmax(logits))", "return int(torch.topk(logits, 2).indices[1])", EV, "greedy not argmax"),
    (PR, 'logits / HR.DECODE["temperature"]', "logits", EV, "temperature ignored"),
    (PR, "manual_seed(HR.conv_seed(self.seed, self.rid, i))", "manual_seed(self.seed)", EV, "one stream per seed"),
    (PR, "torch.multinomial(probs, 1, generator=gen)", "torch.multinomial(probs, 1)", EV, "global RNG used"),
    (PR, 'if nxt == self.end_id:\n                return out, "eot"', "if False:\n                pass", EV,
     "<|end|> does not stop"),
    (PR, 'if nxt in self.eos_ids:\n                return out, "eos"', "if False:\n                pass", EV,
     "EOS does not stop"),
    (PR, 'if nxt in self.role_ids:\n                return out, "role"', "if False:\n                pass", EV,
     "role token does not stop"),
    (PR, 'return out, "role"', 'return out[:-1], "role"', EV, "role token dropped from the text"),
    (PR, "max_new = max(0, min(self.max_new, room))", "max_new = self.max_new", EV, "context room ignored"),
    (PR, "self.model.train(was_training)", "pass", EV, "train mode not restored"),
    (PR, "with torch.no_grad():\n                out, stop", "with nullcontext():\n                out, stop", EV,
     "generation with grad"),
    (PR, "ids |= {tok.token_to_id(s) for s in EOS_STRINGS", "ids |= {-1 for s in EOS_STRINGS", EV,
     "<|endoftext|> not an EOS"),
    (PR, "if want is not None and want != i:", "if False:", EV, "config/tokenizer id clash accepted"),
    (PR, "if PlanckConfig.from_dict(run_cfg[\"model\"]).to_dict() != mcfg.to_dict():", "if False:", HK,
     "config/checkpoint shape clash accepted"),
    (PR, "load_tokenizer(resolve(tok_path, None if tokenizer else base))", "load_tokenizer(tok_path)", HK,
     "relative tokenizer not resolved against the config"),
    (PR, 'model.load_state_dict(ck["model"])', "pass", HK, "checkpoint weights not loaded"),
    ("rc12/runner.py", "if args.ctx is not None and args.ctx > r.ctx:", "if False:", HK, "--ctx > seq_len accepted"),
    (DE, "out = F.scaled_dot_product_attention(q, k_, v_)\n", "out = F.scaled_dot_product_attention(q, k_, v_, "
     "is_causal=True)\n", DC, "decode step sees only key 0"),
    (DE, "cos = m.rope_cos[P:P + T]", "cos = m.rope_cos[:T]", DC, "decode RoPE not offset"),
    (DE, "if i == 0 and cfg.value_residual:", "if i == 1 and cfg.value_residual:", DC, "wrong value-residual v1"),
    (DE, "self.k[i], self.v[i] = k, v", "self.k[i], self.v[i] = (k, v) if self.n == 0 else (self.k[i], self.v[i])",
     DC, "cache not extended"),
    (DE, "blk, scale = m.blocks[u], norm_scale_for(cfg, i)", "blk, scale = m.blocks[u], 1.0", DC,
     "decode ignores norm scaling"),
    (RE, 'key = r["meta"].get("pair_id") if f == "BIND" else r["id"]', 'key = r["id"]', HK, "BIND twins split"),
    (RE, "if not self.every or trainer.step % self.every:", "if not self.every:", HK, "eval every step"),
    (RE, "if any(s is not None for s in self.seeds):", "if False:", HK, "sampled seeds not summarized"),
    (RE, "except Exception as e:", "except ZeroDivisionError as e:", HK, "eval error kills training"),
    (RE, 'if not int(ecfg.get("every", 0)):', "if not ecfg:", HK, "every: 0 turns the hook on"),
    (RE, "self.RN.run(self.recs, resp, self.render, self.seeds, self.seq_len,",
     "self.RN.run(self.recs, resp, self.render, self.seeds, None,", HK, "hook evaluates without the ctx limit"),
    ("harness/trainer.py", "for hook in self.hooks:", "for hook in []:", HK, "trainer never calls hooks"),
    ("harness/train.py", "tr.hooks.append(hook)", "pass", HK, "train.py never installs the hook"),
]


def make_copy(m):
    d = tempfile.mkdtemp(prefix="planck_rc12mut_")
    for sub in ("harness", "rc12"):
        os.makedirs(os.path.join(d, sub))
        for p in glob.glob(os.path.join(ROOT, sub, "*.py")):
            shutil.copy(p, os.path.join(d, sub))
    shutil.copytree(os.path.join(ROOT, "rc12", "dev"), os.path.join(d, "rc12", "dev"))
    os.symlink(os.path.join(ROOT, "experiments"), os.path.join(d, "experiments"))   # rc12 imports E001/E004 code
    if m is not None:
        path = os.path.join(d, m[0])
        src = open(path).read()
        if src.count(m[1]) != 1:
            shutil.rmtree(d)
            raise ValueError(f"{m[4]}: target occurs {src.count(m[1])} times in {m[0]}")
        open(path, "w").write(src.replace(m[1], m[2]))
    return d


def run(m, files):
    d = make_copy(m)
    env = {**os.environ, "PLANCK_TEST_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1"}
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--tb=no", *files],
                           cwd=os.path.join(d, "harness"), env=env, capture_output=True, text=True, timeout=110)
        code, tail = r.returncode, (r.stdout.strip().splitlines() or [""])[-1]
    except subprocess.TimeoutExpired:
        code, tail = -9, "TIMEOUT"
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return code, tail, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="1/1")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    i, n = map(int, a.part.split("/"))
    todo = M[i - 1::n]
    for m in todo:
        shutil.rmtree(make_copy(m))
    code, tail, dt = run(None, [EV, HK, DC])
    print(f"baseline: exit {code} ({tail}) {dt:.0f}s", flush=True)
    if code != 0:
        return 1
    res = list(ThreadPoolExecutor(a.workers).map(lambda m: run(m, [m[3]]), todo))
    bad = 0
    for m, (code, tail, dt) in zip(todo, res):
        ok = code == 1
        bad += not ok
        print(f"{'killed  ' if ok else 'SURVIVED' if code == 0 else 'INVALID '} {m[0]:26s} {m[4]:45s} {dt:4.0f}s "
              f"{tail}", flush=True)
    print(f"{len(todo) - bad}/{len(todo)} killed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
