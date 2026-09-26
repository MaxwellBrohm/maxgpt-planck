"""RC-12 mutation test of hf_batched.py's ENGINE path (notes STEP 9; Max's rule: every test claim is watched failing).
PC ONLY: test_hf_batched.c_engine builds a random toy model on the CPU (refused on macOS). Each mutant replaces one
text in hf_batched.py, is installed in a FRESH process before anything imports it, and runs c_split + c_engine.
KILLED = a failure line; crashed = an exception escaped (not a kill); malformed = text not found exactly once.
The split-rule mutants live in mutation_step9.py (no torch needed).
Usage (PC): <venv python> -B mutation_hf_batched.py   (prints one line per mutant; exit 1 unless the baseline is
clean and every mutant is killed)"""
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MUTANTS = [
    ("torch.multinomial(probs, num_samples=1, generator=g)", "torch.multinomial(probs, num_samples=1)",
     "sampled rows draw from the global RNG, not their own generator"),
    ('HR.conv_seed(seed, rec["id"], i) for', 'HR.conv_seed(seed, rec["id"], 0) for', "per-reply seed ignores the turn"),
    ("            ids[r, width - len(e):] = torch.tensor(e, dtype=torch.long)\n"
     "            mask[r, width - len(e):] = 1\n",
     "            ids[r, :len(e)] = torch.tensor(e, dtype=torch.long)\n            mask[r, :len(e)] = 1\n",
     "right padding"),
    ("            mask[r, width - len(e):] = 1\n", "            mask[r, :] = 1\n", "padding attended"),
    ('RowSampler(torch, seeds, HR.DECODE["temperature"], self.device)', "RowSampler(torch, seeds, 1.0, self.device)",
     "sampling at T 1.0"),
    ("            if g is None:\n                continue\n", "            if False:\n                continue\n",
     "greedy rows sampled from the global RNG"),
    ("            out = out[:j + 1]\n", "            pass\n", "a finished row keeps generate's padding"),
    ("out += self._chunk(reqs[lo:lo + self.max_batch])", "out += self._chunk(reqs[lo:lo + self.max_batch])[::-1]",
     "replies of a chunk come back reversed"),
    ("rows = gen[:, width:].tolist()", "rows = gen[:, width - 1:].tolist()", "the prompt's last token read as reply"),
]
BOOT = r'''
import os, sys, types
sys.path.insert(0, {here!r}); sys.dont_write_bytecode = True
old, new = {old!r}, {new!r}
path = os.path.join({here!r}, "hf_batched.py")
src = open(path).read()
if old is not None:
    if src.count(old) != 1:
        print("MALFORMED", src.count(old)); sys.exit(3)
    m = types.ModuleType("hf_batched"); m.__file__ = path; sys.modules["hf_batched"] = m
    exec(compile(src.replace(old, new), path, "exec"), m.__dict__)
import test_hf_batched as T
fails = T.c_split() + T.c_engine({tok!r})
print("FAILS", len(fails), fails[:2])
sys.exit(1 if fails else 0)
'''


def run(old, new, tok):
    t0 = time.time()
    p = subprocess.run([sys.executable, "-B", "-c", BOOT.format(here=HERE, old=old, new=new, tok=tok)],
                       capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    last = [x for x in p.stdout.splitlines() if x.startswith(("FAILS", "MALFORMED"))]
    return p.returncode, (last[-1] if last else (p.stderr.strip().splitlines() or ["?"])[-1])[:160], time.time() - t0


def main():
    if platform.system() == "Darwin":
        sys.exit("PC only: the engine check builds a toy model")
    tok = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-135M-Instruct"
    rc, msg, sec = run(None, None, tok)
    print(f"baseline rc {rc} {msg} ({sec:.0f} s)", flush=True)
    bad = rc != 0
    for k, (old, new, what) in enumerate(MUTANTS):
        rc, msg, sec = run(old, new, tok)
        state = "KILLED" if rc == 1 and msg.startswith("FAILS") else ("MALFORMED" if rc == 3 else f"SURVIVED/CRASH rc {rc}")
        bad |= state != "KILLED"
        print(f"#{k} {state}: {what} | {msg} ({sec:.0f} s)", flush=True)
    print("ALL HF_BATCHED MUTANTS KILLED" if not bad else "NOT ALL KILLED (or the baseline failed)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
