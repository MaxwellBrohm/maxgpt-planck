"""RC-12 mutation test of hf_batched.py (notes STEP 9, STEP 9f; Max's rule: every test claim is watched failing).
Each mutant replaces one text in hf_batched.py, is installed in a FRESH process before anything imports it, and runs
the checks of test_hf_batched.py (c_engine only when c_split and c_chunk pass). KILLED = a failure line; crashed = an
exception escaped (not a kill); malformed = text not found exactly once.
  CHUNK   the STEP 9f chunker and reply_batch's bookkeeping; checks c_split + c_chunk (pure Python, any machine)
  ENGINE  the generate path; checks c_split + c_chunk + c_engine, which builds a random toy model on the CPU: PC ONLY
The split-rule mutants live in mutation_step9.py (no torch needed).
Usage: python3 -B mutation_hf_batched.py --chunk-only   (Mac or PC)
       <venv python> -B mutation_hf_batched.py [tok]     (PC: both groups)
Prints one line per mutant; exit 1 unless the baseline is clean and every mutant is killed."""
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN = "    for k in sorted(range(len(lengths)), key=lambda k: (-lengths[k], k)):\n"
CUT = "        if cur and (len(cur) + 1 > max_batch or (len(cur) + 1) * lengths[cur[0]] ** 2 > budget):\n"
CHUNK = [
    (PLAN, PLAN.replace("(-lengths[k], k)", "(lengths[k], k)"), "shortest first (width taken from the shortest row)"),
    (PLAN, PLAN.replace("(-lengths[k], k)", "(-lengths[k], -k)"), "ties in reverse request order"),
    (PLAN, "    for k in range(len(lengths)):\n", "no length sort"),
    (CUT, CUT.replace("len(cur) + 1 > max_batch", "len(cur) > max_batch"), "max_batch + 1 rows"),
    (CUT, CUT.replace("len(cur) + 1 > max_batch", "len(cur) + 2 > max_batch"), "cut one row early"),
    (CUT, CUT.replace("(len(cur) + 1) * lengths", "len(cur) * lengths"), "budget checked without the new row"),
    (CUT, CUT.replace("lengths[cur[0]] ** 2", "lengths[cur[0]]"), "budget on rows x width, not width^2"),
    (CUT, CUT.replace("lengths[cur[0]] ** 2", "lengths[k] ** 2"), "width of the newest (shortest) row"),
    (CUT, CUT.replace(" or (len(cur) + 1) * lengths[cur[0]] ** 2 > budget", ""), "budget ignored"),
    ("    if cur:\n        chunks.append(cur)\n    return chunks", "    return chunks", "last chunk dropped"),
    ("            chunks.append(cur)\n            cur = []\n", "            chunks.append(cur)\n",
     "chunk list not reset"),
    ("min(lo + max_batch, len(lengths))", "min(lo + max_batch - 1, len(lengths))", "budget None: runs of max_batch - 1"),
    ("plan_chunks([len(e) for e in enc], self.max_batch, self.token_budget)",
     "plan_chunks([len(e) for e in enc], self.max_batch, None)", "reply_batch ignores the budget"),
    ("for k, res in zip(idx, self._chunk(", "for k, res in zip(sorted(idx), self._chunk(", "replies scattered wrong"),
    ("[enc[k] for k in idx]", "[enc[k] for k in sorted(idx)]", "prompts paired with the wrong requests"),
    ("zip(reqs, enc, out)", "zip(reqs, enc, [out[k] for c in plan for k in c])", "trace in chunk order"),
    ("token_budget=TOKEN_BUDGET, chat_template=None,", "token_budget=None, chat_template=None,",
     "default is the pre-9f chunker"),
    ("                self.torch.cuda.empty_cache()      #", "                pass      #", "CUDA cache never emptied"),
    ("        if cuda:\n            self.torch.cuda.reset_peak_memory_stats()\n",
     "        if cuda:\n            self.torch.cuda.reset_peak_memory_stats()\n            self.torch.cuda.empty_cache()\n",
     "extra empty_cache at the call start (the check counts exactly one per chunk)"),
    ("            if cuda:                               # chunks", "            if cuda and not out[0]:            # chunks",
     "cache emptied before the first chunk of a call only"),
    ('cuda = str(self.device).startswith("cuda")', "cuda = True", "torch.cuda called on the CPU"),
    ('cuda = str(self.device).startswith("cuda")', 'cuda = str(self.device) == "cuda"', "cuda:0 not emptied"),
    ("        self.batches.append(len(reqs))\n        if not reqs", "        self.batches.append(1)\n        if not reqs",
     "batch sizes not recorded"),
]
ENGINE = [
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
    ("stop, got))\n        return res\n", "stop, got))\n        return res[::-1]\n", "replies of a chunk come back reversed"),
    ("for row in gen[:, width:].tolist():", "for row in gen[:, width - 1:].tolist():",
     "the prompt's last token read as reply"),
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
fails = T.c_split() + T.c_chunk()
if not fails and {engine!r}:                   # a mutant the pure checks kill never reaches the toy model
    fails = T.c_engine({tok!r})
print("FAILS", len(fails), fails[:2])
sys.exit(1 if fails else 0)
'''


def run(old, new, tok, engine):
    t0 = time.time()
    code = BOOT.format(here=HERE, old=old, new=new, tok=tok, engine=engine)
    p = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    last = [x for x in p.stdout.splitlines() if x.startswith(("FAILS", "MALFORMED"))]
    return p.returncode, (last[-1] if last else (p.stderr.strip().splitlines() or ["?"])[-1])[:160], time.time() - t0


def main():
    args = [a for a in sys.argv[1:] if a != "--chunk-only"]
    engine = "--chunk-only" not in sys.argv[1:]
    if engine and platform.system() == "Darwin":
        sys.exit("the ENGINE group builds a toy model: PC only (use --chunk-only here)")
    tok = args[0] if args else "HuggingFaceTB/SmolLM2-135M-Instruct"
    rc, msg, sec = run(None, None, tok, engine)
    print(f"baseline rc {rc} {msg} ({sec:.0f} s)", flush=True)
    bad = rc != 0
    groups = [("CHUNK", CHUNK)] + ([("ENGINE", ENGINE)] if engine else [])
    n = 0
    for name, mutants in groups:
        for k, (old, new, what) in enumerate(mutants):
            rc, msg, sec = run(old, new, tok, engine)
            state = ("KILLED" if rc == 1 and msg.startswith("FAILS")
                     else ("MALFORMED" if rc == 3 else f"SURVIVED/CRASH rc {rc}"))
            bad |= state != "KILLED"
            n += 1
            print(f"{name} #{k} {state}: {what} | {msg} ({sec:.0f} s)", flush=True)
    print(f"ALL {n} HF_BATCHED MUTANTS KILLED" if not bad else "NOT ALL KILLED (or the baseline failed)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
