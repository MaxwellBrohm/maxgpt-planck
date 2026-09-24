"""E004: token lengths of the training stream per tokenizer family (tokenizers only, no model), to see how many
examples the trainer's 768-token rejection will drop and whether the drop is even across kinds (notes (a):
kept share within 2 points of the drawn share). usage: python3 len_train.py > ../logs/len_train.txt"""
import os
import sys
sys.dont_write_bytecode = True
os.environ["HF_HUB_OFFLINE"] = "1"
from collections import Counter
from transformers import AutoTokenizer
import train_e004 as T

N, MAX = 3000, 768
exs = T.take(0, N)
texts = [T.prompt(ex) + ex["answer"] for ex in exs]
drawn = Counter(ex["kind"] for ex in exs)
for m in ["HuggingFaceTB/SmolLM2-135M-Instruct", "roneneldan/TinyStories-1M", "EleutherAI/pythia-14m"]:
    tok = AutoTokenizer.from_pretrained(m)
    lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    kept = [ex for ex, n in zip(exs, lens) if n <= MAX]
    kc = Counter(ex["kind"] for ex in kept)
    s = sorted(lens)
    print(f"{m}: n={N} median={s[N // 2]} p95={s[int(N * .95)]} max={s[-1]} over {MAX}: {N - len(kept)} "
          f"({(N - len(kept)) / N:.3f})")
    worst = max(abs(kc[k] / len(kept) - drawn[k] / N) for k in drawn)
    for k in T.KINDS:
        print(f"   {k:12s} drawn {drawn[k] / N:.3f} kept {kc[k] / len(kept):.3f}")
    print(f"   largest kept-vs-drawn share change: {worst:.4f} (limit 0.02)")
