"""E005 stream acceptance on the examples each seed actually trains on (notes.txt WHAT IS RE-CHECKED). No model is
loaded; the SmolLM2 tokenizer is (the 768-token rejection needs it). Per seed 1-5: the first N_KEPT = 6,404 examples
KEPT by e005_sets.example_stream (E005's encoder, lik.cand_ids, max length 768), which is what e005_train consumes
(4 for the first-batch loss self-check, then 400 steps x 16):
  1 oracles  every O and X rule <= 0.70, O1 and O2 <= 0.50, IDEAL 1.00 (oracles_e005.score and gate), every seed
  2 shift    drawn vs kept share per key (kind, block, render, case, placement) moves by <= 2 points (notes), every seed
  3 purity   purity_e005.violations is empty on every kept example (purity_e005 itself stops at 6,404 DRAWN
             examples, and the kept examples run past that by the rejected count)
  info       shares_e005.share_checks on the pooled kept examples (the targets were set on the drawn stream;
             printed, not gated: the pre-registered kept rule is 2)
usage: <venv python> -B validate_e005.py [--seeds 1 2 3 4 5] [--n 6404] > ../logs/validate_e005.stdout"""
import argparse
import os
import sys
from collections import Counter

os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e005_sets as S
import lik
import oracles_e005 as OR
import purity_e005 as PU
import shares_e005 as SH
import train_e005 as T5

SMOL = "HuggingFaceTB/SmolLM2-135M-Instruct"
N_KEPT, MAX_LEN, MAX_SHIFT = 6404, 768, 0.02


def kept_examples(tok, seed, n):
    """-> (the first n kept examples, the stats of the drawn prefix that produced them)"""
    drawn, kept, stats = [], [], S.new_stats()

    def src():
        for ex in T5.stream(seed):
            drawn.append(ex)
            yield ex

    for _ids, _labels, r in S.example_stream(tok, seed, MAX_LEN, stats, lik.cand_ids, source=src()):
        ex = drawn[-1]
        if ex["render"] != r:
            raise SystemExit(f"seed {seed}: kept example and yielded render disagree")
        kept.append(ex)
        if len(kept) == n:
            break
    return kept, stats, len(drawn)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    ap.add_argument("--n", type=int, default=N_KEPT)
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SMOL)
    S.eot_id(tok)  # refuses a tokenizer whose eos is not <|im_end|>
    eg = PU.eval_alias_grams()
    fails, pooled = [], []
    for s in a.seeds:
        kept, stats, nd = kept_examples(tok, s, a.n)
        if len(kept) != a.n:
            fails.append(f"seed {s}: only {len(kept)} kept examples")
        pooled += kept
        overall, per, nb = OR.score(kept)
        bad = OR.gate(overall)
        fails += [f"seed {s} oracle: {b}" for b in bad]
        summ = S.stream_summary(stats)
        shift = summ["max_share_shift"]
        if shift is None or shift > MAX_SHIFT:
            fails.append(f"seed {s}: drawn-vs-kept shift {shift} > {MAX_SHIFT}")
        pv = Counter()
        for ex in kept:
            for axis in PU.violations(ex, eg):
                pv[axis] += 1
        if pv:
            fails.append(f"seed {s}: purity {dict(pv)}")
        mx = max(overall[n] for n, _ in OR.FAKE)
        worst = max(((k, v, abs(x["drawn"] - x["kept"])) for k, d in summ["shares"].items() for v, x in d.items()
                     if x["drawn"] is not None and x["kept"] is not None), key=lambda t: t[2])
        print(f"seed {s}: drawn {nd}, rejected {stats['rejected_long']}, kept {len(kept)} (blocks {nb}); "
              f"render kept {dict(stats['kept']['render'])}; max rule {mx:.3f} O1 {overall['O1 first mention']:.3f} "
              f"O2 {overall['O2 last mention']:.3f} IDEAL {overall['IDEAL']:.2f} {'OK' if not bad else 'FAIL'}; "
              f"largest shift {shift:.4f} ({worst[0]}={worst[1]}); purity violations {sum(pv.values())}")
    print("\nshares on the pooled kept examples (info; targets and tolerances are the drawn-stream ones):")
    for ok, msg in SH.share_checks(pooled):
        print(f"  {'ok  ' if ok else 'off '} {msg}")
    print("\nRESULT: " + ("ALL PASS (oracles, drawn-vs-kept shift <= 2 points, purity 0, on every seed's kept examples)"
                          if not fails else f"{len(fails)} FAILED: {fails}"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
