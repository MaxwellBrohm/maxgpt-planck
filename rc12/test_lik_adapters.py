"""Test of the log-prob arithmetic in lik_adapters.py (HFLik, PlanckLik, seq_logprob). Needs torch, so run it with a
Python that has torch; it loads NO language model: the "model" is a fixed random bigram table (logits at position
t depend only on token t), run on synthetic token ids, never on RC-12 rows. The expected values are computed here
in plain Python from the same table.
  <python with torch> -B test_lik_adapters.py      (exit 1 on any failure; exit 3 if torch is missing)"""
import math
import sys

try:
    import torch
except ImportError:
    print("test_lik_adapters: NOT RUN (no torch in this Python)")
    sys.exit(3)

import lik_adapters as LA

V = 11
G = torch.Generator().manual_seed(1212)
W = torch.randn(V, V, generator=G)
FAILS = []


def check(ok, msg):
    if not ok:
        FAILS.append(msg)
        print("FAIL", msg)


class HFBigram(torch.nn.Module):
    def forward(self, input_ids=None, attention_mask=None):
        return type("Out", (), {"logits": W[input_ids]})()


class PlanckBigram(torch.nn.Module):
    def forward(self, x):
        return (W[x], None)


def expected(pre, cont):
    total, seq = 0.0, list(pre) + list(cont)
    for k in range(len(pre), len(seq)):
        row = [float(v) for v in W[seq[k - 1]]]
        mx = max(row)
        lse = mx + math.log(sum(math.exp(v - mx) for v in row))
        total += row[seq[k]] - lse
    return total


def main():
    hf = LA.HFLik(torch, HFBigram(), "cpu", ctx=12, pad=0)
    pl = LA.PlanckLik(torch, PlanckBigram(), "cpu", ctx=12)
    pre, conts = [3, 1, 4, 1, 5], [[9], [2, 6], [5, 3, 5], [8, 9, 7, 9]]
    for c in conts:
        e = expected(pre, c)
        check(abs(hf(pre, c) - e) < 1e-4, f"HFLik {c}: {hf(pre, c)} != {e}")
        check(abs(pl(pre, c) - e) < 1e-4, f"PlanckLik {c}: {pl(pre, c)} != {e}")
    for name, sc in (("HFLik", hf), ("PlanckLik", pl)):
        got = sc.many(pre, conts)
        check(all(abs(g - expected(pre, c)) < 1e-4 for g, c in zip(got, conts)), f"{name}.many (padded) {got}")
    check(abs(hf([3], [1]) - float(torch.log_softmax(W[3], -1)[1])) < 1e-5, "one-token prefix")
    # logit budget (STEP 14S): rows x widest x vocab x 4 bytes per forward; the scores never depend on the grouping
    widths = [len(pre) + len(c) for c in conts]                        # 6, 7, 8, 9
    for budget, sizes in ((10 ** 9, [4]), (2 * 9 * V * 4, [2, 2]), (2 * 8 * V * 4, [2, 1, 1]), (1, [1, 1, 1, 1])):
        gs = LA.groups(widths, V, budget)
        check([len(g) for g in gs] == sizes, f"groups at {budget}: {gs}")
        h = LA.HFLik(torch, HFBigram(), "cpu", ctx=12, pad=0, vocab=V, budget=budget)
        got = h.many(pre, conts)
        check(h.forwards == sizes, f"forwards at budget {budget}: {h.forwards} != {sizes}")
        check(all(abs(g - expected(pre, c)) < 1e-4 for g, c in zip(got, conts)), f"grouped many at {budget}: {got}")
    check(LA.groups(widths, None, 1) == [[0, 1, 2, 3]], "vocab None splits")
    check(LA.groups([9, 6, 6], V, 2 * 9 * V * 4) == [[0, 1], [2]], "a group's cost uses its widest row")
    check(sum(LA.groups([5] * 7, V, 3 * 5 * V * 4), []) == list(range(7)), "groups lose or reorder rows")
    for name, fn in (("HFLik", lambda: hf(pre, [1] * 8)), ("PlanckLik", lambda: pl(pre, [1] * 8)),
                     ("seq_logprob empty prefix", lambda: LA.seq_logprob(torch, W[[1, 2]], [1, 2], 0, 2))):
        try:
            fn()
            check(False, f"{name}: no error on an over-long or empty-prefix sequence")
        except ValueError:
            pass
    print(f"test_lik_adapters: {len(FAILS)} failures")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
