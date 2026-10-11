"""Likelihood rows: lik_run --selfcheck and the logit budget (rc12/notes.txt STEP 14S; moved here from test_lik_engine.py
at the STEP 14S review, D4 and D5). python3, no torch: stub scorers only, no model and no RC-12 row text beyond the
dev records.
  python3 -B test_lik_selfcheck.py      (mutants: mutation_lik.py, the STEP 14S block)"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile

import lik_adapters as LA
import lik_run as LU
import lik_stubs as ST

FAILS = []


def check(ok, msg):
    if not ok:
        FAILS.append(msg)
        print("FAIL", msg)


class Padded:
    """a scorer whose batched path drifts with the batch size (a padding bug) by `bias` nats per extra row."""

    ctx, info = None, dict(model_type="fake")

    def __init__(self, tok, bias):
        self.inner, self.bias = ST.Recency(tok), bias

    def __call__(self, pre, cont):
        return self.inner(pre, cont)

    def many(self, pre, conts):
        return [self.inner(pre, c) + self.bias * (len(conts) - 1) * (k % 2) for k, c in enumerate(conts)]


class Tied:
    """every candidate scores 0.0 alone (a tie: wrong); the batched path lifts the first candidate by bias."""
    ctx, info = None, dict(model_type="fake")

    def __init__(self, tok, bias):
        self.bias = bias

    def __call__(self, pre, cont):
        return 0.0

    def many(self, pre, conts):
        return [self.bias if k == 0 else 0.0 for k in range(len(conts))]


def selfcheck_run(bias, n=20, dtype="float32", cls=Padded):
    out = tempfile.mkdtemp(prefix="rc12_likself_")
    tok = ST.make_tok("word")
    old = LA.HFLik.load
    LA.HFLik.load = lambda *a: (cls(tok, bias), tok)
    try:
        with contextlib.redirect_stdout(io.StringIO()) as console:
            code = LU.main(["--responder", "hf:q/Qwen", "--untested-ok", "--render", "plain", "--families", "RECALL",
                            "--selfcheck", str(n), "--dtype", dtype, "--out", out])
        s = json.load(open(os.path.join(out, "plain", "summary.json")))
        return code, s.get("selfcheck"), console.getvalue()
    finally:
        LA.HFLik.load = old
        shutil.rmtree(out, ignore_errors=True)


def test_selfcheck():
    code, sc, con = selfcheck_run(0.0)
    check(code == 0 and sc["ok"] and sc["n_rows"] == 20 and sc["max_abs_diff"] == 0.0, f"clean selfcheck {sc}")
    check(sc["n_scores"] >= 40 and "selfcheck 20 rows" in con, f"selfcheck counts {sc}")
    code, sc, con = selfcheck_run(0.5)
    check(code == 4 and not sc["ok"] and sc["max_abs_diff"] >= 0.5 and "SELFCHECK FAILED" in con, f"bias {sc}")
    code, sc, _ = selfcheck_run(0.004)
    check(code == 0 and sc["ok"] and sc["tol"] == 0.01, f"a gap under the fp32 tol {sc}")
    code, sc, _ = selfcheck_run(0.02)
    check(code == 4 and not sc["ok"], f"a gap over the fp32 tol {sc}")
    code, sc, _ = selfcheck_run(0.02, dtype="bfloat16")
    check(code == 0 and sc["tol"] == 0.25, f"bf16 tol {sc}")
    code, sc, _ = selfcheck_run(5000.0, dtype="bfloat16")       # Recency scores are character offsets
    check(code == 4 and sc["flips"] > 0, f"flips counted {sc}")
    code, sc, _ = selfcheck_run(0.001, cls=Tied)
    check(code == 4 and sc["flips"] > 0 and sc["max_abs_diff"] <= sc["tol"], f"a flip under the tol {sc}")
    tok = ST.make_tok("word")
    rows = [r for r in LU.LR.build(LU.LR.load())[0] if r["key"] == "RECALL"][:12]
    n_pre = sorted(LU.LS.score_row(r, ST.Recency(tok), tok, "plain")["n_prompt"] for r in rows)
    ctx = n_pre[6] + 1                                       # about half the rows over ctx
    scored = LU.LS.score_rows(rows, ST.Recency(tok), tok, "plain", ctx)
    live = [r for r in scored if r["scores"] is not None]
    sc = LU.selfcheck(scored, rows, ST.Recency(tok), tok, "plain", ctx, 12, 0.01)
    check(0 < len(live) < 12 and sc["n_rows"] == len(live) and sc["ok"], f"over-ctx rows in selfcheck {sc}")
    empty = LU.selfcheck([], [], Tied(None, 0.0), ST.make_tok("word"), "plain", None, 5, 0.01)
    check(empty["n_rows"] == 0 and empty["ok"] is False, f"no row checked counts as ok {empty}")


def test_flip_rule():
    """review D4: a flip is a change of the MATCHED verdict (right_m, gold against its equal-length foils); the
    all-candidate verdict (right) is reported, never gated. Stored rows are tampered, the scores left equal."""
    tok = ST.make_tok("word")
    rows = [r for r in LU.LR.build(LU.LR.load())[0] if r["key"] == "RECALL"][:40]
    scored = LU.LS.score_rows(rows, ST.Recency(tok), tok, "plain")
    both = [r for r in scored if r["scores"] is not None and r["right_m"] is not None]
    check(len(both) >= 2, f"rows with foils {len(both)}")
    k = scored.index(both[0])
    for field, want_ok in (("right", True), ("right_m", False)):
        tampered = [dict(r) for r in scored]
        tampered[k][field] = not tampered[k][field]
        sc = LU.selfcheck(tampered, rows, ST.Recency(tok), tok, "plain", None, 40, 0.01)
        check(sc["ok"] is want_ok and sc["flips"] == (0 if want_ok else 1) and sc["max_abs_diff"] == 0.0,
              f"{field} flipped in a stored row: {sc}")


def test_logit_budget_pin():
    """review D4: the budget itself is pinned (2 GiB of fp32 logits a forward); test_lik_engine only compared the
    loaded scorer's budget with the same constant."""
    check(LA.LOGIT_BUDGET == 2 * 2 ** 30, f"LOGIT_BUDGET {LA.LOGIT_BUDGET}, not 2 GiB")


def main():
    for t in (test_selfcheck, test_flip_rule, test_logit_budget_pin):
        try:
            t()
        except Exception as e:                      # reported as the test's FAIL, never a silent pass
            check(False, f"{t.__name__} raised {type(e).__name__}: {str(e)[:200]}")
    check("torch" not in sys.modules and "transformers" not in sys.modules, "torch or transformers left imported")
    print(f"test_lik_selfcheck: {len(FAILS)} failures")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
