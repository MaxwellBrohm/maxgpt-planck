"""More tests of the likelihood rows (STEP 11 FIX ROUND): the tokenizer wrappers (moved from test_lik.py), BIND twins
sharing a lead-in, matched foils, the non-finite and tie counts of the Tier 0 guard, lik_compare, the HF attention
choice and the float32 defaults (fake transformers / torch modules, no model), and the OWN copying property. Stub
tokenizers and scorers only. An exception inside a test is reported as that test's FAIL.
  python3 -B test_lik_more.py      (exit 1 on any failure; mutation_lik.py runs it against its mutants)"""
import contextlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import types
from collections import defaultdict

import lik_adapters as LA
import lik_compare as LC
import lik_run as LU
import lik_score as LS
import lik_stubs as ST
import test_lik as TL

ROWS, BY_ID = TL.ROWS, TL.BY_ID
FAILS = []


def check(ok, msg):
    if not ok:
        FAILS.append(msg)
        print("FAIL", msg)


def test_wrappers():
    class FakeHF:
        eos_token_id = None

        def __call__(self, text, add_special_tokens):
            return type("E", (), {"input_ids": ([7] if add_special_tokens else []) + [ord(ch) for ch in text]})()

        def apply_chat_template(self, messages, add_generation_prompt, tokenize, enable_thinking=True):
            return "".join(f"[{m['role']}]{m['content']}" for m in messages) + f"[assistant{enable_thinking}]"
    hf = LA.HFTok(FakeHF(), qwen3=True)
    sc = LS.score_rows(ROWS[:5], ST.length, hf, "plain")
    check([r["n_prompt"] for r in sc] == [len(TL.expected_prompt(r, "plain")) + 1 for r in ROWS[:5]], "HFTok plain")
    check(hf.template(ROWS[0]["messages"]).endswith("[assistantFalse]"), "HFTok: thinking not off for Qwen3")

    class FakePlanck:
        def encode_text(self, text):
            return [ord(ch) for ch in text]

        def encode(self, messages):
            return [9] * len(messages) + [8]
    sc = LS.score_rows(ROWS[:5], ST.length, LA.PlanckTok(FakePlanck()), "template")
    check([r["n_prompt"] for r in sc] == [len(r["messages"]) + 1 + len(r["lead"]) for r in ROWS[:5]],
          "PlanckTok template: role-token ids, then the lead-in text")


class PosByLead:
    """the reviewer's cheater, blind to binding: the FIRST-mentioned candidate after lead-ins in A, else the LAST."""
    def __init__(self, tok, A):
        self.tok, self.A = tok, A

    def __call__(self, pre, cont):
        hist, _, lead = self.tok.decode(pre).rpartition("Assistant: ")
        pos = [m.start() for m in re.finditer(r"\b%s\b" % re.escape(self.tok.decode(cont).strip()), hist, re.I)]
        p = pos[0] if pos else -1
        return -float(p) if lead in self.A else float(p)


def test_bind_lead():
    tw = defaultdict(list)
    for r in ROWS:
        if r["key"] == "BIND":
            tw[r["pair_id"]].append(r)
    check(len(tw) == 30 and all(len(v) == 2 and v[0]["lead"] == v[1]["lead"] for v in tw.values()),
          "BIND twins do not share a lead-in")
    ndiff = {sum(x != y for x, y in zip(a["messages"], b["messages"])) for a, b in tw.values()}
    check(ndiff == {2}, f"BIND twins differ in {ndiff} messages, not exactly the 2 value statements")
    tok = ST.make_tok("word")
    bind = [r for r in ROWS if r["key"] == "BIND"]
    for A in ((), ("It is", "It's", "That's", "The answer would be"), tuple(TL.LP.LEADS)):
        got = LS.summarize(LS.score_rows(bind, PosByLead(tok, set(A)), tok, "plain"))["BIND"]["pairs"]
        check(got == dict(n=30, acc=0.0), f"a lead-dependent position cheater passes BIND pairs with A={A}: {got}")


def rigged(rows, tok, values):
    """score each row with fixed values: values(row, ntok) -> {candidate: score}."""
    out = []
    for r in rows:
        v = values(r, {c: len(tok.encode(" " + c, False)) for c in r["candidates"]})
        out.append(LS.score_row(r, lambda pre, cont, v=v: v[tok.decode(cont).strip()], tok, "plain"))
    return out


def three(gold_v, same_v, other_v):
    """gold, the gold's same-count foils, the other candidates."""
    return lambda r, n: {c: gold_v if c == r["gold"] else same_v if n[c] == n[r["gold"]] else other_v for c in n}


def test_matched():
    tok = ST.make_tok("chunk")
    sc = rigged(ROWS, tok, three(10.0, 0.0, 20.0))    # unequal-count foils beat the gold, same-count foils lose
    for r in sc:
        want = [c for c in r["candidates"] if c != r["gold"] and r["ntok"][c] == r["ntok"][r["gold"]]]
        check(r["foils"] == want and r["chance"] == (1 / (1 + len(want)) if want else None), f"{r['id']} foils")
        check(r["right_m"] == (True if want else None) and r["right"] == r["equal"], f"{r['id']}: {r['right_m']}")
    summ = LS.summarize(sc)
    tot = {k: s["n_matched"] for k, s in summ.items()}
    check(all(s["acc_matched"] in (1.0, None) for s in summ.values()) and sum(tot.values()) > sum(
        s["n_equal"] for s in summ.values()), f"matched rows {tot}")
    check(all(abs(s["tier0"]["mean"] - 10.0) < 1e-9 for s in summ.values() if s["n_matched"]), "tier0 not 10")
    n_un = sum(s["n_unmatched"] for s in summ.values())
    check(n_un > 0 and all(s["n_matched"] + s["n_unmatched"] == s["n"] for s in summ.values()), f"unmatched {n_un}")


def test_nonfinite_ties():
    tok = ST.make_tok("word")
    rec = [r for r in ROWS if r["key"] == "RECALL"]
    bad = {r["id"] for r in rec[::2]}            # the reviewer's case: the gold -inf on 24 of 48 rows
    good, inf = rigged(rec, tok, three(2.0, -1.0, -1.0)), rigged(rec, tok, three(-math.inf, -1.0, -1.0))
    s = LS.summarize([b if b["id"] in bad else g for g, b in zip(good, inf)])["RECALL"]
    check(s["n_nonfinite"] == 24 and s["acc_matched"] == 0.5 and s["tier0"]["n"] == 48 and
          s["tier0"]["mean"] == -math.inf and s["tier0"]["p10"] == -math.inf, f"non-finite gold hidden: {s}")
    s = LS.summarize(inf[:1] + good[1:])["RECALL"]
    check(s["n_nonfinite"] == 1 and s["tier0"]["mean"] == -math.inf, f"one non-finite row hidden by the mean {s}")
    s = LS.summarize(rigged(rec, tok, three(1.0, 1.0, 1.0)))["RECALL"]
    check(s["n_ties"] == 48 and s["acc_matched"] == 0.0 and s["tier0"]["mean"] == 0.0, f"ties {s}")


def test_compare():
    ta, tb = ST.make_tok("chunk"), ST.StubTok(chunk=3)
    va = lambda r, n: {c: 10.0 if c == r["gold"] else 20.0 if n[c] == n[r["gold"]] else 0.0 for c in n}
    rows_a, rows_b = [], []
    for r in ROWS:
        na = {c: len(ta.encode(" " + c, False)) for c in r["candidates"]}
        nb = {c: len(tb.encode(" " + c, False)) for c in r["candidates"]}
        common = [c for c in r["candidates"] if c != r["gold"] and na[c] == na[r["gold"]] and nb[c] == nb[r["gold"]]]
        vb = {c: 10.0 if c == r["gold"] else (5.0 if c in common else 30.0) for c in nb}   # B loses only off-common
        rows_a.append(LS.score_row(r, lambda p, c, v=va(r, na): v[ta.decode(c).strip()], ta, "plain"))
        rows_b.append(LS.score_row(r, lambda p, c, v=vb: v[tb.decode(c).strip()], tb, "plain"))
    res = LC.compare(rows_a, rows_b)
    n_common = sum(bool(LC.common_foils(a, b)) for a, b in zip(rows_a, rows_b))
    check(sum(s["n_common"] for s in res.values()) == n_common > 0, f"common rows {n_common}")
    check(all(s["acc_a"] in (0.0, None) and s["acc_b"] in (1.0, None) for s in res.values()),
          f"compare does not judge on the common foils {res}")
    check(sum(s["n_common"] for s in res.values()) < sum(s["n_matched"] for s in LS.summarize(rows_a).values()) and
          any(set(b["foils"]) - set(LC.common_foils(a, b)) for a, b in zip(rows_a, rows_b)), "no foil dropped")
    bad = [dict(rows_b[0], lead="It is" if rows_b[0]["lead"] != "It is" else "It's")] + rows_b[1:]
    for other, needle in ((bad, "differ between"), (rows_b[1:], "row ids differ")):
        try:
            LC.compare(rows_a, other)
            check(False, f"compare accepted rows that {needle}")
        except ValueError as e:
            check(needle in str(e), f"compare refusal {e}")


def fake_modules(model_types, calls):
    class Model:
        def __init__(self, mt, kw):
            self.config = types.SimpleNamespace(model_type=mt, max_position_embeddings=64,
                                                _attn_implementation=kw.get("attn_implementation", "sdpa"))

            self.to, self.eval = (lambda device: self), (lambda: self)
    tf, th = types.ModuleType("transformers"), types.ModuleType("torch")
    ns = types.SimpleNamespace
    tf.AutoConfig = ns(from_pretrained=lambda m: ns(model_type=model_types[m]))
    tf.AutoTokenizer = ns(from_pretrained=lambda m: ns(pad_token_id=0, eos_token_id=1))
    tf.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda m, **kw: calls.append((m, kw)) or Model(model_types[m], kw))
    th.float32, th.bfloat16 = "F32", "BF16"
    return dict(transformers=tf, torch=th)


def test_attn_dtype():
    calls, saved = [], {k: sys.modules.get(k) for k in ("transformers", "torch")}
    sys.modules.update(fake_modules(dict(g="gemma3_text", q="qwen2"), calls))
    try:
        sc, _ = LA.HFLik.load("g")
        check(calls[-1][1] == dict(torch_dtype="F32", attn_implementation="eager") and sc.info["attn"] == "eager"
              and sc.info["dtype"] == "float32", f"Gemma 3 not loaded eager in float32: {calls[-1]} {sc.info}")
        sc, _ = LA.HFLik.load("q", "bfloat16")
        check(calls[-1][1] == dict(torch_dtype="BF16") and sc.info["attn"] == "sdpa", f"qwen2 {calls[-1]}")
        try:
            LA.HFLik.load("g", attn="sdpa")
            check(False, "Gemma 3 with sdpa not refused")
        except ValueError:
            pass
    finally:
        for k, v in saved.items():
            sys.modules.pop(k, None) if v is None else sys.modules.__setitem__(k, v)
    seen, out = {}, tempfile.mkdtemp(prefix="rc12_likm_")

    class Scorer:
        ctx, info = None, dict(model_type="fake", attn="eager")

        def __call__(self, pre, cont):
            return 0.0

    def fake_load(kind):
        def load(*a):
            seen[kind] = a
            return Scorer(), ST.make_tok("word")
        return load
    old_hf, old_pl = LA.HFLik.load, LA.PlanckLik.load
    LA.HFLik.load, LA.PlanckLik.load = fake_load("hf"), fake_load("planck")
    try:
        with contextlib.redirect_stdout(io.StringIO()) as console:
            LU.main(["--responder", "hf:x/y", "--untested-ok", "--render", "template", "--families", "BIND",
                     "--out", out])
            LU.main(["--responder", "planck:x.pt", "--untested-ok", "--prereg-pushed", "--render", "template",
                     "--families", "BIND", "--out", out + "/p"])
        s = json.load(open(os.path.join(out, "template", "summary.json")))
        check(seen["hf"][1] == "float32" and s["engine"] == Scorer.info, f"lik_run hf defaults {seen} {s['engine']}")
        check("nonfinite 0 ties 60" in console.getvalue(), "the console line lacks n_nonfinite / n_ties")
        check(seen["planck"][4] == "fp32", f"lik_run planck precision default {seen['planck']}")
    finally:
        LA.HFLik.load, LA.PlanckLik.load = old_hf, old_pl
        shutil.rmtree(out, ignore_errors=True)


def test_own_copying():
    tok = ST.make_tok("word")
    sc = LS.score_rows(ROWS, ST.Mentions(tok), tok, "plain")
    right = defaultdict(list)
    for r in sc:
        right[r["key"]].append(r["right"])
    check(all(right["OWN"]) and len(right["OWN"]) == 32, "the most-mentioned heuristic no longer solves OWN")
    check(all(sum(v) <= 4 for k, v in right.items() if k != "OWN"), {k: sum(v) for k, v in right.items()})
    summ = LS.summarize(sc)
    check([k for k, s in summ.items() if "note" in s] == ["OWN"] and "copying" in summ["OWN"]["note"], "OWN note")


def main():
    for t in (test_wrappers, test_bind_lead, test_matched, test_nonfinite_ties, test_compare, test_attn_dtype,
              test_own_copying):
        try:
            t()
        except Exception as e:                      # reported as the test's FAIL, never a silent pass
            check(False, f"{t.__name__} raised {type(e).__name__}: {str(e)[:200]}")
    check("torch" not in sys.modules and "transformers" not in sys.modules, "torch or transformers left imported")
    print(f"test_lik_more: {len(FAILS)} failures")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
