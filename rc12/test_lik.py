"""Tests of the likelihood rows (lik_rows.py, lik_score.py, pools_lik_prefix.py, lik_run.py, lik_stubs.py and the
tokenizer wrappers of lik_adapters.py). Stub tokenizers and stub scorers only: no model, no torch. The expected
prompts are rebuilt here from the dev records (not through lik_rows or render.py), so the IDEAL oracle raises on
any history, render or lead-in that differs. An exception inside a test is reported as that test's FAIL. The
tokenizer-wrapper test and the STEP 11 FIX ROUND tests are in test_lik_more.py.
  python3 -B test_lik.py      (exit 1 on any failure; mutation_lik.py runs it against its mutants)"""
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from os.path import commonprefix

import common as C
import grade_text as T
import lik_adapters as LA  # noqa: F401  (importing it must not import torch: test_cli)
import lik_rows as LR
import lik_score as LS
import lik_stubs as ST
import pools_lik_prefix as LP
import pools_vals as V

HERE = os.path.dirname(os.path.abspath(__file__))
RECS = LR.load()
BY_ID = {r["id"]: r for r in RECS}
ROWS, EXCL = LR.build(RECS)
FAILS = []


def check(ok, msg):
    if not ok:
        FAILS.append(msg)
        print("FAIL", msg)


def lead_of(row):                # BIND twins share the lead-in of their pair (STEP 11 FIX ROUND)
    rec = BY_ID[row["rid"]]
    return LP.lead_for(rec["meta"]["pair_id"] if rec["family"] == "BIND" else rec["id"], row["turn"])


def expected_prompt(row, render):
    rec = BY_ID[row["rid"]]
    turns = sorted(rec["turns"], key=lambda t: t["i"])
    before, q = [t for t in turns if t["i"] < row["turn"]], turns[row["turn"] - 1]["text"]
    if render == "plain":
        lines = [x for t in before for x in (f"User: {t['text']}", f"Assistant: {t['ideal']}")]
        return "\n".join(lines + [f"User: {q}", f"Assistant: {lead_of(row)}"])
    body = "".join(f"<|im_start|>user\n{t['text']}<|im_end|>\n<|im_start|>assistant\n{t['ideal']}<|im_end|>\n"
                   for t in before)
    return body + f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n" + lead_of(row)


def ideal_for(tok, render):
    return ST.Ideal(tok, [(expected_prompt(r, render), r["gold"], r["candidates"]) for r in ROWS])


def test_pool():
    check(LP.SEED != C.SEED, "the prefix seed equals the dev seed")
    values = [v for pool in V.POOLS.values() for v in pool] + [c for r in ROWS for c in r["candidates"]]
    bad = [(lead, v) for lead in LP.LEADS for v in set(values) if V.mentions(lead, v, False)]
    check(not bad, f"lead-ins mention values {bad[:3]}")
    check(all(lead_of(r) == r["lead"] for r in ROWS), "lead draw")
    use = Counter(r["lead"] for r in ROWS)
    check(set(use) == set(LP.LEADS) and min(use.values()) >= len(ROWS) / len(LP.LEADS) / 2, f"lead use {use}")


def test_rows():
    want = dict(RECALL=48, CORR=64, BIND=60, TWOHOP=48, COMPOSE=16, OWN=32, TOPIC=48, ROLE=48, LOOKUP=48)
    check(dict(Counter(r["key"] for r in ROWS)) == want, f"rows per key {Counter(r['key'] for r in ROWS)}")
    ex = {f"{k}:{w}": n for (k, w), n in EXCL.items()}
    check(ex == {"K:knowledge": 64, "LOOKUP:no_gold": 48, "LOOP:no_gold": 480, "OWN:no_gold": 16,
                 "PERSIST:no_gold": 216, "RECALL:no_gold": 12, "RECALL:one_candidate": 12,
                 "ROLE:open_question": 48, "T0:one_candidate": 96}, f"exclusions {ex}")
    for r in ROWS:
        m = r["messages"]
        ok = (r["gold"] in r["candidates"] and len(set(r["candidates"])) == len(r["candidates"]) >= 2 and
              len(m) == 2 * r["turn"] - 1 and
              [x["role"] for x in m] == ["user", "assistant"] * (r["turn"] - 1) + ["user"])
        check(ok, f"{r['id']}: row shape")
        check(not any(V.mentions(r["lead"], c, False) for c in r["candidates"]), f"{r['id']}: lead names a value")
    for r in (x for x in ROWS if x["key"] == "OWN"):
        p = next(p for p in BY_ID[r["rid"]]["probes"] if p["turn"] == r["turn"])
        src = T.norm(BY_ID[r["rid"]]["turns"][p["gold_fn"]["src_turn"] - 1]["ideal"])
        picked = [o for o in p["gold_fn"]["options"] if T.asserted_hits(src, o, p["gold_fn"]["options"])]
        check(picked == [r["gold"]], f"{r['id']}: G-DYN parses {picked} from the IDEAL Q reply, row gold {r['gold']}")
    check(json.dumps(LR.build(LR.load())[0]) == json.dumps(ROWS), "build not deterministic")


def test_ideal():
    for name in ("word", "chunk", "merge", "canonical", "eos"):
        for render in ("plain", "template"):
            tok = ST.make_tok(name)
            sc = LS.score_rows(ROWS, ideal_for(tok, render), tok, render)
            check(all(r["right"] and r["margin"] > 0 for r in sc), f"IDEAL {name} {render}: a row not right")
            summ = LS.summarize(sc)
            for key, s in summ.items():
                check(s["acc_matched"] in (1.0, None) and s["acc_unequal"] in (1.0, None) and
                      s["tier0"]["n"] == s["n_matched"], f"IDEAL {name} {render} {key}: {s}")
            n_uneq, n_split = sum(s["n_unequal"] for s in summ.values()), sum(s["n_split"] for s in summ.values())
            hows = {h for r in sc for h in r["how"]}
            check(name != "word" or n_uneq == 0, f"word tokenizer: {n_uneq} unequal rows")
            check(name != "chunk" or n_uneq > 0, "chunk tokenizer: no unequal rows (the fixture must try)")
            check(name != "merge" or n_split > 0, "merge tokenizer: the split path is never taken")
            check(name not in ("eos", "word") or hows == {"joint"}, f"{name} {render}: {hows}")
            if name == "canonical":
                check(hows == {"canonical"}, f"canonical {render}: {hows}")
                for r in sc:
                    text = expected_prompt(r, render)
                    common = commonprefix([text + " " + c for c in r["candidates"]])
                    want = (render == "plain") + len(common) // 3
                    check(r["n_prompt"] == want, f"{r['id']}: canonical prefix {r['n_prompt']} != {want}")


def test_length():
    for render in ("plain", "template"):
        tok = ST.make_tok("chunk")
        sc = LS.score_rows(ROWS, ST.length, tok, render)
        for r in sc:
            n = {c: len(tok.encode(" " + c, False)) for c in r["candidates"]}
            short = [c for c in n if n[c] == min(n.values())]
            check(r["ntok"] == n and r["equal"] == (len(set(n.values())) == 1), f"{r['id']}: ntok {r['ntok']} {n}")
            check(r["right"] == (short == [r["gold"]]), f"{r['id']}: LENGTH right {r['right']} {n}")
        summ = LS.summarize(sc)
        for key, s in summ.items():
            t = s["tier0"]
            check(s["acc_equal"] in (0.0, None) and t["mean"] in (0.0, None) and t["p10"] in (0.0, None) and
                  s["acc_matched"] in (0.0, None) and s["n_ties"] == s["n_matched"], f"LENGTH {render} {key}: {s}")
        check(sum(r["right"] for r in sc if not r["equal"]) > 0, "LENGTH never right on an unequal row")
        tw = defaultdict(list)
        for r in sc:
            if r["key"] == "BIND":
                tw[r["pair_id"]].append(r)
        n_eq = sum(all(t["equal"] for t in v) for v in tw.values())
        n_credit = sum(all(t["right"] for t in v) for v in tw.values())
        check(summ["BIND"]["pairs"] == dict(n=n_eq, acc=0.0) and n_eq < len(tw) and n_credit > 0,
              f"LENGTH {render} BIND pairs {summ['BIND']['pairs']} (equal pairs {n_eq} of {len(tw)}, "
              f"length-credited pairs {n_credit})")


def last_mention(text, c):
    ends = [m.end() for m in re.finditer(r"\b%s\b" % re.escape(c), text, re.I)]
    return float(ends[-1]) if ends else -1.0


def test_recency():
    tok = ST.make_tok("word")
    sc = LS.score_rows(ROWS, ST.Recency(tok), tok, "plain")
    exp = {}
    for r in sc:
        pos = {c: last_mention(expected_prompt(r, "plain"), c) for c in r["candidates"]}
        others = [v for c, v in pos.items() if c != r["gold"]]
        exp[r["id"]] = all(pos[r["gold"]] > v for v in others)
        check(r["right"] == exp[r["id"]] and r["margin"] == pos[r["gold"]] - max(others), f"{r['id']}: recency")
    tw = defaultdict(list)
    for r in sc:
        if r["key"] == "BIND":
            tw[r["pair_id"]].append(exp[r["id"]])
    want = sum(all(v) for v in tw.values()) / len(tw)
    got = LS.summarize(sc)["BIND"]["pairs"]
    check(got == dict(n=len(tw), acc=want) and any(any(v) and not all(v) for v in tw.values()),
          f"BIND pairs {got}, want {want}")


def test_rules():
    check(LS.p10([5, 1, 4, 2, 3, 6, 7, 8, 9, 10]) == 1 and LS.p10(list(range(11, 0, -1))) == 2, "p10 rank")
    check(LS.p10(list(range(20))) == 1 and LS.p10([]) is None, "p10 rank (20, empty)")
    check(not LS.right({"a": 1.0, "b": 1.0}, "a") and LS.right({"a": 2.0, "b": 1.0}, "a"), "strict rule")
    check(not LS.right({"a": math.nan, "b": 1.0}, "a") and not LS.right({"a": 1.0}, "a"), "nan / one candidate")
    check(LS.margin({"a": 2.0, "b": 1.0, "c": 1.5}, "a") == 0.5 and LS.margin({"a": math.inf, "b": 1}, "a") is None,
          "margin")


def test_bos_ctx_many():
    tok, seen = ST.make_tok("word"), []
    for render in ("plain", "template"):
        LS.score_rows(ROWS[:20], lambda pre, cont: seen.append((render, pre[0] == tok.BOS)) or 0.0, tok, render)
    check(all(bos == (render == "plain") for render, bos in seen), "BOS only on the plain render")
    need = {r["id"]: len(tok.encode(expected_prompt(r, "plain"), True)) +
            max(len(tok.encode(" " + c, False)) for c in r["candidates"]) for r in ROWS}
    ctx = sorted(need.values())[len(need) // 2]
    sc = LS.score_rows(ROWS, ST.length, tok, "plain", ctx)
    check(sum(r["over_ctx"] for r in sc) == sum(v > ctx for v in need.values()) > 0, "over_ctx count")
    check(all((r["scores"] is None) == r["over_ctx"] for r in sc), "over_ctx rows scored")
    for s in LS.summarize(sc).values():
        check(s["n_equal"] + s["n_unequal"] + s["n_over_ctx"] == s["n"], f"over_ctx rows counted {s}")

    class Many:                     # not callable: only lik_score's many() path can score with it
        def many(self, pre, conts):
            return [ST.length(pre, c) for c in conts]
    tok = ST.make_tok("chunk")
    a, b = LS.score_rows(ROWS, Many(), tok, "template"), LS.score_rows(ROWS, ST.length, tok, "template")
    check(a == b, "the many() path differs from the per-candidate path")


def test_cli():
    out = tempfile.mkdtemp(prefix="rc12_lik_")
    run = lambda *a: subprocess.run([sys.executable, "-B", os.path.join(HERE, "lik_run.py"), *a],
                                    capture_output=True, text=True, timeout=110)
    p = run("--responder", "stub:IDEAL", "--tok", "chunk", "--render", "both", "--out", out)
    check(p.returncode == 0, f"lik_run stub:IDEAL exit {p.returncode} {p.stderr[-300:]}")
    for render in ("template", "plain"):
        s = json.load(open(os.path.join(out, render, "summary.json")))
        check(s["rows"] == 412 and all(f["acc_equal"] == 1.0 for f in s["families"].values()), f"cli {render}")
    check(json.load(open(os.path.join(out, "excluded.json")))["K:knowledge"] == 64, "cli excluded.json")
    p = run("--responder", "hf:none/none", "--out", out)
    check(p.returncode != 0 and "UNTESTED" in p.stdout + p.stderr, "hf: without --untested-ok not refused")
    p = run("--responder", "planck:none.pt", "--untested-ok", "--out", out)
    check(p.returncode != 0 and "pushed" in p.stdout + p.stderr, "planck: without --prereg-pushed not refused")
    check("torch" not in sys.modules, "importing the lik modules imported torch")
    shutil.rmtree(out, ignore_errors=True)


def main():
    for t in (test_pool, test_rows, test_ideal, test_length, test_recency, test_rules, test_bos_ctx_many, test_cli):
        try:
            t()
        except Exception as e:                      # reported as the test's FAIL, never a silent pass
            check(False, f"{t.__name__} raised {type(e).__name__}: {str(e)[:200]}")
    print(f"test_lik: {len(FAILS)} failures")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
