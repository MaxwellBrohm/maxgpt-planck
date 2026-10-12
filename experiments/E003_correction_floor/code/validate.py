"""Self-test for E002 (no model is loaded; --tok loads only a tokenizer).

1. Graders (metrics_ft): right() fails an empty answer (ties, missing gold, NaN) and a plausible
   wrong answer; the pass rule, the majority rule, lock-in detection and the paired CI behave as
   specified on hand-built fixtures.
2. Oracle "models" on the real eval items: first-mentioned, last-mentioned, empty, plausible-wrong
   and ideal. Only ideal may pass the pass rule; first-mentioned must score 0 on LW10 and
   last-mentioned 0 on TS10 and NU10 (so neither shortcut can pass).
3. No question (or answer prefix) contains its own answer word: every eval set, kbig, and 20,000
   generated training examples.
4. Held-out wording: no normalized sentence frame, question or answer prefix of the training
   generator equals one in any eval set; object nouns, names and fillers are disjoint.
5. Shortcut audit on the training stream: "first value wins" and "last value wins" are each right
   on at most half of the update-family examples; "latest statement about the asked object" is
   right on all of them.
6. (--tok MODEL) tokenization: labels only on answer tokens, no example over max_len after
   rejection, answer tokens decode to the answer, and eval prompt lengths.
Run: python validate.py [--tok HuggingFaceTB/SmolLM2-135M-Instruct]   (exit 1 on any failure)
"""
import argparse, math, random, re, sys
from collections import Counter

import items as I
import items_new as N
import eval_extra as X
import uprobe_items as UP
import khard_items as KH
import kbig_items as KB
import train_data as TD
import metrics_ft as MF

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
    return cond


def words(s):
    return re.findall(r"[a-z0-9']+", s.lower())


STOP = {"of", "the", "and", "de", "da", "van", "von", "le", "la", "a", "an", "in", "on", "to"}


def answer_leak(question, prefix, gold):
    """True if any word of the gold (or a >=5-letter stem of it) appears in the question or prefix."""
    qw = set(words(question) + words(prefix)) - STOP
    for w in words(gold):
        if w in STOP:
            continue
        if w in qw:
            return True
        if len(w) >= 5 and any(len(x) >= 5 and (x.startswith(w[:5]) and w.startswith(x[:5])) for x in qw):
            return True
    return False


# ---------------- 1. grader fixtures ----------------
def test_graders():
    R = MF.right
    check(not R({}), "right(): empty dict passes")
    check(not R({"gold": -1.0}), "right(): gold-only passes")
    check(not R({"gold": -1.0, "orig": -1.0}), "right(): tie passes (empty answer)")
    check(not R({"gold": float("nan"), "orig": -3.0}), "right(): NaN gold passes")
    check(not R({"gold": -1.0, "orig": float("nan")}), "right(): NaN foil passes")
    check(not R({"gold": -2.0, "orig": -1.0}), "right(): plausible wrong answer passes")
    check(not R({"gold": -1.0, "orig": -3.0, "other": -0.5}), "right(): wrong with 3 candidates passes")
    check(R({"gold": -1.0, "orig": -3.0}), "right(): correct answer fails")
    P = MF.primary
    def recs(var, d, n, k):
        return [dict(render="plain", var=var, d=d, fam="day", sid=f"day|{d}|{i}",
                     scores={"gold": -1.0, "orig": (-2.0 if i < k else 0.0)}) for i in range(n)]
    base = recs("same_k1", 10, 10, 10) + recs("twoslot", 10, 10, 10)
    check(P(base + recs("noupd", 10, 10, 8))["pass"], "primary: 0.8 exactly should pass")
    check(not P(base + recs("noupd", 10, 10, 7))["pass"], "primary: 0.7 should fail")
    d4good = recs("same_k1", 4, 10, 10) + recs("twoslot", 4, 10, 10) + recs("noupd", 4, 10, 10)
    d10bad = recs("same_k1", 10, 10, 0) + recs("twoslot", 10, 10, 10) + recs("noupd", 10, 10, 10)
    check(not P(d4good + d10bad)["pass"], "primary: must be read at d10, not d4")
    check(not P([dict(r, render="chat") for r in base + recs("noupd", 10, 10, 10)])["pass"],
          "primary: must be read on the plain render only")
    check(not P(recs("same_k1", 10, 10, 10) + recs("noupd", 10, 10, 10))["pass"], "primary: missing two-slot must fail")
    V = MF.model_verdict
    check(V([True, True, True, False, False])["pass"], "verdict: 3/5 should pass")
    check(not V([True, True, False, False, False])["pass"], "verdict: 2/5 should fail")
    check(V([True, True, False])["pass"], "verdict: 2/3 should pass")
    check(not V([True, False])["pass"], "verdict: 1/2 should fail (tie is not most)")
    check(not V([])["pass"], "verdict: no seeds should fail")
    L = MF.lock_in_step
    up = lambda s, x: {"step": s, "LW": x, "TS": x, "NU": x}
    check(L([up(0, .1), up(50, .9), up(100, .9)]) == 50, "lock_in: stays high from 50")
    check(L([up(0, .1), up(50, .9), up(100, .5)]) is None, "lock_in: drops at the end -> None")
    check(L([up(0, .1), up(50, .5), up(100, .85)]) == 100, "lock_in: only the last probe")
    check(L([{"step": 0, "LW": .9, "TS": .1, "NU": .9}, {"step": 50, "LW": .9, "TS": .95, "NU": .9}]) == 50,
          "lock_in: requires all three")
    check(L([{"step": 0, "LW": None, "TS": .9, "NU": .9}]) is None, "lock_in: missing metric counts as not locked")
    # paired CI
    mk = lambda i, g, o, h="x": {"id": i, "h": h, "scores": {"gold": g, "foil": o}}
    b = [mk(i, -2.0, -1.0) for i in range(50)]
    a = [mk(i, -1.0, -2.0) for i in range(50)]
    p = MF.paired(b, a, n_boot=500)
    check(p["d_acc"] == 1.0 and p["d_acc_ci95"] == [1.0, 1.0], f"paired: all gained should be +1 {p}")
    check(abs(p["d_margin"] - 2.0) < 1e-9, "paired: margin change should be +2")
    p0 = MF.paired(b, b, n_boot=500)
    check(p0["d_acc"] == 0 and p0["d_acc_ci95"] == [0.0, 0.0] and p0["d_margin"] == 0, "paired: no change should be 0")
    mixed = [mk(i, -1.0, -2.0) if i % 2 else mk(i, -2.0, -1.0) for i in range(50)]
    pm = MF.paired(b, mixed, n_boot=2000)
    check(pm["d_acc_ci95"][0] < pm["d_acc"] < pm["d_acc_ci95"][1], "paired: CI should bracket the mean")
    try:
        MF.paired(b, [mk(i, -1.0, -2.0, h="y") for i in range(50)], n_boot=10)
        check(False, "paired: prompt-hash mismatch not caught")
    except ValueError:
        pass
    try:
        MF.paired(b, a[:-1], n_boot=10)
        check(False, "paired: missing item not caught")
    except ValueError:
        pass


# ---------------- 2. oracles on the real eval items ----------------
def mention_pos(item, val):
    txt = [u + " " + a for u, a in item["turns"]]
    pat = re.compile(r"(?<![A-Za-z])" + re.escape(val) + r"(?![A-Za-z])", re.I)
    pos = [i for i, t in enumerate(txt) if pat.search(t)]
    return pos


def oracle_scores(item, kind):
    c = {lab: v.strip() for lab, v in item["cands"].items()}
    if kind == "ideal":
        return {lab: (0.0 if lab == "gold" else -1.0) for lab in c}
    if kind == "empty":
        return {lab: -1.0 for lab in c}
    if kind == "wrong":
        return {lab: (-3.0 if lab == "gold" else -1.0) for lab in c}
    if kind == "first":
        return {lab: -float(min(mention_pos(item, v))) for lab, v in c.items()}
    if kind == "last":
        return {lab: float(max(mention_pos(item, v))) for lab, v in c.items()}
    raise ValueError(kind)


def test_oracles():
    its = N.build()
    res = {}
    for kind in ("ideal", "empty", "wrong", "first", "last"):
        recs = [dict(render="plain", var=x["var"], d=x["d"], fam=x["fam"], sid=x["sid"], scores=oracle_scores(x, kind)) for x in its]
        pr = MF.primary(recs)
        res[kind] = {k: (v["acc"] if isinstance(v, dict) else v) for k, v in pr.items()}
    check(res["ideal"]["pass"] and res["ideal"]["LW10"] == 1.0, f"oracle ideal should pass: {res['ideal']}")
    for k in ("empty", "wrong", "first", "last"):
        check(not res[k]["pass"], f"oracle {k} passes the pass rule: {res[k]}")
    check(res["empty"]["LW10"] == 0 and res["empty"]["TS10"] == 0 and res["empty"]["NU10"] == 0, f"empty should score 0: {res['empty']}")
    check(res["wrong"]["LW10"] == 0 and res["wrong"]["NU10"] == 0, f"wrong should score 0: {res['wrong']}")
    check(res["first"]["LW10"] == 0 and res["first"]["TS10"] == 0 and res["first"]["NU10"] == 1.0, f"first-mentioned: {res['first']}")
    check(res["last"]["LW10"] == 1.0 and res["last"]["TS10"] == 0 and res["last"]["NU10"] == 0, f"last-mentioned: {res['last']}")
    # the incidental no-update variant: last-mentioned must fail, first-mentioned passes
    ex = X.build()
    for kind, want in (("last", 0.0), ("first", 1.0), ("empty", 0.0)):
        a = sum(MF.right(oracle_scores(x, kind)) for x in ex) / len(ex)
        check(a == want, f"noupd_incid oracle {kind}: {a} != {want}")
    # crossed pair (post hoc): only per-object tracking passes the pair
    cr = X.build_crossed()
    def first_corr(x):   # "the first correction wins" heuristic: value in the first 'Actually' turn
        for u, a in x["turns"]:
            if u.startswith("Actually"):
                for lab, v in x["cands"].items():
                    if re.search(r"\b" + re.escape(v.strip()) + r"\b", u):
                        return {l: (0.0 if l == lab else -1.0) for l in x["cands"]}
    def last_corr(x):
        best = None
        for u, a in x["turns"]:
            if u.startswith("Actually"):
                for lab, v in x["cands"].items():
                    if re.search(r"\b" + re.escape(v.strip()) + r"\b", u):
                        best = lab
        return {l: (0.0 if l == best else -1.0) for l in x["cands"]}
    def pair_rate(scorer):
        by = {}
        for x in cr:
            by.setdefault(x["sid"], []).append(MF.right(scorer(x)))
        return sum(all(v) for v in by.values()) / len(by)
    for name, scorer, want in (("ideal", lambda x: oracle_scores(x, "ideal"), 1.0),
                               ("first", lambda x: oracle_scores(x, "first"), 0.0),
                               ("last", lambda x: oracle_scores(x, "last"), 0.0),
                               ("empty", lambda x: oracle_scores(x, "empty"), 0.0),
                               ("first_correction", first_corr, 0.0), ("last_correction", last_corr, 0.0)):
        r = pair_rate(scorer)
        check(r == want, f"cross_pair oracle {name}: {r} != {want}")
    for x in cr:
        check(not answer_leak(x["question"], x["prefix"], x["cands"]["gold"]), f"cross leak {x['sid']}")
        check(len({v.strip() for v in x["cands"].values()}) == 4, f"cross duplicate values {x['sid']}")
    return res


# ---------------- 3. answer leaks ----------------
def test_leaks(train_exs):
    n = 0
    for it in N.build() + X.build():
        n += 1
        check(not answer_leak(it["question"], it["prefix"], it["cands"]["gold"]), f"eval leak: {it['task']} {it['sid']}")
    for it in I.build() + UP.build() + KH.build():
        q = it["prompt"].split("\n")[-2][len("User: "):]
        pre = it["prompt"].split("\n")[-1][len("Assistant: "):]
        n += 1
        check(not answer_leak(q, pre, it["cands"]["gold"]), f"eval leak: {it['task']} {q!r} -> {it['cands']['gold']}")
    for it in KB.build():
        n += 1
        check(not answer_leak(it["question"], it["prefix"], it["cands"]["gold"]), f"kbig leak: {it['question']} -> {it['cands']['gold']}")
        check(it["cands"]["gold"].strip().lower() != it["cands"]["foil"].strip().lower(), f"kbig gold == foil: {it['question']}")
    qs = Counter(it["question"] for it in KB.build())
    check(max(qs.values()) == 1, f"kbig duplicate questions: {[q for q, c in qs.items() if c > 1]}")
    for ex in train_exs:
        n += 1
        check(not answer_leak(ex["question"], ex["prefix"], ex["gold"]), f"train leak: {ex['question']} -> {ex['gold']}")
    return n


# ---------------- 4. held-out wording ----------------
EVAL_OBJ = ["dentist appointment", "dentist", "appointment", "haircut", "new car", "car", "new bike", "bike", "cat",
            "sister", "aunt"]
TRAIN_OBJ = sorted({o for o, _ in TD.DAY_EVENTS + TD.COLOR_OBJECTS} | {al for _, al in TD.DAY_EVENTS + TD.COLOR_OBJECTS} |
                   {p for p, r in TD.T_OWN} | {r for p, r in TD.T_OWN} | {x for pair in TD.T_REL for x in pair}, key=len, reverse=True)
VALUES = sorted(set(TD.DAYS + TD.COLORS + I.USER_NAMES + I.FEMALE_NAMES + I.ASSIST_NAMES + I.PET_NAMES + I.JOBS +
                    TD.T_USER + TD.T_OTHER + TD.T_ASSIST + TD.T_PETS + TD.T_JOBS), key=len, reverse=True)
OBJ_RE = re.compile(r"\b(" + "|".join(re.escape(o) for o in sorted(set(EVAL_OBJ + TRAIN_OBJ), key=len, reverse=True)) + r")\b", re.I)
VAL_RE = re.compile(r"\b(" + "|".join(re.escape(v) for v in VALUES) + r")\b", re.I)


def norm(s):
    """Normalized frame: values -> <v>, object nouns -> <e>. Some words are both (the job 'dentist'
    in 'dentist appointment'), so frames are compared under BOTH replacement orders (see nset)."""
    s = VAL_RE.sub("<v>", s)
    s = OBJ_RE.sub("<e>", s)
    return re.sub(r"\s+", " ", s.strip().lower())


def norm_b(s):
    s = OBJ_RE.sub("<e>", s)
    s = VAL_RE.sub("<v>", s)
    return re.sub(r"\s+", " ", s.strip().lower())


def nset(s):
    return {norm(s), norm_b(s)}


def has_value(s):
    return VAL_RE.search(s) is not None


def eval_frames():
    fr = set()
    for it in N.build() + X.build() + X.build_crossed():
        for u, a in it["turns"]:
            if has_value(u + " " + a):
                fr |= nset(u) | nset(a)
        fr |= nset(it["question"]) | nset(it["prefix"])
    for it in I.build() + UP.build():
        lines = it["prompt"].split("\n")
        for ln in lines:
            body = ln.split(": ", 1)[1] if ": " in ln else ln
            if has_value(body):
                fr |= nset(body)
        fr |= nset(lines[-2].split(": ", 1)[1]) | nset(lines[-1].split(": ", 1)[1])
    return fr


def train_frames(exs):
    fr = Counter()
    for ex in exs:
        for u, a in ex["turns"]:
            if has_value(u + " " + a):
                for x in nset(u) | nset(a):
                    fr[x] += 1
        for x in nset(ex["question"]) | nset(ex["prefix"]):
            fr[x] += 1
    return fr


def ngrams(s, n):
    w = s.split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def test_heldout(exs):
    ev, tr = eval_frames(), train_frames(exs)
    shared = sorted(set(tr) & ev)
    check(not shared, f"held-out: {len(shared)} training frames equal eval frames: {shared[:8]}")
    # informational: shared 5-grams between frames
    ev5 = set().union(*(ngrams(f, 5) for f in ev))
    tr5 = set().union(*(ngrams(f, 5) for f in tr))
    # objects, names and fillers
    eval_names = set(I.USER_NAMES + I.FEMALE_NAMES + I.ASSIST_NAMES + I.PET_NAMES + I.JOBS)
    train_names = set(TD.T_USER + TD.T_OTHER + TD.T_ASSIST + TD.T_PETS + TD.T_JOBS)
    check(not (eval_names & train_names), f"held-out: shared names/jobs {eval_names & train_names}")
    ev_fill = {norm(q) for q, a in I.DISTRACTORS + N.EXTRA_DISTRACTORS}
    tr_fill = {norm(q) for q, a in TD.T_DISTRACT}
    check(not (ev_fill & tr_fill), f"held-out: shared fillers {ev_fill & tr_fill}")
    bad = re.compile(r"\b(dentist|appointment|haircut|car|bike|cat|sister|aunt)\b", re.I)
    leaks = [ex for ex in exs if any(bad.search(u + " " + a) for u, a in ex["turns"]) or bad.search(ex["question"] + " " + ex["prefix"])]
    check(not leaks, f"held-out: {len(leaks)} training dialogues use eval object words, e.g. {leaks[:1]}")
    vr = re.compile(r"\b(" + "|".join(TD.DAYS + TD.COLORS + ["grey", "pink", "brown"]) + r")\b", re.I)
    for q, a in TD.T_DISTRACT:
        check(not vr.search(q + " " + a), f"training filler carries a value: {q}")
    return {"eval_frames": len(ev), "train_frames": len(tr), "shared_frames": len(shared), "shared_5grams": sorted(ev5 & tr5)}


# ---------------- 5. shortcut audit ----------------
def test_shortcuts(exs):
    upd = [ex for ex in exs if ex["family"] == "update"]
    first = sum(ex["mentions"][0][0] == ex["gold"] for ex in upd) / len(upd)
    last = sum(ex["mentions"][-1][0] == ex["gold"] for ex in upd) / len(upd)
    latest = sum([v for v, o in ex["mentions"] if o == ex["asked"]][-1] == ex["gold"] for ex in upd) / len(upd)
    check(first <= 0.5, f"shortcut: first-mentioned right on {first:.3f} of update items (> 0.5)")
    check(last <= 0.5, f"shortcut: last-mentioned right on {last:.3f} of update items (> 0.5)")
    check(latest == 1.0, f"shortcut: latest-statement rule right on only {latest:.3f}")
    # every gold is in context, mentioned by the value-carrying turns
    for ex in exs:
        ctx = " ".join(u + " " + a for u, a in ex["turns"])
        check(re.search(r"\b" + re.escape(ex["gold"]) + r"\b", ctx) is not None, f"train gold not in context: {ex['kind']}")
    kinds = Counter(ex["kind"] for ex in exs)
    fams = Counter(ex["family"] for ex in exs)
    by_kind = {}
    for k in sorted({ex["kind"] for ex in upd}):
        xs = [ex for ex in upd if ex["kind"] == k]
        by_kind[k] = {"n": len(xs), "first": round(sum(ex["mentions"][0][0] == ex["gold"] for ex in xs) / len(xs), 3),
                      "last": round(sum(ex["mentions"][-1][0] == ex["gold"] for ex in xs) / len(xs), 3)}
    return {"first_mentioned_right": round(first, 3), "last_mentioned_right": round(last, 3), "latest_rule_right": latest,
            "kinds": dict(kinds), "families": dict(fams), "by_kind": by_kind}


# ---------------- 6. tokenization ----------------
def test_tok(model_id, exs, max_len=768):
    from transformers import AutoTokenizer
    import lik
    import ft_test as FT
    tok = AutoTokenizer.from_pretrained(model_id)
    lens, rej = [], 0
    for ex in exs[:3000]:
        ids, labels, how = FT.encode(tok, ex)
        n_lab = sum(1 for y in labels if y != -100)
        ans_ids = [y for y in labels if y != -100]
        check(len(ids) == len(labels), "tok: ids/labels length mismatch")
        check(n_lab >= 1, "tok: no answer tokens labelled")
        check(tok.decode(ans_ids).strip() == ex["answer"].strip(), f"tok: answer decodes to {tok.decode(ans_ids)!r} not {ex['answer']!r}")
        check(all(y == -100 for y in labels[: len(ids) - n_lab]), "tok: a prompt token is labelled")
        if len(ids) > max_len:
            rej += 1
        else:
            lens.append(len(ids))
    rng = random.Random(5)
    stats = {"n": 0, "rejected_long": 0, "kinds": {}, "len_sum": 0, "len_max": 0, "split": 0}
    st = FT.example_stream(tok, rng, max_len, stats)
    start = tok("User:", add_special_tokens=True).input_ids
    for _ in range(2000):
        ids, labels = next(st)
        check(len(ids) <= max_len, "tok: stream yielded an example over max_len")
        check(ids[: len(start)] == start, "tok: a streamed example does not start at the dialogue start (truncated?)")
        n_lab = sum(1 for y in labels if y != -100)
        check(n_lab >= 1 and all(y == -100 for y in labels[: len(ids) - n_lab]), "tok: streamed labels not answer-only")
    # the length limit must not bias the long-distance mix: first-mentioned-right at d >= 8 among KEPT examples
    kept_first, all_first = [], []
    for ex in exs:
        if ex["family"] == "update" and ex["d"] >= 8:
            ids, labels, how = FT.encode(tok, ex)
            all_first.append(ex["mentions"][0][0] == ex["gold"])
            if len(ids) <= max_len:
                kept_first.append(ex["mentions"][0][0] == ex["gold"])
    fr = sum(kept_first) / max(1, len(kept_first))
    fa = sum(all_first) / max(1, len(all_first))
    check(abs(fr - fa) <= 0.03, f"tok: length limit shifts the d>=8 mix toward first-mentioned golds ({fa:.3f} -> {fr:.3f})")
    ev_lens = {}
    for it in N.build():
        if it["d"] == 10:
            p, _ = lik.render(it, "plain", tok, model_id)
            ev_lens.setdefault("new_plain_d10", []).append(len(tok(p).input_ids))
            p, _ = lik.render(it, "chat", tok, model_id)
            ev_lens.setdefault("new_chat_d10", []).append(len(tok(p, add_special_tokens=False).input_ids))
    return {"d8plus_first_mentioned_right_all": round(fa, 3), "d8plus_first_mentioned_right_kept": round(fr, 3), "raw_over_max_len": rej, "raw_n": min(3000, len(exs)), "kept_mean_len": round(sum(lens) / len(lens), 1),
            "kept_max_len": max(lens), "stream_rejected": stats["rejected_long"], "stream_n": stats["n"],
            "stream_split_answers": stats["split"],
            "eval_len": {k: {"mean": round(sum(v) / len(v), 1), "max": max(v)} for k, v in ev_lens.items()}}


def run(tok_model=None, n_train=20000, verbose=True):
    del FAILS[:]
    rng = random.Random(123)
    exs = [TD.gen(rng) for _ in range(n_train)]
    out = {}
    test_graders()
    import gen_probe as GP
    bad = GP.self_test()
    check(not bad, f"gen_probe grader self-test: {bad}")
    out["oracles"] = test_oracles()
    out["leak_checks"] = test_leaks(exs)
    out["heldout"] = test_heldout(exs)
    out["shortcuts"] = test_shortcuts(exs)
    if tok_model:
        out["tokenization"] = test_tok(tok_model, exs)
    out["fails"] = list(FAILS)
    if verbose:
        import json
        print(json.dumps({k: v for k, v in out.items() if k != "fails"}, indent=1)[:6000])
        print(f"{len(FAILS)} failures")
        for f in FAILS[:30]:
            print("  FAIL", f)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", default=None)
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()
    r = run(a.tok, a.n)
    sys.exit(1 if r["fails"] else 0)
