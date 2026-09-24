"""Self-test for E003 (no model is loaded or built; --tok loads tokenizers only; params reads weight-file
headers / meta shapes only).

 1. E002's model-free suite, unchanged (validate.py copied byte-for-byte): graders (empty, tie, NaN and
    plausible-wrong answers fail), pass-rule boundary, d10-only, plain-only, majority rule, lock-in, paired CI,
    oracle models on the real eval items, answer leaks, held-out wording, training-stream shortcut audit.
 2. The copied E002 modules are byte-identical to the checksums recorded when they were copied.
 3. Item identity: every E003 eval set renders to the SAME prompts (hash, order, variant, distance) as E002's
    scored SmolLM2-135M baseline records, so E003 and E002 numbers are on identical items.
 4. The LR-selection dev draw: 320 items (LW 192, TS 64, NU 64, all d10), no prompt shared with any eval set or
    the in-training probe; oracle models on it behave like on the eval items (only ideal passes).
 5. The LR pick rule (pick_lr.py) on synthetic runs: min-cell criterion, tie -> smaller LR, NaN-loss runs
    ineligible, extension exactly when the pick is the top LR and below 0.8, NONE when nothing is eligible.
 6. The analysis (analyze_e003.py) on oracle records: empty and wrong answers fail, ideal passes, the crossed
    pair needs both answers, the knowledge change is after-minus-before on paired items.
 7. Parameter counts (params.py) equal the tensor shapes stored in every downloaded weight file.
 8. The queue runs the LR search on dev only and the scored seeds on eval only.
 9. The loss self-check comparator close() rejects a 1% difference and NaN.
10. (--tok) per tokenizer family: labels only on answer tokens, answers decode exactly, streamed examples
    start at the dialogue start, the 768 limit does not shift the d>=8 mix, eval lengths vs context.
Run: python validate_e003.py [--tok] [--fast]   (exit 1 on any failure)
"""
import argparse, hashlib, json, math, os, random, re, shutil, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
E002 = os.path.join(os.path.dirname(EXP), "E002_ft_test")
sys.path.insert(0, HERE)
import validate as V          # E002's suite (copy)
import items as I
import items_new as N
import eval_extra as X
import uprobe_items as UP
import khard_items as KH
import kbig_items as KB
import train_data as TD
import metrics_ft as MF
import lik
import params as PR
import pick_lr as PL
import analyze_e003 as AZ

check = V.check
FAILS = V.FAILS
E002_BASE = "HuggingFaceTB__SmolLM2-135M-Instruct__base"
TOK_FAMILIES = {"pythia": ["EleutherAI/pythia-14m", "EleutherAI/pythia-31m", "EleutherAI/pythia-70m", "EleutherAI/pythia-160m"],
                "tinystories": ["roneneldan/TinyStories-1M", "roneneldan/TinyStories-3M", "roneneldan/TinyStories-8M",
                                "roneneldan/TinyStories-33M"]}
TRAINED_CTX = {"pythia": 2048, "tinystories": 512}
QUEUE = os.path.join(HERE, "queue_e003.sh")


def phash(s):
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def prompt_of(it):
    return lik.render(it, "plain", None, "")[0] if "turns" in it else it["prompt"]


def eval_sets():
    import e003_ft_test as FT
    return {s: its for s, r, its in FT.eval_sets({"eval"}, False)}


def dev_items():
    import e003_ft_test as FT
    return FT.dev_items()


# ---------------- 2. copies ----------------
def test_copies():
    rec = {}
    for ln in open(os.path.join(EXP, "logs", "copied_from_E002.sha256")):
        h, f = ln.split()
        rec[f] = h
    same_as_e002_now = {}
    for f, h in rec.items():
        got = hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest()
        check(got == h, f"copy: {f} changed since it was copied from E002")
        p2 = os.path.join(E002, "code", f)
        same_as_e002_now[f] = os.path.exists(p2) and hashlib.sha256(open(p2, "rb").read()).hexdigest() == h
    return {"copied": sorted(rec), "identical_to_E002_now": same_as_e002_now}


# ---------------- 3. item identity with E002 ----------------
EXPECTED_SETS = {"new", "extra", "old", "uprobe", "khard", "kbig", "cross"}


def test_identity():
    out = {}
    es = eval_sets()
    check(set(es) == EXPECTED_SETS, f"identity: E003 eval sets {sorted(es)} are not E002's plain sets {sorted(EXPECTED_SETS)}")
    for s, its in es.items():
        p = os.path.join(E002, "out", f"{E002_BASE}__{s}__plain.jsonl")
        if not check(os.path.exists(p), f"identity: E002 baseline records missing for {s}"):
            continue
        e2 = [json.loads(l) for l in open(p) if l.strip()]
        check(len(e2) == len(its), f"identity: {s} has {len(its)} items, E002 scored {len(e2)}")
        bad = 0
        for it, r in zip(its, e2):
            same = phash(prompt_of(it)) == r["h"] and all(it.get(k) == r.get(k) for k in ("var", "d", "sid", "task"))
            bad += not same
        check(bad == 0, f"identity: {bad} of {len(its)} {s} items differ from E002's")
        out[s] = {"n": len(its), "differ": bad}
    return out


# ---------------- 4. dev draw ----------------
def test_dev():
    import e003_ft_test as FT
    dv = dev_items()
    n = {name: sum(1 for x in dv if x["var"] in vs and x["d"] == 10)
         for name, vs in (("LW", MF.LW_VARS), ("TS", ("twoslot",)), ("NU", ("noupd",)))}
    check(len(dv) == 320 and n == {"LW": 192, "TS": 64, "NU": 64}, f"dev: wrong composition {len(dv)} {n}")
    check(all(x["d"] == 10 for x in dv), "dev: not all d10")
    dev_h = {phash(prompt_of(x)) for x in dv}
    check(len(dev_h) == len(dv), "dev: duplicate prompts inside the dev draw")
    ev_h = {phash(prompt_of(x)) for its in eval_sets().values() for x in its}
    pr_h = {phash(prompt_of(x)) for x in FT.probe_items(False)}
    check(not (dev_h & ev_h), f"dev: {len(dev_h & ev_h)} dev prompts are also eval prompts")
    check(not (dev_h & pr_h), f"dev: {len(dev_h & pr_h)} dev prompts are also probe prompts")
    ev_turns = {json.dumps(x["turns"]) for its in eval_sets().values() for x in its if "turns" in x}
    check(not ({json.dumps(x["turns"]) for x in dv} & ev_turns), "dev: a dev dialogue equals an eval dialogue")
    res = {}
    for kind in ("ideal", "empty", "wrong", "first", "last"):
        recs = [dict(render="plain", var=x["var"], d=x["d"], fam=x["fam"], sid=x["sid"], scores=V.oracle_scores(x, kind)) for x in dv]
        pr = MF.primary(recs)
        res[kind] = {k: (v["acc"] if isinstance(v, dict) else v) for k, v in pr.items()}
    check(res["ideal"]["pass"], f"dev oracle ideal should pass {res['ideal']}")
    for k in ("empty", "wrong", "first", "last"):
        check(not res[k]["pass"], f"dev oracle {k} passes: {res[k]}")
    check(res["empty"]["LW10"] == 0 and res["empty"]["TS10"] == 0 and res["empty"]["NU10"] == 0, f"dev empty should be 0 {res['empty']}")
    return {"n": len(dv), "cells": n, "oracles": res}


# ---------------- 5. LR pick rule ----------------
def _write_run(out, model, lr, kind_by_var, losses=None, n_dev=None):
    """kind_by_var: {var: oracle kind or fraction right (float)}."""
    dv = dev_items()
    tag = PL.lrtag(lr)
    stem = os.path.join(out, f"{PL.slug(model)}__{tag}")
    recs = []
    cnt = {}
    for x in dv:
        k = kind_by_var.get(x["var"], "empty")
        if isinstance(k, float):
            j = cnt.get(x["var"], 0)
            cnt[x["var"]] = j + 1
            tot = sum(1 for y in dv if y["var"] == x["var"])
            k = "ideal" if j < round(k * tot) else "wrong"
        recs.append(dict(set="dev", render="plain", var=x["var"], d=x["d"], fam=x["fam"], sid=x["sid"],
                         scores=V.oracle_scores(x, k)))
    recs = recs[:n_dev] if n_dev is not None else recs
    with open(stem + "__dev__plain.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    L = losses if losses is not None else [1.0] * 400
    json.dump({"steps": 400, "losses": L, "eval_counts": {"dev/plain": len(recs)}, "lock_in_step": None},
              open(stem + "__run.json", "w"))


def test_pick():
    m = "test/model"
    ALL = {"same_k1": "ideal", "same_k2": "ideal", "same_k3": "ideal", "twoslot": "ideal", "noupd": "ideal"}
    WRONG = {v: "wrong" for v in ALL}
    EMPTY = {v: "empty" for v in ALL}
    cases = []

    def run(name, setup, grid, want, final=False):
        d = tempfile.mkdtemp(prefix="e003pick_")
        try:
            for lr, spec in setup.items():
                _write_run(d, m, lr, *spec) if isinstance(spec, tuple) else _write_run(d, m, lr, spec)
            got = PL.pick(m, grid, d, final)["line"]
            check(got == want, f"pick_lr {name}: got {got!r}, want {want!r}")
            cases.append((name, got))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    g3 = ["5e-05", "3e-04", "1e-03"]
    run("ideal in the middle", {5e-5: EMPTY, 3e-4: ALL, 1e-3: WRONG}, g3, "CHOSEN 3e-04")
    run("tie -> smaller LR", {5e-5: EMPTY, 3e-4: ALL, 1e-3: ALL}, g3, "CHOSEN 3e-04")
    run("all-zero runs (empty or wrong answers score 0): tie -> smallest LR, no extension",
        {5e-5: EMPTY, 3e-4: WRONG, 1e-3: EMPTY}, g3, "CHOSEN 5e-05")
    part = {"same_k1": 0.5, "same_k2": 0.5, "same_k3": 0.5, "twoslot": 0.5, "noupd": "ideal"}
    run("top LR best but < 0.8 -> extend", {5e-5: EMPTY, 3e-4: EMPTY, 1e-3: part}, g3, "EXTEND 3e-03")
    run("after extension -> pick, never extend again", {5e-5: EMPTY, 3e-4: EMPTY, 1e-3: part, 3e-3: EMPTY},
        g3 + ["3e-03"], "CHOSEN 1e-03", final=True)
    run("top LR passes -> no extension", {5e-5: EMPTY, 3e-4: part, 1e-3: ALL}, g3, "CHOSEN 1e-03")
    g4 = g3 + ["3e-03"]
    run("4-LR grid extends to 1e-02", {5e-5: EMPTY, 3e-4: EMPTY, 1e-3: EMPTY, 3e-3: part}, g4, "EXTEND 1e-02")
    run("NaN-loss run is ineligible", {5e-5: EMPTY, 3e-4: part, 1e-3: (ALL, [1.0] * 399 + [float("nan")])}, g3, "CHOSEN 3e-04")
    run("short run is ineligible", {5e-5: part, 3e-4: (ALL, [1.0] * 200)}, ["5e-05", "3e-04"], "CHOSEN 5e-05")
    run("incomplete dev scoring is ineligible", {5e-5: part, 3e-4: (ALL, None, 100)}, ["5e-05", "3e-04"], "CHOSEN 5e-05")
    run("nothing eligible -> NONE", {}, g3, "NONE no eligible LR-search run")
    # mean prefers lopsided (0.83 vs 0.70), min prefers even (0.70 vs 0.50): the rule must pick even
    lopsided = {"same_k1": "ideal", "same_k2": "ideal", "same_k3": "ideal", "twoslot": "ideal", "noupd": 0.5}
    even = {"same_k1": 0.7, "same_k2": 0.7, "same_k3": 0.7, "twoslot": 0.7, "noupd": 0.7}
    run("min-cell, not mean", {5e-5: lopsided, 3e-4: even}, ["5e-05", "3e-04"], "CHOSEN 3e-04", final=True)
    run("min-cell, not mean (order swapped)", {5e-5: even, 3e-4: lopsided}, ["5e-05", "3e-04"], "CHOSEN 5e-05", final=True)
    return cases


# ---------------- 6. analysis ----------------
def _oracle_sets(kind):
    sets = {}
    for s, its in eval_sets().items():
        sets[f"{s}__plain"] = [dict(set=s, render="plain", id=i, h=phash(prompt_of(x)), seq_len=100,
                                    **{k: x[k] for k in ("task", "cond", "d", "fam", "var", "sid", "k", "cat") if k in x},
                                    scores=V.oracle_scores(x, kind)) for i, x in enumerate(its)]
    return sets


def test_analysis():
    out = {}
    acc = lambda c: c["acc"] if isinstance(c, dict) else None   # a missing cell reads as None, never as a crash
    for kind in ("ideal", "empty", "wrong"):
        s = AZ.summarize_run(_oracle_sets(kind), None)
        out[kind] = s["primary_plain"]["pass"]
        cp = acc((s.get("cross_posthoc") or {}).get("cross_pair|d10"))
        if kind == "ideal":
            check(s["primary_plain"]["pass"] and cp == 1.0, f"analysis: ideal should pass with cross_pair 1.0 ({cp})")
            check(acc(s["old"].get("owner_pair_d10")) == 1.0, "analysis: ideal old-battery owner pair should be 1.0")
            check(all(acc(s["primary_plain"][k]) == 1.0 for k in ("LW10", "TS10", "NU10")), "analysis: ideal cells should be 1.0")
        else:
            check(not s["primary_plain"]["pass"], f"analysis: {kind} answers pass the pass rule")
            check(cp == 0.0, f"analysis: {kind} answers get cross_pair {cp}")
            check(acc(s["primary_plain"]["LW10"]) == 0.0, f"analysis: {kind} answers get LW10 {acc(s['primary_plain']['LW10'])}")
    # crossed pair needs BOTH answers: A right, B wrong -> pair 0
    sets = _oracle_sets("ideal")
    check(bool(sets.get("cross__plain")), "analysis: no crossed-control records to test")
    for r in sets.get("cross__plain", []):
        if r["var"] == "cross_B":
            r["scores"] = {k: (-5.0 if k == "gold" else -1.0) for k in r["scores"]}
    cr = AZ.cross_summary(sets.get("cross__plain", []))
    sb = AZ.summarize_run(sets, None)
    check(sb["primary_plain"]["pass"] and not sb["pass_and_cross"], "analysis: secondary label must need the crossed pair")
    check(AZ.summarize_run(_oracle_sets("ideal"), None)["pass_and_cross"], "analysis: ideal should get the secondary label")
    check(acc(cr.get("cross_pair|d10")) == 0.0 and acc(cr.get("cross_A|d10")) == 1.0, f"analysis: cross pair with B wrong {cr.get('cross_pair|d10')}")
    # knowledge change = after - before on the same items
    kn = AZ.knowledge(_oracle_sets("wrong"), _oracle_sets("ideal")).get("kbig_all") or {}
    check(kn.get("d_acc") == 1.0 and kn.get("gained") == 441, f"analysis: knowledge gain sign {kn}")
    kn2 = AZ.knowledge(_oracle_sets("ideal"), _oracle_sets("wrong")).get("kbig_all") or {}
    check(kn2.get("d_acc") == -1.0, "analysis: knowledge loss sign")
    # length diagnostic splits on the pretraining context
    new = [dict(r) for r in _oracle_sets("ideal").get("new__plain", [])]
    for i, r in enumerate(new):
        r["seq_len"] = 600 if i % 2 else 300
        if i % 2:
            r["scores"] = {k: (-5.0 if k == "gold" else -1.0) for k in r["scores"]}
    ld = AZ.length_split(new, 512)
    check(acc(ld.get("LW10|fits<=512")) == 1.0 and acc(ld.get("LW10|over512")) == 0.0, f"analysis: length split {ld}")
    check(ld.get("pass_on_fitting_items") is True, "analysis: fitting items all right -> secondary pass")
    for r in new:
        if r["seq_len"] == 300 and r["var"] == "noupd":
            r["scores"] = {k: (-5.0 if k == "gold" else -1.0) for k in r["scores"]}
    check(AZ.length_split(new, 512).get("pass_on_fitting_items") is False, "analysis: a failed fitting NU cell must fail the secondary")
    v = MF.model_verdict([True, True, False])
    check(v["pass"], "analysis: 2 of 3 seeds should pass")
    check(not MF.model_verdict([True, False, False])["pass"], "analysis: 1 of 3 seeds should fail")
    return out


# ---------------- 7. params ----------------
def test_params():
    rows = PR.table(PR.E003_MODELS, do_verify=True)
    for r in rows:
        check(r["verify"]["match"], f"params: {r['model']} formula {r['total']}/{r['embedding']} vs file "
                                    f"{r['verify']['stored_total']}/{r['verify']['stored_embedding']}")
    bodies = [r["body"] for r in rows]
    check(bodies == sorted(bodies), f"params: queue order is not smallest body first {bodies}")
    return {r["model"]: {"total": r["total"], "embedding": r["embedding"], "body": r["body"]} for r in rows}


# ---------------- 8. queue static check ----------------
def queue_text():
    return open(QUEUE).read() if os.path.exists(QUEUE) else ""


def test_queue():
    q = queue_text()
    if not check(bool(q), "queue: queue_e003.sh missing"):
        return {}
    lrline = [l for l in q.splitlines() if "LRSEARCH_ARGS=" in l]
    scline = [l for l in q.splitlines() if "SCORED_ARGS=" in l]
    check(len(lrline) == 1 and "--sets dev" in lrline[0] and "eval" not in lrline[0].split("--sets")[1].split()[0],
          f"queue: LR-search runs must score dev only: {lrline}")
    check(len(scline) == 1 and "--sets eval" in scline[0] and "dev" not in scline[0].split("--sets")[1].split()[0],
          f"queue: scored seeds must score eval only: {scline}")
    check("--seed 0" in lrline[0] if lrline else False, "queue: LR search must use seed 0")
    check(re.search(r"for s in 1 2 3", q) is not None, "queue: scored seeds must be 1 2 3")
    common = [l for l in q.splitlines() if l.startswith("COMMON=")]
    check(len(common) == 1 and all(x in common[0] for x in ("--bs 4", "--accum 4", "--max-len 768", "--grad-ckpt")),
          f"queue: training settings differ from E002 135M: {common}")
    check("--steps 400" in q or "--steps" not in (lrline[0] if lrline else ""), "queue: steps must be the default 400")
    return {"lr_args": lrline, "scored_args": scline}


# ---------------- 9. close() ----------------
def test_close():
    import e003_ft_test as FT
    check(FT.close(1.0, 1.0005), "close: 0.05% should match")
    check(not FT.close(1.0, 1.01), "close: 1% should not match")
    check(not FT.close(float("nan"), float("nan")), "close: NaN must not match")
    check(not FT.close(2.0, 0.5), "close: 2.0 vs 0.5 must not match")


# ---------------- 10. tokenization ----------------
def test_tok(family, exs, max_len=768):
    from transformers import AutoTokenizer
    import e003_ft_test as FT
    ids_ = TOK_FAMILIES[family]
    snaps = [hashlib.sha256(open(os.path.join(PR.snapshot_dir(m), "tokenizer.json"), "rb").read()).hexdigest() for m in ids_]
    same_tok = len(set(snaps)) == 1
    if not same_tok:   # the files may differ cosmetically; require identical encodings instead
        toks = [AutoTokenizer.from_pretrained(m) for m in ids_]
        probe = [TD.transcript(ex["turns"], ex["question"], ex["prefix"]) + ex["answer"] for ex in exs[:200]]
        same_tok = all(t(p).input_ids == toks[0](p).input_ids for t in toks[1:] for p in probe)
    check(same_tok, f"tok: the {family} models do not share one tokenization")
    model_id = ids_[0]
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
    # eval lengths: every scored sequence must fit the model's positions; report what exceeds pretraining context
    n_ctx = PR.count(PR.load_config(model_id))["n_ctx"]
    ctx = TRAINED_CTX[family]
    ev = {}
    for s, its in list(eval_sets().items()) + [("dev", dev_items())]:
        L = []
        for it in its:
            p = prompt_of(it)
            L.append(max(len(tok(p + c).input_ids) for c in it["cands"].values()))
        check(max(L) <= n_ctx, f"tok: {s} has a sequence of {max(L)} tokens > n_ctx {n_ctx}")
        ev[s] = {"max": max(L), "over_trained_ctx": sum(l > ctx for l in L), "n": len(L)}
        if s in ("new", "dev"):
            for name, vs in (("LW10", MF.LW_VARS), ("TS10", ("twoslot",)), ("NU10", ("noupd",))):
                sel = [l for l, it in zip(L, its) if it["var"] in vs and it["d"] == 10]
                ev[s][f"{name}_over_trained_ctx"] = f"{sum(l > ctx for l in sel)}/{len(sel)}"
    return {"family": family, "tokenizer_shared": same_tok, "raw_over_max_len": rej, "raw_n": min(3000, len(exs)),
            "kept_mean_len": round(sum(lens) / len(lens), 1), "kept_max_len": max(lens), "stream_rejected": stats["rejected_long"],
            "stream_n": stats["n"], "d8plus_first_right_all": round(fa, 3), "d8plus_first_right_kept": round(fr, 3),
            "trained_ctx": ctx, "n_ctx": n_ctx, "eval_len": ev}


def run(tok=False, fast=False, verbose=True, n_train=None):
    del FAILS[:]
    n_train = n_train or (6000 if fast else 20000)
    rng = random.Random(123)
    exs = [TD.gen(rng) for _ in range(n_train)]
    out = {}
    V.test_graders()
    out["e002_oracles"] = V.test_oracles()
    out["leak_checks"] = V.test_leaks(exs)
    out["heldout"] = V.test_heldout(exs)
    out["shortcuts"] = V.test_shortcuts(exs)
    out["copies"] = test_copies()
    out["identity"] = test_identity()
    out["dev"] = test_dev()
    out["pick"] = test_pick()
    out["analysis"] = test_analysis()
    out["params"] = test_params()
    out["queue"] = test_queue()
    test_close()
    if tok:
        out["tokenization"] = {f: test_tok(f, exs) for f in TOK_FAMILIES}
    out["fails"] = list(FAILS)
    if verbose:
        print(json.dumps({k: v for k, v in out.items() if k != "fails"}, indent=1, default=str)[:12000])
        print(f"{len(FAILS)} failures")
        for f in FAILS[:30]:
            print("  FAIL", f)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    r = run(a.tok, a.fast)
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1, default=str)
    sys.exit(1 if r["fails"] else 0)
