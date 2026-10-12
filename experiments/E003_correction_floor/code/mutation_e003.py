"""Mutation test for E003 (no model): break each grader, pass-rule detail, LR-pick rule, analysis step, dev draw,
item-identity guard, parameter formula, queue argument and tokenization property on purpose, and confirm
validate_e003.run() goes red. A test that has never been watched failing is not evidence. A crash is recorded as
a crash, not a kill. Exits non-zero if any mutant survives or the unmutated suite is not green.
Run: python mutation_e003.py [--tok]      writes ../logs/mutation_e003.json and prints one line per mutant
"""
import argparse, importlib, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import items as I
import items_new as N
import eval_extra as X
import train_data as TD
import metrics_ft as MF
import lik
import params as PR
import pick_lr as PL
import analyze_e003 as AZ
import e003_ft_test as FT
import validate as V
import validate_e003 as VE

ORDER = [I, N, X, TD, MF, lik, PR, PL, AZ, FT, V, VE]


def fresh():
    for m in ORDER:
        importlib.reload(m)


# ---------------- graders (metrics_ft, copied from E002) ----------------
def g_ties_pass():
    def right(scores):
        if not scores or "gold" not in scores or len(scores) < 2:
            return False
        g = scores["gold"]
        return all(g >= v for k, v in scores.items() if k != "gold")
    MF.right = right

def g_only_first_foil():
    def right(scores):
        if not scores or "gold" not in scores or len(scores) < 2:
            return False
        foil = [v for k, v in scores.items() if k != "gold"][0]
        return scores["gold"] > foil
    MF.right = right

def g_gold_only_passes():
    orig = MF.right
    MF.right = lambda s: True if (s and list(s) == ["gold"]) else orig(s)

def g_empty_passes():
    orig = MF.right
    MF.right = lambda s: True if not s else orig(s)

def g_nan_passes():
    def right(scores):
        if not scores or "gold" not in scores or len(scores) < 2:
            return False
        g = scores["gold"]
        return all(not (v > g) for k, v in scores.items() if k != "gold") and not all(v == g for v in scores.values())
    MF.right = right

def g_primary_no_controls():
    orig = MF.primary
    def primary(recs, render="plain"):
        p = orig(recs, render)
        p["pass"] = p["LW10"] is not None and p["LW10"]["acc"] >= 0.8
        return p
    MF.primary = primary

def g_primary_d4():
    def primary(recs, render="plain"):
        lw, ts, nu = MF.cell(recs, render, MF.LW_VARS, 4), MF.cell(recs, render, ("twoslot",), 4), MF.cell(recs, render, ("noupd",), 4)
        return {"LW10": lw, "TS10": ts, "NU10": nu, "pass": all(c is not None and c["acc"] >= 0.8 for c in (lw, ts, nu))}
    MF.primary = primary

def g_primary_strict():
    def primary(recs, render="plain"):
        lw, ts, nu = MF.cell(recs, render, MF.LW_VARS, 10), MF.cell(recs, render, ("twoslot",), 10), MF.cell(recs, render, ("noupd",), 10)
        return {"LW10": lw, "TS10": ts, "NU10": nu, "pass": all(c is not None and c["acc"] > 0.8 for c in (lw, ts, nu))}
    MF.primary = primary

def g_verdict_tie():
    MF.model_verdict = lambda sp: {"seeds": len(sp), "passing": sum(sp), "lock_in_rate": None,
                                   "pass": len(sp) > 0 and sum(sp) >= len(sp) / 2}

def g_paired_no_hash():
    orig = MF.paired
    def paired(before, after, **kw):
        after = [dict(r, h=b.get("h")) for r, b in zip(after, before)]
        return orig(before, after, **kw)
    MF.paired = paired

def g_paired_sign():
    orig = MF.paired
    MF.paired = lambda before, after, **kw: orig(after, before, **kw)

# ---------------- LR pick ----------------
def p_tie_larger():
    PL.key = lambda r: (r["min"], r["mean"], r["lr"])

def p_by_mean():
    PL.key = lambda r: (r["mean"], -r["lr"])

def p_nan_eligible():
    orig = PL.load_run
    def load_run(out_dir, model, lr):
        stem = os.path.join(out_dir, f"{PL.slug(model)}__{PL.lrtag(lr)}")
        if os.path.exists(stem + "__run.json"):
            meta = json.load(open(stem + "__run.json"))
            meta["losses"] = [1.0 if not (isinstance(x, float) and math.isfinite(x)) else x for x in meta["losses"]]
            json.dump(meta, open(stem + "__run.json", "w"))
        return orig(out_dir, model, lr)
    PL.load_run = load_run

def p_never_extend():
    PL.THRESH = -1.0

def p_extend_when_passing():
    PL.THRESH = 1.01

def p_dev_nu_d4():
    orig = PL.dev_cells
    def dev_cells(recs):
        c = orig(recs)
        nu = MF.cell(recs, "plain", ("noupd",), 4)
        c["NU"] = nu["acc"] if nu else 1.0
        accs = [c["LW"], c["TS"], c["NU"]]
        c["min"], c["mean"] = min(accs), round(sum(accs) / 3, 4)
        return c
    PL.dev_cells = dev_cells

# ---------------- analysis ----------------
def a_cross_any():
    orig = AZ.cross_summary
    def cross_summary(cross):
        cr = orig(cross)
        for d in (0, 4, 10):
            by = {}
            for r in cross:
                if r["d"] == d:
                    by.setdefault(r["sid"], []).append(MF.right(r["scores"]))
            cr[f"cross_pair|d{d}"] = MF.acc([any(v) for v in by.values()])
        return cr
    AZ.cross_summary = cross_summary

def a_knowledge_self():
    orig = AZ.knowledge
    AZ.knowledge = lambda base_sets, sets: orig(sets, sets)

def a_primary_wrong_set():
    orig = AZ.summarize_run
    AZ.summarize_run = lambda sets, meta: orig(dict(sets, new__plain=sets.get("extra__plain", [])), meta)

def a_length_split_flipped():
    orig = AZ.length_split
    def ls(new, ctx):
        o = orig(new, ctx)
        return {k: (o[k.replace(f"fits<={ctx}", f"over{ctx}")] if "fits<=" in k else o[k.replace(f"over{ctx}", f"fits<={ctx}")])
                for k in o}
    AZ.length_split = ls

def a_secondary_ignores_cross():
    orig = AZ.summarize_run
    def sr(sets, meta):
        s = orig(sets, meta)
        s["pass_and_cross"] = s["primary_plain"]["pass"]
        return s
    AZ.summarize_run = sr

def a_fitting_uses_over():
    orig = AZ.length_split
    def ls(new, ctx):
        o = orig(new, ctx)
        o["pass_on_fitting_items"] = all(o[f"{n}|over{ctx}"] is not None and o[f"{n}|over{ctx}"]["acc"] >= 0.8
                                         for n in ("LW10", "TS10", "NU10"))
        return o
    AZ.length_split = ls

# ---------------- dev draw ----------------
def d_dev_is_eval_items():
    # (setting DEV_SEED to the eval seed 2026 is NOT an overlap: the dev build draws d10 only, so its rng stream
    # differs; the real leak is reusing the eval items themselves)
    FT.dev_items = lambda: [x for x in N.build() if x["var"] in FT.DEV_VARS and x["d"] == 10]

def d_dev_no_twoslot():
    FT.DEV_VARS = ("same_k1", "same_k2", "same_k3", "noupd")

# ---------------- item identity ----------------
def i_filler_dropped():
    N.EXTRA_DISTRACTORS = N.EXTRA_DISTRACTORS[:-1]

def i_cross_missing():
    orig = FT.eval_sets
    FT.eval_sets = lambda names, dry: [x for x in orig(names, dry) if x[0] != "cross"]

def i_new_seed():
    orig = N.build
    N.build = lambda n_scen=32, distances=(0, 4, 10), seed=2026, families=("day", "color"): \
        orig(n_scen, distances, 2027 if seed == 2026 else seed, families)

# ---------------- params ----------------
def r_neo_no_pos():
    orig = PR.count
    def count(cfg):
        c = orig(cfg)
        if cfg["model_type"] == "gpt_neo":
            c["embedding"] -= c["pos_emb"]; c["total"] -= c["pos_emb"]; c["pos_emb"] = 0
        return c
    PR.count = count

def r_pythia_tied():
    orig = PR.count
    def count(cfg):
        return orig(dict(cfg, tie_word_embeddings=True) if cfg["model_type"] == "gpt_neox" else cfg)
    PR.count = count

# ---------------- queue ----------------
def q_lr_on_eval():
    orig = VE.queue_text
    VE.queue_text = lambda: orig().replace("LRSEARCH_ARGS=(--seed 0 --sets dev", "LRSEARCH_ARGS=(--seed 0 --sets eval,dev")

def q_seed0_scored():
    orig = VE.queue_text
    VE.queue_text = lambda: orig().replace("for s in 1 2 3", "for s in 0 1 2")

def q_bs_changed():
    orig = VE.queue_text
    VE.queue_text = lambda: orig().replace("COMMON=(--bs 4 --accum 4", "COMMON=(--bs 16 --accum 1")

# ---------------- loss self-check comparator ----------------
def c_close_always():
    FT.close = lambda a, b: True

# ---------------- tokenization (need --tok) ----------------
def k_left_truncate():
    orig = FT.encode
    def enc(tok, ex):
        ids, labels, how = orig(tok, ex)
        return ids[-300:], labels[-300:], how
    FT.encode = enc

def k_label_prompt():
    orig = FT.encode
    def enc(tok, ex):
        ids, labels, how = orig(tok, ex)
        return ids, list(ids[:-2]) + labels[-2:], how
    FT.encode = enc

def k_short_maxlen_bias():
    orig = VE.test_tok
    VE.test_tok = lambda family, exs, max_len=768: orig(family, exs, max_len=512)


MUTANTS = [
    ("ties count as right (empty answer passes)", g_ties_pass, False),
    ("gold compared with the first foil only (plausible wrong answer passes)", g_only_first_foil, False),
    ("gold-only record passes", g_gold_only_passes, False),
    ("empty record passes", g_empty_passes, False),
    ("NaN scores pass", g_nan_passes, False),
    ("pass rule ignores two-slot and no-update", g_primary_no_controls, False),
    ("pass rule read at d4", g_primary_d4, False),
    ("pass rule uses > 0.8", g_primary_strict, False),
    ("'most seeds' counts a tie", g_verdict_tie, False),
    ("paired CI ignores prompt hashes", g_paired_no_hash, False),
    ("paired change sign flipped", g_paired_sign, False),
    ("LR pick: tie -> larger LR", p_tie_larger, False),
    ("LR pick: by mean instead of min cell", p_by_mean, False),
    ("LR pick: NaN-loss run eligible", p_nan_eligible, False),
    ("LR pick: never extends", p_never_extend, False),
    ("LR pick: extends even when the top LR passes", p_extend_when_passing, False),
    ("LR pick: dev NU read at d4", p_dev_nu_d4, False),
    ("analysis: crossed pair = either answer", a_cross_any, False),
    ("analysis: knowledge change vs itself, not the untouched model", a_knowledge_self, False),
    ("analysis: pass rule read from the wrong set", a_primary_wrong_set, False),
    ("analysis: length split flipped", a_length_split_flipped, False),
    ("analysis: secondary label ignores the crossed pair", a_secondary_ignores_cross, False),
    ("analysis: fitting-items secondary reads the over-context items", a_fitting_uses_over, False),
    ("dev draw reuses the scored eval items", d_dev_is_eval_items, False),
    ("dev draw drops two-slot", d_dev_no_twoslot, False),
    ("eval items changed (a filler dropped)", i_filler_dropped, False),
    ("eval sets missing the crossed control", i_cross_missing, False),
    ("eval items drawn with another seed", i_new_seed, False),
    ("params: GPT-Neo position table not counted", r_neo_no_pos, False),
    ("params: Pythia head counted as tied", r_pythia_tied, False),
    ("queue: LR search scores the eval items", q_lr_on_eval, False),
    ("queue: scored seeds include the LR-search seed 0", q_seed0_scored, False),
    ("queue: micro-batch differs from E002", q_bs_changed, False),
    ("loss self-check comparator always matches", c_close_always, False),
    ("examples left-truncated (old bug)", k_left_truncate, True),
    ("prompt tokens labelled", k_label_prompt, True),
    ("max_len 512 (biased long-distance mix)", k_short_maxlen_bias, True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", action="store_true")
    a = ap.parse_args()
    fresh()
    base = VE.run(a.tok, fast=True, verbose=False)
    print(f"unmutated: {len(base['fails'])} failures {base['fails'][:3]}", flush=True)
    ok = not base["fails"]
    results = []
    for name, fn, needs_tok in MUTANTS:
        if needs_tok and not a.tok:
            results.append({"mutant": name, "status": "skipped (no --tok)"})
            continue
        fresh()
        fn()
        try:
            r = VE.run(needs_tok, fast=True, verbose=False)
            killed, why = bool(r["fails"]), r["fails"][:2]
        except Exception as e:  # a crash is not a kill
            killed, why = False, [f"CRASH {type(e).__name__}: {e}"]
        results.append({"mutant": name, "status": "killed" if killed else "SURVIVED", "by": why})
        print(("killed   " if killed else "SURVIVED ") + name + (f"  <- {why[0][:110]}" if killed else f"  {why}"), flush=True)
        ok &= killed
    fresh()
    n_k = sum(r["status"] == "killed" for r in results)
    print(f"{n_k} of {len(results)} mutants killed")
    json.dump({"unmutated_fails": base["fails"], "mutants": results, "killed": n_k, "total": len(results)},
              open(os.path.join(os.path.dirname(HERE), "logs", "mutation_e003.json"), "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
