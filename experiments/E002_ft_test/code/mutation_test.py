"""Mutation test for E002: break each grader, pass-rule detail and training-generator property on
purpose and confirm validate.run() goes red. A test that has never been watched failing is not
evidence. Exits non-zero if any mutant survives or the unmutated suite is not green.
Run: python mutation_test.py [--tok HuggingFaceTB/SmolLM2-135M-Instruct]
"""
import argparse, importlib, json, math, sys

import metrics_ft as MF
import train_data as TD
import ft_test as FT
import validate as V

TOK = None


import eval_extra as X
import gen_probe as GP


def fresh():
    for m in (MF, TD, X, GP, FT, V):
        importlib.reload(m)


# ---------------- grader mutants ----------------
def g_ties_pass():
    MF.right = lambda s: bool(s) and "gold" in s and len(s) >= 2 and all(s["gold"] >= v for k, v in s.items() if k != "gold")

def g_only_first_foil():
    def r(s):
        if not s or "gold" not in s or len(s) < 2:
            return False
        others = [k for k in s if k != "gold"]
        return s["gold"] > s[others[0]]
    MF.right = r

def g_gold_only_passes():
    orig = MF.right
    MF.right = lambda s: True if (s and "gold" in s and len(s) == 1) else orig(s)

def g_empty_dict_passes():
    orig = MF.right
    MF.right = lambda s: True if not s else orig(s)

def g_pass_without_controls():
    orig = MF.primary
    def p(recs, render="plain"):
        out = orig(recs, render)
        out["pass"] = out["LW10"] is not None and out["LW10"]["acc"] >= MF.THRESH
        return out
    MF.primary = p

def g_pass_without_twoslot():
    orig = MF.primary
    def p(recs, render="plain"):
        out = orig(recs, render)
        out["pass"] = all(c is not None and c["acc"] >= MF.THRESH for c in (out["LW10"], out["NU10"]))
        return out
    MF.primary = p

def g_read_at_d4():
    def p(recs, render="plain"):
        lw = MF.cell(recs, render, MF.LW_VARS, 4)
        ts = MF.cell(recs, render, ("twoslot",), 4)
        nu = MF.cell(recs, render, ("noupd",), 4)
        return {"LW10": lw, "TS10": ts, "NU10": nu, "pass": all(c is not None and c["acc"] >= MF.THRESH for c in (lw, ts, nu))}
    MF.primary = p

def g_strict_threshold():
    def p(recs, render="plain"):
        lw = MF.cell(recs, render, MF.LW_VARS, 10)
        ts = MF.cell(recs, render, ("twoslot",), 10)
        nu = MF.cell(recs, render, ("noupd",), 10)
        return {"LW10": lw, "TS10": ts, "NU10": nu, "pass": all(c is not None and c["acc"] > MF.THRESH for c in (lw, ts, nu))}
    MF.primary = p

def g_any_render():
    orig_cell = MF.cell
    MF.cell = lambda recs, render, vars_, d, fam=None: orig_cell([dict(r, render=render) for r in recs], render, vars_, d, fam)

def g_majority_ge_half():
    MF.model_verdict = lambda sp: {"seeds": len(sp), "passing": sum(sp), "lock_in_rate": None,
                                   "pass": len(sp) > 0 and sum(sp) >= len(sp) / 2}

def g_lockin_first_touch():
    def L(traj, keys=("LW", "TS", "NU")):
        for t in traj:
            if all(t.get(k) is not None and t[k] >= MF.THRESH for k in keys):
                return t["step"]
        return None
    MF.lock_in_step = L

def g_lockin_ignores_ts():
    orig = MF.lock_in_step
    MF.lock_in_step = lambda traj, keys=("LW", "TS", "NU"): orig(traj, keys=("LW", "NU"))

def g_paired_no_hash():
    orig = MF.paired
    def p(before, after, key="id", n_boot=10000, seed=0):
        return orig([dict(r, h=None) for r in before], [dict(r, h=None) for r in after], key, n_boot, seed)
    MF.paired = p

def g_paired_sign_flip():
    orig = MF.paired
    def p(*a, **k):
        out = orig(*a, **k)
        out["d_acc"] = -out["d_acc"]
        out["d_margin"] = -out["d_margin"]
        return out
    MF.paired = p

# ---------------- generator mutants ----------------
def x_cross_b_gold_last_corr_of_a():
    import eval_extra as X
    orig = X.build_crossed
    def b(*a, **k):
        its = orig(*a, **k)
        for it in its:
            if it["var"] == "cross_B":
                c = it["cands"]
                c["gold"], c["other_corr"] = c["other_corr"], c["gold"]
        return its
    X.build_crossed = b

def gp_ignore_other_values():
    import gen_probe as GP
    orig = GP.grade
    def g(reply, cands):
        if not reply or not reply.strip():
            return False, False
        ok = bool(GP.mentions(reply, cands["gold"].strip()))
        return ok, orig(reply, cands)[1]
    GP.grade = g

def gp_no_negation():
    import gen_probe as GP
    GP.NEG = __import__("re").compile(r"(?!x)x")

def gp_empty_passes():
    import gen_probe as GP
    orig = GP.grade
    GP.grade = lambda reply, cands: (True, True) if not reply.strip() else orig(reply, cands)

def gp_lenient_last_mention():
    import gen_probe as GP
    orig = GP.grade
    def g(reply, cands):
        s, _ = orig(reply, cands)
        vals = {lab: v.strip() for lab, v in cands.items()}
        pos = {lab: GP.mentions(reply, v) for lab, v in vals.items()}
        last = max(((p[-1], lab) for lab, p in pos.items() if p), default=(None, None))[1]
        return s, last == "gold"
    GP.grade = g

def t_no_noupd():
    TD.UPDATE_KINDS = {"upd": 0.45, "twoslot": 0.25, "twoslot_b": 0.10, "mid": 0.10, "revert": 0.10}

def t_twoslot_gold_is_last():
    orig = TD.gen_update
    def g(rng, kind=None, vtype=None, d=None):
        ex = orig(rng, kind, vtype, d)
        if ex["kind"] == "twoslot":
            ex["gold"] = ex["mentions"][-1][0]
            ex["answer"] = " " + ex["gold"] + "."
        return ex
    TD.gen_update = g

def t_only_single_object():
    TD.UPDATE_KINDS = {"upd": 1.0}

def t_eval_frame_leak():
    TD.T["day"]["orig"] = TD.T["day"]["orig"] + [("My {o} is on {v}.", "Okay, your {o} is on {v}.")]

def t_eval_question_leak():
    TD.T["day"]["ask"] = TD.T["day"]["ask"] + [("What day is my {o}?", "Your {o} is on")]

def t_eval_filler_leak():
    TD.T_DISTRACT = TD.T_DISTRACT + [("Why do cats purr?", "Cats purr when they are relaxed, but also to calm themselves when they are stressed or hurt.")]

def t_filler_with_value():
    TD.T_DISTRACT = TD.T_DISTRACT + [("Is the gym busy on Monday?", "It is usually busiest in the early evening.")]

def t_eval_object_leak():
    TD.T["day"]["objects"] = TD.T["day"]["objects"] + [("dentist appointment", "dentist")]

def t_question_contains_gold():
    orig = TD.gen_update
    def g(rng, kind=None, vtype=None, d=None):
        ex = orig(rng, kind, vtype, d)
        ex["question"] = ex["question"][:-1] + f" (was it {ex['gold']}?)"
        return ex
    TD.gen_update = g

def t_eval_name_leak():
    TD.T_PETS = TD.T_PETS[:-1] + ["Biscuit"]

# ---------------- tokenization / stream mutants (need --tok) ----------------
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
    orig = V.test_tok
    V.test_tok = lambda model_id, exs, max_len=768: orig(model_id, exs, max_len=512)


MUTANTS = [
    ("ties count as right (empty answer passes)", g_ties_pass, False),
    ("gold compared with the first foil only", g_only_first_foil, False),
    ("gold-only record passes", g_gold_only_passes, False),
    ("empty record passes", g_empty_dict_passes, False),
    ("pass rule ignores two-slot and no-update", g_pass_without_controls, False),
    ("pass rule ignores two-slot", g_pass_without_twoslot, False),
    ("pass rule read at d4 instead of d10", g_read_at_d4, False),
    ("pass rule uses > 0.8 instead of >= 0.8", g_strict_threshold, False),
    ("pass rule pools renders", g_any_render, False),
    ("'most seeds' counts a tie", g_majority_ge_half, False),
    ("lock-in = first touch, not stays", g_lockin_first_touch, False),
    ("lock-in ignores two-slot", g_lockin_ignores_ts, False),
    ("paired CI ignores prompt hashes", g_paired_no_hash, False),
    ("paired CI sign flipped", g_paired_sign_flip, False),
    ("crossed item B asks for A's correction (post-hoc control broken)", x_cross_b_gold_last_corr_of_a, False),
    ("generation grader ignores other candidate values", gp_ignore_other_values, False),
    ("generation grader ignores negation", gp_no_negation, False),
    ("generation grader passes an empty reply", gp_empty_passes, False),
    ("generation lenient grader uses the last mention", gp_lenient_last_mention, False),
    ("training has no no-update items", t_no_noupd, False),
    ("training two-slot gold = last mention", t_twoslot_gold_is_last, False),
    ("training has only single-object updates (old shortcut)", t_only_single_object, False),
    ("eval statement frame leaked into training", t_eval_frame_leak, False),
    ("eval question/prefix leaked into training", t_eval_question_leak, False),
    ("eval filler leaked into training", t_eval_filler_leak, False),
    ("training filler carries a weekday", t_filler_with_value, False),
    ("eval object noun leaked into training", t_eval_object_leak, False),
    ("training question contains its answer", t_question_contains_gold, False),
    ("eval pet name leaked into training", t_eval_name_leak, False),
    ("examples left-truncated (old bug)", k_left_truncate, True),
    ("prompt tokens labelled", k_label_prompt, True),
    ("max_len 512 (biased long-distance mix)", k_short_maxlen_bias, True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", default=None)
    a = ap.parse_args()
    fresh()
    base = V.run(a.tok, n_train=6000, verbose=False)
    print(f"unmutated: {len(base['fails'])} failures {base['fails'][:3]}")
    ok = not base["fails"]
    results = []
    for name, fn, needs_tok in MUTANTS:
        if needs_tok and not a.tok:
            results.append({"mutant": name, "status": "skipped (no --tok)"})
            continue
        fresh()
        fn()
        try:
            r = V.run(a.tok if needs_tok else None, n_train=6000, verbose=False)
            killed = bool(r["fails"])
            why = r["fails"][:2]
        except Exception as e:  # a crash is not a kill: record it separately
            killed, why = False, [f"CRASH {type(e).__name__}: {e}"]
        results.append({"mutant": name, "status": "killed" if killed else "SURVIVED", "by": why})
        print(("killed   " if killed else "SURVIVED ") + name + ("" if killed else f"  {why}"), flush=True)
        ok &= killed
    fresh()
    n_k = sum(r["status"] == "killed" for r in results)
    print(f"{n_k} of {len(results)} mutants killed")
    json.dump({"unmutated_fails": base["fails"], "mutants": results, "killed": n_k, "total": len(results)},
              open("../logs/mutation_test.json", "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
