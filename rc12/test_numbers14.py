"""Pins the dev numbers recorded on 2026-10-11 (rc12/notes.txt STEP 14N): the extras (dev_extras14.py), the panel
(dev_regrade.py + dev_panel.py) and the likelihood rows, as committed in logs/extras14_20261011.jsonl,
logs/panel_20261011.jsonl and logs/lik_rows_20261011.jsonl.
  A  the results agree with themselves: R = 100 x mean of the 9 families and S7 = 100 x mean of the 7 in every row;
     D = R - R_qwen, the CI part = lower bound >= -3, the s12 clause and the LOOKUP bar read their numbers; every
     pinned extra is there (the Base plain only); dev_extras14's comparator R equals the panel's Qwen2.5 R
  B  the likelihood rows: 12 models x 2 renders, LFM2-2.6B absent, every self-check passed in float32, and every
     key's matched / all-equal counts and chance and the common-foil counts equal the tokenizer-only measurement
     (logs/lik_tokcheck_pc.json)
  C  the text carries the numbers: prereg s4, s10b and s12 (fragments built from the results), and notes.txt every
     line of tables14.all_tables()
  D  (only with raw copies) rederive14.py, written apart from the scorer, gives exactly the recorded numbers from the
     raw files: --raw <extras run tree> (models with 0 rows that regrade differently, and minimind-3's blank plain
     share in s12), --panel <regraded panel copy>, --lik <rc12_lik copy>
  python3 -B test_numbers14.py [--raw R] [--panel P] [--lik L] [--logs D] [--prereg F] [--notes F] [--pins F]"""
import argparse
import json
import os
import re
import sys

import rederive14 as RD
import tables14 as T

HERE = os.path.dirname(os.path.abspath(__file__))
NINE, SEVEN = RD.NINE, RD.SEVEN
NEW = {"Falcon-H1-Tiny-90M-Instruct-Curriculum": "-Instruct-Curriculum", "minimind-3": "minimind-3",
       "Falcon-H1-Tiny-90M-Instruct-pre-DPO": "-Instruct-pre-DPO", "cRia-LM-75M-Instruct": "cRia-LM-75M-Instruct",
       "Falcon-H1-Tiny-90M-Instruct-Curriculum-pre-DPO": "-Instruct-Curriculum-pre-DPO"}
OLD = {"Veyra2-Blueberry-5M-Instruct": ("Veyra2-Blueberry-5M-Instruct (4,984,192 parameters)", "Blueberry-5M"),
       "BananaMind-2-Nano-Chat": ("BananaMind-2-Nano-Chat", "Nano"),
       "Veyra2-Blueberry-10M-Instruct": ("Veyra2-Blueberry-10M-Instruct", "Blueberry-10M"),
       "Vertex-0.6-15M-Instruct": ("Vertex-0.6-15M-Instruct", "Vertex-15M"),
       "Veyra2-Mango-15M-Instruct": ("Veyra2-Mango-15M-Instruct", "Mango-15M"),
       "Loom-Spark-3.2": ("Loom-Spark-3.2",) * 2,
       "BananaMind-2-Mini-Chat": ("BananaMind-2-Mini-Chat", "Mini"), "Swen-28M": ("Swen-28M",) * 2,
       "Veyra2-Mango-30M-Instruct": ("Veyra2-Mango-30M-Instruct", "Mango-30M")}
CARD = "Vertex-0.6-15M-Instruct@card"
PLAIN_ONLY = "Falcon-H1-Tiny-90M-Base"


def flat(text):
    return re.sub(r"\s+", " ", text)


def section(text, start, end):
    return flat(text[text.index(start):text.index(end, text.index(start))])


def close(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol


def check_results(ex, pa, pins, fails):
    m3 = [r for r in ex + pa if r["kind"] in ("mean3", "per_seed")]
    for r in m3:
        R = 100 * sum(r["F_" + f] for f in NINE) / 9
        S7 = 100 * sum(r["F_" + f] for f in SEVEN) / 7
        if not (close(R, r["R"]) and close(S7, r["S7"])):
            fails.append(f"A {r['model']} {r['render']} {r['seed']}: R {r['R']} S7 {r['S7']} vs families {R} {S7}")
    for c in (x for x in ex if x["kind"] == "ci"):
        lo = c["ci"]["lo"]
        if not close(c["D"], c["R"] - c["R_qwen"]) or c["ci_part_met"] != (lo >= -3.0):
            fails.append(f"A {c['model']}: D or the CI part ({c['D']}, {lo}, {c['ci_part_met']})")
        cl = c["s12_clause"]
        if cl["passes"] != (cl["S7"] >= 40 and cl["PERSIST"] >= 0.40) or c["lookup"]["met"] != (c["lookup"]["value"]
                                                                                              >= 0.60):
            fails.append(f"A {c['model']}: the clause or the LOOKUP bar does not read its numbers")
    tm = {r["model"] for r in ex if r["kind"] == "mean3" and r["render"] == "template"}
    pm = {r["model"] for r in ex if r["kind"] == "mean3" and r["render"] == "plain"}
    want = {m.split("/")[-1] for m in pins["models"]}
    if tm != (want - {PLAIN_ONLY}) | {CARD} or pm != want:
        fails.append(f"A extras present: template {sorted(tm ^ ((want - {PLAIN_ONLY}) | {CARD}))}, plain "
                     f"{sorted(pm ^ want)} differ from pins.json")
    for r in ex:
        if r["kind"] == "mean3" and r["model"] in list(NEW) + [PLAIN_ONLY] and r["regraded_rows_differing"] != 0:
            fails.append(f"A {r['model']} {r['render']}: {r['regraded_rows_differing']} rows regrade differently")
        if r["kind"] == "per_seed" and r["scores_jsonl_ok"] is not True:
            fails.append(f"A {r['model']} {r['render']} {r['seed']}: stored scores.jsonl differs")
    q = [r["R"] for r in pa if r["kind"] == "mean3" and r["model"] == "Qwen2.5-0.5B-Instruct"
         and r["render"] == "template"]
    meta = [r for r in ex if r["kind"] == "meta"][0]
    if len(q) != 1 or not close(q[0], meta["comparator_R"]["template"], 1e-12):
        fails.append(f"A comparator R: extras {meta['comparator_R']['template']} vs panel {q}")


def check_lik(lk, tok, pa, fails):
    got = sorted((r["model"], r["render"]) for r in lk)
    ids = [r for r in pa if r["kind"] == "meta"][0]["model_ids"].values()
    want = sorted((m, r) for m in ids if m != "LiquidAI/LFM2-2.6B" for r in ("template", "plain"))
    if got != want or len(got) != 24:
        fails.append(f"B models: {sorted(set(got) ^ set(want))}")
    for r in lk:
        sc, t = r["selfcheck"], tok["models"].get(r["model"], {})
        if not (sc["ok"] and sc["flips"] == 0 and sc["tol"] == 0.01 and sc["n_rows"] == 48 and sc["n_scores"] == 144
                and sc["max_abs_diff"] <= 0.01 and r["engine"]["dtype"] == "float32" and r["rows"] == 412):
            fails.append(f"B {r['model']} {r['render']}: self-check or dtype {sc} {r['engine']['dtype']}")
        for k, v in r["families"].items():
            m = t.get("renders", {}).get(r["render"], {}).get(k, {})
            if (v["n_matched"], v["n_equal"], v["n"]) != (m.get("n_matched"), m.get("n_equal"), m.get("n")) \
                    or not close(v["chance_matched"], m.get("chance"), 1e-12):
                fails.append(f"B {r['model']} {r['render']} {k}: counts differ from the tokenizer-only measurement")
            if r["vs_qwen"] and r["render"] == "template" and \
                    r["vs_qwen"][k]["n_common"] != t["vs_comparator"][k]["n_common"]:
                fails.append(f"B {r['model']} {k}: common-foil rows {r['vs_qwen'][k]['n_common']} vs tokcheck")


def frag_new(c, label):
    lk = c["lookup"]
    return (f"{label} R {c['R']:.2f} ({c['ci']['lo']:+.2f} to {c['ci']['hi']:+.2f}), S7 {c['S7']:.2f} "
            f"({c['s7_ci']['lo']:.2f} to {c['s7_ci']['hi']:.2f}), PERSIST {c['s12_clause']['PERSIST']:.3f}, "
            f"LOOKUP {lk['value']:.3f} (P {lk['P']:.3f}, X {lk['X']:.3f})")


def check_text(ex, pa, lk, prereg, notes, fails):
    s12 = section(prereg, "12. BASELINE PANEL", "13. SEALED SPLIT")
    s10b = section(prereg, "The comparison: every public panel model's LOOKUP", "11. HEADROOM RULE")
    s4 = section(prereg, "Likelihood rows (diagnostic, P-004;", "5. FAMILIES AND DEV COUNTS")
    ci = {c["model"]: c for c in ex if c["kind"] == "ci"}
    m3 = {(r["model"], r["render"]): r for r in ex if r["kind"] == "mean3"}
    want = [(s12, frag_new(ci[m], lab)) for m, lab in NEW.items()]
    b = m3[(PLAIN_ONLY, "plain")]
    mm = m3[("minimind-3", "plain")]
    want += [(s12, f"-Base, plain only, R {b['R']:.2f}, S7 {b['S7']:.2f}"),
             (s12, f"Plain (diagnostic) R {mm['R']:.2f}: {mm['EMPTY']:.3f} of its plain replies"),
             (s12, f"every extra added before the lock at most {max(ci[m]['S7'] for m in NEW):.2f}"),
             (s10b, f"P at most {max(ci[m]['lookup']['P'] for m in NEW):.3f}, X at most "
                    f"{max(ci[m]['lookup']['X'] for m in NEW):.3f}"),
             (s10b, f"P at most {max(ci[m]['lookup']['P'] for m in list(OLD) + [CARD]):.3f}, X at most "
                    f"{max(ci[m]['lookup']['X'] for m in list(OLD) + [CARD]):.3f}")]
    for m, (name, short) in OLD.items():
        c = ci[m]
        want += [(s12, f"{name} {c['R']:.2f} ({c['ci']['lo']:+.2f} to {c['ci']['hi']:+.2f}"),
                 (s12, f"{short} {c['S7']:.2f} ({c['s7_ci']['lo']:.2f} to {c['s7_ci']['hi']:.2f}), "
                       f"{c['s12_clause']['PERSIST']:.3f}")]
    c = ci[CARD]
    want += [(s12, f"card row {c['R']:.2f}, {c['ci']['lo']:+.2f} to {c['ci']['hi']:+.2f}"),
             (s12, f"card row {c['S7']:.2f}, {c['s7_ci']['lo']:.2f} to {c['s7_ci']['hi']:.2f}, "
                   f"{c['s12_clause']['PERSIST']:.3f}"),
             (s12, f"(R {m3[('Swen-28M', 'plain')]['R']:.2f}, S7 {m3[('Swen-28M', 'plain')]['S7']:.2f}; a diagnostic)")]
    lt = {r["model"].split("/")[-1]: r["families"] for r in lk if r["render"] == "template"}
    for m in ("Qwen3-0.6B", "Qwen3.5-0.8B", "LFM2-1.2B"):
        f = lt[m]["LOOKUP"]
        want.append((s4, f"{m} {f['acc_matched']:.2f} against {f['chance_matched']:.2f}"))
        gen = [r["F_LOOKUP"] for r in pa if r["kind"] == "mean3" and r["render"] == "template" and r["model"] == m]
        if gen != [] and gen[0] > 0.014:
            fails.append(f"C s4: {m}'s generated LOOKUP {gen[0]} is above 0.014")
    co, cc = [lt[m]["CORR"]["acc_matched"] for m in lt], [lt[m]["CORR"]["chance_matched"] for m in lt]
    want += [(s4, f"CORR stays near chance for every model ({min(co):.2f} to {max(co):.2f} against {min(cc):.2f} to "
                  f"{max(cc):.2f})"),
             (s4, f"predicts ({min(lt[m]['OWN']['acc_matched'] for m in lt):.2f} to "
                  f"{max(lt[m]['OWN']['acc_matched'] for m in lt):.2f})")]
    for where, frag in want:
        if frag not in where:
            fails.append(f"C prereg lacks: {frag}")
    for name, lines in T.all_tables(ARGS.logs).items():
        for line in lines:
            if line not in notes:
                fails.append(f"C notes.txt lacks {name} line: {line}")


def check_raw(ex, pa, lk, fails):
    n = 0
    rec = {(r["model"], r["render"], r["seed"]): r for r in ex + pa if r["kind"] in ("mean3", "per_seed")}
    jobs = []
    if ARGS.raw:
        jobs += [(ARGS.raw, r["model"], r["render"]) for r in ex if r["kind"] == "mean3" and r["model"] != CARD
                 and r["regraded_rows_differing"] == 0]
    if ARGS.panel:
        jobs += [(ARGS.panel, r["model"], r["render"]) for r in pa if r["kind"] == "mean3"]
    for root, m, rend in jobs:
        d = RD.dev(root, m, rend)
        for s, v in list(d["seeds"].items()) + [("mean3", d["mean3"])]:
            r = rec.get((m, rend, s))
            if r is None:       # the panel file keeps mean3 rows only
                continue
            for k in NINE + ["LOOKUP", "T0", "OWN_ungated", "R", "S7", "loop"]:
                n += 1
                key = k if k in ("R", "S7", "T0", "loop", "OWN_ungated") else "F_" + k
                if not close(v[k], r[key], 1e-12):
                    fails.append(f"D {m} {rend} {s} {k}: rederived {v[k]} recorded {r[key]}")
    if ARGS.raw:
        x = RD.blank_after_role_cut(ARGS.raw, "minimind-3", "plain")
        n += 1
        if f"{x:.3f} of all because it opens a new" not in section(open(ARGS.prereg).read(), "12. BASELINE PANEL",
                                                                  "13. SEALED SPLIT"):
            fails.append(f"D prereg s12 lacks minimind-3's blank-after-role-cut share {x:.3f}")
    if ARGS.lik:
        for r in lk:
            mine = RD.lik(RD.read(os.path.join(ARGS.lik, r["model"].replace("/", "__"), r["render"], "rows.jsonl")))
            for k, v in r["families"].items():
                pairs = [(mine[k]["n_matched"], v["n_matched"]), (mine[k]["acc_matched"], v["acc_matched"]),
                         (mine[k]["chance_matched"], v["chance_matched"]), (mine[k]["n_equal"], v["n_equal"]),
                         (mine[k]["acc_equal"], v["acc_equal"]), (mine[k]["tier0_mean"], v["tier0"]["mean"]),
                         (mine[k]["tier0_p10"], v["tier0"]["p10"]), (mine[k]["foil_mismatch"], 0)]
                if k == "BIND":
                    pairs += [(mine[k]["pairs_n"], v["pairs"]["n"]), (mine[k]["pairs_acc"], v["pairs"]["acc"])]
                for a, b in pairs:
                    n += 1
                    if not (a == b or close(a, b, 1e-12)):
                        fails.append(f"D lik {r['model']} {r['render']} {k}: rederived {a} recorded {b}")
    return n


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    for a in ("raw", "panel", "lik"):
        ap.add_argument("--" + a)
    ap.add_argument("--logs", default=T.LOGS)
    ap.add_argument("--prereg", default=os.path.join(HERE, "..", "prereg", "RC-12.draft.txt"))
    ap.add_argument("--notes", default=os.path.join(HERE, "notes.txt"))
    ap.add_argument("--pins", default=os.path.join(HERE, "pins.json"))
    ARGS = ap.parse_args()
    ex, pa, lk = (T.load(w, ARGS.logs) for w in ("extras", "panel", "lik"))
    fails, n = [], 0
    for name, run in (("A", lambda: check_results(ex, pa, json.load(open(ARGS.pins)), fails)),
                      ("B", lambda: check_lik(lk, json.load(open(os.path.join(ARGS.logs, "lik_tokcheck_pc.json"))),
                                              pa, fails)),
                      ("C", lambda: check_text(ex, pa, lk, open(ARGS.prereg).read(), open(ARGS.notes).read(), fails)),
                      ("D", lambda: check_raw(ex, pa, lk, fails))):
        try:
            n = run() or n
        except Exception as e:      # a missing row or key is a failure of that part, never a pass
            fails.append(f"{name} stopped: {type(e).__name__} {e}")
    for f in fails[:40]:
        print("FAIL", f)
    print(f"test_numbers14: {'PASS' if not fails else f'{len(fails)} failures'}; raw numbers compared: {n}"
          f"{'' if n else ' (no raw copy given: part D not run)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
