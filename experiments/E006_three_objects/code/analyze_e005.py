"""E005 analysis (notes.txt PASS RULE, ALIAS READING, EXTRA REPORTS); CPU only, loads no model. Reads E005's out/
and, read-only, E004's out/ (its seeds are rescored by this same code for every comparison) and E002's out/.
  rule       E004's pass rule and reading labels (rules_e004 via rules_e005) on the 5 planned seeds; a FAIL prints
             the trigger facts and the item evidence, not E004's fixed strings
  alias      the pre-registered alias reading (DATA GAP / REAL LIMIT / INCONCLUSIVE) + per-family values, E004 alongside
  evidence   per seed: H1/H2 by reference form, alias adjacency split, ellipsis with another object after, H5 splits,
             wrong picks (evidence_e005)
  chat       chat-render strict GEN per family, its first-line variant, the capped share, and the stopping yardstick
  collateral format, knowledge (McNemar p unrounded; E005 vs E004 per item), continuity, length split and chat
             probe (analyze_e004_b), continuity reproduction by prompt hash (analyze_e005_b), E005 - E004 per family
A run counts only when its e004 LIK and GEN plain files hold 640 records (E004's rule); a planned seed without
one counts as failing. The untouched baseline is E004's, copied into E005's out/ (tag base) by the queue.
usage: python3 -B analyze_e005.py [--out-dir D] [--e004-out D] [--results F] [--tables F]"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import analyze_e004 as A
import analyze_e004_b as B
import analyze_e005_b as B5
import evidence_e005 as V
import rules_e005 as R5

MODEL, N_PLANNED = "HuggingFaceTB/SmolLM2-135M-Instruct", 5
SEEDS = [f"s{s}" for s in range(1, N_PLANNED + 1)]
E004_OUT = os.path.join(os.path.dirname(EXP), "E004_general_updating", "out")
f3 = R5.f3


def recs(out_dir, tag, name):
    return A.load(os.path.join(out_dir, f"{A.slug(MODEL)}__{tag}__{name}.jsonl"))


def run_block(out_dir, tag):
    """-> dict for one run, or None when its plain e004 LIK or GEN file is missing or incomplete."""
    cells = A.run_cells(out_dir, MODEL, tag)
    if cells is None:
        return None
    lik, gen = recs(out_dir, tag, "e004__plain"), recs(out_dir, tag, "gen_e004__plain")
    ev = V.evidence(V.join(lik, gen))
    chat_gen = recs(out_dir, tag, "gen_e004__chat")
    chat = V.chat_cells(chat_gen, gen) if chat_gen and len(chat_gen) == A.N_EVAL else None
    al = ev["alias"]["all"]
    return {"cells": cells, "evidence": ev, "alias": (al["LIK"]["acc"], al["GEN"]["acc"]), "chat": chat,
            "chat_cells": A.run_cells(out_dir, MODEL, tag, "chat")}


def side(out_dir):
    return {t: run_block(out_dir, t) for t in ["base"] + SEEDS}


def readings(runs):
    seeds = [runs[t] for t in SEEDS]
    cells = [r["cells"] if r else None for r in seeds]
    if all(c is None for c in cells):
        rule = {"label": "NOT_SCORED", "detail": "no complete scored seed run", "sub": []}
    else:
        rule = R5.reading(cells, N_PLANNED, [c["ID_LIK"] if c else None for c in cells], MODEL,
                          [r["evidence"] if r else None for r in seeds])
    alias = R5.alias_reading([r["alias"] if r else None for r in seeds], N_PLANNED)
    stop = R5.stopping([(r["chat"] and {f: v.get("strict") for f, v in r["chat"].items()},
                         r["cells"]["GEN"]) if r and r["chat"] else None for r in seeds], N_PLANNED)
    return {"rule": rule, "alias": alias, "stopping": stop, "seed_pass": {t: R5.RU.seed_pass(c) for t, c in zip(SEEDS, cells)}}


def rule_lines(name, runs, rd):
    L = [f"== {name}: reading {rd['rule']['label']} ({rd['rule'].get('detail', '')})"]
    L += [f"   sub-reading: {s}" for s in rd["rule"].get("sub", [])]
    L.append("   run   part " + " ".join(f"{f[:9]:>9}" for f in R5.PASS) + "        ID  pass")
    for t in ["base"] + SEEDS:
        r = runs[t]
        if r is None:
            L.append(f"   {t:<5} (missing or incomplete)")
            continue
        c = r["cells"]
        for part in ("LIK", "GEN"):
            idv = c["ID_LIK"] if part == "LIK" else c["ID_GEN"]
            v = ("PASS" if R5.RU.seed_pass(c) else "fail") if part == "LIK" else ""
            L.append(f"   {t:<5} {part:<4} " + " ".join(f"{f3(c[part][f]):>9}" for f in R5.PASS) + f" {f3(idv):>9}  {v}")
    return L


def alias_lines(name, runs, rd):
    a = rd["alias"]
    L = [f"   alias reading ({name}): {a['label']} (HIGH {a['n_high']}, LOW {a['n_low']} of {N_PLANNED} planned)"
         + (f"; {a['note']}" if a["note"] else "")]
    for t in ["base"] + SEEDS:
        r = runs[t]
        if r:
            e = r["evidence"]["alias"]
            cls = R5.alias_class(*r["alias"]) if t != "base" else "-"
            L.append(f"   alias {t}: LIK {f3(e['all']['LIK']['acc'])} GEN {f3(e['all']['GEN']['acc'])} (n "
                     f"{e['all']['LIK']['n']}) [{cls}]; H1 {f3(e['H1']['LIK']['acc'])}/{f3(e['H1']['GEN']['acc'])}, "
                     f"H2 {f3(e['H2']['LIK']['acc'])}/{f3(e['H2']['GEN']['acc'])}; placement LIK/GEN " + ", ".join(
                         f"{p} {f3(v['LIK']['acc'])}/{f3(v['GEN']['acc'])} (n {v['LIK']['n']})"
                         for p, v in e["place"].items()))
    return L


def evidence_run_lines(t, ev):
    L = []
    for f in ("H1", "H2"):
        L.append(f"   {t} {f} by form LIK/GEN: " + ", ".join(
            f"{k} {f3(v['LIK']['acc'])}/{f3(v['GEN']['acc'])} (n {v['LIK']['n']})" for k, v in ev["form"][f].items()))
    L.append(f"   {t} ellipsis latest, another object after (True) or not, LIK: " + "; ".join(
        f"{f} " + " ".join(f"{k}={f3(v['LIK']['acc'])}/{v['LIK']['n']}" for k, v in d.items())
        for f, d in sorted(ev["ell_other_after"].items())))
    h5 = ev["h5"]
    L.append(f"   {t} H5 LIK: other object's indirect correction after the asked latest " + ", ".join(
        f"{k} {f3(v['LIK']['acc'])} (n {v['LIK']['n']})" for k, v in h5["other_ind_after"].items())
        + "; by asked latest form " + ", ".join(f"{k} {f3(v['LIK']['acc'])} (n {v['LIK']['n']})" for k, v in h5["form"].items()))
    return L


def chat_lines(name, runs, rd):
    L = [f"   chat ({name}, not ruled): stopping learned = {rd['stopping']['learned']} "
         f"(seeds within {R5.STOP_TOL} of plain GEN on every pass family: {rd['stopping']['seeds_stop']})"]
    for t in ["base"] + SEEDS:
        r = runs[t]
        if r and r["chat"]:
            ch = r["chat"]
            for k in ("strict", "first_line", "capped"):
                L.append(f"   chat {t} {k:<10} " + " ".join(f"{f3(ch.get(f, {}).get(k)):>6}" for f in R5.PASS + ["ID"]))
            gap = [ch[f]["strict"] - r["cells"]["GEN"][f] for f in R5.PASS if ch.get(f)]
            L.append(f"   chat {t} strict - plain GEN over pass families: min {f3(min(gap))} max {f3(max(gap))}")
    return L


def analyze(out_dir, e004_out):
    runs5, runs4 = side(out_dir), side(e004_out)
    rd5, rd4 = readings(runs5), readings(runs4)
    L = ["E005 tables (analyze_e005.py; pass rule notes.txt PASS RULE, alias reading notes.txt ALIAS READING)", ""]
    L += rule_lines(f"{MODEL} E005", runs5, rd5) + alias_lines("E005", runs5, rd5) + chat_lines("E005", runs5, rd5)
    if rd5["rule"]["label"] != "FAIL":
        L += [f"   evidence: {x}" for x in R5.evidence_lines([runs5[t] and runs5[t]["evidence"] for t in SEEDS])]
    L += [f"   (E004 rescored by this code: reading {rd4['rule']['label']})"]
    L += [f"   E004 evidence: {x}" for x in R5.evidence_lines([runs4[t] and runs4[t]["evidence"] for t in SEEDS])]
    L += alias_lines("E004, same code", runs4, rd4) + chat_lines("E004, same code", runs4, rd4)
    for name, runs in (("E005", runs5), ("E004", runs4)):
        for t in SEEDS:
            if runs[t]:
                L += evidence_run_lines(f"{name} {t}", runs[t]["evidence"])
    fd = B5.family_diff({t: r and r["cells"] for t, r in runs5.items() if t != "base"},
                        {t: r and r["cells"] for t, r in runs4.items() if t != "base"})
    L += B5.family_diff_lines(fd)
    col = B.collateral(MODEL, N_PLANNED, out_dir)
    col.pop("e002_repro_flips", None)
    kn = col.pop("knowledge")
    L += B5.knowledge_lines(kn)
    k54 = B5.e005_vs_e004_knowledge(out_dir, [t for t in SEEDS if runs5[t]], e004_out, [t for t in SEEDS if runs4[t]], MODEL)
    if k54:
        L.append(f"   knowledge kbig441 E005 vs E004 (per item mean over seeds): {k54['acc_e005']:.3f} vs "
                 f"{k54['acc_e004']:.3f} d={k54['d']:+.3f} CI{k54['ci95']} (seeds {k54['seeds_e005']}/{k54['seeds_e004']})")
    L += B.table_collateral(dict(col, knowledge={}))
    fl = {"untouched vs E002 base": B5.repro_flips(out_dir, MODEL, B5.E002_OUT),
          "untouched vs E004 base (the copy)": B5.repro_flips(out_dir, MODEL, e004_out)}
    for label, v in fl.items():
        L += B5.flips_lines(v, label)
    res = {"e005": {"reading": rd5, "runs": runs5}, "e004_same_code": {"reading": rd4, "runs": runs4},
           "family_diff": fd, "knowledge": kn, "knowledge_e005_vs_e004": k54, "collateral": col, "repro_flips": fl}
    return res, L


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(EXP, "out"))
    ap.add_argument("--e004-out", default=E004_OUT)
    ap.add_argument("--results", default=os.path.join(EXP, "results.json"))
    ap.add_argument("--tables", default=os.path.join(EXP, "logs", "tables.txt"))
    a = ap.parse_args(argv)
    res, lines = analyze(a.out_dir, a.e004_out)
    json.dump(res, open(a.results, "w"), indent=1, default=str)
    open(a.tables, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
