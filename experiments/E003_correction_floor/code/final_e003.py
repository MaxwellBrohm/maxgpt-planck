"""E003 final reading (CPU only, loads no model; run after analyze_e003.py). Plan: ../notes.txt, 2026-10-02 entry.
(1) integrity of the finished queue, (2) the pre-registered verdicts and floor re-derived from ../results.json,
(3) POST-HOC descriptive uncertainty (labelled; never a verdict).
Writes ../logs/final_e003.json and ../logs/final_e003.txt.   usage: python final_e003.py
"""
import datetime as DT, glob, json, math, os, re, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
OUT, LOGS = os.path.join(EXP, "out"), os.path.join(EXP, "logs")
sys.path.insert(0, HERE)
import metrics_ft as MF
import params as PR
import pick_lr as PL

SHORT = {"roneneldan/TinyStories-1M": "ts1m", "EleutherAI/pythia-14m": "p14m", "roneneldan/TinyStories-3M": "ts3m",
         "EleutherAI/pythia-31m": "p31m", "roneneldan/TinyStories-8M": "ts8m", "EleutherAI/pythia-70m": "p70m",
         "roneneldan/TinyStories-33M": "ts33m", "EleutherAI/pythia-160m": "p160m"}
EVAL_N = {"new/plain": 1728, "extra/plain": 192, "old/plain": 1116, "uprobe/plain": 576, "khard/plain": 80,
          "kbig/plain": 441, "cross/plain": 384}
DEV_N = {"dev/plain": 320}
CELLS = (("LW10", MF.LW_VARS), ("TS10", ("twoslot",)), ("NU10", ("noupd",)))
slug = PL.slug
problems = []


def bad(msg):
    problems.append(msg)


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(c - h, 3), round(c + h, 3)


def lines_of(path):
    return sum(1 for l in open(path) if l.strip())


def ts(s):
    return DT.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()


def queue_attempts():
    att = defaultdict(lambda: {"done": 0, "killed": []})
    for l in open(os.path.join(LOGS, "queue.txt")):
        m = re.match(r"^\S+ \S+ (\S+) exit=(\d+) killed=(\S+)", l)
        if m:
            if m.group(2) == "0" and m.group(3) == "None":
                att[m.group(1)]["done"] += 1
            else:
                att[m.group(1)]["killed"].append(m.group(3))
    return att


def check_job(model, job, tag, want_counts, want_args, train):
    """Integrity of one finished job; returns a short dict."""
    g = json.load(open(os.path.join(LOGS, f"{job}.guard.json")))
    if g["exit"] != 0 or g["killed"] is not None:
        bad(f"{job}: final guard record exit={g['exit']} killed={g['killed']}")
    cmd = g["cmd"]
    if cmd[3] != model:
        bad(f"{job}: model {cmd[3]}")
    for k, v in want_args.items():
        if k not in cmd or cmd[cmd.index(k) + 1] != v:
            bad(f"{job}: {k} != {v} in {cmd}")
    stem = os.path.join(OUT, f"{slug(model)}__{tag}")
    meta = json.load(open(stem + "__run.json"))
    if meta.get("fatal"):
        bad(f"{job}: fatal {meta['fatal']}")
    if meta.get("eval_counts") != want_counts:
        bad(f"{job}: eval_counts {meta.get('eval_counts')}")
    for k, n in want_counts.items():
        p = f"{stem}__{k.replace('/', '__')}.jsonl"
        if not os.path.exists(p) or lines_of(p) != n:
            bad(f"{job}: {p} missing or not {n} records")
    t0, t1 = ts(g["started"]) - 2, ts(g["ended"]) + 2
    for p in glob.glob(stem + "__*"):
        if not t0 <= os.path.getmtime(p) <= t1:
            bad(f"{job}: {os.path.basename(p)} written outside the final run window")
    sc = None
    if train:
        L = meta.get("losses") or []
        if len(L) != 400 or meta.get("steps") != 400 or not all(math.isfinite(x) for x in L):
            bad(f"{job}: losses not 400 finite")
        c = meta.get("loss_selfcheck") or {}
        sc = abs(c.get("hf", 0) - c.get("answer_only", 1e9))
        if sc > 1e-3 * max(1.0, abs(c.get("hf", 0))):
            bad(f"{job}: loss self-check {c}")
    return {"started": g["started"], "ended": g["ended"], "selfcheck_absdiff": sc, "lr": meta.get("lr")}


def recs(model, tag):
    return [json.loads(l) for l in open(os.path.join(OUT, f"{slug(model)}__{tag}__new__plain.jsonl")) if l.strip()]


def family_split(rs):
    """POST-HOC: each pass-rule cell by item family, and which candidate the model put on top."""
    out = {}
    for name, vars_ in CELLS:
        sel = [r for r in rs if r["render"] == "plain" and r["d"] == 10 and r["var"] in vars_]
        for fam in sorted({r["fam"] for r in sel}):
            c = MF.acc([MF.right(r["scores"]) for r in sel if r["fam"] == fam])
            out[f"{name}|{fam}"] = c["acc"]
        top = defaultdict(int)
        for r in sel:
            top[max(r["scores"].items(), key=lambda kv: kv[1])[0]] += 1
        out[f"{name}|top"] = {k: round(v / len(sel), 3) for k, v in sorted(top.items())}
    return out


def main():
    res = json.load(open(os.path.join(EXP, "results.json")))
    att = queue_attempts()
    F = {"integrity": {}, "models": {}, "attempts": {}}
    for model in PR.E003_MODELS:
        s, M = SHORT[model], res["models"][model]
        body = M["params"]["body"]
        integ = {"base": check_job(model, f"base_{s}", "base", {**EVAL_N, **DEV_N}, {"--steps": "0", "--tag": "base"}, False)}
        pk = json.load(open(os.path.join(LOGS, f"lr_pick_{slug(model)}.json")))
        pf_path = os.path.join(LOGS, f"lr_pick_{slug(model)}_final.json")
        pf = json.load(open(pf_path)) if os.path.exists(pf_path) else None
        rec = PL.pick(model, pk["grid"], OUT, False)
        if rec["line"] != pk["line"]:
            bad(f"{model}: pick recomputed {rec['line']} vs logged {pk['line']}")
        lrs = list(pk["grid"])
        if pf:
            lrs.append(pk["extend"])
            recf = PL.pick(model, lrs, OUT, True)
            if recf["line"] != pf["line"]:
                bad(f"{model}: final pick recomputed {recf['line']} vs logged {pf['line']}")
        for lr in lrs:
            integ[f"lr{lr}"] = check_job(model, f"lr_{s}_{lr}", f"lr{lr}_s0", DEV_N,
                                         {"--lr": lr, "--seed": "0", "--sets": "dev"}, True)
        chosen = (pf or pk)["chosen"]
        if chosen != M["chosen_lr"]:
            bad(f"{model}: results.json chosen_lr {M['chosen_lr']} vs pick {chosen}")
        for n in (1, 2, 3):
            integ[f"s{n}"] = check_job(model, f"ft_{s}_s{n}", f"s{n}", EVAL_N,
                                       {"--lr": chosen, "--seed": str(n), "--sets": "eval", "--tag": f"s{n}"}, True)
        F["integrity"][model] = integ
        for job in [f"base_{s}"] + [f"lr_{s}_{lr}" for lr in lrs] + [f"ft_{s}_s{n}" for n in (1, 2, 3)]:
            a = att.get(job, {"done": 0, "killed": []})
            if a["done"] != 1:
                bad(f"{job}: {a['done']} completed attempts in queue.txt")
            if a["killed"]:
                F["attempts"][job] = a["killed"]
        # (2) pre-registered reading
        v = M["verdict"]
        dev_rows = (pf or pk)["rows"]
        kb = [M["seeds"][t]["knowledge"]["kbig_all"] for t in M["done_seeds"]]
        base_kbig = kb[0]["acc_before"] if kb else None
        m = {"body": body, "total": M["params"]["total"], "ladder": "pythia" if "pythia" in model else "tinystories",
             "chosen_lr": chosen, "seeds_pass": f"{v['passing']}/{v['seeds']}", "verdict": "PASS" if v["pass"] else "FAIL",
             "best_dev_min": max(r["min"] for r in dev_rows), "best_dev_lr": f"{max(dev_rows, key=PL.key)['lr']:.0e}",
             "fails_every_lr": (not v["pass"]) and all(r["min"] < 0.8 for r in dev_rows),
             "per_object_tracking": M["verdict_pass_and_cross"]["passing"],
             "fitting_items_diag": (M.get("verdict_fitting_items") or {}).get("passing"),
             "lock_in_steps": M["lock_in_steps"], "base_kbig_acc": base_kbig,
             "knowledge_informative": base_kbig is not None and base_kbig >= 0.6,
             "kbig_dacc": {t: (k["d_acc"], k["d_acc_ci95"]) for t, k in zip(M["done_seeds"], kb)}, "seeds": {}}
        # (3) POST-HOC
        for t in M["done_seeds"]:
            p = M["seeds"][t]["primary_plain"]
            cells = {}
            for name, _ in CELLS:
                c = p[name]
                k = round(c["acc"] * c["n"])
                cells[name] = {"acc": c["acc"], "n": c["n"], "wilson95": wilson(k, c["n"])}
            failing = [nm for nm in cells if cells[nm]["acc"] < 0.8]
            m["seeds"][t] = {"cells": cells, "failing": failing,
                             "fail_robust": any(cells[nm]["wilson95"][1] < 0.8 for nm in failing),
                             "family_split": family_split(recs(model, t))}
        m["base_family_split"] = family_split(recs(model, "base"))
        F["models"][model] = m
    # floor (pre-registered) and component ladders (POST-HOC)
    F["floor"], F["component_floor"] = {}, {}
    for lad in ("pythia", "tinystories"):
        ms = sorted([(m["body"], mo) for mo, m in F["models"].items() if m["ladder"] == lad])
        passing = [b for b, mo in ms if F["models"][mo]["verdict"] == "PASS"]
        F["floor"][lad] = {"floor_body": min(passing) if passing else None, "largest_tested_body": ms[-1][0],
                           "reading": (f"floor = {min(passing):,}" if passing else f"no pass: floor > {ms[-1][0]:,} body")}
        for name, _ in CELLS:
            ok = [b for b, mo in ms if sum(F["models"][mo]["seeds"][t]["cells"][name]["acc"] >= 0.8
                                           for t in F["models"][mo]["seeds"]) >= 2]
            F["component_floor"][f"{lad}|{name}"] = min(ok) if ok else None
    F["e002_reference"] = res.get("e002_reference")
    F["clopper_pearson_0of3"] = {"one_sided95_upper": round(1 - 0.05 ** (1 / 3), 3),
                                 "two_sided95_upper": round(1 - 0.025 ** (1 / 3), 3)}
    F["problems"] = problems
    json.dump(F, open(os.path.join(LOGS, "final_e003.json"), "w"), indent=1)
    open(os.path.join(LOGS, "final_e003.txt"), "w").write(report(F) + "\n")
    print(report(F))


def report(F):
    L = ["E003 final reading (code/final_e003.py; plan in notes.txt 2026-10-02)", "",
         f"INTEGRITY: {len(F['problems'])} problems" + "".join(f"\n  {p}" for p in F["problems"]),
         "  killed attempts before each job's one completed run (all re-run from scratch, same command):"]
    for job, ks in F["attempts"].items():
        L.append(f"    {job}: {len(ks)} ({', '.join(sorted(set(ks)))[:90]})")
    L += ["", "PRE-REGISTERED READING", f"{'model':28s} {'body':>11s} LR     seeds verdict  best-dev-min(LR)  fails-every-LR  "
          f"per-object  fit-diag  base-kbig  knowledge"]
    for mo, m in F["models"].items():
        L.append(f"{mo:28s} {m['body']:11,d} {m['chosen_lr']}  {m['seeds_pass']}   {m['verdict']:7s}  {m['best_dev_min']:.2f} "
                 f"({m['best_dev_lr']})      {str(m['fails_every_lr']):5s}           {m['per_object_tracking']}/3        "
                 f"{'-' if m['fitting_items_diag'] is None else str(m['fitting_items_diag']) + '/3'}       "
                 f"{m['base_kbig_acc']:.3f}      {'read' if m['knowledge_informative'] else 'uninformative (<0.6)'}")
    for lad, f in F["floor"].items():
        L.append(f"  floor, {lad} ladder: {f['reading']}")
    e2 = F.get("e002_reference") or {}
    L.append(f"  E002 reference (other family, instruct-tuned): SmolLM2-135M-Instruct body 106,203,456 "
             f"{e2.get('passing')}/{e2.get('seeds')} PASS={e2.get('pass')}")
    L += ["", "POST-HOC (descriptive, never a verdict)",
          f"  0/3 seeds: per-seed pass probability one-sided 95% upper bound {F['clopper_pearson_0of3']['one_sided95_upper']}",
          "  component ladders (smallest body where the cell alone is >= 0.8 on 2+ of 3 seeds): " +
          "; ".join(f"{k} {v:,}" if v else f"{k} none" for k, v in F["component_floor"].items()),
          "  model / seed: LW10 TS10 NU10 [Wilson 95%] failing cells; robust = a failing cell's upper bound < 0.8"]
    for mo, m in F["models"].items():
        for t, s in m["seeds"].items():
            c = s["cells"]
            L.append(f"  {SHORT[mo]:5s} {t}: " + "  ".join(f"{nm} {c[nm]['acc']:.2f} [{c[nm]['wilson95'][0]:.2f},"
                                                         f"{c[nm]['wilson95'][1]:.2f}]" for nm in c)
                     + f"  failing {','.join(s['failing']) or '-'} robust={s['fail_robust']}")
    L.append("  family split and top candidate at d10 (base, then seeds):")
    for mo, m in F["models"].items():
        for t, fs in [("base", m["base_family_split"])] + [(t, s["family_split"]) for t, s in m["seeds"].items()]:
            L.append(f"  {SHORT[mo]:5s} {t:4s}: " + "; ".join(
                f"{nm} day {fs[nm + '|day']:.2f} color {fs[nm + '|color']:.2f} top {fs[nm + '|top']}" for nm, _ in CELLS))
    return "\n".join(L)


if __name__ == "__main__":
    main()
