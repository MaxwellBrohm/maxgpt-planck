"""E004 analysis (notes.txt (c)-(e)); CPU only, loads no model. Reads ../out and ../logs, applies rules_e004.py.
  default      writes ../results.json and ../logs/tables.txt (whatever exists so far; missing runs are listed)
  --gate M     prints the reading label of model M only (PASS, PARTIAL-G, PARTIAL-X, FAIL, NOT_SCORED); the
               queue uses it to decide whether the tiny ladder runs seeds (a SmolLM2 FAIL: LR search only)
  --gate-base M  prints UNTOUCHED_PASS, UNTOUCHED_FAIL or UNTOUCHED_MISSING (an untouched pass needs no fine-tune)
A scored run counts only when its e004 LIK (plain) and GEN (plain) files both hold 640 records (10 families x 64).
A planned seed without a complete run counts as a failing seed (majority over the PLANNED seeds).
Collateral, continuity, chat probe and length splits: analyze_e004_b.py."""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import metrics_ft as MF
import rules_e004 as RU

FAMILIES = RU.PASS + ["ID"]
N_EVAL = 640
# (model id, short name, planned scored seeds); tiny ladder in the pre-registered order (largest body first)
MODELS = [("HuggingFaceTB/SmolLM2-135M-Instruct", "135m", 5),
          ("roneneldan/TinyStories-8M", "ts8m", 3), ("EleutherAI/pythia-31m", "p31m", 3),
          ("roneneldan/TinyStories-3M", "ts3m", 3), ("EleutherAI/pythia-14m", "p14m", 3),
          ("roneneldan/TinyStories-1M", "ts1m", 3)]


def slug(m):
    return m.replace("/", "__")


def load(path):
    if not os.path.exists(path):
        return None
    return [json.loads(l) for l in open(path) if l.strip()]


def load_json(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def acc(xs):
    return round(sum(xs) / len(xs), 4) if xs else None


def by_family(recs, fn):
    return {f: acc([fn(r) for r in recs if r.get("family") == f]) for f in FAMILIES}


def run_cells(out_dir, model, tag, render="plain"):
    """-> cells dict for one scored run, or None when its e004 LIK or GEN file is missing or incomplete."""
    stem = os.path.join(out_dir, f"{slug(model)}__{tag}")
    lik, gen = load(f"{stem}__e004__{render}.jsonl"), load(f"{stem}__gen_e004__{render}.jsonl")
    if lik is None or gen is None or len(lik) != N_EVAL or len(gen) != N_EVAL:
        return None
    L = by_family(lik, lambda r: MF.right(r["scores"]))
    G = by_family(gen, lambda r: bool(r["strict"]))
    chance = by_family(lik, lambda r: 1.0 / len(r["scores"]) if r.get("scores") else 0.0)
    ref = {}
    for f in ("H1", "H2"):
        for part, recs, fn in (("LIK", lik, lambda r: MF.right(r["scores"])), ("GEN", gen, lambda r: bool(r["strict"]))):
            forms = sorted({r.get("latest_ref") for r in recs if r.get("family") == f} - {None})
            ref[f"{f} {part}"] = {x: acc([fn(r) for r in recs if r.get("family") == f and r.get("latest_ref") == x])
                                  for x in forms}
    return {"LIK": {f: L[f] for f in RU.PASS}, "GEN": {f: G[f] for f in RU.PASS}, "ID_LIK": L["ID"],
            "ID_GEN": G["ID"], "chance": chance, "lenient": by_family(gen, lambda r: bool(r["lenient"])),
            "ref_split": ref, "failing": RU.failing({"LIK": L, "GEN": G}),
            "below_chance": [f for f in FAMILIES if RU.below_chance(L[f], chance[f])]}


def run_meta(out_dir, model, tag):
    m = load_json(os.path.join(out_dir, f"{slug(model)}__{tag}__run.json")) or {}
    ts = m.get("train_stats") or {}
    return {k: v for k, v in dict(lr=m.get("lr"), steps=m.get("steps"), lock_in_step=m.get("lock_in_step"),
                                  loss_finite=m.get("loss_finite"), max_share_shift=ts.get("max_share_shift"),
                                  rejected_long=ts.get("rejected_long"), total_s=m.get("total_s"),
                                  s_per_step=m.get("s_per_step"), fatal=m.get("fatal")).items() if v is not None}


def model_block(model, n_planned, out_dir, logs_dir):
    base = run_cells(out_dir, model, "base")
    seeds = {f"s{s}": run_cells(out_dir, model, f"s{s}") for s in range(1, n_planned + 1)}
    blk = {"model": model, "n_planned": n_planned, "base": base, "seeds": seeds,
           "untouched_pass": RU.untouched_pass(base) if base else None,
           "meta": {t: run_meta(out_dir, model, t) for t in ["base"] + list(seeds)},
           "lr_pick": load_json(os.path.join(logs_dir, f"lr_pick_{slug(model)}_final.json"))
           or load_json(os.path.join(logs_dir, f"lr_pick_{slug(model)}.json")),
           "missing": [t for t, c in [("base", base)] + list(seeds.items()) if c is None]}
    if all(c is None for c in seeds.values()):
        blk["reading"] = {"label": "NOT_SCORED", "detail": "no complete scored seed run", "sub": []}
    else:
        blk["reading"] = RU.reading(list(seeds.values()), n_planned,
                                    [c["ID_LIK"] if c else None for c in seeds.values()], model)
    blk["seed_pass"] = {t: RU.seed_pass(c) for t, c in seeds.items()}
    blk["below_chance_flags"] = {t: c["below_chance"] for t, c in seeds.items() if c and c["below_chance"]}
    return blk


def f2(x):
    return "-" if x is None else f"{x:.2f}"


def table_rule(blk):
    lines = [f"== {blk['model']}: reading {blk['reading']['label']} ({blk['reading'].get('detail', '')})"]
    for s in blk["reading"].get("sub", []):
        lines.append(f"   sub-reading: {s}")
    hdr = "   run   part " + " ".join(f"{f[:8]:>8}" for f in RU.PASS) + "       ID  pass"
    lines.append(hdr)
    for tag, c in [("base", blk["base"])] + list(blk["seeds"].items()):
        if c is None:
            lines.append(f"   {tag:<5} (missing or incomplete)")
            continue
        for part in ("LIK", "GEN"):
            row = " ".join(f"{f2(c[part][f]):>8}" for f in RU.PASS)
            idv = c["ID_LIK"] if part == "LIK" else c["ID_GEN"]
            verdict = ("PASS" if RU.seed_pass(c) else "fail") if part == "LIK" else ""
            lines.append(f"   {tag:<5} {part:<4} {row} {f2(idv):>8}  {verdict}")
        if c["below_chance"]:
            lines.append(f"   {tag:<5} BELOW-CHANCE (LIK): {', '.join(c['below_chance'])}")
    base = blk["base"]
    if base:
        lines.append("   chance    " + " ".join(f"{f2(base['chance'][f]):>8}" for f in RU.PASS))
    lines.append(f"   untouched model passes (needs no fine-tune): {blk['untouched_pass']}")
    for t, m in blk["meta"].items():
        if m:
            lines.append(f"   meta {t}: " + ", ".join(f"{k}={v}" for k, v in m.items()))
    if blk["lr_pick"]:
        p = blk["lr_pick"]
        rows = ", ".join(f"{lr}: min {r['min']:.2f} mean {r['mean']:.2f}" if r else f"{lr}: ineligible"
                         for lr, r in p["rows"].items())
        lines.append(f"   LR pick: {p['line']} ({rows})")
    for tag, c in blk["seeds"].items():
        if c:
            lines.append(f"   {tag} H1/H2 by reference form: " + "; ".join(
                f"{k} " + " ".join(f"{x}={f2(v)}" for x, v in d.items()) for k, d in c["ref_split"].items()))
    return lines


def analyze(out_dir, logs_dir, models=MODELS):
    import analyze_e004_b as B
    res, lines = {"models": {}}, ["E004 tables (analyze_e004.py; pass rule per notes.txt (c))", ""]
    for model, short, n in models:
        blk = model_block(model, n, out_dir, logs_dir)
        blk["collateral"] = B.collateral(model, n, out_dir)
        res["models"][short] = blk
        lines += table_rule(blk) + B.table_collateral(blk["collateral"]) + [""]
    return res, lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate")
    ap.add_argument("--gate-base")
    ap.add_argument("--out-dir", default=os.path.join(EXP, "out"))
    ap.add_argument("--logs-dir", default=os.path.join(EXP, "logs"))
    ap.add_argument("--results", default=os.path.join(EXP, "results.json"))
    ap.add_argument("--tables", default=os.path.join(EXP, "logs", "tables.txt"))
    a = ap.parse_args(argv)
    if a.gate or a.gate_base:
        m = a.gate or a.gate_base
        n = dict((x, k) for x, _, k in MODELS).get(m, 3)
        if a.gate:
            print(model_block(m, n, a.out_dir, a.logs_dir)["reading"]["label"])
        else:
            c = run_cells(a.out_dir, m, "base")
            print("UNTOUCHED_MISSING" if c is None else ("UNTOUCHED_PASS" if RU.untouched_pass(c) else "UNTOUCHED_FAIL"))
        return
    res, lines = analyze(a.out_dir, a.logs_dir)
    json.dump(res, open(a.results, "w"), indent=1)
    open(a.tables, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
