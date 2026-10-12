"""E004 learning-rate pick (notes.txt (d); rules in rules_e004.py). CPU only, loads no model.

Reads the seed-0 dev-draw runs ../out/<slug>__lr<LR>_s0__run.json and __dev__plain.jsonl (LR as the exact string
the queue passed, e.g. lr1.5e-04_s0) and prints ONE line: "CHOSEN <lr>", "EXTEND <lr>" or "NONE <reason>".
  eligible   run.json present, no 'fatal', steps == 400, 400 finite training losses, 320 dev records
             (10 families x 32) and eval_counts["dev/plain"] == 320
  score      LIK per pass family on the dev draw (metrics_ft.right: ties / NaN / missing fail); min and mean over
             the 9 pass families
usage: pick_lr_e004.py <model> --mode 135m --lrs 5e-05 1.5e-04
       pick_lr_e004.py <model> --mode grid --lrs 5e-05 3e-04 1e-03 [3e-03] [--ext 1e-02] [--final]
Writes ../logs/lr_pick_<slug>[_final].json.
"""
import argparse, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import metrics_ft as MF
import rules_e004 as RU

STEPS, N_DEV = 400, 320


def slug(m):
    return m.replace("/", "__")


def dev_accs(recs):
    out = {}
    for f in RU.PASS:
        xs = [MF.right(r["scores"]) for r in recs if r.get("family") == f]
        out[f] = round(sum(xs) / len(xs), 4) if xs else None
    return out


def load_run(out_dir, model, lr):
    """-> (row dict or None, reason for ineligibility or None)."""
    stem = os.path.join(out_dir, f"{slug(model)}__lr{lr}_s0")
    rj, dj = stem + "__run.json", stem + "__dev__plain.jsonl"
    if not os.path.exists(rj):
        return None, "no run.json"
    meta = json.load(open(rj))
    if meta.get("fatal"):
        return None, f"fatal: {meta['fatal']}"
    L = meta.get("losses") or []
    if meta.get("steps") != STEPS or len(L) != STEPS:
        return None, f"{len(L)} losses, steps={meta.get('steps')} (not {STEPS})"
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in L):
        return None, "non-finite training loss"
    if not os.path.exists(dj):
        return None, "no dev records"
    recs = [json.loads(l) for l in open(dj) if l.strip()]
    if len(recs) != N_DEV or (meta.get("eval_counts") or {}).get("dev/plain") != N_DEV:
        return None, f"{len(recs)} dev records (not {N_DEV})"
    accs = dev_accs(recs)
    row = dict(RU.dev_score(accs), fams=accs, lr=lr, lock_in_step=meta.get("lock_in_step"),
               loss_first10=round(sum(L[:10]) / 10, 4), loss_last50=round(sum(L[-50:]) / 50, 4))
    return row, None


def pick(model, mode, lrs, ext=None, final=False, out_dir=os.path.join(EXP, "out")):
    runs = list(lrs) + ([ext] if ext else [])
    rows, bad = {}, {}
    for lr in runs:
        row, why = load_run(out_dir, model, lr)
        rows[lr] = row
        if why:
            bad[lr] = why
    if mode == "135m":
        dec, val = RU.pick_135m({k: v for k, v in rows.items()})
    else:
        dec, val = RU.pick_grid(rows, list(lrs), final=final or bool(ext))
    return {"model": model, "mode": mode, "lrs": runs, "rows": rows, "ineligible": bad, "final": final or bool(ext),
            "decision": dec, "value": val, "line": f"{dec} {val}"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--mode", choices=["135m", "grid"], required=True)
    ap.add_argument("--lrs", nargs="+", required=True)
    ap.add_argument("--ext", default=None, help="the extension LR (its run is read too; the pick is then final)")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--out", default=os.path.join(EXP, "out"))
    ap.add_argument("--log", default=os.path.join(EXP, "logs"))
    a = ap.parse_args(argv)
    res = pick(a.model, a.mode, a.lrs, a.ext, a.final, a.out)
    os.makedirs(a.log, exist_ok=True)
    name = f"lr_pick_{slug(a.model)}{'_final' if res['final'] else ''}.json"
    json.dump(res, open(os.path.join(a.log, name), "w"), indent=1)
    print(res["line"])


if __name__ == "__main__":
    main()
