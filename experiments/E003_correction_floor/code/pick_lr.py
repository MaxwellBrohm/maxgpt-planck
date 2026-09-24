"""E003 learning-rate pick (pre-registered in notes.txt; CPU only, loads no model).

For one model, reads the seed-0 LR-search runs (tag lr<LR>_s0) and applies the rule:
  eligible  run finished (run.json, no 'fatal'), all training losses finite, 400 steps, 320 dev items scored
  dev cells LW = same_k1+k2+k3 at d10 (n=192), TS = twoslot d10 (n=64), NU = noupd d10 (n=64), plain render,
            on the dev draw (items_new seed 3003), graded by metrics_ft.right (ties/NaN/missing fail)
  score     min(LW, TS, NU); ties broken by the higher mean(LW, TS, NU), then by the SMALLER learning rate
  extension (at most once) if the pick is the LARGEST grid LR and its min < 0.8, run the next half-decade LR
            (1e-03 -> 3e-03, 3e-03 -> 1e-02) and pick again
            over all runs (no further extension)
  none      no eligible run -> NONE (the model gets no scored seeds; reported as "no usable LR")
Prints one line: "CHOSEN <lr>", "EXTEND <lr>" or "NONE <reason>", and writes ../logs/lr_pick_<slug>.json.
usage: pick_lr.py <model> --grid 5e-05 3e-04 1e-03 [--final]
"""
import argparse, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import metrics_ft as MF

STEPS = 400
N_DEV = 320
THRESH = 0.8
NEXT_LR = {"1e-03": 3e-3, "3e-03": 1e-2}   # the next half-decade step above the top of a grid


def lrtag(lr):
    return f"lr{float(lr):.0e}_s0"


def slug(m):
    return m.replace("/", "__")


def load_run(out_dir, model, lr):
    stem = os.path.join(out_dir, f"{slug(model)}__{lrtag(lr)}")
    rj, dj = stem + "__run.json", stem + "__dev__plain.jsonl"
    if not os.path.exists(rj):
        return None, "no run.json"
    meta = json.load(open(rj))
    if meta.get("fatal"):
        return None, f"fatal: {meta['fatal']}"
    L = meta.get("losses") or []
    if len(L) != STEPS or meta.get("steps") != STEPS:
        return None, f"{len(L)} losses (not {STEPS})"
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in L):
        return None, "non-finite training loss"
    if not os.path.exists(dj):
        return None, "no dev records"
    recs = [json.loads(l) for l in open(dj) if l.strip()]
    if len(recs) != N_DEV or (meta.get("eval_counts") or {}).get("dev/plain") != N_DEV:
        return None, f"{len(recs)} dev records (not {N_DEV})"
    return (meta, recs), None


def dev_cells(recs):
    lw = MF.cell(recs, "plain", MF.LW_VARS, 10)
    ts = MF.cell(recs, "plain", ("twoslot",), 10)
    nu = MF.cell(recs, "plain", ("noupd",), 10)
    accs = [c["acc"] if c else 0.0 for c in (lw, ts, nu)]
    return {"LW": accs[0], "TS": accs[1], "NU": accs[2], "n": [c["n"] if c else 0 for c in (lw, ts, nu)],
            "min": min(accs), "mean": round(sum(accs) / 3, 4)}


def key(r):
    return (r["min"], r["mean"], -r["lr"])


def pick(model, grid, out_dir, final=False):
    grid = sorted(float(x) for x in grid)
    rows, bad = [], {}
    for lr in grid:
        run, why = load_run(out_dir, model, lr)
        if run is None:
            bad[f"{lr:.0e}"] = why
            continue
        meta, recs = run
        c = dev_cells(recs)
        L = meta["losses"]
        rows.append(dict(lr=lr, **c, loss_first10=round(sum(L[:10]) / 10, 4), loss_last50=round(sum(L[-50:]) / 50, 4),
                         lock_in_step=meta.get("lock_in_step")))
    res = {"model": model, "grid": [f"{x:.0e}" for x in grid], "rows": rows, "ineligible": bad, "final": final}
    if not rows:
        res["decision"] = "NONE"
        res["line"] = "NONE no eligible LR-search run"
        return res
    best = max(rows, key=key)
    top = max(grid)
    if (not final) and best["lr"] == top and best["min"] < THRESH:
        ext = NEXT_LR.get(f"{top:.0e}", float(f"{3 * top:.0e}"))
        res.update(decision="EXTEND", extend=f"{ext:.0e}", line=f"EXTEND {ext:.0e}")
        return res
    res.update(decision="CHOSEN", chosen=f"{best['lr']:.0e}", line=f"CHOSEN {best['lr']:.0e}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--grid", nargs="+", required=True)
    ap.add_argument("--final", action="store_true", help="after an extension: pick, never extend again")
    ap.add_argument("--out", default=os.path.join(EXP, "out"))
    ap.add_argument("--log", default=os.path.join(EXP, "logs"))
    a = ap.parse_args()
    res = pick(a.model, a.grid, a.out, a.final)
    os.makedirs(a.log, exist_ok=True)
    json.dump(res, open(os.path.join(a.log, f"lr_pick_{slug(a.model)}{'_final' if a.final else ''}.json"), "w"), indent=1)
    print(res["line"])


if __name__ == "__main__":
    main()
