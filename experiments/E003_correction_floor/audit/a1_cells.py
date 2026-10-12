"""E003 audit A1: pass-rule cells, verdicts and LR picks re-derived from the raw out/ records.

Own code: own grader, own cell sums, own LR-pick rule (written from the notes.txt text, not pick_lr.py).
Item modules are imported only to REBUILD the item lists (stdlib), so every record can be matched to its item
by prompt hash, variant, distance, family, scenario id and candidate labels. No model, no torch.
usage: python3 -B a1_cells.py   (writes a1_out.json next to this file)
"""
import hashlib, json, math, os, sys

AUD = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(AUD)
OUT = os.path.join(EXP, "out")
sys.path.insert(0, os.path.join(EXP, "code"))
import items as I            # noqa: E402  (stdlib only)
import items_new as N        # noqa: E402
import eval_extra as X       # noqa: E402
assert "torch" not in sys.modules and "transformers" not in sys.modules

MODELS = [("roneneldan/TinyStories-1M", "ts1m", 64), ("EleutherAI/pythia-14m", "p14m", 128),
          ("roneneldan/TinyStories-3M", "ts3m", 128), ("EleutherAI/pythia-31m", "p31m", 256),
          ("roneneldan/TinyStories-8M", "ts8m", 256), ("EleutherAI/pythia-70m", "p70m", 512),
          ("roneneldan/TinyStories-33M", "ts33m", 768), ("EleutherAI/pythia-160m", "p160m", 768)]
LWV = ("same_k1", "same_k2", "same_k3")


def ok(sc):
    """right: gold present, >= 2 candidates, every score a finite number, gold strictly above all others."""
    if not isinstance(sc, dict) or "gold" not in sc or len(sc) < 2:
        return False
    for v in sc.values():
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            return False
    return all(sc["gold"] > v for k, v in sc.items() if k != "gold")


def h(s):
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def items_with_hash(its):
    return [dict(it, _h=h(I.transcript(it["turns"], it["question"], it["prefix"]))) for it in its]


NEW = items_with_hash(N.build())
DEV = items_with_hash([x for x in N.build(n_scen=32, distances=(10,), seed=3003)
                       if x["var"] in ("same_k1", "same_k2", "same_k3", "twoslot", "noupd")])
CROSS = items_with_hash(X.build_crossed())
ITEMS = {"new": NEW, "dev": DEV, "cross": CROSS}


def load(model, tag, set_):
    p = os.path.join(OUT, f"{model.replace('/', '__')}__{tag}__{set_}__plain.jsonl")
    if not os.path.exists(p):
        return None
    return [json.loads(l) for l in open(p) if l.strip()]


def match(recs, set_):
    """-> number of mismatches between records and the rebuilt items (order, hash, metadata, candidate labels)."""
    its = ITEMS[set_]
    bad = 0
    if len(recs) != len(its):
        return 10 ** 6
    for i, (r, it) in enumerate(zip(recs, its)):
        if (r["id"] != i or r["h"] != it["_h"] or r["var"] != it["var"] or r["d"] != it["d"] or r["fam"] != it["fam"]
                or r["sid"] != it["sid"] or set(r["scores"]) != set(it["cands"]) or r.get("render") != "plain"):
            bad += 1
    return bad


def cells(recs, fam=None):
    def c(vs):
        xs = [ok(r["scores"]) for r in recs if r["var"] in vs and r["d"] == 10 and (fam is None or r["fam"] == fam)]
        return (sum(xs), len(xs))
    return {"LW": c(LWV), "TS": c(("twoslot",)), "NU": c(("noupd",))}


def acc(kn):
    return kn[0] / kn[1] if kn[1] else None


def lrs_for(model):
    tags = sorted(p.split("__")[2] for p in os.listdir(OUT) if p.startswith(model.replace("/", "__") + "__lr")
                  and p.endswith("__run.json"))
    return [t for t in tags]


def my_pick(rows, grid, final):
    best = max(rows, key=lambda r: (r["min"], r["mean"], -r["lr"]))
    if not final and best["lr"] == max(grid) and best["min"] < 0.8:
        return ("EXTEND", {1e-3: 3e-3, 3e-3: 1e-2}[max(grid)])
    return ("CHOSEN", best["lr"])


def main():
    res = json.load(open(os.path.join(EXP, "results.json")))
    out, problems = {}, []
    for model, short, dm in MODELS:
        M = {"seeds": {}, "lr": {}}
        # baseline + seeds on the eval set
        for tag in ("base", "s1", "s2", "s3"):
            new, cross = load(model, tag, "new"), load(model, tag, "cross")
            mm = match(new, "new") + match(cross, "cross")
            if mm:
                problems.append(f"{short} {tag}: {mm} record/item mismatches")
            c = cells(new)
            a = {k: acc(v) for k, v in c.items()}
            # crossed pair at d10 (per-object tracking secondary label)
            by = {}
            for r in cross:
                if r["d"] == 10:
                    by.setdefault(r["sid"], []).append(ok(r["scores"]))
            cp = sum(1 for v in by.values() if len(v) == 2 and all(v)) / len(by)
            M["seeds"][tag] = dict(counts=c, acc=a, passes=all(x >= 0.8 for x in a.values()), cross_pair10=cp,
                                   fam={f: {k: acc(v) for k, v in cells(new, f).items()} for f in ("day", "color")})
            stored = (res["models"][model]["baseline"] if tag == "base" else res["models"][model]["seeds"][tag])["primary_plain"]
            for k, sk in (("LW", "LW10"), ("TS", "TS10"), ("NU", "NU10")):
                if abs(round(a[k], 4) - stored[sk]["acc"]) > 1e-9 or c[k][1] != stored[sk]["n"]:
                    problems.append(f"{short} {tag} {k}: mine {a[k]:.4f} stored {stored[sk]['acc']}")
        seeds = [M["seeds"][t]["passes"] for t in ("s1", "s2", "s3")]
        M["verdict"] = "PASS" if sum(seeds) > len(seeds) / 2 else "FAIL"
        M["n_pass"] = sum(seeds)
        if M["verdict"] != ("PASS" if res["models"][model]["verdict"]["pass"] else "FAIL"):
            problems.append(f"{short}: verdict differs from results.json")
        # LR search on the dev draw
        grid = [5e-5, 3e-4, 1e-3] + ([3e-3] if dm <= 128 else [])
        rows = []
        for tag in lrs_for(model):
            lr = float(tag[2:].split("_")[0])
            dev = load(model, tag, "dev")
            meta = json.load(open(os.path.join(OUT, f"{model.replace('/', '__')}__{tag}__run.json")))
            mm = match(dev, "dev")
            if mm:
                problems.append(f"{short} {tag}: {mm} dev mismatches")
            for s in ("new", "cross", "kbig"):
                if load(model, tag, s) is not None:
                    problems.append(f"{short} {tag}: LR-search run has eval set {s}")
            L = meta.get("losses") or []
            elig = (not meta.get("fatal") and len(L) == 400 and meta.get("steps") == 400 and all(math.isfinite(x) for x in L)
                    and len(dev) == 320 and meta.get("seed") == 0 and meta.get("sets") == ["dev"])
            c = cells(dev)
            a = {k: acc(v) for k, v in c.items()}
            row = dict(lr=lr, tag=tag, eligible=elig, **a, min=min(a.values()), mean=sum(a.values()) / 3,
                       loss_last50=sum(L[-50:]) / 50, loss_first10=sum(L[:10]) / 10)
            rows.append(row)
            M["lr"][f"{lr:.0e}"] = row
        g_rows = [r for r in rows if r["lr"] in grid and r["eligible"]]
        d1 = my_pick(g_rows, grid, False)
        M["pick_steps"] = [d1[0] + f" {d1[1]:.0e}"]
        extra = sorted(set(r["lr"] for r in rows) - set(grid))
        if d1[0] == "EXTEND":
            if extra != [d1[1]]:
                problems.append(f"{short}: extension runs {extra} but rule says {d1[1]:.0e}")
            d2 = my_pick([r for r in rows if r["eligible"]], grid + [d1[1]], True)
            M["pick_steps"].append(f"{d2[0]} {d2[1]:.0e}")
            chosen = d2[1]
        else:
            if extra:
                problems.append(f"{short}: extra LR runs {extra} without an EXTEND")
            chosen = d1[1]
        M["chosen"] = chosen
        for t in ("s1", "s2", "s3"):
            meta = json.load(open(os.path.join(OUT, f"{model.replace('/', '__')}__{t}__run.json")))
            if abs(meta["lr"] - chosen) > 1e-12 or meta["seed"] != int(t[1]) or meta.get("sets") != ["eval"]:
                problems.append(f"{short} {t}: lr {meta['lr']} seed {meta['seed']} sets {meta.get('sets')} vs chosen {chosen}")
        stored_pick = (res["models"][model].get("lr_search_final") or res["models"][model]["lr_search"])["chosen"]
        if float(stored_pick) != chosen:
            problems.append(f"{short}: stored chosen {stored_pick} mine {chosen:.0e}")
        M["best_dev_min"] = max(r["min"] for r in rows)
        M["any_dev_min_ge_0.8"] = any(r["min"] >= 0.8 for r in rows)
        M["n_lr_runs"] = len(rows)
        out[model] = M
    json.dump({"models": out, "problems": problems}, open(os.path.join(AUD, "a1_out.json"), "w"), indent=1)
    print(f"items rebuilt: new {len(NEW)}, dev {len(DEV)}, cross {len(CROSS)}")
    print("model    LR     LW10(s1/s2/s3)        TS10                  NU10                  pass  verdict  bestdevmin  LRruns")
    for model, short, _ in MODELS:
        M = out[model]
        f = lambda k: "/".join(f"{M['seeds'][t]['acc'][k]:.3f}" for t in ("s1", "s2", "s3"))
        mean = lambda k: sum(M['seeds'][t]['acc'][k] for t in ("s1", "s2", "s3")) / 3
        print(f"{short:6} {M['chosen']:.0e} {f('LW')} ({mean('LW'):.2f}) {f('TS')} ({mean('TS'):.2f}) {f('NU')} ({mean('NU'):.2f})"
              f"  {M['n_pass']}/3  {M['verdict']}  {M['best_dev_min']:.3f}  {M['n_lr_runs']}  {' -> '.join(M['pick_steps'])}")
    print("LR rows (dev): model lr min LW TS NU loss_first10 loss_last50 eligible")
    for model, short, _ in MODELS:
        for k, r in sorted(out[model]["lr"].items(), key=lambda kv: kv[1]["lr"]):
            print(f"  {short:6} {k} min {r['min']:.3f} LW {r['LW']:.3f} TS {r['TS']:.3f} NU {r['NU']:.3f} "
                  f"loss {r['loss_first10']:.3f}->{r['loss_last50']:.3f} {r['eligible']}")
    print(f"PROBLEMS: {len(problems)}")
    for p in problems:
        print("  " + p)


if __name__ == "__main__":
    main()
