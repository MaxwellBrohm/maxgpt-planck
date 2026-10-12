"""E005 analysis, part b: comparisons with E004 and E002 (CPU only, no model; E002/E004 files are read, never written).
  flips_by_hash  the continuity reproduction check, fixed (audit Q6): E004 compared two runs position by position and
                 skipped a set whose record counts differed ("items differ (384 vs 184)" on cross/chat, where E002's
                 file holds the first 184 of the 384 items). Records are now matched by prompt hash; the matched
                 count, the decision flips on them and the unmatched counts on each side are all reported, and a
                 hash seen twice in one file with different scores is an error
  knowledge      per seed vs the untouched model (analyze_e004_b.paired: bootstrap CI, margin t interval), exact
                 McNemar p printed unrounded; and E005 vs E004 per item (mean correctness over seeds, paired
                 bootstrap over items, 95% CI)
  family_diff    every family, E005 minus E004, mean over the seeds each has (LIK and GEN, plain)"""
import os
import random

import metrics_ft as MF
import analyze_e004 as A
import analyze_e004_b as B
import rules_e005 as R5

CONT_SETS = B.CONT_SETS
E002_OUT = B.E002_OUT


def flips_by_hash(a, b):
    """a, b: record lists (each with "h" and "scores") -> matched, flips, only_a, only_b, max_abs_score_diff."""
    def index(rs, name):
        d = {}
        for r in rs:
            if r["h"] in d and d[r["h"]]["scores"] != r["scores"]:
                raise ValueError(f"hash {r['h']} twice in {name} with different scores")
            d[r["h"]] = r
        return d
    da, db = index(a, "a"), index(b, "b")
    m = sorted(set(da) & set(db))
    flips = sum(MF.right(da[h]["scores"]) != MF.right(db[h]["scores"]) for h in m)
    diff = max((abs(da[h]["scores"][k] - db[h]["scores"].get(k, float("inf")))
                for h in m for k in da[h]["scores"]), default=None)
    return {"n_a": len(a), "n_b": len(b), "matched": len(m), "flips": flips, "only_a": len(set(da) - set(db)),
            "only_b": len(set(db) - set(da)), "max_abs_score_diff": diff}


def repro_flips(out_dir, model, other_out, tag="base", other_tag="base"):
    """continuity LIK sets of (out_dir, tag) vs (other_out, other_tag), every set and render present in both."""
    out = {}
    for s in CONT_SETS:
        for render in ("plain", "chat"):
            a = B.recs(out_dir, model, tag, f"{s}__{render}")
            b = A.load(os.path.join(other_out, f"{A.slug(model)}__{other_tag}__{s}__{render}.jsonl")) or []
            if a and b:
                out[f"{s}/{render}"] = flips_by_hash(a, b)
    return out


def p_str(p):
    return "-" if p is None else (f"{p:.2e}" if p < 1e-3 else f"{p:.4f}")


def knowledge_lines(kn):
    L = []
    for t, k in kn.items():
        for name, p in k.items():
            if "error" in p:
                continue
            mp = B.mcnemar_p(p["gained"], p["lost"])
            L.append(f"   knowledge {t} {name}: {p['acc_before']:.3f} -> {p['acc_after']:.3f} d={p['d_acc']:+.3f} "
                     f"CI{p['d_acc_ci95']} margin {p['d_margin']:+.3f} CI{p.get('d_margin_ci95')} gained {p['gained']} "
                     f"lost {p['lost']} McNemar p={p_str(mp)}")
    return L


def item_means(out_dir, model, tags, name="kbig__plain"):
    per = {}
    for t in tags:
        for r in B.recs(out_dir, model, t, name):
            per.setdefault(r["id"], []).append(MF.right(r["scores"]))
    return {i: sum(v) / len(v) for i, v in per.items() if len(v) == len(tags)}


def e005_vs_e004_knowledge(out5, tags5, out4, tags4, model, n_boot=2000, seed=0):
    a, b = item_means(out5, model, tags5), item_means(out4, model, tags4)
    ids = sorted(set(a) & set(b))
    if not ids or not tags5 or not tags4:
        return None
    d = [a[i] - b[i] for i in ids]
    rnd = random.Random(seed)
    boots = sorted(sum(d[rnd.randrange(len(d))] for _ in d) / len(d) for _ in range(n_boot))
    return {"n_items": len(ids), "seeds_e005": len(tags5), "seeds_e004": len(tags4),
            "acc_e005": round(sum(a[i] for i in ids) / len(ids), 4), "acc_e004": round(sum(b[i] for i in ids) / len(ids), 4),
            "d": round(sum(d) / len(d), 4), "ci95": [round(boots[int(0.025 * n_boot)], 4), round(boots[int(0.975 * n_boot) - 1], 4)]}


def family_diff(cells5, cells4):
    """cells*: {tag: run_cells dict or None} -> {part: {family: (mean E005, mean E004, diff)}}."""
    out = {}
    for part, idkey in (("LIK", "ID_LIK"), ("GEN", "ID_GEN")):
        out[part] = {}
        for f in A.FAMILIES:
            def mean(cs):
                xs = [(c[idkey] if f == "ID" else c[part][f]) for c in cs.values() if c]
                xs = [x for x in xs if x is not None]
                return sum(xs) / len(xs) if xs else None
            m5, m4 = mean(cells5), mean(cells4)
            out[part][f] = (m5, m4, None if m5 is None or m4 is None else round(m5 - m4, 4))
    return out


def family_diff_lines(fd):
    L = []
    for part, d in fd.items():
        L.append(f"   E005 - E004 {part} (mean over seeds): " + " ".join(
            f"{f}={'-' if v[2] is None else f'{v[2]:+.3f}'}" for f, v in d.items()))
    return L


def flips_lines(fl, label):
    return [f"   {label} {k}: records {v['n_a']} vs {v['n_b']}, distinct hashes matched {v['matched']}, flips "
            f"{v['flips']}, only here {v['only_a']}, only there {v['only_b']}, max |score diff| "
            f"{R5.f3(v['max_abs_score_diff'])}" for k, v in fl.items()]
