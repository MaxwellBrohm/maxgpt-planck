from common import *
import hashlib
from collections import defaultdict


def rt(sc):
    vals = list(sc.values())
    if len(vals) < 2 or any(v is None or math.isnan(v) or math.isinf(v) for v in vals):
        return False
    return all(sc["gold"] > v for k, v in sc.items() if k != "gold")


def mg(sc):
    return sc["gold"] - max(v for k, v in sc.items() if k != "gold")


def sets_of(exp, tag):
    K = recs(exp, tag, "kbig")
    O = recs(exp, tag, "old")
    H = recs(exp, tag, "khard")
    s = {
        "kbig441": [(r["h"], rt(r["scores"]), mg(r["scores"]), r.get("right")) for r in K],
        "khard40": [(r["h"], rt(r["scores"]), mg(r["scores"]), r.get("right")) for r in K if r["cat"] == "khard40"],
        "khard40_via_khard": [(r["h"], rt(r["scores"]), mg(r["scores"]), r.get("right")) for r in H if r["task"] == "K4_closed"],
        "K_closedbook60": [(r.get("h", r["id"]), rt(r["scores"]), mg(r["scores"]), r.get("right")) for r in O if r["task"] == "K_closedbook"],
    }
    return s


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


print("base record identity E005 copy vs E004 original:")
for k in ["kbig", "khard", "old"]:
    a = f"{E5}/out/{SLUG}__base__{k}__plain.jsonl"
    b = f"{E4}/out/{SLUG}__base__{k}__plain.jsonl"
    print(f"  {k}: {sha(a)} vs {sha(b)} equal={sha(a) == sha(b)}")

base = sets_of("E005", "base")
for name in ["kbig441", "khard40", "khard40_via_khard", "K_closedbook60"]:
    b = base[name]
    n = len(b)
    stored_mis = sum(x[3] is not None and x[1] != x[3] for x in b)
    print(f"\n== {name} (n={n}); base acc {sum(x[1] for x in b) / n:.4f} (my 'right' vs stored 'right' mismatches: {stored_mis})")
    mean_d = {}
    per_item = {}
    for exp in ["E004", "E005"]:
        ds = []
        per_item[exp] = [0.0] * n
        for t in SEEDS:
            a = sets_of(exp, t)[name]
            assert [x[0] for x in a] == [x[0] for x in b], "item order/hash mismatch"
            mis = sum(x[3] is not None and x[1] != x[3] for x in a)
            diffs = [int(y[1]) - int(x[1]) for x, y in zip(b, a)]
            gained = sum(d == 1 for d in diffs)
            lost = sum(d == -1 for d in diffs)
            acc = sum(y[1] for y in a) / n
            lo, hi = boot_ci(diffs, B=4000, seed=hash(name + exp + t) % 10**6 if False else 1000 + SEEDS.index(t))
            p = binom_two_sided(gained, lost)
            dm = [y[2] - x[2] for x, y in zip(b, a)]
            ds.append(acc - sum(x[1] for x in b) / n)
            for i, y in enumerate(a):
                per_item[exp][i] += y[1] / 5
            print(f"  {exp} {t}: acc {acc:.4f} d {acc - sum(x[1] for x in b) / n:+.4f} CI [{lo:+.4f}, {hi:+.4f}] "
                  f"gained {gained} lost {lost} McNemar p {p:.3g}; mean margin change {sum(dm) / n:+.3f}; stored-right mismatches {mis}")
        mean_d[exp] = sum(ds) / 5
        print(f"  {exp} mean change over seeds {mean_d[exp]:+.4f} (range {min(ds):+.4f} to {max(ds):+.4f})")
    d_items = [e5 - e4 for e5, e4 in zip(per_item["E005"], per_item["E004"])]
    lo, hi = boot_ci(d_items, B=4000, seed=777)
    m5 = sum(per_item["E005"]) / n
    m4 = sum(per_item["E004"]) / n
    print(f"  E005 vs E004 (per-item mean over seeds, paired bootstrap over items): {m5:.4f} vs {m4:.4f} d {m5 - m4:+.4f} CI [{lo:+.4f}, {hi:+.4f}]")
    # same-seed pairs (E005 sN vs E004 sN) exact McNemar on pooled discordant
    g = l = 0
    for t in SEEDS:
        a5 = sets_of("E005", t)[name]
        a4 = sets_of("E004", t)[name]
        g += sum(y[1] and not x[1] for x, y in zip(a4, a5))
        l += sum(x[1] and not y[1] for x, y in zip(a4, a5))
    print(f"  same-seed pairs pooled: E005 right & E004 wrong {g}, reverse {l}")
