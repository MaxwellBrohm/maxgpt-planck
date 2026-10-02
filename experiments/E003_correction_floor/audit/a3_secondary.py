"""E003 audit A3: secondary labels, knowledge change, lock-in and the post-hoc numbers, from raw records (own code).

  per-object tracking   pass AND crossed pair at d10 >= 0.8 (uses a1_out.json, which a1_cells.py computed)
  fitting-items         TinyStories only: the three pass-rule cells on items with seq_len <= 512, pass rule on them
  knowledge             kbig441 paired change vs the untouched model, own 10,000-resample bootstrap (seed 1)
  lock-in               first in-training probe step from which LW, TS and NU all stay >= 0.8 (run.json probe)
  post-hoc              Wilson 95% bounds; Clopper-Pearson bound for 0/3; family split; noupd picks on p160m s2
usage: python3 -B a3_secondary.py   (run a1_cells.py first)
"""
import json, math, os, random, sys

AUD = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(AUD)
OUT = os.path.join(EXP, "out")
sys.path.insert(0, AUD)
from a1_cells import ok, MODELS, LWV  # noqa: E402
A1 = json.load(open(os.path.join(AUD, "a1_out.json")))["models"]


def load(model, tag, set_):
    p = os.path.join(OUT, f"{model.replace('/', '__')}__{tag}__{set_}__plain.jsonl")
    return [json.loads(l) for l in open(p) if l.strip()]


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - hw, c + hw


def boot(diffs, n=10000, seed=1):
    rng = random.Random(seed)
    m = len(diffs)
    bs = sorted(sum(diffs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return bs[int(0.025 * n)], bs[int(0.975 * n) - 1]


def main():
    print(f"Clopper-Pearson one-sided 95% upper bound after 0/3: {1 - 0.05 ** (1 / 3):.3f}")
    res = json.load(open(os.path.join(EXP, "results.json")))["models"]
    for model, short, _ in MODELS:
        M = A1[model]
        print(f"\n== {short}")
        # per-object tracking secondary label
        pc = [M["seeds"][t]["passes"] and M["seeds"][t]["cross_pair10"] >= 0.8 for t in ("s1", "s2", "s3")]
        print(f"  per-object tracking: {sum(pc)}/3; crossed pair d10 by seed "
              + " ".join(f"{M['seeds'][t]['cross_pair10']:.3f}" for t in ("s1", "s2", "s3"))
              + f"; base {M['seeds']['base']['cross_pair10']:.3f}")
        # Wilson: a failing cell whose upper bound < 0.8
        for t in ("s1", "s2", "s3"):
            c = M["seeds"][t]["counts"]
            ub = {k: wilson(*c[k])[1] for k in c}
            robust = [k for k in c if c[k][0] / c[k][1] < 0.8 and ub[k] < 0.8]
            print(f"  {t}: " + " ".join(f"{k} {c[k][0]}/{c[k][1]} ub {ub[k]:.3f}" for k in c) + f"  robust-failing {robust}")
        # family split
        for t in ("base", "s1", "s2", "s3"):
            f = M["seeds"][t]["fam"]
            print(f"  fam {t:4}: day LW {f['day']['LW']:.2f} TS {f['day']['TS']:.2f} NU {f['day']['NU']:.2f} | "
                  f"color LW {f['color']['LW']:.2f} TS {f['color']['TS']:.2f} NU {f['color']['NU']:.2f}")
        # fitting-items diagnostic (TinyStories)
        if short.startswith("ts"):
            fits = []
            for t in ("s1", "s2", "s3"):
                new = [r for r in load(model, t, "new") if r["d"] == 10 and r["seq_len"] <= 512]
                cc = {}
                for k, vs in (("LW", LWV), ("TS", ("twoslot",)), ("NU", ("noupd",))):
                    xs = [ok(r["scores"]) for r in new if r["var"] in vs]
                    cc[k] = (sum(xs), len(xs))
                fits.append(all(a / n >= 0.8 for a, n in cc.values()))
                print(f"  fits<=512 {t}: " + " ".join(f"{k} {a}/{n}={a / n:.2f}" for k, (a, n) in cc.items()))
            print(f"  fitting-items pass: {sum(fits)}/3")
        # knowledge: kbig paired change vs base
        kb_b = {r["id"]: r for r in load(model, "base", "kbig")}
        base_acc = sum(ok(r["scores"]) for r in kb_b.values()) / len(kb_b)
        line = []
        for t in ("s1", "s2", "s3"):
            kb_a = {r["id"]: r for r in load(model, t, "kbig")}
            assert set(kb_a) == set(kb_b) and all(kb_a[i]["h"] == kb_b[i]["h"] for i in kb_a)
            d = [int(ok(kb_a[i]["scores"])) - int(ok(kb_b[i]["scores"])) for i in sorted(kb_b)]
            lo, hi = boot(d)
            st = res[model]["seeds"][t]["knowledge"]["kbig_all"]
            line.append(f"{t} {100 * sum(d) / len(d):+.1f} [{100 * lo:+.1f},{100 * hi:+.1f}] excl0={hi < 0 or lo > 0} "
                        f"(stored {100 * st['d_acc']:+.1f} [{100 * st['d_acc_ci95'][0]:+.1f},{100 * st['d_acc_ci95'][1]:+.1f}])")
        print(f"  kbig base acc {base_acc:.3f} ({'read' if base_acc >= 0.6 else 'uninformative'}): " + "; ".join(line))
        # lock-in from the in-training probe
        li = []
        for t in ("s1", "s2", "s3"):
            pr = json.load(open(os.path.join(OUT, f"{model.replace('/', '__')}__{t}__run.json")))["probe"]
            step = None
            for i, p in enumerate(pr):
                if all(all(q[k] is not None and q[k] >= 0.8 for k in ("LW", "TS", "NU")) for q in pr[i:]):
                    step = p["step"]
                    break
            li.append(step)
            last = pr[-1]
        print(f"  lock-in {li}; probe steps {[p['step'] for p in pr][:3]}..{pr[-1]['step']} ({len(pr)} points);"
              f" last probe {t}: {last}")
    # pythia-160m s2: what does it pick on noupd d10?
    recs = [r for r in load("EleutherAI/pythia-160m", "s2", "new") if r["var"] == "noupd" and r["d"] == 10]
    top = sum(1 for r in recs if max(r["scores"], key=r["scores"].get) == "later")
    print(f"\npythia-160m s2 noupd d10: 'later' (the other object's later value) on top in {top}/{len(recs)}")
    # pythia 31/70/160 colour items: candidate on top in the failing cells
    for model in ("EleutherAI/pythia-31m", "EleutherAI/pythia-70m", "EleutherAI/pythia-160m"):
        for t in ("s1", "s2", "s3"):
            rs = [r for r in load(model, t, "new") if r["d"] == 10 and r["fam"] == "color" and r["var"] in ("twoslot", "noupd")]
            wrong = [max(r["scores"], key=r["scores"].get) for r in rs if not ok(r["scores"])]
            later = sum(1 for w in wrong if w in ("other", "later"))
            print(f"  {model.split('-')[-1]} {t} colour TS+NU d10: wrong {len(wrong)}/{len(rs)}, other object's later value on top {later}")


if __name__ == "__main__":
    main()
