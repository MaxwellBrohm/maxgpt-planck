"""A3. Q-H5 (P vs C) from raw records: BIG H5 plain LIK (recomputed from scores) and GEN (stored strict), my own
paired bootstrap, the label by the notes' text, L4 - C, the device shift, per seed, H5L, the mechanism checks
(introduction order, after_ind, NL split, reweighting, wrong picks that are the dialogue's last statement)."""
import json, re
from common import *

code_path()
import items_e004 as I

BIG = [json.loads(l) for l in open(os.path.join(E6, "big", "big_items.jsonl"))]
ITEMS = {"big": {f: [x for x in BIG if x["family"] == f] for f in ("H5", "C_noupd", "C_twoslot")},
         "h5l": {"H5L": [x for x in BIG if x["family"] == "H5L"]}, "e004": I.draw("eval")}


def intro_order(it):
    seen = []
    for st in it["stmts"]:
        if st["obj"] not in seen:
            seen.append(st["obj"])
    return seen.index(it["asked"]) + 1


def nl_value(it):
    last = it["stmts"][-1]
    for st in reversed(it["stmts"]):
        if st["obj"] != last["obj"]:
            return st["value"]


def pick(r):
    return r["cand_vals"][max(r["scores"], key=lambda k: r["scores"][k])]


def gen_pick(reply, it):
    hits = [(m.start(), v) for v in it["values"] for m in [re.search(r"(?<![A-Za-z])" + re.escape(v) +
            r"(?![A-Za-z])", reply, re.I)] if m]
    return min(hits)[1] if hits else None


def label(dL, dG, pl4L, pl4G, dev):
    both = lambda f: f(dL) and f(dG)
    gain = both(lambda d: d[0] >= 0.06 and d[1] > 0)
    if gain and (dev is None or abs(dev) >= 0.03):
        return "GAIN (RESTORED VS PARTLY NOT READ)"
    if gain and pl4L[2] > 0 and pl4G[2] > 0:
        return "RESTORED"
    if gain:
        return "PARTLY RESTORED"
    if both(lambda d: d[2] < 0):
        return "WORSE"
    if both(lambda d: d[2] < 0.06):
        return "NOT THE CAUSE"
    return "INCONCLUSIVE"


if __name__ == "__main__":
    M = {(a, kd): matrix(a, "big", "plain", kd, ["H5"])["H5"] for a in ("C", "P", "e004w", "e005w")
         for kd in ("LIK", "GEN")}
    H = {(a, kd): matrix(a, "h5l", "plain", kd, ["H5L"])["H5L"] for a in ("C", "P") for kd in ("LIK", "GEN")}
    d = {kd: boot_pair(M[("P", kd)] - M[("C", kd)]) for kd in ("LIK", "GEN")}
    pl4 = {kd: boot_pair(M[("P", kd)] - M[("e004w", kd)]) for kd in ("LIK", "GEN")}
    h5l = {kd: boot_pair(H[("P", kd)] - H[("C", kd)]) for kd in ("LIK", "GEN")}
    dev = M[("C", "LIK")].mean() - M[("e005w", "LIK")].mean()
    l4c = M[("e004w", "LIK")].mean() - M[("C", "LIK")].mean()
    for kd in ("LIK", "GEN"):
        print(f"P - C BIG H5 {kd}: {fmt(d[kd])} | P - L4 {fmt(pl4[kd])} | H5L P - C {fmt(h5l[kd])}")
    print(f"L4 - C {l4c:.3f} (flag if < 0.05: {l4c < 0.05}); dev C - e005w {dev:+.4f}")
    print("LABEL:", label(d["LIK"], d["GEN"], pl4["LIK"], pl4["GEN"], dev))
    for a in ("C", "P", "e004w", "e005w"):
        print(f"pooled {a}: LIK {M[(a, 'LIK')].mean():.3f} GEN {M[(a, 'GEN')].mean():.3f} per seed LIK"
              f" {[round(float(x), 3) for x in M[(a, 'LIK')].mean(axis=1)]}")
    for kd in ("LIK", "GEN"):
        ps = M[("P", kd)].mean(axis=1) - M[("C", kd)].mean(axis=1)
        print(f"per seed P - C {kd}: {[f'{x:+.3f}' for x in ps]} sd {ps.std(ddof=1):.3f}")
    # mechanism checks on BIG H5 (LIK), from the item file
    items = ITEMS["big"]["H5"]
    io = np.array([intro_order(x) for x in items])
    ai = np.array([x["meta"]["after_ind"] for x in items])
    nl = np.array([nl_value(x) == x["gold"] for x in items])
    print("BIG H5 intro order counts", {k: int((io == k).sum()) for k in (1, 2, 3)},
          "gold is last statement:", sum(x["stmts"][-1]["value"] == x["gold"] for x in items))
    for a in ("C", "P", "e004w", "e005w"):
        m = M[(a, "LIK")]
        by = {k: m[:, io == k].mean() for k in (1, 2, 3)}
        rew = sum(by[k] * w for k, w in ((1, 27), (2, 20), (3, 17))) / 64
        recs = [load(tag_of(a, s), "big", "plain") for s in SEEDS]
        wrong = last = 0
        for rs in recs:
            for r in rs:
                if r["family"] != "H5" or lik_right(r):
                    continue
                it = items[r["idx"]]
                wrong += 1
                last += pick(r) == it["stmts"][-1]["value"]
        grecs = [load(tag_of(a, s), "gen_big", "plain") for s in SEEDS]
        gw = glast = 0
        for rs in grecs:
            for r in rs:
                if r["family"] != "H5" or r["strict"]:
                    continue
                gw += 1
                glast += gen_pick(r["reply"], items[r["idx"]]) == items[r["idx"]]["stmts"][-1]["value"]
        print(f"{a}: intro {by[1]:.3f}/{by[2]:.3f}/{by[3]:.3f} reweighted {rew:.3f} after_ind F/T"
              f" {m[:, ~ai].mean():.3f}/{m[:, ai].mean():.3f} NL right/wrong {m[:, nl].mean():.3f}/"
              f"{m[:, ~nl].mean():.3f} | LIK wrong {wrong}, = last statement {last} | GEN wrong {gw},"
              f" first value named = last statement {glast}")
    for kd in ("LIK", "GEN"):
        for k in (1, 2, 3):
            print(f"  P - C {kd} intro {k}: {fmt(boot_pair((M[('P', kd)] - M[('C', kd)])[:, io == k]))}")
    # D4004 H5
    d4 = ITEMS["e004"]["H5"]
    io4 = np.array([intro_order(x) for x in d4])
    for a in ("C", "P", "e004w", "e005w", "E005", "E004"):
        m = matrix(a, "e004", "plain", "LIK", ["H5"])["H5"]
        print(f"D4004 H5 LIK {a}: {m.mean():.3f} intro {[round(float(m[:, io4 == k].mean()), 3) for k in (1, 2, 3)]}")
    for fam in ("C_noupd", "C_twoslot"):
        for kd in ("LIK", "GEN"):
            a = matrix("P", "big", "plain", kd, [fam])[fam]
            b = matrix("C", "big", "plain", kd, [fam])[fam]
            print(f"control BIG {fam} {kd} P - C: {fmt(boot_pair(a - b))}")
