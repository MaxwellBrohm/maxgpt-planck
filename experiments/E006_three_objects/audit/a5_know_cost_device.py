"""A5. Knowledge (K = G - C on kbig's unexposed items), the cost tables (P - C, G - C; COST / LOWER marks), and
Q-device (C - E005 Mac records, count of intervals clear of 0; scoring-device flips; base vs E004; TF32 twins),
from raw scores with my own bootstrap. Optional argv[1]: pc_exposure.out for the broad exposure sensitivity."""
import json, re, sys
from common import *

EXP = json.load(open(os.path.join(E6, "replay_ref", "exposed_kbig.json")))


def kmat(arm, set_name="kbig"):
    M = []
    for s in SEEDS:
        rs = sorted(load(tag_of(arm, s), set_name, "plain", root_of(arm)), key=lambda r: r["id"])
        M.append([int(lik_right(r)) for r in rs])
    return np.array(M, dtype=float)


def base_vec(set_name="kbig"):
    return np.array([int(lik_right(r)) for r in sorted(load("base", set_name, "plain"), key=lambda r: r["id"])])


def mcnemar_p(gained, lost):
    from math import comb
    n = gained + lost
    k = min(gained, lost)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def knowledge(extra=None):
    G, C = kmat("G"), kmat("C")
    b = base_vec()
    keep = np.array([i not in set(EXP["strict"]) for i in range(G.shape[1])])
    loose = np.array([i not in set(EXP["loose"]) for i in range(G.shape[1])])
    K = boot_pair(G[:, keep] - C[:, keep])
    lab = ("REDUCES THE COST" if K[0] >= 0.03 and K[1] > 0 else "SMALL REDUCTION" if K[1] > 0
           else "INCREASES THE COST" if K[2] < 0 else "NO EFFECT")
    print(f"K unexposed (n {keep.sum()}): {fmt(K)} -> {lab}")
    print(f"K all 441: {fmt(boot_pair(G - C))}; loose-unexposed (n {loose.sum()}): {fmt(boot_pair(G[:, loose] - C[:, loose]))}")
    print(f"accuracy all 441: base {b.mean():.3f} C {C.mean():.3f} G {G.mean():.3f} P {kmat('P').mean():.3f}"
          f" e005w {kmat('e005w').mean():.3f}")
    for s in range(5):
        g = int(((G[s] == 1) & (b == 0)).sum()); l = int(((G[s] == 0) & (b == 1)).sum())
        print(f"  G{s + 1} vs base: gained {g} lost {l} McNemar p {mcnemar_p(g, l):.4f}")
    if extra:
        txt = open(extra).read()
        for name in ("pool", "s1", "s2", "s3", "s4", "s5"):
            m = re.search(rf"^{name} missing (\d+) gold_any (\[.*?\]) co (\[.*?\])$", txt, re.M)
            if not m:
                continue
            ga, co = set(json.loads(m.group(2))), set(json.loads(m.group(3)))
            for nm, S in (("gold anywhere", ga), ("gold + entity in one thread", co)):
                k2 = np.array([i not in S for i in range(G.shape[1])])
                print(f"  broad exposure [{name}, {nm}]: {len(S)} exposed; K on the other {k2.sum()}:"
                      f" {fmt(boot_pair(G[:, k2] - C[:, k2]))}; K on exposed: {fmt(boot_pair(G[:, ~k2] - C[:, ~k2]))}"
                      f" (missing ids {m.group(1)})")


def cost():
    for arm in ("P", "G"):
        marks = []
        for set_name, fams in (("e004", FAM10), ("big", ["H5", "C_noupd", "C_twoslot"]), ("h5l", ["H5L"])):
            for render in ("plain", "chat"):
                for kd in ("LIK", "GEN"):
                    A = matrix(arm, set_name, render, kd, fams)
                    B = matrix("C", set_name, render, kd, fams)
                    for f in fams:
                        d = boot_pair(A[f] - B[f])
                        m = "COST" if d[0] <= -0.05 and d[2] < 0 else "LOWER" if d[2] < 0 else "-"
                        if m != "-" or d[0] <= -0.025:
                            marks.append(f"{set_name} {f} {kd} {render} {fmt(d)} [{m}]")
        print(f"cost {arm} - C: marks or d <= -0.025:", *marks, sep="\n   ")


def device():
    n_clear, n = 0, 0
    lines = []
    for render in ("plain", "chat"):
        for kd in ("LIK", "GEN"):
            A, B = matrix("C", "e004", render, kd, FAM10), matrix("E005", "e004", render, kd, FAM10)
            for f in FAM10:
                d = boot_pair(A[f] - B[f]); n += 1; n_clear += d[1] > 0 or d[2] < 0
                lines.append((f, kd, render, d))
    E5AL = os.path.join(E5, "al", "out")
    for render in ("plain", "chat"):
        A, B = {}, {}
        for s in SEEDS:
            for D, rs in ((A, load(f"C{s}", "al", render)),
                          (B, [json.loads(l) for l in open(os.path.join(E5AL, f"e005_s{s}__al__{render}.jsonl"))])):
                for r in sorted(rs, key=lambda r: r["id"]):
                    D.setdefault(r["cell"], {}).setdefault(s, []).append(int(lik_right(r)))
        for f in sorted(A):
            a = np.array([A[f][s] for s in SEEDS], float); b = np.array([B[f][s] for s in SEEDS], float)
            d = boot_pair(a - b); n += 1; n_clear += d[1] > 0 or d[2] < 0
            lines.append((f, "LIK", render, d))
    for st in ("new", "extra", "old", "uprobe", "cross", "kbig", "khard"):
        A = np.vstack([[lik_right(r) for r in sorted(load(f"C{s}", st, "plain"), key=lambda r: r["id"])] for s in SEEDS])
        B = np.vstack([[lik_right(r) for r in sorted(load(f"s{s}", st, "plain", root_of("E005")), key=lambda r: r["id"])]
                       for s in SEEDS])
        d = boot_pair(A.astype(float) - B); n += 1; n_clear += d[1] > 0 or d[2] < 0
        lines.append((st, "LIK", "plain", d))
    nz = [(f, kd, r, d) for f, kd, r, d in lines if abs(d[0]) > 1e-9]
    print(f"Q-device: {n} intervals (chat probe's 4 not included here), clear of 0: {n_clear};"
          f" nonzero d: {len(nz)} (positive {sum(d[0] > 0 for *_, d in nz)}); largest |d|:"
          f" {max(nz, key=lambda x: abs(x[3][0]))}")
    for s in SEEDS:   # scoring device: e005w (CUDA) vs E005 (Mac), same weights
        for render in ("plain", "chat"):
            a = {r["id"]: r for r in load(f"e005w{s}", "e004", render)}
            b = {r["id"]: r for r in load(f"s{s}", "e004", render, root_of("E005"))}
            flips = sum(lik_right(a[i]) != lik_right(b[i]) for i in a)
            mx = max(abs(a[i]["scores"][k] - b[i]["scores"][k]) for i in a for k in a[i]["scores"])
            ga = {r["id"]: r["reply"] for r in load(f"e005w{s}", "gen_e004", render)}
            gb = {r["id"]: r["reply"] for r in load(f"s{s}", "gen_e004", render, root_of("E005"))}
            print(f"  e005w{s} {render}: LIK flips {flips}/{len(a)} max|diff| {mx:.2e}; identical replies"
                  f" {sum(ga[i] == gb.get(i) for i in ga)}/{len(ga)}")
    for st, rends in (("new", ("plain", "chat")), ("extra", ("plain", "chat")), ("old", ("plain",)),
                      ("uprobe", ("plain",)), ("cross", ("plain", "chat"))):
        for render in rends:
            a = {r["h"]: r for r in load("base", st, render)}
            b = {r["h"]: r for r in load("base", st, render, root_of("E004"))}
            common_h = [h for h in a if h in b]
            flips = sum(lik_right(a[h]) != lik_right(b[h]) for h in common_h)
            print(f"  base {st}/{render}: {len(common_h)} shared prompts, flips {flips}")
    for t, s in (("C1t", 1), ("C2t", 2)):
        for f in ("H5", "C_noupd", "C_twoslot"):
            a = np.array([[lik_right(r) for r in sorted(load(t, "big", "plain"), key=lambda r: r["id"]) if r["family"] == f]], float)
            b = np.array([[lik_right(r) for r in sorted(load(f"C{s}", "big", "plain"), key=lambda r: r["id"]) if r["family"] == f]], float)
            print(f"  twin {t} - C{s} BIG {f} LIK: {fmt(boot_pair(a - b))}, items differing {int((a != b).sum())}")


if __name__ == "__main__":
    knowledge(sys.argv[1] if len(sys.argv) > 1 else None)
    cost()
    device()
