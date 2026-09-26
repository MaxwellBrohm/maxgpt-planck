import json, itertools
C = json.load(open("q1_cells.json"))
PASSF = ["H1","H2","H3","H4","H5","H6","H7","C_noupd","C_twoslot"]
XAX = ["H3","H4","H5","H6","H7"]
N = 64
def ok(k): return k / N >= 0.8   # k >= 52
def fails(tag):
    c = C[tag]; return [(f, p) for f in PASSF for p, j in (("LIK", 0), ("GEN", 2)) if not ok(c[f][j])]
def fam_fail(tag): return sorted({f for f, _ in fails(tag)}, key=PASSF.index)
def label(seeds, planned=5):
    done = [s for s in seeds]  # seeds with complete results; missing planned seeds count as failing
    maj = lambda n: 2 * n > planned
    npass = sum(1 for s in done if not fails(s))
    if maj(npass): return "PASS", npass
    nlik = sum(1 for s in done if all(ok(C[s][f][0]) for f in PASSF))
    if maj(nlik): return "PARTIAL-G", nlik
    for size in (1, 2):
        best = None
        for X in itertools.combinations(XAX, size):
            n = sum(1 for s in done if set(fam_fail(s)) <= set(X))
            if maj(n) and (best is None or n > best[1]): best = (X, n)
        if best: return "PARTIAL-X " + "+".join(best[0]), best[1]
    return "FAIL", None
for s in ["base","s1","s2","s3","s4","s5"]:
    print(s, "failing cells:", ", ".join(f"{f} {p} {C[s][f][0 if p=='LIK' else 2]}/64={C[s][f][0 if p=='LIK' else 2]/64:.3f}" for f, p in fails(s)) if s != "base" else f"{len(fails(s))} of 18")
    if s != "base":
        thin = [f"{f} {p} {C[s][f][j]}/64" for f in PASSF + ["ID"] for p, j in (("LIK",0),("GEN",2)) if 52 <= C[s][f][j] <= 53]
        print("   thin (.80-.83):", ", ".join(thin))
print("5 seeds:", label(["s1","s2","s3","s4","s5"]))
print("4 seeds (s5 incomplete):", label(["s1","s2","s3","s4"]))
for miss in ["s1","s2","s3","s4","s5"]:
    print(" without", miss, label([s for s in ["s1","s2","s3","s4","s5"] if s != miss]))
print("seeds whose only failing family is H5:", [s for s in ["s1","s2","s3","s4","s5"] if fam_fail(s) == ["H5"]])
# control / H1 / H2 LIK failures on any seed (strict reading of 'never partial')
print("LIK failures on H1/H2/controls:", [(s, f, C[s][f][0]) for s in ["s1","s2","s3","s4","s5"] for f in ["H1","H2","C_noupd","C_twoslot"] if not ok(C[s][f][0])])
print("min LIK on H1/H2/controls per seed:", {s: min((C[s][f][0], f) for f in ["H1","H2","C_noupd","C_twoslot"]) for s in ["s1","s2","s3","s4","s5"]})
# X of size 2 alternatives
for X in itertools.combinations(XAX, 2):
    n = sum(1 for s in ["s1","s2","s3","s4","s5"] if set(fam_fail(s)) <= set(X))
    if n >= 3: print("size-2 X", X, n)
