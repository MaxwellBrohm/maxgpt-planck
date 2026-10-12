from common import *
from collections import Counter

flat = items_flat()
rng = random.Random(4)
print("30 random E005 chat replies (6 per seed; ids drawn without looking):")
for t in SEEDS:
    C = recs("E005", t, "gen_e004", "chat")
    ids = rng.sample(range(640), 6)
    for i in sorted(ids):
        r = C[i]
        print(f"  {t} id {i:3d} {r['family']:9} gold={flat[i]['gold']:<10} stop={r['stop']:4} n_new={r['n_new']:2d} strict={r['strict']!s:5} | {r['reply']!r}")

print("\nleak / runaway scan over all E005 chat replies (5 x 640):")
pat_role = re.compile(r"user|assistant|system|<\|im_|im_start|im_end", re.I)
for t in ["base"] + SEEDS:
    C = recs("E005", t, "gen_e004", "chat")
    multi = sum("\n" in r["reply"] for r in C)
    role = [r["reply"] for r in C if pat_role.search(r["reply"])]
    empty = sum(not r["reply"].strip() for r in C)
    nw = Counter(len(WORD.findall(r["reply"].lower())) for r in C)
    nn = Counter(r["n_new"] for r in C)
    sent = sum(len(re.findall(r"[.!?](\s|$)", r["reply"])) > 1 for r in C)
    print(f"  {t}: multi-line {multi}, role/template words {len(role)}, empty {empty}, >1 sentence {sent}, "
          f"n_new range {min(nn)}-{max(nn)}, words range {min(nw)}-{max(nw)}")
    if role[:3]:
        print("    e.g.", role[:3])

print("\nwrong E005 chat replies: 12 random ones (all seeds)")
wrong = []
for t in SEEDS:
    C = recs("E005", t, "gen_e004", "chat")
    wrong += [(t, i, r) for i, r in enumerate(C) if not r["strict"]]
print("  total wrong:", len(wrong))
for t, i, r in rng.sample(wrong, 12):
    print(f"  {t} id {i:3d} {r['family']:9} gold={flat[i]['gold']:<10} fails={r['fails']} | {r['reply']!r}")

print("\nmost common reply shapes (value replaced by <V>), E005 chat pooled:")
shape = Counter()
for t in SEEDS:
    C = recs("E005", t, "gen_e004", "chat")
    for i, r in enumerate(C):
        s = r["reply"]
        for v in flat[i]["values"]:
            s = re.sub(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])", "<V>", s, flags=0 if v[:1].isupper() else re.I)
        shape[s] += 1
for s, n in shape.most_common(12):
    print(f"  {n:4d} {s!r}")
print("  distinct shapes:", len(shape))

print("\nE004 chat, 3 examples per seed for the comparison:")
for t in SEEDS:
    C = recs("E004", t, "gen_e004", "chat")
    for i in [0, 200, 400]:
        print(f"  E004 {t} id {i} stop={C[i]['stop']} n_new={C[i]['n_new']} | {C[i]['reply'][:160]!r}")
