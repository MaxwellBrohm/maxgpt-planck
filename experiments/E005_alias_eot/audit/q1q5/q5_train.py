import sys, itertools
sys.dont_write_bytecode = True
sys.path.insert(0, "REPO/experiments/E005_alias_eot/code")
import train_e005 as T5
import train_e004 as T4
from collections import Counter
def stats(stream, n, label):
    c = Counter()
    for ex in itertools.islice(stream, n):
        st = [s for s in ex["stmts"]]
        c["n"] += 1
        c["nobj%d" % ex["n_obj"]] += 1
        if ex["n_obj"] != 2: continue
        intro = []
        for s in st:
            if s["obj"] not in intro: intro.append(s["obj"])
        if len(intro) < 2: continue
        p = intro.index(ex["asked"])
        about = [s for s in st if s["obj"] == ex["asked"]]
        lat = about[-1]
        other_after = any(s["obj"] != ex["asked"] and s["turn"] > lat["turn"] for s in st)
        gold_last = st[-1]["value"] == ex["gold"] and st[-1]["obj"] == ex["asked"]
        c[f"2obj_asked{p+1}"] += 1
        c[f"2obj_asked{p+1}_otherafter"] += other_after
        c[f"2obj_asked{p+1}_goldislast"] += gold_last
    print(label, dict(c))
    for p in (1, 2):
        a = c[f"2obj_asked{p}"]
        if a: print(f"   2-object, asked introduced {p}: {a} ({a/c['n']:.3f} of all); another object's statement after the asked latest {c[f'2obj_asked{p}_otherafter']/a:.3f}; gold = dialogue's last statement {c[f'2obj_asked{p}_goldislast']/a:.3f}")
for s in (1, 2, 3):
    stats(T5.stream(s), 6404, f"E005 seed {s} (first 6,404 drawn)")
    stats(T4.stream(s), 6404, f"E004 seed {s} (first 6,404 drawn)")
