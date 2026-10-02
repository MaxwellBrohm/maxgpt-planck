"""E003 audit A5: mutation tests of the audit's own checks (a1, a2, a4). Each mutant must raise problems or hits;
the unmutated run must raise none. Mutations are in memory; a1's output is redirected to a scratch folder.
usage: python3 -B a5_mutants.py <scratch dir>
"""
import contextlib, importlib, io, json, os, random, re, sys

AUD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AUD)
SCR = sys.argv[1]
os.makedirs(SCR, exist_ok=True)


def run_a1(patch):
    a1 = importlib.reload(importlib.import_module("a1_cells"))
    a1.AUD = SCR
    patch(a1)
    with contextlib.redirect_stdout(io.StringIO()):
        a1.main()
    return len(json.load(open(os.path.join(SCR, "a1_out.json")))["problems"])


def run_a2(patch):
    a2 = importlib.reload(importlib.import_module("a2_integrity"))
    patch(a2)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        a2.main()
    return int(re.search(r"PROBLEMS: (\d+)", buf.getvalue()).group(1))


def none(m):
    pass


def a1_d4(m):
    orig = m.cells
    m.cells = lambda recs, fam=None: {k: (sum(m.ok(r["scores"]) for r in recs if r["var"] in v and r["d"] in (4, 10) and (fam is None or r["fam"] == fam)),
                                          sum(1 for r in recs if r["var"] in v and r["d"] in (4, 10) and (fam is None or r["fam"] == fam)))
                                      for k, v in (("LW", m.LWV), ("TS", ("twoslot",)), ("NU", ("noupd",)))}


def a1_hash(m):
    orig = m.load

    def load(model, tag, set_):
        r = orig(model, tag, set_)
        if r and model.endswith("pythia-70m") and tag == "s2" and set_ == "new":
            r[500]["h"] = "000000000000"
        return r
    m.load = load


def a1_mean_pick(m):
    m.my_pick.__globals__["my_pick"] = lambda rows, grid, final: (
        lambda b: ("EXTEND", {1e-3: 3e-3, 3e-3: 1e-2}[max(grid)]) if (not final and b["lr"] == max(grid) and b["min"] < 0.8)
        else ("CHOSEN", b["lr"]))(max(rows, key=lambda r: (r["mean"], -r["lr"])))


def a1_thresh(m):
    orig = m.acc
    m.main.__globals__["acc"] = lambda kn: min(1.0, orig(kn) + 0.3) if kn[1] else None


def a1_tie(m):
    m.main.__globals__["ok"] = lambda sc: isinstance(sc, dict) and "gold" in sc and all(sc["gold"] >= v - 0.05 for v in sc.values())


def a2_py(m):
    m.PY = "/usr/bin/python3"


def a2_window(m):
    orig = m.ts
    m.main.__globals__["ts"] = lambda s: orig(s) + 3600


def a2_twice(m):
    orig = m.queue_history

    def qh():
        h = orig()
        h["ft_p31m_s2"].append(h["ft_p31m_s2"][-1])
        return h
    m.main.__globals__["queue_history"] = qh


def a2_count(m):
    m.EVAL["new/plain"] = 1727


print("a1 unmutated problems:", run_a1(none))
for name, f in (("d4 pooled with d10", a1_d4), ("one record hash changed", a1_hash), ("LR pick by mean", a1_mean_pick),
                ("accuracies +0.3", a1_thresh), ("ties and near-ties count as right", a1_tie)):
    k = run_a1(f)
    print(f"a1 mutant {name}: problems {k} -> {'KILLED' if k else 'SURVIVED'}")
print("a2 unmutated problems:", run_a2(none))
for name, f in (("wrong interpreter", a2_py), ("guard window shifted 1 h", a2_window),
                ("a job completed twice", a2_twice), ("pre-registered count 1727", a2_count)):
    k = run_a2(f)
    print(f"a2 mutant {name}: problems {k} -> {'KILLED' if k else 'SURVIVED'}")
# a4: plant one eval key statement and one eval prompt into a training draw
a4src = open(os.path.join(AUD, "a4_leaks.py")).read()
TD = importlib.import_module("train_data")
real_gen = TD.gen


def planted(rng):
    ex = real_gen(rng)
    if rng.random() < 0.001:
        ex = dict(ex, turns=[("Actually, my dentist appointment is on Friday now.", "Got it.")] + list(ex["turns"]))
    return ex


TD.gen = planted
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(a4src.replace("import train_data as TD", "TD = sys.modules['train_data']"), "a4", "exec"),
         {"__name__": "a4", "__file__": os.path.join(AUD, "a4_leaks.py")})
hits = [int(x) for x in re.findall(r"eval key-statement template (\d+)", buf.getvalue())]
print(f"a4 mutant planted key statement: hits per seed {hits} -> {'KILLED' if all(hits) else 'SURVIVED'}")
TD.gen = real_gen
