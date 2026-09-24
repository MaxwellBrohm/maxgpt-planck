"""E004 static self-check of the training / scoring entry point, part A. Loads NO model and NO tokenizer.
  1 argument parsing: valid command lines resolve to the pre-registered settings; bad ones exit
  2 --plan: the sets, item counts and output paths of a scored run, a baseline, an LR-search run and a dry run
  3 import hygiene: resolving and planning a run does not import torch or transformers (checked in a subprocess)
  4 grader on real items: on every E004 eval/dev/probe generation item, the item's own forced prefix + gold
    (a natural sentence) passes; a reply naming the gold and another in-context value fails (clause 4); the
    user's voice "My <object> is <gold>." fails (clause 7); the same on the continuity generation items;
    and every training answer of 2,000 seed-1 examples passes the grader for its own example
Part B (selfcheck_e004_b.py, venv python) covers tokenizers and a stub model.
usage: python3 -B selfcheck_e004.py   (writes ../logs/selfcheck_e004.txt; exit 0 only if ALL PASS)"""
import contextlib, io, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e004_ft_test as M
import e004_sets as S
import gen_grade as G
import train_e004 as T
from pools_train import POOLS

LOG = os.path.join(os.path.dirname(HERE), "logs", "selfcheck_e004.txt")
SMOL = "HuggingFaceTB/SmolLM2-135M-Instruct"
out, fails = [], []


def check(ok, msg):
    out.append(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


def exits(argv):
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            M.resolve(argv)
        except SystemExit as e:
            return e.code != 0
    return False


def part_parse():
    a = M.resolve([SMOL, "--seed", "1"])
    want = dict(steps=400, lr=5e-5, bs=4, accum=4, max_len=768, probe_every=50, tag="s1", max_new=48, chat=False,
                dry=False, plan=False, device="mps", save=0)
    got = {k: getattr(a, k) for k in want}
    check(got == want and a.set_names == {"eval", "cont", "know"}, f"defaults {got} sets {sorted(a.set_names)}")
    check(a.bs * a.accum == 16, "effective batch 16")
    check(M.resolve([SMOL, "--seed", "0", "--steps", "0"]).tag == "base", "--steps 0 -> tag base")
    d = M.resolve([SMOL, "--seed", "0", "--dry"])
    check(d.steps == 5 and d.tag == "dry", "--dry -> 5 steps, tag dry")
    lr = M.resolve([SMOL, "--seed", "0", "--lr", "1.5e-4", "--sets", "dev", "--tag", "lr1.5e-04_s0"])
    check(lr.set_names == {"dev"} and lr.lr == 1.5e-4 and lr.tag == "lr1.5e-04_s0", "LR-search command line")
    bad = [[SMOL], [SMOL, "--seed", "1", "--sets", "bogus"], [SMOL, "--seed", "1", "--sets", "dev,eval"],
           [SMOL, "--seed", "1", "--max-new", "64"], [SMOL, "--seed", "1", "--bs", "0"],
           [SMOL, "--seed", "1", "--lr", "0"], [SMOL, "--seed", "1", "--sets", ""], ["--seed", "1"]]
    for argv in bad:
        check(exits(argv), f"refused: {' '.join(argv)}")


def run_plan(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        M.main(argv + ["--plan"])
    return json.loads(buf.getvalue())


def part_plan():
    p = run_plan([SMOL, "--seed", "1", "--chat"])
    lik, gen = {(s, r): n for s, r, n in p["lik"]}, {(s, r): n for s, r, n in p["gen"]}
    check(lik.get(("e004", "plain")) == 640 and lik.get(("e004", "chat")) == 640, f"scored run e004 LIK {lik}")
    check(all(lik.get(k) for k in [("new", "plain"), ("extra", "plain"), ("old", "plain"), ("uprobe", "plain"),
                                   ("cross", "plain"), ("khard", "plain"), ("kbig", "plain")]), "continuity + knowledge LIK sets")
    check(lik.get(("kbig", "plain")) == 441, f"kbig 441 items ({lik.get(('kbig', 'plain'))})")
    check(gen == {("e004", "plain"): 640, ("e004", "chat"): 640, ("cont", "plain"): 416}, f"GEN sets {gen}")
    check(("dev", "plain") not in lik and p["probe"] == 144, f"no dev draw in a scored run; probe {p['probe']}")
    check(len(set(p["outputs"])) == len(p["outputs"]) and all("__s1__" in o for o in p["outputs"]), "unique outputs")
    b = run_plan([SMOL, "--seed", "0", "--steps", "0"])
    check(b["probe"] == 0 and b["tag"] == "base" and ("e004", "chat") not in {(s, r) for s, r, _ in b["lik"]},
          "baseline plan: no probe, tag base, plain only without --chat")
    lr = run_plan(["EleutherAI/pythia-31m", "--seed", "0", "--lr", "3e-4", "--sets", "dev", "--tag", "lr3e-04_s0"])
    check([tuple(x) for x in lr["lik"]] == [("dev", "plain", 320)] and lr["gen"] == [], f"LR plan {lr['lik']} {lr['gen']}")
    dr = run_plan([SMOL, "--seed", "0", "--dry"])
    check(all(n <= 2 for _, _, n in dr["lik"] + dr["gen"]) and dr["probe"] == 9 and dr["steps"] == 5,
          f"dry plan: <= 2 items per set, probe {dr['probe']}")


def part_imports():
    code = ("import sys; sys.path.insert(0, %r); import e004_ft_test as M, contextlib, io\n"
            "with contextlib.redirect_stdout(io.StringIO()): M.main(['x/y', '--seed', '1', '--plan'])\n"
            "print(sorted(m for m in ('torch', 'transformers') if m in sys.modules))") % HERE
    r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=100)
    check(r.returncode == 0 and r.stdout.strip() == "[]", f"--plan imports neither torch nor transformers "
                                                          f"({r.stdout.strip() or r.stderr.strip()[-200:]})")


def grade_item(it, reply):
    return G.grade(reply, "newline", it["gold"], it["pool"], it["obj"], cands=it.get("cands"))


def part_grader_items():
    items = []
    for name in ("eval", "dev", "probe"):
        items += S.e004_gen(name)
    raw = {(it["family"], it["idx"], name): it for name in ("eval", "dev", "probe")
           for its in S.draw_items(name).values() for it in its}
    n_ok = n_other = n_voice = 0
    bad = []
    for g in items:
        src = raw[(g["family"], g["idx"], g["set"])]
        good = f"{src['prefix']} {g['gold']}."
        other = next(v for v in g["cands"] if v != g["gold"])
        r1, r2 = grade_item(g, good), grade_item(g, f"{g['gold']}, or {other}.")
        r3 = grade_item(g, f"My {src['objects'][src['asked']][0]} is {g['gold']}.")
        n_ok += r1["strict"]
        n_other += (not r2["strict"]) and 4 in r2["fails"]
        n_voice += (not r3["strict"]) and 7 in r3["fails"]
        if not r1["strict"] and len(bad) < 3:
            bad.append((good, r1["fails"]))
    n = len(items)
    check(n_ok == n, f"E004 prefix + gold passes on {n_ok}/{n} generation items {bad}")
    check(n_other == n, f"gold + another in-context value fails clause 4 on {n_other}/{n}")
    check(n_voice == n, f"user's voice fails clause 7 on {n_voice}/{n}")
    cont = S.cont_gen()
    import items_new as N
    pre = {it["var"] + it["fam"]: it["prefix"] for it in N.build() if it["d"] == 10}
    ok = sum(grade_item(c, f"{c.get('prefix_cut') or pre[c['var'] + c['fam']]} {c['gold']}.")["strict"] for c in cont)
    check(ok == len(cont), f"continuity prefix + gold passes on {ok}/{len(cont)}")
    exs = T.take(1, 2000)
    ok, bad = 0, []
    for ex in exs:
        ph, hd = ex["objects"][ex["asked"]]
        r = G.grade(ex["answer"].strip(), "newline", ex["gold"], POOLS[ex["vtype"]]["values"], G.obj_words_of(ph, hd))
        ok += r["strict"]
        if not r["strict"] and len(bad) < 3:
            bad.append((ex["answer"].strip(), r["fails"]))
    check(ok == len(exs), f"training answers pass the grader: {ok}/{len(exs)} {bad}")


def main():
    for part in (part_parse, part_plan, part_imports, part_grader_items):
        out.append(f"-- {part.__name__}")
        part()
    out.append(f"{len(fails)} failures; {'ALL PASS' if not fails else 'FAIL'}")
    with open(LOG, "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
