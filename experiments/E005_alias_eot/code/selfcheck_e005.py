"""E005 static self-check of the entry point (e005_ft_test.py). Loads NO model and NO tokenizer (system python).
  1 argument parsing: E004's checks on the E005 parser; the LR defaults to 1.5e-4 (pre-registered); bad lines exit
  2 --plan: a scored run (with --chat) plans E004's sets (eval LIK/GEN plain + chat, continuity, knowledge, probe)
    plus the E005 training fields; a baseline plans no training; a dry run plans <= 2 items per set
  3 import hygiene: resolving and planning does not import torch or transformers (subprocess)
  4 dry runs exercise both renders: on seeds 0-5 the examples a dry run consumes (first batch + 5 steps x 16) hold
    both renders (>= 20 each), and the side iterator of the self-check finds bs plain and bs chat examples early
  5 wiring: the entry point trains through e005_train / e005_sets (not E004's loop), and e005_train's loop is E004's
    (same optimizer, schedule, clip, loss, probe, dry padding lines)
usage: python3 -B selfcheck_e005.py   (writes ../logs/selfcheck_e005.txt; exit 0 only if ALL PASS)"""
import contextlib, io, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e005_ft_test as M
import train_e005 as T5

LOG = os.path.join(os.path.dirname(HERE), "logs", "selfcheck_e005.txt")
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


def run_plan(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        M.main(argv + ["--plan"])
    return json.loads(buf.getvalue())


def part_parse():
    a = M.resolve([SMOL, "--seed", "1"])
    want = dict(steps=400, lr=1.5e-4, bs=4, accum=4, max_len=768, probe_every=50, tag="s1", max_new=48, chat=False,
                dry=False, plan=False, device="mps", save=0)
    got = {k: getattr(a, k) for k in want}
    check(got == want and a.set_names == {"eval", "cont", "know"}, f"defaults {got} sets {sorted(a.set_names)}")
    check(M.resolve([SMOL, "--seed", "0", "--steps", "0"]).tag == "base", "--steps 0 -> tag base")
    d = M.resolve([SMOL, "--seed", "0", "--dry", "--chat"])
    check(d.steps == 5 and d.tag == "dry" and d.chat and d.lr == 1.5e-4, "--dry -> 5 steps, tag dry, LR 1.5e-4")
    bad = [[SMOL], [SMOL, "--seed", "1", "--sets", "bogus"], [SMOL, "--seed", "1", "--sets", "dev,eval"],
           [SMOL, "--seed", "1", "--max-new", "64"], [SMOL, "--seed", "1", "--bs", "0"],
           [SMOL, "--seed", "1", "--lr", "0"], [SMOL, "--seed", "1", "--sets", ""], ["--seed", "1"]]
    for argv in bad:
        check(exits(argv), f"refused: {' '.join(argv)}")


def part_plan():
    p = run_plan([SMOL, "--seed", "1", "--chat"])
    lik, gen = {(s, r): n for s, r, n in p["lik"]}, {(s, r): n for s, r, n in p["gen"]}
    check(lik.get(("e004", "plain")) == 640 and lik.get(("e004", "chat")) == 640 and lik.get(("kbig", "plain")) == 441,
          f"scored run LIK sets {lik}")
    check(gen == {("e004", "plain"): 640, ("e004", "chat"): 640, ("cont", "plain"): 416}, f"GEN sets {gen}")
    check(p["probe"] == 144 and p["lr"] == 1.5e-4 and p.get("p_chat") == 0.5 and p.get("blocks") == T5.BLOCKS,
          f"probe 144, LR 1.5e-4, training fields {p.get('p_chat')} {p.get('blocks')}")
    check(all("/E005_alias_eot/out/" in o and "__s1__" in o for o in p["outputs"]) and
          len(set(p["outputs"])) == len(p["outputs"]), "outputs unique and inside E005's out/")
    b = run_plan([SMOL, "--seed", "0", "--steps", "0"])
    check(b["probe"] == 0 and b["tag"] == "base" and "train_stream" not in b, "baseline plan: no probe, no training")
    dr = run_plan([SMOL, "--seed", "0", "--dry", "--chat"])
    check(all(n <= 2 for _, _, n in dr["lik"] + dr["gen"]) and dr["probe"] == 9 and dr["steps"] == 5 and
          ("e004", "chat", 2) in [tuple(x) for x in dr["gen"]], f"dry plan: <= 2 items per set incl. chat GEN")


def part_imports():
    code = ("import sys; sys.path.insert(0, %r); import e005_ft_test as M, contextlib, io\n"
            "with contextlib.redirect_stdout(io.StringIO()): M.main(['x/y', '--seed', '1', '--plan'])\n"
            "print(sorted(m for m in ('torch', 'transformers') if m in sys.modules))") % HERE
    r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=100)
    check(r.returncode == 0 and r.stdout.strip() == "[]", f"--plan imports neither torch nor transformers "
                                                          f"({r.stdout.strip() or r.stderr.strip()[-200:]})")


def part_dry_renders():
    rows = []
    for seed in range(6):
        exs = T5.take(seed, 4 + 5 * 16)
        n = {r: sum(e["render"] == r for e in exs) for r in ("plain", "chat")}
        side = T5.take(seed, 40)
        early = min(sum(e["render"] == r for e in side) for r in ("plain", "chat"))
        rows.append((seed, n, early))
    ok = all(min(n.values()) >= 20 and early >= 4 for _, n, early in rows)
    check(ok, f"dry-run examples per seed hold both renders; check batches fill within 40 draws {rows}")


def body(src, name):
    m = re.search(rf"\ndef {name}\(.*?(?=\n(?:def |[A-Za-z_]+ = ))", src, re.S)
    return m.group(0) if m else ""


def part_wiring():
    src5 = open(os.path.join(HERE, "e005_ft_test.py")).read()
    check("import e005_train as TR" in src5 and "e004_train" not in src5.replace("e004_train.py", ""),
          "the entry point trains through e005_train, never e004_train")
    t4, t5 = open(os.path.join(HERE, "e004_train.py")).read(), open(os.path.join(HERE, "e005_train.py")).read()
    lines = ["opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)",
             "opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))",
             "gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()",
             "loss = C.answer_loss(model, ids.to(a.device), att.to(a.device), lab.to(a.device)) / a.accum",
             "if (step + 1) % a.probe_every == 0 or step == a.steps - 1:",
             'model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})']
    miss = [ln for ln in lines if not (ln in t4 and ln in t5)]
    check(not miss, f"e005_train keeps E004's optimizer, schedule, clip, loss, probe and checkpointing lines {miss}")
    check("pad_to=a.max_len if a.dry else None" in body(t5, "train"), "dry steps are padded to --max-len")


def main():
    for part in (part_parse, part_plan, part_imports, part_dry_renders, part_wiring):
        out.append(f"-- {part.__name__}")
        part()
    out.append(f"{len(fails)} failures; {'ALL PASS' if not fails else 'FAIL'}")
    with open(LOG, "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
