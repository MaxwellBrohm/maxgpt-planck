"""E006 mutation runner (notes.txt CODE TO WRITE, mutation_e006). No model; the SmolLM2 tokenizer is loaded for the
replay-encoding mutants (tokshim_e006.load: E006_TOK=hf on the PC, shim elsewhere). Each mutant is ONE exact source
edit (the old text must occur exactly once) of a generator, checker or rule module, executed as a fresh module and
handed to the test suite that must catch it; a queue mutant is a queue_e006.sh text run by test_queue_e006.sh in a
temporary copy. A mutant is KILLED when the suite reports a failure; a crash is not a
kill (it is reported as CRASHED and fails the run); a mutant the suite passes SURVIVED (fails the run). The unmutated
modules must pass every suite first (baseline).
usage: python -B mutation_e006.py [--only ID-PREFIXES]   (exit 0 = baselines pass and every mutant is killed)"""
import os
import sys
import time
import traceback
import types
from collections import Counter
sys.dont_write_bytecode = True

import test_train_e006p as TT
import test_replay_e006 as TR
import test_rules_e006 as TRU
import position_e006 as PO0
import train_e006p as TP0
import tokshim_e006 as TK

HERE = os.path.dirname(os.path.abspath(__file__))
M = [  # (id, module, old, new, suite)
    ("P1 split drawn from R_new", "train_e006p", "if len(b) > 1 and rp.random() < q:", "if len(b) > 1 and rng.random() < q:", "p"),
    ("P2 split moves B's original too", "train_e006p", "[b[0]] + head + [al] + tail + b[1:], \"A\", True",
     "head + [al] + tail + b, \"A\", True", "p"),
    ("P3 other_obj placement falls into the split branch", "train_e006p", 'if place == "other_obj":\n            b[0]',
     'if place == "other_obj" and rp.random() > 2:\n            b[0]', "p"),
    ("P4 y-first keeps an indirect first correction", "train_e006p", 'if post[0]["ref"] in ("pron", "ell"):', "if False:", "p"),
    ("P5 y-first drops the planned gap", "train_e006p", 'post[0]["gap"] = y[0]["gap"]', 'post[0]["gap"] = None', "p"),
    ("P6 y-first puts Y's corrections before X", "train_e006p", "[dict(y[0], gap=None)] + x + post",
     "[dict(y[0], gap=None)] + post + x", "p"),
    ("P7 y-first changes the ask", "train_e006p", "return asked, y_first(rp, x, y), True",
     'return ("Y" if asked == "X" else "X"), y_first(rp, x, y), True', "p"),
    ("P8 an unchanged ALIAS example is marked", "train_e006p", "    if changed:\n        ex[\"p_change\"] = \"split\"",
     "    if True:\n        ex[\"p_change\"] = \"split\"", "p"),
    ("P9 R_pos seeded like R_new", "train_e006p", "R_POS = 47", "R_POS = 45", "p"),
    ("P10 the render draw taken from R_pos", "train_e006p", 'ex["render"] = "chat" if r_mix.random() < P_CHAT',
     'ex["render"] = "chat" if r_pos.random() < P_CHAT', "p"),
    ("P11 the value type drawn from R_pos", "train_e006p", "    vtype = rng.choice(T.VTYPES)\n    labs",
     "    vtype = rp.choice(T.VTYPES)\n    labs", "p"),
    ("P12 F_YFIRST 0.30 -> 0.40", "train_e006p", "F_YFIRST = 0.30", "F_YFIRST = 0.40", "p"),
    ("O1 introduction order off by one", "position_e006", 'return seen.index(ex["asked"]) + 1', 'return seen.index(ex["asked"]) + 2', "pos"),
    ("O2 gate tolerance loosened", "position_e006", "if v is None or v > lim:", "if v is None or v > lim + 0.2:", "pos"),
    ("R1 replay label span starts on the header newline", "e006_sets", "labels[len(a):len(b)] = ids[len(a):len(b)]",
     "labels[len(a) - 1:len(b)] = ids[len(a) - 1:len(b)]", "enc"),
    ("R2 <|im_end|> not in the target", "e006_sets", 'end = pre + msgs[i]["content"] + EOT', 'end = pre + msgs[i]["content"]', "enc"),
    ("R3 cut keeps a dangling user turn", "e006_sets", "ids, labels = encode_prefix(tok, turns[:2 * m])",
     "ids, labels = encode_prefix(tok, turns[:2 * m + 1])", "enc"),
    ("R4 replay order not seeded per seed", "e006_sets", "random.Random(1000 * seed + REPLAY_ORDER).shuffle(xs)",
     "random.Random(REPLAY_ORDER).shuffle(xs)", "enc"),
    ("R10 a lossy thread is not dropped", "e006_sets", "    if tok.decode(ids, skip_special_tokens=False) != full:",
     "    if False:", "enc"),
    ("R11 the whole input is not checked", "e006_sets", "    if spans and tok.decode(ids, skip_special_tokens=False) != tpl:",
     "    if False:", "enc"),
    ("R5 probe containment needs 8 words", "replay_e006", "NGRAM, PROBE_MIN_WORDS = 8, 3", "NGRAM, PROBE_MIN_WORDS = 8, 8", "drop"),
    ("R6 eval overlap by 9-grams", "replay_e006", "NGRAM, PROBE_MIN_WORDS = 8, 3", "NGRAM, PROBE_MIN_WORDS = 9, 3", "drop"),
    ("R7 surnames matched case-sensitively", "replay_e006", '+ r")(?![A-Za-z])", re.I)', '+ r")(?![A-Za-z])")', "drop"),
    ("R8 reserve leak read from assistant turns", "replay_e006", 'for t in turns if t["role"] == "user")):',
     'for t in turns if t["role"] == "assistant")):', "drop"),
    ("R9 strict exposure needs only the gold", "replay_e006", "if cw and any(all(_has_word(t, w) for w in cw) for t in hits):",
     "if hits:", "drop"),
    ("B1 H5L gold not forced last", "items_big_e006", "if order[-1][0] != asked or intro_order(order, asked) != f[\"intro\"]:",
     "if intro_order(order, asked) != f[\"intro\"]:", "big"),
    ("B2 H5L strata ignored", "items_big_e006", "if order[-1][0] != asked or intro_order(order, asked) != f[\"intro\"]:",
     "if order[-1][0] != asked:", "big"),
    ("Q1 gain threshold 0.06 -> 0.05", "rules_e006", "GAIN, NTC, DROP, DEV_TOL, TRADE = 0.06,", "GAIN, NTC, DROP, DEV_TOL, TRADE = 0.05,", "rules"),
    ("Q2 gain CI low >= 0", "rules_e006", "x[0] >= GAIN and x[1] > 0", "x[0] >= GAIN and x[1] >= 0", "rules"),
    ("Q3 NOT THE CAUSE with CI high <= 0.06", "rules_e006", "x[2] < NTC", "x[2] <= NTC", "rules"),
    ("Q4 the WORSE branch unreachable", "rules_e006", "elif both(lambda x: x[2] < 0):\n        label = \"WORSE\"",
     "elif both(lambda x: x[2] < NTC and False):\n        label = \"WORSE\"", "rules"),
    ("Q5 paired difference sign", "rules_e006", "float(X[s][i]) - float(Y[s][i])", "float(Y[s][i]) - float(X[s][i])", "rules"),
    ("Q6 pairing by item broken", "rules_e006", "float(X[s][i]) - float(Y[s][i]) for i in items]",
     "float(X[s][i]) - float(Y[s][items[0]]) for i in items]", "rules"),
    ("Q7 chat mean per conversation, not per turn", "rules_e006", "d = float(S.sum() / N.sum())", "d = float(S.sum() / S.size)", "rules"),
    ("Q8 TF ratio 0.5 -> 0.6", "rules_e006", "HALF = 0.03, 0.10, -3.0, 0.5", "HALF = 0.03, 0.10, -3.0, 0.6", "rules"),
    ("Q9 loop margin inclusive", "rules_e006", "if not loop[2] < LOOP_MARGIN:", "if not loop[2] <= LOOP_MARGIN:", "rules"),
    ("Q10 checks read on the CI high side", "rules_e006", "if not checks[1] > CHECKS_MARGIN:", "if not checks[2] > CHECKS_MARGIN:", "rules"),
    ("Q11 device shift without abs", "rules_e006", "abs(dev_shift_lik) >= DEV_TOL", "dev_shift_lik >= DEV_TOL", "rules"),
    ("Q12 recency trade on either condition", "rules_e006", "h5l[m][0] <= TRADE and h5l[m][2] < 0",
     "h5l[m][0] <= TRADE or h5l[m][2] < 0", "rules"),
    ("Q13 knowledge big effect 0.03 -> 0.02", "rules_e006", "HALF = 0.03, 0.10, -3.0, 0.5", "HALF = 0.02, 0.10, -3.0, 0.5",
     "rules"),
    ("Q14 NO DROP flag at > 0.05", "rules_e006", "c_l4_gap_lik >= DROP", "c_l4_gap_lik > DROP", "rules"),
    ("Q15 L4 not read raises the NO DROP flag", "rules_e006", 'flags = ["L4 NOT READ"]',
     'flags = ["NO DROP TO RESTORE ON CUDA"]', "rules"),
    ("G1 replacement instead of addition (3 update micro-batches)", "e006_train", "    for _ in range(accum):\n        yield \"u\"",
     "    for _ in range(accum - 1):\n        yield \"u\"", "sched"),
    ("G2 the replay micro-batch first", "e006_train",
     '    for _ in range(accum):\n        yield "u", [next(update_stream) for _ in range(bs)]\n    yield "r", [next(replay_iter) for _ in range(bs)]',
     '    yield "r", [next(replay_iter) for _ in range(bs)]\n    for _ in range(accum):\n        yield "u", [next(update_stream) for _ in range(bs)]',
     "sched"),
    ("U1 a temp verdict does not stop the queue", "queue_e006.sh", "case \"$killed\" in temp|gpu_mem|host_mem|smi)",
     "case \"$killed\" in none)", "queue"),
    ("U2 finished jobs are not skipped on a restart", "queue_e006.sh", "if [ -e ../logs/$name.guard.json ]; then",
     "if false; then", "queue"),
    ("U3 a GPU job runs without the lock", "queue_e006.sh", "$FLOCK -w 7200 $LOCK $PY -B guard_e006_pc.py",
     "$PY -B guard_e006_pc.py", "queue"),
    ("U4 a failed scored job stops the queue", "queue_e006.sh", '--save 1 || log "GAP: $arm$s"', '--save 1 || stop "GAP: $arm$s"',
     "queue"),
]


def queue_suite(text):
    """runs test_queue_e006.sh against a queue text in a temporary copy -> failures ([] = the test passed)."""
    import shutil
    import subprocess
    import tempfile
    tmp = tempfile.mkdtemp(prefix="e006_qm_")
    try:
        for f in ("test_queue_e006.sh", "fake_py_e006.py"):
            shutil.copy(os.path.join(HERE, f), tmp)
        open(os.path.join(tmp, "queue_e006.sh"), "w").write(text)
        r = subprocess.run(["bash", os.path.join(tmp, "test_queue_e006.sh")], capture_output=True, text=True, timeout=900)
        return [] if r.returncode == 0 else ([x for x in r.stdout.splitlines() if x.startswith("FAIL")] or ["exit != 0"])
    finally:
        shutil.rmtree(tmp)


def mutant(mod, old, new):
    if mod.endswith(".sh"):
        src = open(os.path.join(HERE, mod)).read()
        if src.count(old) != 1:
            raise ValueError(f"{mod}: the mutated text occurs {src.count(old)} times")
        return src.replace(old, new)
    src = open(os.path.join(HERE, mod + ".py")).read()
    if src.count(old) != 1:
        raise ValueError(f"{mod}: the mutated text occurs {src.count(old)} times")
    m = types.ModuleType(mod)
    m.__file__ = os.path.join(HERE, mod + ".py")
    exec(compile(src.replace(old, new), m.__file__, "exec"), m.__dict__)
    return m


def big_suite(BG):
    items = BG.build_h5l()
    F = []
    if any(it["stmts"][-1]["obj"] != it["asked"] for it in items):
        F.append("H5L: an item whose asked latest is not last")
    if Counter(BG.intro_order(it["stmts"], it["asked"]) for it in items) != {1: 64, 2: 64, 3: 64}:
        F.append("H5L: introduction-order strata are not 64/64/64")
    return F


def run_suite(kind, m, tok):
    if kind == "p":
        TT.TP = m
        try:
            return TT.run(n_eq=1500, n_pair=500, n_struct=300, seeds=(1, 2), numbers=("P",))[0]
        finally:
            TT.TP = TP0
    if kind == "pos":
        TT.PO = m
        try:
            return TT.run(n_eq=10, n_pair=10, n_struct=10, seeds=(1,), numbers=("C", "P"))[0]
        finally:
            TT.PO = PO0
    if kind == "enc":
        return TR.encode_suite(tok, S=m)
    if kind == "drop":
        return TR.drops_suite(R=m)
    if kind == "big":
        return big_suite(m)
    if kind == "rules":
        return TRU.suite(m)
    if kind == "queue":
        return queue_suite(m)
    if kind == "sched":
        return TR.schedule_suite(tok, TR=m)
    raise ValueError(kind)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    only = argv[argv.index("--only") + 1].split(",") if "--only" in argv else None
    todo = [m for m in M if only is None or any(m[0].startswith(p) for p in only)]
    tok = TK.load()
    import e006_sets, e006_train, replay_e006, items_big_e006, rules_e006
    base = {"p": TP0, "pos": PO0, "enc": e006_sets, "drop": replay_e006, "big": items_big_e006, "rules": rules_e006,
            "queue": open(os.path.join(HERE, "queue_e006.sh")).read(), "sched": e006_train}
    bad = []
    for kind, m in base.items():
        if kind not in {x[4] for x in todo}:
            continue
        t = time.time()
        f = run_suite(kind, m, tok)
        print(f"baseline {kind}: {'PASS' if not f else 'FAIL ' + str(f[:2])} ({time.time() - t:.0f}s)", flush=True)
        if f:
            bad.append(f"baseline {kind}")
    res = Counter()
    for mid, mod, old, new, kind in todo:
        t = time.time()
        try:
            f = run_suite(kind, mutant(mod, old, new), tok)
            v = "KILLED" if f else "SURVIVED"
            why = f[0][:110] if f else ""
        except Exception as e:  # a crash is not a kill
            v, why = "CRASHED", f"{type(e).__name__}: {str(e)[:100]}"
            traceback.print_exc(limit=1)
        res[v] += 1
        print(f"{v:8s} {mid} ({time.time() - t:.0f}s) {why}", flush=True)
        if v != "KILLED":
            bad.append(mid)
    print(f"\n{len(todo)} mutants: {dict(res)}")
    print("RESULT: " + ("ALL PASS (baselines pass, every mutant killed)" if not bad else f"FAILED: {bad}"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
