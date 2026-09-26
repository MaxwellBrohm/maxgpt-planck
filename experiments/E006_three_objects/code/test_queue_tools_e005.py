"""Mutation test of the two step-4 queue tools (no model; tokenizer only). Every mutant must make its named check
fail (a crash is not a kill), and the clean baselines must pass.
  validate_e005   on seed 1, first 300 kept: baseline 0; shift limit 0, oracle limit 0.30, one purity violation -> 1
  reuse_base_e005 on a scratch copy of E004's code, base records, guard verdicts and transcript (E004 is only read):
                  baseline 0; one LIK prompt hash, one GEN family, a missing base file, a killed guard verdict,
                  run.json steps 400, a .py newer than the base run, a set builder that drops one item, a chat
                  transcript with 52 lines -> 1 with the named FAIL line; copy mode copies with equal sha256, is
                  idempotent, and refuses an existing copy with other bytes
usage: <venv python> -B test_queue_tools_e005.py <scratch dir> > ../logs/test_queue_tools_e005.txt"""
import contextlib, io, json, os, shutil, sys, time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import purity_e005 as PU
import oracles_e005 as OR
import reuse_base_e005 as RB
import validate_e005 as V

RESULTS = []


def run(fn, argv):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = fn(argv)
    except BaseException as e:  # a crash is reported, never counted as a kill
        return "crash", f"{type(e).__name__}: {e}"
    return rc, buf.getvalue()


def expect(name, rc, out, want_rc, want_line=None):
    good = rc == want_rc and (want_line is None or want_line in out)
    RESULTS.append(good)
    print(f"{'ok  ' if good else 'BAD '} {name}: rc={rc}" + (f" (wants {want_line!r})" if want_line else ""))
    if not good:
        print("     " + str(out)[-400:].replace("\n", "\n     "))


def validate_mutants():
    args = ["--seeds", "1", "--n", "300"]
    expect("validate baseline", *run(V.main, args), 0)
    old = V.MAX_SHIFT
    V.MAX_SHIFT = 0.0
    expect("validate shift limit 0", *run(V.main, args), 1, "drawn-vs-kept shift")
    V.MAX_SHIFT = old
    old = OR.MAX_RULE
    OR.MAX_RULE = 0.30
    expect("validate oracle limit 0.30", *run(V.main, args), 1, "oracle:")
    OR.MAX_RULE = old
    real, hit = PU.violations, []

    def one(ex, eg):
        v = real(ex, eg)
        if not hit:
            hit.append(1)
            v = dict(v, **{"eval surname": ["injected"]})
        return v
    PU.violations = one
    expect("validate one purity violation", *run(V.main, args), 1, "purity")
    PU.violations = real


def scratch_tree(root, real):
    """-> (E4, EXP) scratch copies of the REAL trees (real = RB's original E4, E4CODE, E4OUT, EXP)"""
    r4, r4code, r4out, rexp = real
    if os.path.exists(root):
        shutil.rmtree(root)
    e4, exp = os.path.join(root, "E004_general_updating"), os.path.join(root, "E005_alias_eot")
    shutil.copytree(r4code, os.path.join(e4, "code"), copy_function=shutil.copy2)
    os.makedirs(os.path.join(e4, "out"))
    for f in os.listdir(r4out):
        if f.startswith(f"{RB.SLUG}__base__"):
            shutil.copy2(os.path.join(r4out, f), os.path.join(e4, "out", f))
    os.makedirs(os.path.join(e4, "logs"))
    for f in ("base_135m.guard.json", "chat_135m_base.guard.json"):
        shutil.copy2(os.path.join(r4, "logs", f), os.path.join(e4, "logs", f))
    os.makedirs(os.path.join(e4, "transcripts"))
    shutil.copy2(os.path.join(r4, "transcripts", RB.TRANSCRIPT), os.path.join(e4, "transcripts", RB.TRANSCRIPT))
    os.makedirs(os.path.join(exp, "logs"))
    shutil.copy2(os.path.join(rexp, "logs", "copied_from_e004_sha256.txt"), os.path.join(exp, "logs"))
    return e4, exp


def point(e4, exp):
    RB.E4, RB.E4CODE, RB.E4OUT, RB.EXP = e4, os.path.join(e4, "code"), os.path.join(e4, "out"), exp


def edit_json(path, fn):
    d = json.load(open(path))
    fn(d)
    json.dump(d, open(path, "w"))


def rewrite(path, fn):
    t = open(path).read()
    u = fn(t)
    assert u != t, f"mutant did not change {path}"
    open(path, "w").write(u)


def edit_line(path, i, fn):
    ls = open(path).read().splitlines()
    d = json.loads(ls[i])
    fn(d)
    ls[i] = json.dumps(d)
    open(path, "w").write("\n".join(ls) + "\n")


def reuse_mutants(root):
    real = (RB.E4, RB.E4CODE, RB.E4OUT, RB.EXP)
    base = lambda s: os.path.join(RB.E4OUT, f"{RB.SLUG}__base__{s}")
    muts = [
        ("one LIK prompt hash", lambda: edit_line(base("e004__chat.jsonl"), 7, lambda d: d.update(h="000000000000")),
         "FAIL  5 LIK"),
        ("one GEN family", lambda: edit_line(base("gen_e004__plain.jsonl"), 3, lambda d: d.update(family="H9")),
         "FAIL  5 GEN"),
        ("a missing base file", lambda: os.remove(base("khard__plain.jsonl")), "FAIL  3 base files"),
        ("a killed guard verdict", lambda: edit_json(os.path.join(RB.E4, "logs", "base_135m.guard.json"),
                                                     lambda d: d.update(killed="free_memory")), "FAIL  2 base_135m"),
        ("run.json steps 400", lambda: edit_json(base("run.json"), lambda d: d.update(steps=400)), "FAIL  2 base run.json"),
        ("a .py newer than the base run", lambda: os.utime(os.path.join(RB.E4CODE, "lik.py"), (time.time(),) * 2),
         "FAIL  1 every E004 .py"),
        ("a set builder that drops one item", lambda: rewrite(os.path.join(RB.E4CODE, "e004_sets.py"), lambda t: t.replace(
            '("kbig", "plain", KB.build())', '("kbig", "plain", KB.build()[:-1])', 1)), "FAIL  4 set hashes"),
        ("a 52-line chat transcript", lambda: rewrite(os.path.join(RB.E4, "transcripts", RB.TRANSCRIPT),
                                                      lambda t: "".join(t.splitlines(True)[:52])), "FAIL  3 untouched"),
    ]
    try:
        point(*scratch_tree(root, real))
        expect("reuse baseline (scratch copy)", *run(RB.main, ["--check-only"]), 0, "REUSE OK")
        for name, mutate, line in muts:
            point(*scratch_tree(root, real))
            assert RB.E4.startswith(root) and RB.E4OUT.startswith(root) and RB.EXP.startswith(root)
            mutate()
            expect(f"reuse {name}", *run(RB.main, ["--check-only"]), 1, line)
        e4, exp = scratch_tree(root, real)
        point(e4, exp)
        rc, out = run(RB.main, [])
        n = sum(1 for f in os.listdir(os.path.join(exp, "out")))
        same = all(RB.sha(os.path.join(e4, "out", f)) == RB.sha(os.path.join(exp, "out", f)) for f in os.listdir(os.path.join(exp, "out")))
        expect(f"reuse copy mode ({n} files in out/, sha256 equal {same})", rc if (n == 16 and same) else "bad", out, 0, "REUSE OK")
        expect("reuse copy mode again (idempotent)", *run(RB.main, []), 0, "REUSE OK")
        with open(os.path.join(exp, "out", f"{RB.SLUG}__base__run.json"), "a") as f:
            f.write(" ")
        expect("reuse refuses an existing copy with other bytes", *run(RB.main, []), 1, "exists with other bytes")
    finally:
        RB.E4, RB.E4CODE, RB.E4OUT, RB.EXP = real
        shutil.rmtree(root, ignore_errors=True)


def main():
    root = os.path.join(sys.argv[1], "reuse_mut")
    validate_mutants()
    reuse_mutants(root)
    n_bad = RESULTS.count(False)
    print(f"\nRESULT: {'ALL PASS' if not n_bad else f'{n_bad} FAILED'} ({len(RESULTS)} cases)")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
