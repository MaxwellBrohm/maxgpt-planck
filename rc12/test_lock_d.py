"""RC-12 lock tests, part 4: the S5 pack (sealed_pack.py) on a fixture work copy, and test_gens' s-ending value rule
(rc12/notes.txt STEP 15). Stdlib, no model. The rebuild check runs build_dev in a real work copy; here rebuild_items
is replaced by a stand-in that returns the items (or not), the only part not run for real (the dry run ran it).
  python3 -B test_lock_d.py     (exit 1 on any failure)"""
import gzip
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.pop("RC12_SPLIT", None)
import sealed_crypto as CR                   # noqa: E402
import sealed_kit as K                       # noqa: E402
import sealed_pack as PK                     # noqa: E402
from lockfix import check, dies, full_gates, write, repo, quiet, report   # noqa: E402

ITEMS = '{"split": "sealed", "n_turns": 12, "turns": []}\n'
SEED = "cd" * 32 + "\n"


def fixture(tmp, split="sealed", s1="PASS"):
    """a packable work copy: items, content, expect, a full gate table naming its inputs, a passing S1 file, and
    a draw log whose last run accepted this seed."""
    r = repo(os.path.join(tmp, "r"))
    work = os.path.join(tmp, "w")
    quiet(K.prepare, work, "HEAD", repo=r)
    items = ITEMS.replace('"sealed"', f'"{split}"')
    write(work, K.ITEMS, items)
    for m in K.CONTENT_MODULES:
        write(work, f"rc12/content_{m}.py", f"X = {m!r}\n")
    write(work, "expect.json", "{}")
    write(tmp, "seed.txt", SEED)
    gates = full_gates().replace("commit x split sealed", f"commit {K.commit_of(work)} split {split}")
    inputs = K.gated_inputs(work, os.path.join(tmp, "seed.txt"), os.path.join(work, "expect.json"))
    write(work, "GATES.txt", gates + "".join(f"input sha256 {K.sha256(p)}  {n}\n" for n, p in inputs))
    write(work, "OVERLAP.txt", f"commit x; items sha256 {CR.sha(items.encode())}\nS1 overall: {s1} (0 findings)\n")
    write(tmp, "draws.txt", f"commit {K.commit_of(work)} split {split}: seed draws\ndraw 1: PASS; ceiling S7 30.0\n"
          f"accepted draw 1; seed sha256 {CR.sha(SEED.strip().encode())}\n")
    write(tmp, "oodh.jsonl", '{"turns": []}\n')
    open(os.path.join(tmp, "grams.gz"), "wb").write(gzip.compress(b"# head; 7 hashes\n" + b"a" * 64 + b"\n"))
    return r, work


def run_pack(tmp, r, work, out, rebuild=None):
    """pack with the rebuild stand-in; None when it packed, else its refusal message."""
    real = PK.rebuild_items
    PK.rebuild_items = lambda *a: rebuild or open(os.path.join(work, K.ITEMS), "rb").read()
    try:
        quiet(PK.pack, work, os.path.join(tmp, "seed.txt"), os.path.join(tmp, "draws.txt"),
              os.path.join(tmp, "oodh.jsonl"), os.path.join(tmp, "grams.gz"), out, repo=r)
        return None
    except SystemExit as e:
        return str(e.code)
    finally:
        PK.rebuild_items = real


def t_pack_good(tmp):
    r, work = fixture(tmp)
    out = os.path.join(tmp, "pack")
    got = run_pack(tmp, r, work, out)
    check(got is None, f"pack: a good work copy was refused: {got}")
    s5 = open(os.path.join(out, "SEALED_HASHES.txt")).read()
    tar = open(os.path.join(out, CR.TAR), "rb").read()
    names = ["rc12_sealed.jsonl", "expect.json", "generator_seed.txt", "GATES.txt", "OVERLAP.txt", "SEED_ATTEMPTS.txt",
             CR.NGRAMS, CR.OODH] + [f"content_{m}.py" for m in K.CONTENT_MODULES]
    check(all(re.search(rf"^sha256 [0-9a-f]{{64}}  {re.escape(n)}$", s5, re.M) for n in names),
          "pack: S5 does not name every archived file")
    check(f"sha256 {CR.sha(SEED.strip().encode())}  generator seed" in s5,
          "pack: the seed hash is not the hex string's")
    check(f"sha256 {CR.sha(tar)}  {CR.TAR}" in s5 and "decon13 hashes 7 " in s5 and "; S1 PASS" in s5 and
          f"commit {K.commit_of(work)}" in s5, "pack: tar hash, S6 count, S1 or commit missing")
    check(os.stat(os.path.join(out, CR.TAR)).st_mode & 0o777 == 0o600, "pack: the plaintext tar is not private")


def reasons(msg):
    """the refusal reasons of a pack message (one per check that refused)."""
    return msg.split("not packed: ", 1)[1].split("; ") if "not packed: " in msg else [msg]


def edit_file(path, old, new):
    text = open(path).read()
    assert text.count(old) >= 1, old
    open(path, "w").write(text.replace(old, new, 1))


def quick_table(w, t):
    g = open(os.path.join(w, "GATES.txt")).read().splitlines()
    write(w, "GATES.txt", "\n".join([g[0], "test_gens              exit   0 PASS  x", "failing steps: 0"] +
                                     [x for x in g if x.startswith("input ")]) + "\n")


def t_pack_refusals(tmp):
    """each spoiled fixture is refused for exactly its own reason (no other check can stand in for it)."""
    gates = lambda w: os.path.join(w, "GATES.txt")
    cases = [("a failing gate table", "full, clean S4 gate table",
              lambda w, t: edit_file(gates(w), "mutation_gens          exit   0 PASS",
                                     "mutation_gens          exit   1 FAIL")),
             ("a quick-gates-only table", "full, clean S4 gate table", quick_table),
             ("a gate table of another commit", "written for this commit",
              lambda w, t: edit_file(gates(w), f"commit {K.commit_of(w)}", "commit " + "f" * 40)),
             ("content changed after the gates", "content_balance.py is not the file the S4 gates ran on",
              lambda w, t: write(w, "rc12/content_balance.py", "X = 'new'\n")),
             ("a draw log without this acceptance", "draw log's last run",
              lambda w, t: edit_file(os.path.join(t, "draws.txt"), "accepted draw 1", "no seed passed")),
             ("an S1 file of other items", "OVERLAP.txt is not the S1 check of these items",
              lambda w, t: write(w, "OVERLAP.txt", "items sha256 x\nS1 overall: PASS\n")),
             ("a work copy unlike its commit", "differs from its commit",
              lambda w, t: write(w, "rc12/a.py", "X = 9\n"))]
    for i, (why, frag, spoil) in enumerate(cases):
        d = os.path.join(tmp, f"case{i}")
        r, work = fixture(d)
        spoil(work, d)
        got = run_pack(d, r, work, os.path.join(d, "pack"))
        check(got is not None and len(reasons(got)) == 1 and frag in got, f"pack: {why}: {got}")
    d = os.path.join(tmp, "seed")
    r, work = fixture(d)
    write(d, "seed.txt", "ef" * 32 + "\n")
    got = run_pack(d, r, work, os.path.join(d, "pack"))
    check(got is not None and "generator_seed.txt is not the file" in got and "draw log's last run" in got,
          f"pack: a seed other than the gated and accepted one: {got}")
    d = os.path.join(tmp, "outside")
    r, work = fixture(d)
    got = run_pack(d, r, work, os.path.join(r, "rc12", "pack"))
    check(got is not None and "under sealed/" in got, f"pack: a plaintext tar inside the repo outside sealed/: {got}")
    d = os.path.join(tmp, "rebuild")
    r, work = fixture(d)
    got = run_pack(d, r, work, os.path.join(d, "pack"), rebuild=b"other\n")
    check(got is not None and reasons(got) == ["the seed does not rebuild the items byte for byte"],
          f"pack: rebuild: {got}")
    d = os.path.join(tmp, "s1fail")
    r, work = fixture(d, s1="FAIL")
    got = run_pack(d, r, work, os.path.join(d, "pack"))
    check(got is not None and reasons(got) == ["S1 did not pass (OVERLAP.txt)"], f"pack: a failed S1: {got}")
    d = os.path.join(tmp, "dry")
    r, work = fixture(d, split="dryrun", s1="FAIL")
    check(run_pack(d, r, work, os.path.join(d, "pack")) is None and "S1 NOT PASSED (a dry-run split)" in open(
        os.path.join(d, "pack", "SEALED_HASHES.txt")).read(), "pack: a dry-run split with a failed S1 not packed "
          "with its label")


def t_s_value(tmp):
    """test_gens refuses pools with no s-ending instrument, food or project value (the singular rule's case),
    PROJECTS keys unlike PROJECT, and avoid lists that are not their items' own object words (a stale word, an item
    without its word, a word through several items; case ignored, {fields} not words)."""
    import test_gens as T
    saved = {n: list(getattr(T.V, n)) for n in ("INSTRUMENT", "FOOD", "PROJECT")}
    try:
        T.FAILS.clear()
        T.test_pools()
        check(not T.FAILS, f"s-value: dev pools refused: {T.FAILS[:2]}")
        for n, vals in saved.items():
            getattr(T.V, n)[:] = [v[:-1] if v.endswith("s") else v for v in vals]
        T.FAILS.clear()
        T.test_pools()
        check(any("ends in s" in f for f in T.FAILS), "s-value: pools without an s-ending value pass")
        import pools_misc as M
        keep = dict(M.PROJECTS)
        try:
            M.PROJECTS.pop(next(iter(M.PROJECTS)))
            T.FAILS.clear()
            T.test_pools()
            check(any("PROJECTS keys" in f for f in T.FAILS), "content rules: PROJECTS keys unlike PROJECT pass")
        finally:
            M.PROJECTS.clear()
            M.PROJECTS.update(keep)
        for n, vals in saved.items():                       # the pools back to dev before the avoid-list cases
            getattr(T.V, n)[:] = vals
        import fam_bind as FB, fam_own_topic as FO, fam_role_lookup as FL
        last = FO.OWN_AVOID.index("team")                  # quiz still guards the quiz-team items without it
        mine = ["my"] * (len(FO.OWN_AVOID) - 1) + ["team"]
        cases = [(FB, "BIND_AVOID", 0, "zorvex", "BIND_AVOID word is no BIND", True),   # a word no template uses
                 (FB, "BIND_AVOID", 0, "the", "BIND_AVOID word is no BIND", True),      # a word that is no object
                 (FB, "BIND_AVOID", 0, "Bike", None, True),         # the build lowercases avoid words: allowed
                 (FO, "OWN_AVOID", last, "zorvex", "is in no template", True),
                 (FO, "OWN_AVOID", 0, "puppy", "share no word", True),  # boat's items left unguarded, all in use
                 (FL, "LOOKUP_AVOID", 0, "menu", "share no word", True), (FO, "OWN_AVOID", 0, "name", "share no word",
                                                                          True),
                 (FO, "OWN_AVOID", None, mine, "runs through several items", True),   # "my" runs through 3 items
                 (FL, "LOOKUP_AVOID", 0, "age", "is in no template", False),   # a {field} name is no word
                 (FO, "OWN_AVOID", 0, "Boat", None, True), (FL, "LOOKUP_AVOID", 0, "Rota", None, True),   # case
                 (FO, "OWN_AVOID", last, "my", "runs through several items", True),   # a spare slot, common word
                 (FO, "OWN_AVOID", last, "My", "runs through several items", True)]   # ... in any case
        for mod, name, i, word, frag, alone in cases:       # each refused by its own rule (alone where it can be)
            keep = list(getattr(mod, name))
            try:
                getattr(mod, name)[i if i is not None else slice(None)] = word
                T.FAILS.clear()
                T.test_pools()
                ok = not T.FAILS if frag is None else (len(T.FAILS) == 1 or not alone) and any(
                    frag in f for f in T.FAILS)
                check(ok, f"content rules: {name} with {word!r} not judged by its own rule: {T.FAILS[:3]}")
            finally:
                getattr(mod, name)[:] = keep
    finally:
        for n, vals in saved.items():
            getattr(T.V, n)[:] = vals
        T.FAILS.clear()


def t_g6_coverage(tmp):
    """G6 keeps every fixture kind the items can give, whatever the record order: with the BIND perspective records
    that ask the assistant's name moved behind four that ask the user's, the "given" fixtures are still built."""
    import grade_voice as GV
    import mutation_graders as MG
    recs = MG.load()
    persp = [r for r in recs if (r["family"], r["cell"]) == ("BIND", "perspective")]
    user = [r for r in persp if all(GV.mode_of(p) == "user" for p in r["probes"])]
    given = [r for r in persp if r not in user]
    rest = [r for r in recs if r not in persp]
    fx, _ = MG.build(rest + user + given)        # (only BIND gives "given" VAL fixtures today; another family that
    check(len(user) >= MG.PER_CELL and given and any(f["label"].startswith("given_") and f["family"] == "BIND"
                                                     for f in fx),                 # did would fail this, rightly)
          "G6: the assistant-name fixtures vanish when such records come after the first four of their cell")
    fx, index = MG.build(recs)
    check((len(index), len(fx)) == (124, 9075), f"G6: dev's fixture set changed: {len(index)} records, {len(fx)}")
    mut = ("grade_fmt_dyn", "len(picked) != 1", "len(picked) < 1")      # a source turn with 2+ picks passes
    picks = [r for r in recs if (r["family"], r["cell"]) == ("OWN", "pick")]
    two = [dict(f, grader=p["grader"], family=r["family"]) for r in picks for p in MG.pick_probes(r)
           for f in MG.BUILD[p["grader"]](r, p) if f["label"] == "source_two"]
    idx = dict(index, **{r["id"]: r for r in picks})
    wrong = lambda: {f["rid"] for f in two if MG.run_fixture(f, idx)["ok"] != f["expect"]}
    base = wrong()                                          # the real graders refuse every source_two reply
    killed = MG.with_source(*mut, wrong)
    check(not base, f"G6: source_two gets the wrong verdict from the real graders on {sorted(base)[:3]}")
    pos = {p["gold_fn"]["options"].index(p["ideal_gold"]) for r in picks for p in r["probes"] if p.get("gold_fn")}
    check(pos == {0, 1, 2} and killed == {r["id"] for r in picks},
          f"G6: source_two does not kill the two-picks source mutant on every OWN pick record (gold first, second "
          f"and third: {sorted(pos)}): {len(killed)} of {len(picks)}")


def main():
    tmp = tempfile.mkdtemp(prefix="rc12lockd_")
    try:
        for t in (t_pack_good, t_pack_refusals, t_s_value, t_g6_coverage):
            t(os.path.join(tmp, t.__name__))
    finally:
        shutil.rmtree(tmp)
    return report("test_lock_d")


if __name__ == "__main__":
    sys.exit(main())
