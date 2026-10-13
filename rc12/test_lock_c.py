"""RC-12 lock tests, part 3: lock_check (draft and final) and lock_finalize, on fixture git repositories and a copy
of the real prereg draft in a temporary folder, and the S1 findings (rc12/notes.txt STEP 15). Stdlib, no model.
  python3 -B test_lock_c.py     (exit 1 on any failure)"""
import contextlib
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.pop("RC12_SPLIT", None)
import lock_check as LC                      # noqa: E402
import sealed_overlap as SO                 # noqa: E402
import split_content as SPC                 # noqa: E402
import sealed_kit as K                       # noqa: E402
from lockfix import check, full_gates, write, git, repo, quiet, report   # noqa: E402


def lock_fixture(tmp):
    r = repo(os.path.join(tmp, "l"))
    dev = '{"a": 1}\n'
    write(r, "rc12/dev/rc12_dev.jsonl", dev)
    write(r, "oodh/HASHES.txt", f"sealed/oodh/oodh_part1.jsonl sha256 {'a' * 64}\nsorted tree ids (x) sha256 "
          f"{'b' * 64}\nthreads 150; probes 415: H-FACT 226, H-ASK 96, H-CORR 1, H-ABS 92\n")
    body = (f"oodh {'a' * 64} tree {'b' * 64}; 150 threads, 415 probes (H-FACT 226, H-ASK 96, H-CORR 1, H-ABS\n"
            f"  92); dev {hashlib.sha256(dev.encode()).hexdigest()}\n")
    write(r, LC.DRAFT, "RC-12 PRE-REGISTRATION: DRAFT, NOT LOCKED\n" + body)
    git(r, "add", "-A")
    git(r, "commit", "-qm", "draft")
    return r, body


def make_final(r, built, body, gates=None, draws=None, split="sealed; records 1; twelve-turn 1; S1 PASS",
               gates_commit=None, seed="c" * 64):
    """a consistent locked tree: S5 naming every archived file (GATES and the draw log with their real hashes), the
    split and S1 line, the ciphertext's hash, their public copies, the final prereg quoting every S5 hash, and the
    manifest. The keyword arguments spoil one part at a time."""
    gates = (gates or full_gates()).replace("commit x split sealed", f"commit {gates_commit or built} split sealed")
    draws = draws or f"commit {built} split sealed: seed draws\naccepted draw 2; seed sha256 {seed}\n"
    files = {"GATES.txt": gates, "SEED_ATTEMPTS.txt": draws}
    names = ["rc12_sealed.jsonl", "expect.json", "generator_seed.txt", "decon13_sha256.txt.gz", "RC-12-sealed.tar",
             "OVERLAP.txt"] + [f"content_{m}.py" for m in K.CONTENT_MODULES]
    archive = b"Salted__" + os.urandom(64)
    s5 = [f"commit {built} (x)", f"split {split}"]
    s5 += [f"sha256 {hashlib.sha256(n.encode()).hexdigest()}  {n}" for n in names]
    s5 += [f"sha256 {hashlib.sha256(v.encode()).hexdigest()}  {n}" for n, v in files.items()]
    s5 += [f"sha256 {'a' * 64}  oodh_part1.jsonl", f"sha256 {'c' * 64}  generator seed", "decon13 hashes 123 (x)",
           f"sha256 {hashlib.sha256(archive).hexdigest()}  RC-12-sealed.tar.enc (the ciphertext)"]
    write(r, LC.S5, "\n".join(s5) + "\n")
    write(r, LC.GATES, gates)
    write(r, LC.DRAWS, draws)
    open(os.path.join(r, LC.ARCHIVE), "wb").write(archive)
    final = "RC-12 PRE-REGISTRATION: LOCKED\n" + body + "\n".join(s5) + "\n"
    write(r, LC.FINAL, final)
    quiet(LC.write_manifest, r)
    return final


def fails_of(r, dryrun=False):
    """the FAIL lines of a final check (the function prints them)."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        LC.check(r, True, None, dryrun)
    return [ln[5:] for ln in out.getvalue().splitlines() if ln.startswith("FAIL ")]


def only(r, frag, why):
    """exactly one failure, and it is the one this case is about (so no other check can stand in for it)."""
    got = fails_of(r)
    check(len(got) == 1 and frag in got[0], f"lock_check: {why}: failures {got}")


def t_lock_check(tmp):
    r, body = lock_fixture(tmp)
    check(quiet(LC.check, r, False) == 0, "lock_check: a good draft fails")
    write(r, LC.DRAFT, body.replace("415 probes", "417 probes"))
    check(quiet(LC.check, r, False) == 1, "lock_check: stale OOD-H counts pass")
    write(r, LC.DRAFT, "RC-12 PRE-REGISTRATION: DRAFT, NOT LOCKED\n" + body)
    write(r, "sealed/x.txt", "secret\n")
    git(r, "add", "-f", "sealed/x.txt")
    check(quiet(LC.check, r, False) == 1, "lock_check: a tracked sealed file passes")
    git(r, "rm", "-q", "--cached", "sealed/x.txt")
    built = git(r, "rev-parse", "HEAD")
    git(r, "mv", LC.DRAFT, LC.FINAL)
    make_final(r, built, body)
    check(fails_of(r) == [], f"lock_check: a good final tree fails: {fails_of(r)}")
    quick = "commit x split sealed\ntest_gens              exit   0 PASS  x\nfailing steps: 0\n"
    cases = [(dict(gates=full_gates("validate_e2e_G1_G4")), None, "full, clean S4 gate table", "a failing gate table"),
             (dict(gates=quick), None, "full, clean S4 gate table", "a quick-gates-only table"),
             ({}, lambda f: f + "a (proposal) left\n", "prereg still carries", "a marker"),
             ({}, lambda f: f.replace("c" * 64, "d" * 64), "lacks the S5 hash of generator", "a missing S5 hash"),
             ({}, lambda f: f.replace(body, body.replace("H-ABS\n  92", "H-ABS\n  91")), "OOD-H counts",
              "wrong OOD-H counts"),
             (dict(split="dryrun; records 1; twelve-turn 1; S1 NOT PASSED (a dry-run split)"), None,
              "the split is not sealed", "a dry-run split without --dryrun"),
             (dict(gates_commit="f" * 40), None, "different commits", "a gate table of another commit"),
             (dict(seed="e" * 64), None, "accepted seed is not", "a draw log accepting another seed")]
    for kw, text, frag, why in cases:
        final = make_final(r, built, body, **kw)
        if text is not None:
            write(r, LC.FINAL, text(final))
            quiet(LC.write_manifest, r)
        only(r, frag, why)
    make_final(r, built, body)
    write(r, LC.S5, "".join(x for x in open(os.path.join(r, LC.S5)).readlines() if not x.startswith("decon13")))
    quiet(LC.write_manifest, r)
    only(r, "S6 list's count", "an S5 file without the S6 count")
    make_final(r, built, body)
    write(r, LC.GATES, open(os.path.join(r, LC.GATES)).read() + "\n")
    quiet(LC.write_manifest, r)
    only(r, "is not the GATES.txt the archive holds", "a public gate table unlike the archived one")
    make_final(r, built, body, split="dryrun; records 1; twelve-turn 1; S1 NOT PASSED (a dry-run split)")
    check(fails_of(r, dryrun=True) == [], "lock_check: --dryrun refuses a dry-run split")
    make_final(r, built, body)
    open(os.path.join(r, LC.ARCHIVE), "ab").write(b"x")
    quiet(LC.write_manifest, r)
    only(r, "is not the ciphertext", "a ciphertext S5 does not name")
    make_final(r, built, body)
    write(r, "rc12/content_pools_vals.py", "X = 1\n")
    git(r, "add", "rc12/content_pools_vals.py")
    quiet(LC.write_manifest, r)
    only(r, "must never be in git: rc12/content_pools_vals.py", "a tracked content file")
    git(r, "rm", "-q", "--cached", "rc12/content_pools_vals.py")
    os.remove(os.path.join(r, "rc12/content_pools_vals.py"))
    final = make_final(r, built, body)
    write(r, LC.FINAL, final + "late edit\n")
    only(r, "is stale", "a stale manifest")
    for sub, path, text in (("", "rc12/a.py", "X = 3\n"), ("o", "oodh/gate.py", "G = 2\n"),
                            ("q", "corpus/oodh.py", "R = 2\n"), ("k", "rc12/lockstep.py", "S = 2\n")):
        if sub:          # OOD-H's code, its reserve code and code merely named rc12/lock* are held to the build too
            r, body = lock_fixture(os.path.join(tmp, sub))
            built = git(r, "rev-parse", "HEAD")
            git(r, "mv", LC.DRAFT, LC.FINAL)
        make_final(r, built, body)
        write(r, path, text)
        git(r, "add", "-A")
        git(r, "commit", "-qm", f"{path} after the sealed build")
        quiet(LC.write_manifest, r)
        only(r, f"code changed since the sealed build's commit: ['{path}']", f"{path} changed after the build")


def t_finalize(tmp):
    if not os.path.exists(os.path.join(HERE, "..", LC.DRAFT)):     # after the lock: the draft is gone, so check the
        text = open(os.path.join(HERE, "..", LC.FINAL)).read()      # locked file finalize wrote instead
        check(not [m for m in LC.MARKERS if re.search(m, text)] and "\n20. LOCK RECORD (" in text,
              "finalize (post-lock): the locked prereg carries a marker or lacks its lock record")
        return
    r = os.path.join(tmp, "f")
    write(r, "prereg/RC-12.md", open(os.path.join(HERE, "..", LC.DRAFT)).read())
    write(r, LC.S5, f"commit {'e' * 40} (x)\nsha256 {'f' * 64}  rc12_sealed.jsonl\n")
    write(r, LC.DRAWS, "draw 1: PASS; ceiling S7 35.1\naccepted draw 1\nsuperseded: content fix; void\ndraw 2: FAIL\n"
          "draw 3: PASS; ceiling S7 36.42; e2e fails: none\naccepted draw 3\n")
    run = lambda: subprocess.run([sys.executable, "-B", os.path.join(HERE, "lock_finalize.py"), "--date", "2026-10-13",
                                  "--sealed-date", "2026-10-12", "--root", r], capture_output=True, text=True)
    p = run()
    text = open(os.path.join(r, "prereg/RC-12.md")).read()
    left = [m for m in LC.MARKERS if re.search(m, text)]
    check(p.returncode == 0 and not left, f"finalize: markers left {left} ({p.stdout[-200:]}{p.stderr[-200:]})")
    check("20. LOCK RECORD (2026-10-13" in text and f"sha256 {'f' * 64}  rc12_sealed.jsonl" in text and
          "draw 3 was accepted, of 3 draws in all, 1 restart(s)" in " ".join(text.split()) and "is 36.42." in text,
          "finalize: the lock record is missing or wrong")
    p = run()
    check(p.returncode != 0 and open(os.path.join(r, "prereg/RC-12.md")).read() == text,
          "finalize: a second run (a drifted draft) is not refused unchanged")
    write(r, "prereg/RC-12.md", open(os.path.join(HERE, "..", LC.DRAFT)).read())
    write(r, LC.DRAWS, "draw 1: PASS; ceiling S7 1\naccepted draw 1\ndraw 2: PASS; ceiling S7 2\naccepted draw 2\n")
    check(run().returncode != 0, "finalize: two accepted draws without a superseded line between them pass")
    write(r, LC.DRAWS, "draw 1: PASS; e2e fails: none\naccepted draw 1\n")
    p = run()
    check(p.returncode != 0 and "carries no shortcut ceiling" in p.stdout + p.stderr,
          "finalize: an accepted draw without its shortcut ceiling is not refused by name")


def t_overlap(tmp):
    dev = SO.dev_content()
    same = SO.findings(dev, dev, ["one two three four five six"], ["one two three four five six"])[:3]
    check(all(same), f"overlap: dev against itself not caught: {[len(x) for x in same]}")
    def q(x):                                                            # every string changed
        if isinstance(x, str):
            return "Q" + x
        if isinstance(x, dict):
            return {q(k): q(v) for k, v in x.items()}
        return type(x)(q(v) for v in x) if isinstance(x, (list, tuple)) else x
    new = {m: {n: q(v) for n, v in c.items()} for m, c in dev.items()}
    new["pools_vals"]["COLOUR"][0] = "red"                                  # one shared value
    new["pools_vals"]["WEEKDAY"] = dev["pools_vals"]["WEEKDAY"]             # a closed class, allowed
    new["fam_persist"]["INSTR"]["caps"] = dev["fam_persist"]["INSTR"]["caps"]
    new["pools_misc"]["ROLE_P"] = dev["pools_misc"]["ROLE_P"]           # a short question kept from dev
    new["pools_misc"]["K_ITEMS"] = dev["pools_misc"]["K_ITEMS"]         # its pinned kind and closed answers
    a, b, c, d = SO.findings(dev, new, ["the cat sat on the mat today"], ["a cat sat on the mat again", "x y"])
    check(a == ["pools_vals.COLOUR: red", "pools_vals.COLOUR: red (a dev value of another pool)"],
          f"overlap (a): {a[:4]}")
    check(b == ["cat sat on the mat"], f"overlap (b): {b}")
    check(c == [" ".join(SO.words(dev["fam_persist"]["INSTR"]["caps"]))], f"overlap (c): {c[:3]}")
    answers = {a.lower() for k in dev["pools_misc"]["K_ITEMS"] for a in k[2]}
    answers -= {x.lower() for x in dev["pools_vals"]["LANGUAGE"]}        # a closed class is never flagged
    want_d = sorted({dev["pools_misc"]["ROLE_P"].lower()} | {k[1].lower() for k in dev["pools_misc"]["K_ITEMS"]} |
                    answers)
    check(d == want_d, f"overlap (d): {d[:6]} (want the kept question, the K countries and capitals only)")
    valid = {m: {n: q(v) for n, v in c.items()} for m, c in dev.items()}     # what a valid sealed set keeps:
    for m, n in (("pools_vals", x) for x in SO.CLOSED):                       # the closed classes, pinned fields,
        valid[m][n] = dev[m][n]                                               # filler labels, avoid lists and
    for m, n in (("fam_bind", "BIND_AVOID"), ("fam_own_topic", "OWN_AVOID"),     # BIND's name-probe object words
                 ("fam_role_lookup", "LOOKUP_AVOID")):
        valid[m][n] = dev[m][n]
    for m, n in (("pools_fill_a", "FILLERS_A"), ("pools_fill_b", "FILLERS_B")):
        valid[m][n] = [(a, b, lab) for (a, b, _), (_, _, lab) in zip(valid[m][n], dev[m][n])]
    valid["fam_bind"]["BIND_NAME"]["objw"] = dev["fam_bind"]["BIND_NAME"]["objw"]
    for path, fields in SPC.PINS.items():
        m, n = path.split(".")
        rows = valid[m][n].values() if isinstance(dev[m][n], dict) else valid[m][n]
        devrows = dev[m][n].values() if isinstance(dev[m][n], dict) else dev[m][n]
        for k, (row, drow) in enumerate(zip(list(rows), devrows)):
            if isinstance(row, dict):
                row.update({f: drow[f] for f in fields})
            else:
                valid[m][n][k] = tuple(drow[f] if f in fields else x for f, x in enumerate(row))
    got = SO.findings(dev, valid, [], [])
    check(not any(got), f"overlap: a valid sealed set (only the fixed slots kept) fails S1: {[g[:3] for g in got]}")


def main():
    tmp = tempfile.mkdtemp(prefix="rc12lockc_")
    try:
        for t in (t_lock_check, t_finalize, t_overlap):
            t(tmp)
    finally:
        shutil.rmtree(tmp)
    return report("test_lock_c")


if __name__ == "__main__":
    sys.exit(main())
