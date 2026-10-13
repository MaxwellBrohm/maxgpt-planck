"""RC-12 lock tests, part 2: the work copy, the crypto kit and the seed draws, on a fixture git repository and files
in a temporary folder (rc12/notes.txt STEP 15). Stdlib and openssl, no model. Part 3: test_lock_c.py.
  python3 -B test_lock_b.py     (exit 1 on any failure)"""
import contextlib
import gzip
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
import sealed_crypto as CR                   # noqa: E402
import sealed_kit as K                       # noqa: E402
import sealed_pack as PK                     # noqa: E402
import sealed_search as SS                   # noqa: E402
from lockfix import check, dies, git, write, repo, quiet, report   # noqa: E402


def t_kit(tmp):
    r = repo(tmp)
    work = os.path.join(tmp, "work")
    quiet(K.prepare, work, "HEAD", repo=r)
    check(os.path.isfile(os.path.join(work, "rc12/a.py")) and K.commit_of(work) == git(r, "rev-parse", "HEAD"),
          "kit: prepare did not extract the commit and mark it")
    check(K.check(work, repo=r) == [], f"kit: a fresh copy differs: {K.check(work, repo=r)}")
    for rel in ("rc12/content_pools_vals.py", "rc12/logs/x.txt", "GATES.txt", "OVERLAP.txt", "expect.json"):
        write(work, rel, "ok\n")
    check(K.check(work, repo=r) == [], f"kit: allowed outputs flagged: {K.check(work, repo=r)}")
    write(work, "rc12/a.py", "X = 2\n")
    write(work, "rc12/b.py", "Y = 1\n")
    check(sorted(K.check(work, repo=r)) == ["differs rc12/a.py", "extra rc12/b.py"],
          "kit: a changed or extra file passes")
    check(dies(K.check_place, os.path.join(r, "rc12", "w"), r),
          "kit: a work copy inside the repo outside sealed/ passes")
    check(not dies(K.check_place, os.path.join(r, "sealed", "w"), r), "kit: a work copy under sealed/ refused")
    check(dies(K.prepare, work, "HEAD", repo=r), "kit: prepare into a non-empty folder passes")
    check(K.public_line("build_stdout", '{"turns": [{"text": "secret"}]}', True) == "the build reproduces the items",
          "kit: the build's output (the items) reaches the public table")
    check("secret" not in K.public_line("x", 'FAIL "secret" here', False) and
          K.public_line("x", "0 failures over 640 records", True) == "0 failures over 640 records",
          "kit: a public gate line leaks quoted text or withholds a plain summary")
    check(K.public_line("x", "rows {see} [x] here", True).startswith("(last line withheld"),
          "kit: a public gate line with braces or brackets is published")
    check(K.public_line("x", "0 failures over zorvex records", True, {"zorvex"}).startswith("(last line withheld")
          and SS.quiet_text("G2 X: family zorvex=0.5", {"zorvex"}) == "G2 X: family (withheld)=0.5",
          "kit: a sealed content word reaches a public line")
    import sealed_overlap as SO
    import sealed_public as PUB
    devw = SO.words(SO.dev_content()["pools_vals"]["COLOUR"][0])[0].lower()
    write(work, "rc12/content_pools_vals.py", f'"""doc."""\nX = ["Zorvex zab", "{devw} the", {{"kwem": "it"}}]\n')
    got = PUB.content_words(work)
    check(got == {"zorvex", "zab", "kwem"}, f"kit: content_words is not the content's own non-dev words: {got}")
    os.remove(os.path.join(work, "rc12/content_pools_vals.py"))
    os.remove(os.path.join(work, "rc12/b.py"))
    write(work, "rc12/a.py", "X = 1\n")
    os.makedirs(os.path.join(work, "rc12", "__pycache__"))
    os.symlink(os.path.join(work, "rc12/a.py"), os.path.join(work, "rc12", "link.py"))
    got = K.check(work, repo=r)
    check(sorted(got) == ["bytecode or link rc12/__pycache__", "bytecode or link rc12/link.py"],
          f"kit: bytecode or a symlink passes the copy check ({got})")
    work2 = os.path.join(tmp, "work2")
    quiet(K.prepare, work2, "HEAD", repo=r)
    seed = os.path.join(tmp, "kseed.txt")
    write(tmp, "kseed.txt", "ab" * 16 + "\n")
    check(dies(quiet, K.gates, work, seed, None, "dryrun", repo=r), "kit: gates ran on a work copy unlike its commit")
    real, K.GATES = K.GATES, [("fail", ["fail.py"]), ("mark", ["mark.py"])]
    real_check, K.check = K.check, lambda w, repo=r: []
    try:
        n = quiet(K.gates, work2, seed, None, "dryrun", stop=True)
        check(n == 1 and not os.path.exists(os.path.join(work2, "rc12", "ran")), "kit: gates did not stop early")
        n = quiet(K.gates, work2, seed, None, "dryrun")
        table = open(os.path.join(work2, "GATES.txt")).read()
        check(n == 1 and os.path.exists(os.path.join(work2, "rc12", "ran")) and "failing steps: 1" in table,
              "kit: gates skipped a step")
        check(f"input sha256 {K.sha256(seed)}  generator_seed.txt" in table, "kit: the gate table lacks its inputs")
    finally:
        K.GATES, K.check = real, real_check


def t_crypto(tmp):
    r = repo(os.path.join(tmp, "c"))
    s = os.path.join(tmp, "seed.txt")
    quiet(CR.seed, s)
    check(re.fullmatch(r"[0-9a-f]{64}\n", open(s).read()) and os.stat(s).st_mode & 0o777 == 0o600, "crypto: seed form")
    check(dies(CR.seed, s), "crypto: a seed file overwritten")
    check(dies(CR.keygen, os.path.join(r, "k.key")), "crypto: a key inside a git work tree accepted")
    key, key2 = os.path.join(tmp, "keys", "k.key"), os.path.join(tmp, "keys", "k2.key")
    quiet(CR.keygen, key)
    quiet(CR.keygen, key2)
    check(re.fullmatch(r"[0-9a-f]{64}\n", open(key).read()) and os.stat(key).st_mode & 0o777 == 0o600,
          "crypto: key form")
    files = [("rc12_sealed.jsonl", b'{"a": 1}\n'), ("expect.json", b"{}")]
    tar = PK.det_tar(files)
    check(tar == PK.det_tar(files), "crypto: the tar is not deterministic")
    plain, enc = os.path.join(tmp, CR.TAR), os.path.join(tmp, "a.enc")
    open(plain, "wb").write(tar)
    lines = [f"sha256 {CR.sha(d)}  {n}" for n, d in files] + [f"sha256 {CR.sha(tar)}  {CR.TAR} (plain)",
                                                              f"sha256 {CR.sha(b'x')}  generator seed"]
    h = os.path.join(tmp, "S5.txt")
    open(h, "w").write("\n".join(lines[:2]) + "\n")
    check(dies(quiet, CR.encrypt, plain, key, enc, h), "crypto: a tar S5 does not name was encrypted")
    open(h, "w").write("\n".join(lines) + "\n")
    quiet(CR.encrypt, plain, key, enc, h)
    ctext = open(enc, "rb").read()
    check(ctext[:8] == b"Salted__" and tar not in ctext, "crypto: not ciphertext")
    check(f"sha256 {CR.sha(ctext)}  {CR.ENC_NAME} (the ciphertext)" in open(h).read(),
          "crypto: the ciphertext's hash was not appended to S5")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        CR.keygen(os.path.join(tmp, "keys", "k3.key"))
    check(open(os.path.join(tmp, "keys", "k3.key")).read().strip() not in out.getvalue(),
          "crypto: keygen printed the key")
    check(quiet(CR.verify, enc, key, h) == 0, "crypto: verify of a good archive fails")
    check(quiet(CR.verify, enc, key2, h) == 1, "crypto: verify with the wrong key passes")
    bent = os.path.join(tmp, "bent.enc")
    open(bent, "wb").write(ctext[:40] + bytes([ctext[40] ^ 1]) + ctext[41:])
    check(quiet(CR.verify, bent, key, h) == 1, "crypto: verify of a tampered ciphertext passes")
    base = open(h).read()
    open(h, "w").write(base.replace(CR.sha(b"{}"), "0" * 64))
    check(quiet(CR.verify, enc, key, h) == 1, "crypto: verify against a wrong S5 hash passes")
    open(h, "w").write(base + f"sha256 {'1' * 64}  missing.txt\n")
    check(quiet(CR.verify, enc, key, h) == 1, "crypto: a file named in S5 but not archived passes")
    work = os.path.join(tmp, "w")
    write(work, K.ITEMS, '{"turns": [{"text": "A B, c d e f g h i j k l m n."}]}\n')
    write(tmp, "o.jsonl", '{"turns": [{"text": "x", "ideal": "q r s t u v w x y z aa bb cc"}]}\n')
    out = os.path.join(tmp, "private", "d.gz")
    quiet(CR.ngrams, work, os.path.join(tmp, "o.jsonl"), out)
    got = gzip.decompress(open(out, "rb").read()).decode().splitlines()[1:]
    want = sorted(CR.sha(x.encode()) for x in ("a b c d e f g h i j k l m", "b c d e f g h i j k l m n",
                                               "q r s t u v w x y z aa bb cc"))
    check(got == want, "crypto: 13-gram hashes differ from the known answer")


def t_search(tmp):
    work = os.path.join(tmp, "sw")
    write(work, K.MARKER, "commit abc\n")
    write(work, K.ITEMS, '{"turns": [{"kind": "D", "asks": true}]}\n')
    calls, logp = {"n": 0, "builds": 0}, os.path.join(tmp, "draws.txt")
    real = (K.build, K.gates, SS.summary)

    def build(w, f, s):
        calls["builds"] += 1
        if calls["builds"] == 2:                  # the second draw's build refuses its seed
            raise SystemExit("builder refused")

    def gates(w, f, e, s, only, stop):
        calls["n"] += 1
        calls["only"] = set(only)
        if calls["n"] == -50:                     # the kit refuses mid-search (the copy changed)
            raise SystemExit("the work copy differs from its commit")
        if calls["n"] == 2:                       # the third draw's gates time out
            raise subprocess.TimeoutExpired("gate", 1)
        return 0 if calls["n"] == 3 else 2
    K.build, K.gates = build, gates
    SS.summary = lambda w: "ceiling S7 1.0"
    try:
        seed = os.path.join(tmp, "accepted.txt")
        check(dies(quiet, SS.search, work, seed, [1, 2, 3], 5, "dryrun", os.path.join(work, "draws.txt")),
              "search: a draw log inside the work copy accepted")
        got = quiet(SS.search, work, seed, [1, 2, 3], 5, "dryrun", logp)
        log = open(logp).read()
        check(got == 4 and os.path.isfile(seed) and not [x for x in os.listdir(tmp) if ".try" in x],
              "search: the accepted seed not kept or a discarded seed left")
        check(log.count("FAIL") == 3 and "draw 2: FAIL; the build refused this seed" in log and
              "draw 3: FAIL; a gate timed out" in log and "draw 4: PASS" in log and "accepted draw 4" in log,
              f"search: log {log!r}")
        check(open(os.path.join(work, "expect.json")).read() == '{"filler_labels": {"question": 1, "request": 2, '
              '"statement": 3}, "asks": [1, 0]}', "search: expect.json not from the build")
        calls["n"] = -100
        open(os.path.join(tmp, "s2.txt.try9"), "w").write("left by a cut\n")
        check(dies(quiet, SS.search, work, os.path.join(tmp, "s2.txt"), [1, 2, 3], 2, "dryrun", logp),
              "search: a new run on a log with draws allowed without --supersede")
        check(dies(quiet, SS.search, work, os.path.join(tmp, "s2.txt"), [1, 2, 3], 2, "dryrun", logp,
                   'fix "a quoted item"'), "search: a restart reason with quoted text accepted")
        got = quiet(SS.search, work, os.path.join(tmp, "s2.txt"), [1, 2, 3], 2, "dryrun", logp, "S1 (b) pools_fill_a")
        log = open(logp).read()
        check(got == 0 and "no seed passed in 2 draws" in log and len(re.findall(r"^draw \d+:", log, re.M)) == 6
              and "draw 6: FAIL" in log and log.count("superseded: S1 (b) pools_fill_a") == 1,
              "search: the draw cap, the numbering or the supersede line is wrong")
        check(not os.path.exists(os.path.join(tmp, "s2.txt.try9")), "search: a cut search's trial seed left")
        check(calls["only"] == {"test_gens", "audit_rules_G10", "validate_e2e_G1_G4", "mutation_audit_1",
                                "mutation_audit_2", "mutation_graders_G6"},
              f"search: the draw gates are not prereg s13 S3 (iv)'s: {calls['only']}")
        calls["n"] = -51
        check(dies(quiet, SS.search, work, os.path.join(tmp, "s3.txt"), [1, 2, 3], 2, "dryrun", logp, "stop test"),
              "search: a kit refusal mid-search does not stop the search")
        check("draw 7: STOPPED" in open(logp).read() and not [x for x in os.listdir(tmp) if "s3.txt" in x],
              "search: a stopped draw is not logged or its trial seed is left")
        check(SS.quiet_text('FAIL G1 rc12-sealed-x: reply "my secret text" here') ==
              "FAIL G1 rc12-sealed-x: reply '(withheld)' here", "search: quoted text not withheld")
    finally:
        K.build, K.gates, SS.summary = real


def t_summary(tmp):
    """the draw line withholds sealed words in the free-text parts only: a sealed word that is also one of the
    line's fixed words ("ceiling") must not blank the line lock_finalize reads."""
    work = os.path.join(tmp, "sum")
    write(work, "gate_logs/validate_e2e_G1_G4.txt", "Level R shortcut ceiling (best SHORTCUT fake): S7 36.4\n"
          "FAIL G2 zorvex family 0.31\n")
    write(work, "gate_logs/audit_rules_G10.txt", "GATE (G10): FAIL zorvex row\n")
    real = SS.content_words
    SS.content_words = lambda w: {"ceiling", "zorvex", "none", "killed", "fail"}
    try:
        got = SS.summary(work)
    finally:
        SS.content_words = real
    check("ceiling S7 36.4;" in got and "e2e fails: G2 (withheld) family 0.31;" in got and
          "G10 FAIL (withheld) row;" in got and "zorvex" not in got, f"summary: {got}")


def main():
    tmp = tempfile.mkdtemp(prefix="rc12lock_")
    try:
        for t in (t_kit, t_crypto, t_search, t_summary):
            t(tmp)
    finally:
        shutil.rmtree(tmp)
    return report("test_lock_b")


if __name__ == "__main__":
    sys.exit(main())
