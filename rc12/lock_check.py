"""RC-12 lock: the machinery manifest and the pre-push checks (prereg s13 S9, s19 steps 8-9). Stdlib, no model.
It never opens sealed/: it reads only public files (the prereg, oodh/HASHES.txt, the dev split, code, and the
sealed session's public outputs in rc12/lock/).

  python3 -B rc12/lock_check.py manifest [--root R]        write rc12/lock/MANIFEST.txt
  python3 -B rc12/lock_check.py check [--root R] [--final] [--prereg P]
      draft mode (default): the checks that hold before the lock (OOD-H and dev hashes quoted right, ignores,
      nothing sealed or private tracked); --final: also the prereg is locked (no DRAFT, (proposal), OPEN or LOCK
      CANDIDATE left; every S5 hash and the generator-seed hash written in it), the sealed outputs are present and
      consistent, the code that built the sealed split is the code at HEAD, and the manifest is current.
--root points at another checkout (the dry run uses a scratch clone). Exit 1 on any failure."""
import argparse
import hashlib
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sealed_kit as K                      # noqa: E402

LOCK = "rc12/lock"
ARCHIVE, S5, GATES, DRAWS, MANIFEST = (f"{LOCK}/RC-12-sealed.tar.enc", f"{LOCK}/SEALED_HASHES.txt",
                                       f"{LOCK}/SEALED_GATES.txt", f"{LOCK}/SEED_ATTEMPTS.txt", f"{LOCK}/MANIFEST.txt")
PUBLISHED = {GATES: "GATES.txt", DRAWS: "SEED_ATTEMPTS.txt"}      # public copies of archived files
DRAFT, FINAL = "prereg/RC-12.draft.txt", "prereg/RC-12.md"
SUFFIX = (".py", ".sh", ".json", ".jsonl")
CORPUS = "corpus/oodh.py"          # the OOD-H reserve code oodh/gate.py imports
MARKERS = [r"\bDRAFT\b", r"\(proposal", r"proposals? until the push", r"\bOPEN\b", r"LOCK CANDIDATE",
           r"NOT LOCKED"]


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tracked(root):
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True).stdout
    return [p for p in out.decode().split("\0") if p]


def manifest_lines(root):
    """sha256 of the machinery: every tracked code, config and data file under the work-copy paths (rc12/ and the
    E001 / E004 code it imports; rc12/logs/ and rc12/lock/ aside), oodh/*.py and HASHES.txt, the prereg (draft or
    final) and the sealed session's public outputs."""
    files = [p for p in tracked(root) if p.startswith(tuple(a + "/" for a in K.ARCHIVE_PATHS)) and p.endswith(SUFFIX)
             and not p.startswith(("rc12/logs/", LOCK + "/"))]
    files += [p for p in tracked(root) if p.startswith("oodh/") and (p.endswith(".py") or p == "oodh/HASHES.txt")]
    files += [p for p in tracked(root) if p == CORPUS]
    files += [p for p in (DRAFT, FINAL, ARCHIVE, S5, GATES, DRAWS) if os.path.isfile(os.path.join(root, p))]
    return [f"sha256 {sha_file(os.path.join(root, p))}  {p}" for p in sorted(set(files))]


def write_manifest(root):
    lines = manifest_lines(root)
    with open(os.path.join(root, MANIFEST), "w") as f:
        f.write("RC-12 lock manifest (prereg s13 S9): sha256 of the code, configs, splits and sealed outputs at the "
                "lock.\nRegenerate: python3 -B rc12/lock_check.py manifest\n" + "\n".join(lines) + "\n")
    print(f"{len(lines)} files in {MANIFEST}")


def oodh_hashes(root):
    text = open(os.path.join(root, "oodh/HASHES.txt")).read()
    part1 = re.search(r"oodh_part1\.jsonl sha256 ([0-9a-f]{64})", text).group(1)
    tree = re.search(r"sorted tree ids .*? sha256 ([0-9a-f]{64})", text).group(1)
    counts = re.search(r"threads (\d+); probes (\d+): H-FACT (\d+), H-ASK (\d+), H-CORR (\d+), H-ABS (\d+)", text)
    return part1, tree, counts


def check(root, final, prereg=None, dryrun=False):
    bad = []
    prereg = prereg or os.path.join(root, FINAL if final else DRAFT)
    text = open(prereg, encoding="utf-8").read()
    part1, tree, c = oodh_hashes(root)
    for h in (part1, tree):
        if h not in text:
            bad.append(f"prereg lacks the OOD-H hash {h[:12]} of oodh/HASHES.txt")
    want = (f"{c.group(1)} threads, {c.group(2)} probes (H-FACT {c.group(3)}, H-ASK {c.group(4)}, "
            f"H-CORR {c.group(5)}, H-ABS {c.group(6)})")
    if want not in re.sub(r"\s+", " ", text):
        bad.append(f"prereg does not quote the OOD-H counts as '{want}'")
    dev = sha_file(os.path.join(root, "rc12/dev/rc12_dev.jsonl"))
    if dev not in re.sub(r"\s+", "", text):
        bad.append(f"prereg does not quote the dev split's sha256 {dev[:12]}")
    if any(subprocess.run(["git", "check-ignore", "-q", p], cwd=root).returncode for p in ("sealed/x", "private/x")):
        bad.append(".gitignore does not cover sealed/ and private/")
    for p in tracked(root):
        if p.startswith(("sealed/", "private/")) or "/sealed/" in p or p.endswith((".key", ".tar")) or \
                re.search(r"(^|/)(content_[a-z_]+\.py|rc12_sealed[^/]*|decon13[^/]*)$", p):
            bad.append(f"tracked file that must never be in git: {p}")
    if final:
        bad += check_final(root, text, dryrun)
    for b in bad:
        print("FAIL", b)
    print(f"lock_check {'final' if final else 'draft'}: {'PASS' if not bad else f'{len(bad)} failures'}")
    return 1 if bad else 0


def check_final(root, text, dryrun=False):
    bad = [f"prereg still carries {m}" for m in MARKERS if re.search(m, text)]
    if os.path.exists(os.path.join(root, DRAFT)):
        bad.append(f"{DRAFT} still exists (the locked file is {FINAL})")
    for p in (ARCHIVE, S5, GATES, DRAWS, MANIFEST):
        if not os.path.isfile(os.path.join(root, p)):
            bad.append(f"missing {p}")
    if bad:
        return bad
    if open(os.path.join(root, ARCHIVE), "rb").read(8) != b"Salted__":
        bad.append(f"{ARCHIVE} is not openssl ciphertext")
    s5 = open(os.path.join(root, S5)).read()
    hashes = re.findall(r"^sha256 ([0-9a-f]{64})  (\S+)", s5, re.M)
    names = {n for _, n in hashes}
    if not re.search(r"^decon13 hashes \d+ ", s5, re.M):
        bad.append(f"{S5} does not publish the S6 list's count")
    enc = re.search(r"^sha256 ([0-9a-f]{64})  RC-12-sealed\.tar\.enc ", s5, re.M)
    if not enc or enc.group(1) != sha_file(os.path.join(root, ARCHIVE)):
        bad.append(f"{ARCHIVE} is not the ciphertext {S5} names")
    split = re.search(r"^split (\w+); .*; S1 (PASS|NOT PASSED)", s5, re.M)
    if not split or (split.groups() != ("sealed", "PASS") and not dryrun):
        bad.append(f"{S5}: the split is not sealed with S1 passed ({split.groups() if split else 'no split line'})")
    draws = open(os.path.join(root, DRAWS)).read()
    last_run = draws if "\ncommit " not in draws else "commit " + draws.rsplit("\ncommit ", 1)[1]
    commits = [re.search(r"^commit ([0-9a-f]{40})", x, re.M) for x in (s5, open(os.path.join(root, GATES)).read(),
                                                                         last_run)]
    if None in commits or len({m.group(1) for m in commits}) != 1:
        bad.append("S5, the gate table and the draw log's last run name different commits")
    seed = re.search(r"^sha256 ([0-9a-f]{64})  generator seed", s5, re.M)
    acc = re.findall(r"^accepted draw \d+; seed sha256 ([0-9a-f]{64})$", last_run, re.M)
    if not seed or acc != [seed.group(1)]:
        bad.append("the draw log's accepted seed is not the generator seed S5 names")
    for n in ["rc12_sealed.jsonl", "expect.json", "generator_seed.txt", "generator", "GATES.txt", "SEED_ATTEMPTS.txt",
              "OVERLAP.txt",
              "decon13_sha256.txt.gz", "oodh_part1.jsonl", "RC-12-sealed.tar"] + \
            [f"content_{m}.py" for m in K.CONTENT_MODULES]:
        if n not in names:
            bad.append(f"{S5} lacks {n}")
    flat = re.sub(r"\s+", "", text)
    bad += [f"prereg lacks the S5 hash of {n}" for h, n in hashes if h not in flat]
    part1 = oodh_hashes(root)[0]
    if "oodh_part1.jsonl" in names and dict((n, h) for h, n in hashes)["oodh_part1.jsonl"] != part1:
        bad.append("the archived OOD-H Part 1 is not the file oodh/HASHES.txt names")
    for path, name in PUBLISHED.items():
        if sha_file(os.path.join(root, path)) != dict((n, h) for h, n in hashes).get(name):
            bad.append(f"{path} is not the {name} the archive holds")
    if not K.gates_clean(open(os.path.join(root, GATES)).read()):
        bad.append(f"{GATES} is not a full, clean S4 gate table")
    built = re.search(r"^commit ([0-9a-f]{40})", s5, re.M)
    paths = list(K.ARCHIVE_PATHS) + ["oodh", CORPUS]
    if not built:
        bad.append(f"{S5} names no commit")
    else:
        diff = subprocess.run(["git", "diff", "--name-only", built.group(1), "HEAD", "--", *paths], cwd=root,
                              capture_output=True, text=True)
        changed = [p for p in diff.stdout.split() if p.endswith(SUFFIX) and not p.startswith(("rc12/logs/", LOCK + "/"))
                   or p.startswith("oodh/") and p.endswith(".py") or p == CORPUS]
        if diff.returncode or changed:
            bad.append(f"code changed since the sealed build's commit: {changed[:5] or diff.stderr.strip()}")
    cur = manifest_lines(root)
    have = [ln for ln in open(os.path.join(root, MANIFEST)).read().splitlines() if ln.startswith("sha256 ")]
    if have != cur:
        bad.append(f"{MANIFEST} is stale (regenerate it after the last change)")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["manifest", "check"])
    ap.add_argument("--root", default=os.path.dirname(HERE))
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--prereg")
    ap.add_argument("--dryrun", action="store_true", help="a dry-run lock: accept a dryrun split and a failed S1")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    if a.cmd == "manifest":
        write_manifest(root)
    else:
        sys.exit(check(root, a.final, a.prereg, a.dryrun))


if __name__ == "__main__":
    main()
