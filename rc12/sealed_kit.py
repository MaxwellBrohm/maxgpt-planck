"""RC-12 sealed-session kit, part 1: the work copy, the build and the gates (prereg s13 S1-S4). Stdlib, no model.

Everything runs in a WORK COPY: a git archive of one commit's rc12/ (and the E001 and E004 code it imports) under the
gitignored sealed/ (or, for a dry run, outside the repository). The content files go into its rc12/ and its
rc12/dev/rc12_dev.jsonl holds the split it builds, so every gate reads it unchanged; nothing else may differ from the
commit (check), so S9's code hashes describe the code that built and gated the sealed items.

  python3 -B rc12/sealed_kit.py prepare WORK [--commit REV]
  python3 -B rc12/sealed_kit.py install WORK CONTENT_DIR       (copies the content_*.py files, all of them)
  python3 -B rc12/sealed_kit.py check WORK                     (the copy equals its commit, content aside)
  python3 -B rc12/sealed_kit.py build WORK --seed-file F [--split sealed]
  python3 -B rc12/sealed_kit.py gates WORK --seed-file F --expect E.json [--split sealed] [--only a,b]
  python3 -B rc12/sealed_kit.py overlap WORK                   (s13 S1 against this repository's dev split)
Other parts: sealed_crypto.py (seed, key, 13-grams, encryption), sealed_overlap.py, sealed_search.py, sealed_pack.py."""
import argparse
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile

from sealed_public import content_words, public_line        # noqa: F401 (K.public_line for callers)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ARCHIVE_PATHS = ["rc12", "experiments/E001_battery_and_probes/code", "experiments/E004_general_updating/code"]
CONTENT_MODULES = ["pools_vals", "pools_slots", "pools_misc", "pools_fill_a", "pools_fill_b", "balance", "fam_bind",
                   "fam_persist", "fam_twohop", "fam_own_topic", "fam_role_lookup"]
MARKER = ".rc12_workcopy"
ITEMS = "rc12/dev/rc12_dev.jsonl"
# allowed to differ from the commit inside a work copy: the items, gate outputs, bytecode, the content modules
FREE = ("rc12/logs/", "rc12/runs/", "rc12/dev/rc12_dev.jsonl")
OUTPUTS = (MARKER, "GATES.txt", "expect.json", "OVERLAP.txt")     # the kit's own files at the copy's root
# the S4 gate list: G1-G11 and the generator, grader, audit and end-to-end mutation tests (s8, s13 S4)
GATES = [("build_stdout", ["build_dev.py", "--stdout"]), ("test_gens", ["test_gens.py"]),
         ("audit_rules_G10", ["audit_rules.py"]), ("audit_graders_G11", ["audit_graders.py"]),
         ("validate_e2e_G1_G4", ["validate_e2e.py"]), ("validate_machinery", ["validate_machinery.py"]),
         ("mutation_graders_G6", ["mutation_graders.py"]), ("mutation_gens", ["mutation_gens.py"]),
         ("mutation_audit_1", ["mutation_audit.py", "--part", "1/2"]),
         ("mutation_audit_2", ["mutation_audit.py", "--part", "2/2"])] + \
        [(f"mutation_e2e_{i}", ["mutation_e2e.py", "--part", f"{i}/8"]) for i in range(1, 9)]


def die(msg):
    sys.exit(f"sealed_kit: {msg}")


def inside(path, root):
    path, root = os.path.realpath(path), os.path.realpath(root)
    return path == root or path.startswith(root + os.sep)


def check_place(work, repo=REPO):
    """a work copy lives under REPO/sealed/ (gitignored) or outside the repository (a dry run); never elsewhere."""
    if inside(work, repo) and not inside(work, os.path.join(repo, "sealed")):
        die(f"{work}: a work copy inside the repository must be under sealed/ (gitignored)")


def git(*args, cwd=REPO, binary=False):
    out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=True)
    return out.stdout if binary else out.stdout.decode()


def commit_of(work):
    path = os.path.join(work, MARKER)
    if not os.path.isfile(path):
        die(f"{work} is not a work copy (no {MARKER})")
    return open(path).read().split()[1]


def prepare(work, rev, repo=REPO):
    check_place(work, repo)
    if os.path.exists(work) and os.listdir(work):
        die(f"{work} exists and is not empty")
    sha = git("rev-parse", "--verify", rev + "^{commit}", cwd=repo).strip()
    tar = git("archive", "--format=tar", sha, *ARCHIVE_PATHS, cwd=repo, binary=True)
    os.makedirs(work, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(work, filter="data")
    with open(os.path.join(work, MARKER), "w") as f:
        f.write(f"commit {sha}\n")
    print(f"work copy of {sha[:12]} at {work}")


def install(work, src):
    commit_of(work)
    names = sorted(n for n in os.listdir(src) if n.startswith("content_"))
    want = sorted(f"content_{m}.py" for m in CONTENT_MODULES)
    if names != want:
        die(f"{src} must hold exactly {want}; has {names}")
    for n in names:
        shutil.copyfile(os.path.join(src, n), os.path.join(work, "rc12", n))
    print(f"installed {len(names)} content modules")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(work, repo=REPO):
    """every file of the commit is in the copy unchanged (FREE aside); the only extra files are content modules,
    FREE outputs and the kit's OUTPUTS; no bytecode, __pycache__ or symlink anywhere (a planted .pyc is imported
    even under -B). Returns the list of problems."""
    sha = commit_of(work)
    tracked = git("ls-tree", "-r", "--name-only", sha, "--", *ARCHIVE_PATHS, cwd=repo).split("\n")
    tracked = [p for p in tracked if p]
    bad = []
    for p in tracked:
        here = os.path.join(work, p)
        if p.startswith(FREE):
            continue
        if not os.path.isfile(here):
            bad.append(f"missing {p}")
        elif hashlib.sha256(git("show", f"{sha}:{p}", cwd=repo, binary=True)).hexdigest() != sha256(here):
            bad.append(f"differs {p}")
    allowed = set(tracked) | {f"rc12/content_{m}.py" for m in CONTENT_MODULES} | set(OUTPUTS)
    for root, dirs, files in os.walk(work):
        for n in dirs + files:
            full = os.path.join(root, n)
            p = os.path.relpath(full, work)
            if os.path.islink(full) or n == "__pycache__" or n.endswith((".pyc", ".pyo")):
                bad.append(f"bytecode or link {p}")
            elif n in files and p not in allowed and not p.startswith(FREE) and not p.startswith("gate_logs/"):
                bad.append(f"extra {p}")
    return bad


def env_for(seed_file, split, expect=None):
    seed = open(seed_file).read().strip()
    env = dict(os.environ, RC12_SPLIT=split, RC12_GEN_SEED=seed, PYTHONDONTWRITEBYTECODE="1", RC12_MAX_WORKERS="4")
    if expect:
        env["RC12_EXPECT"] = os.path.abspath(expect)
    return env


def run_in(work, argv, env, log=None, timeout=3600):
    p = subprocess.run([sys.executable, "-B", *argv], cwd=os.path.join(work, "rc12"), env=env, timeout=timeout,
                       capture_output=True, text=True)
    if log:
        with open(log, "w") as f:
            f.write(p.stdout + ("\n[stderr]\n" + p.stderr if p.stderr else ""))
    return p


def build(work, seed_file, split):
    commit_of(work)
    p = run_in(work, ["build_dev.py"], env_for(seed_file, split))
    print(p.stdout + p.stderr)
    if p.returncode:
        die("build failed")


def gates_clean(text):
    """a gate table passes only with a PASS row for every S4 gate and 0 failing steps (the seed draws write a table
    of their quick gates only, so 'failing steps: 0' alone would let a skipped full run through)."""
    rows = {m.group(1) for m in re.finditer(r"^(\S+)\s+exit\s+0 PASS", text, re.M)}
    return "failing steps: 0" in text and all(name in rows for name, _ in GATES)


def gates(work, seed_file, expect, split, only=None, stop=False, repo=REPO):
    """runs GATES in the copy; writes WORK/GATES.txt (step, exit code, last output line): the table that may
    leave the session (S4), never item text. stop: end at the first failing step (the seed draws). Returns the
    number of failing steps."""
    bad = check(work, repo=repo)
    if bad:
        die("the work copy differs from its commit: " + "; ".join(bad[:5]))
    env = env_for(seed_file, split, expect)
    logs = os.path.join(work, "gate_logs")
    os.makedirs(logs, exist_ok=True)
    rows, fails, words = [], 0, content_words(work)
    for name, argv in GATES:
        if only and name not in only:
            continue
        p = run_in(work, argv, env, os.path.join(logs, name + ".txt"))
        ok = p.returncode == 0
        if name == "build_stdout":                       # the build is deterministic: it reproduces the items
            ok = ok and p.stdout == open(os.path.join(work, ITEMS)).read()
        fails += not ok
        last = public_line(name, p.stdout, ok, words)
        rows.append(f"{name:22s} exit {p.returncode:3d} {'PASS' if ok else 'FAIL'}  {last}")
        print(rows[-1], flush=True)
        if stop and not ok:
            break
    with open(os.path.join(work, "GATES.txt"), "w") as f:
        f.write(f"commit {commit_of(work)} split {split}\n" + "\n".join(rows) + f"\nfailing steps: {fails}\n" +
                "".join(f"input sha256 {sha256(path)}  {name}\n" for name, path in gated_inputs(work, seed_file, expect)
                        if os.path.isfile(path)))
    return fails


def gated_inputs(work, seed_file, expect):
    """(name, path) of everything the gates read that is not code: the items, the content files, the pinned counts
    and the seed. GATES.txt records their sha256, so pack can refuse anything that changed after the gates ran."""
    out = [("rc12_sealed.jsonl", os.path.join(work, ITEMS))]
    out += [(f"content_{m}.py", os.path.join(work, "rc12", f"content_{m}.py")) for m in CONTENT_MODULES]
    out += [("expect.json", expect)] if expect else []
    return out + [("generator_seed.txt", seed_file)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "install", "check", "build", "gates", "overlap"])
    ap.add_argument("work")
    ap.add_argument("src", nargs="?")
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--seed-file")
    ap.add_argument("--expect")
    ap.add_argument("--split", default="sealed")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    work = os.path.abspath(a.work)
    if a.cmd != "prepare":
        check_place(work)
    if a.cmd == "prepare":
        prepare(work, a.commit)
    elif a.cmd == "install":
        install(work, a.src)
    elif a.cmd == "check":
        bad = check(work)
        print("\n".join(bad) or "work copy equals its commit (content modules and items aside)")
        sys.exit(1 if bad else 0)
    elif a.cmd == "build":
        build(work, a.seed_file, a.split)
    elif a.cmd == "gates":
        sys.exit(1 if gates(work, a.seed_file, a.expect, a.split, set(filter(None, a.only.split(",")))) else 0)
    elif a.cmd == "overlap":
        import sealed_overlap
        sys.exit(sealed_overlap.main(work))


if __name__ == "__main__":
    main()
