"""E3 notes PART 3 CUDA PARITY: the copied PC log says what the entry says (A PASS with every comparison identical, B and
C FAIL), the entry quotes the log's sha256, numbers and commits, the trees' train.py hashes equal git's blobs at those
commits (skipped without git), and the same checks fail on edited copies of the log and of the entry.
  ~/.venvs/planck/bin/python -m pytest -q experiments/E3_seed_noise/tests/test_parity_cuda.py
"""
import hashlib
import os
import re
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
E3 = os.path.join(ROOT, "experiments", "E3_seed_noise")
LOG = os.path.join(E3, "logs", "parity_cuda", "e3p3_parity_cuda.log")
SUMS = os.path.join(E3, "logs", "parity_cuda", "p3par.sha256")
REF, NEW = "27d8cb7505248ff4391a1dee9a51bad617938195", "77ab43f3ca836793d0cde94146c6dc18d1192fc5"
HEAD = "\nPART 3 CUDA PARITY (2026-10-11"
PARITY_SRC = open(os.path.join(ROOT, "harness", "default_parity.py")).read()
GIT_ENV = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_COMMON_DIR")


def entry(notes: str) -> str:
    i = notes.index(HEAD)
    j = notes.find("\n\n", i + 1)
    return " ".join(notes[i:j if j > 0 else None].split())


def runs(log: str) -> dict:
    """{A|B|C: (rc, verdict, comparison lines)} from the driver's log."""
    out, cur = {}, None
    for ln in log.splitlines():
        m = re.match(r"\S+ \S+ run ([ABC]): ", ln)
        if m:
            cur = m[1]
            out[cur] = {"lines": []}
            continue
        m = re.match(r"\S+ \S+ run ([ABC]) rc (\d+) (PASS|FAIL|NONE)$", ln)
        if m:
            out[m[1]].update(rc=int(m[2]), verdict=m[3])
            cur = None
        elif cur and ln.startswith("[default_parity] "):
            out[cur]["lines"].append(ln[len("[default_parity] "):])
    return out


def problems(log: str, notes: str, sums: str, parity_src: str | None = None) -> list[str]:
    parity_src = PARITY_SRC if parity_src is None else parity_src
    bad, e, r = [], entry(notes), runs(log)
    sha = hashlib.sha256(log.encode()).hexdigest()
    if f"sha256 {sha[:16]}..." not in e:
        bad.append("entry does not quote the log's sha256")
    if f"ref commit {REF}; new commit {NEW}" not in log:
        bad.append("log commits")
    if f"ref = git archive of {REF[:7]}" not in e or f"new = git archive of {NEW[:7]}" not in e:
        bad.append("entry commits")
    a = r.get("A", {})
    want = ["log: identical", "checkpoint: identical", "start: identical", "new_imports: identical", "PASS"]
    if a.get("rc") != 0 or a.get("verdict") != "PASS" or any(w not in a.get("lines", []) for w in want):
        bad.append("A is not a PASS with every comparison identical")
    m = re.match(r"cuda: (\d+) log records, final loss (\S+) vs (\S+);", (a.get("lines") or [""])[0])
    if not m or m[1] != "24" or m[2] != m[3] or f"final loss {m[2]} in both" not in e:
        bad.append("A's records or final loss")
    for k in "BC":
        x = r.get(k, {})
        if x.get("rc") != 1 or x.get("verdict") != "FAIL" or "FAIL" not in x.get("lines", []):
            bad.append(f"{k} is not a comparison FAIL")
    mb = re.match(r"cuda: \d+ log records, final loss \S+ vs (\S+);", (r.get("B", {}).get("lines") or [""])[0])
    if not mb or f"final loss {mb[1]})" not in e:
        bad.append("B's final loss")
    if "start: [\" keys ['data_seed']\"]" not in r.get("C", {}).get("lines", []) or "start keys ['data_seed']" not in e:
        bad.append("C's start record")
    for k in "ABC":
        if f"RESULT {k} rc " not in log:
            bad.append(f"RESULT {k} line")
    if not log.rstrip().endswith("DONE ok"):
        bad.append("log does not end DONE ok")
    held = re.search(r"(\d\d:\d\d:\d\d) gpu\.lock held \(GPU (\d+) C, (\d+) MiB", log)
    rel = re.search(r"(\d\d:\d\d:\d\d) gpu\.lock released", log)
    if not held or not rel or f"one gpu.lock hold {held[1]} to {rel[1]} (GPU {held[2]} C, {held[3]} MiB" not in e:
        bad.append("hold times")
    for ln in sums.splitlines():
        h, f = ln.split()
        if f"{f} {h[:16]}..." not in e:
            bad.append(f"tarball sha256 {f}")
    if "\u2014" in e or "\u2013" in e:
        bad.append("dash")
    if not m or f"PASS: {m[1]} log records" not in e:          # added 2026-10-12 after the independent check
        bad.append("A's record count in the entry")
    bl = r.get("B", {}).get("lines", [])
    if not any(x.startswith("log: [") for x in bl) or not any(x.startswith("checkpoint: [") for x in bl) or \
            "FAIL as wanted (log and checkpoint differ;" not in e:
        bad.append("B's differences")
    if f"the final checkpoint ({', '.join(ck_phrases(parity_src))})" not in e:
        bad.append("checkpoint keys")
    return bad


CK_PHRASE = {"model": "model", "optimizer": "optimizer", "data_state": "data_state", "rng_torch": "torch RNG",
             "rng_cuda": "CUDA RNG", "step": "step", "tokens": "token counts", "sup_tokens": "token counts"}


def ck_phrases(src: str) -> list[str]:
    """The entry's words for the checkpoint keys harness/default_parity.py compares, in its order (rng_cuda added on
    CUDA); a key with no words here is shown as itself, so the entry cannot match it."""
    m = re.search(r'"ck": \{k: ck\[k\] for k in \(([^)]*)\)\}', src)
    keys = re.findall(r'"(\w+)"', m[1]) if m else ["<no key list>"]
    if 'out[tag]["ck"]["rng_cuda"] = ck["rng_cuda"]' in src:
        keys.insert(keys.index("rng_torch") + 1 if "rng_torch" in keys else len(keys), "rng_cuda")
    both = "rng_torch" in keys and "rng_cuda" in keys
    out = []
    for k in keys:
        w = "torch and CUDA RNG" if both and k in ("rng_torch", "rng_cuda") else CK_PHRASE.get(k, k)
        if w not in out:
            out.append(w)
    return out


def read():
    return open(LOG).read(), open(os.path.join(E3, "notes.txt")).read(), open(SUMS).read()


def test_the_record_matches_the_log():
    assert problems(*read()) == []


@pytest.mark.parametrize("old,new,where,want", [
    ("[default_parity] PASS\n", "[default_parity] FAIL\n", "log", "A is not a PASS"),
    ("run A rc 0 PASS", "run A rc 1 PASS", "log", "A is not a PASS"),
    ("[default_parity] checkpoint: identical\n", "[default_parity] checkpoint: ['/model/x']\n", "log", "A is not a PASS"),
    ("run B rc 1 FAIL", "run B rc 0 FAIL", "log", "B is not a comparison FAIL"),
    ("run C rc 1 FAIL", "run C rc 1 NONE", "log", "C is not a comparison FAIL"),
    ("DONE ok", "DONE NOT OK", "log", "log does not end DONE ok"),
    ("RESULT B rc ", "RESULT X rc ", "log", "RESULT B line"),
    ("ref commit 27d8cb75", "ref commit 27d8cb76", "log", "log commits"),
    ("[default_parity] start: identical\n", "[default_parity] start: [' keys']\n", "log", "A is not a PASS"),
    ("final loss 3.07388 vs 3.07388", "final loss 3.07388 vs 3.07389", "log", "A's records or final loss"),
    ("final loss 3.07388 in both", "final loss 3.07389 in both", "notes", "A's records or final loss"),
    ("final loss\n                                                                2.93466)", "final loss 2.93467)", "notes",
     "B's final loss"),
    ("start keys ['data_seed']", "start keys ['seed']", "notes", "C's start record"),
    ("c2049729f8075af5...", "c2049729f8075af6...", "notes", "entry does not quote the log's sha256"),
    ("hold 22:33:45 to 22:34:10", "hold 22:33:45 to 22:34:11", "notes", "hold times"),
    ("p3par_new.tgz 19a91afc66e4fadb...", "p3par_new.tgz 19a91afc66e4fadc...", "notes", "tarball sha256 p3par_new.tgz"),
    ("ref = git archive of 27d8cb7", "ref = git archive of 93bc10b", "notes", "entry commits"),
    ("Tarballs (harness,", "Tarballs \u2014 (harness,", "notes", "dash"),
    ("PASS: 24 log records", "PASS: 25 log records", "notes", "A's record count"),
    ("(log and checkpoint differ;", "(log identical, checkpoint differs;", "notes", "B's differences"),
    ("[default_parity] log: ['[0]/loss'", "[default_parity] log: identical ['[0]/loss'", "log", "B's differences"),
    ("[default_parity] checkpoint: ['/model/tok_emb", "[default_parity] checkpoint: identical ['/model/tok_emb", "log",
     "B's differences"),
    ("model, optimizer, data_state, torch", "model, optimizer, torch", "notes", "checkpoint keys"),
])
def test_each_check_fails_on_an_edited_copy(old, new, where, want):
    """Each edit is caught by the check it targets: a log edit gets the entry's sha256 updated to the edited log's, so
    the sha256 check alone cannot catch it."""
    log, notes, sums = read()
    if where == "log":
        assert log.count(old) >= 1, old
        sha0 = hashlib.sha256(log.encode()).hexdigest()[:16]
        log = log.replace(old, new, 1)
        notes = notes.replace(f"sha256 {sha0}...", f"sha256 {hashlib.sha256(log.encode()).hexdigest()[:16]}...")
    else:
        assert notes.count(old) == 1, old
        notes = notes.replace(old, new)
    got = problems(log, notes, sums)
    assert any(g.startswith(want) for g in got), got


GIT_ROOT = os.environ.get("E3_PARITY_GIT_ROOT") or ROOT    # a mutation copy points this at a checkout


def _git(*a):
    env = {k: v for k, v in os.environ.items() if k not in GIT_ENV}
    env.setdefault("DEVELOPER_DIR", "/Library/Developer/CommandLineTools")
    return subprocess.run(["git", "-C", GIT_ROOT, *a], capture_output=True, env=env)


def git_problems(log: str, notes: str, pairs=(("93bc10b", REF), ("4962426", NEW))) -> list[str]:
    bad, e = [], entry(notes)
    for tag, c in (("ref", REF), ("new", NEW)):
        p = _git("show", f"{c}:harness/train.py")
        h = hashlib.sha256(p.stdout).hexdigest()[:16] if p.returncode == 0 else "no blob"
        if f"{tag} {h}" not in e:
            bad.append(f"entry train.py {tag}")
        if f"{'tree new' if tag == 'new' else 'ref'} train.py sha256 {h}" not in log:
            bad.append(f"log train.py {tag}")
    for a, b in pairs:
        if _git("diff", "--quiet", a, b, "--", "harness", "data_prep", "tokenizer", "corpus").returncode != 0:
            bad.append(f"trees {a} {b[:7]}")
    return bad


HAS_GIT = _git("rev-parse", "--git-dir").returncode == 0


@pytest.mark.skipif(not HAS_GIT, reason="needs a git checkout")
def test_train_py_hashes_are_the_commits_blobs():
    assert git_problems(*read()[:2]) == []


@pytest.mark.skipif(not HAS_GIT, reason="needs a git checkout")
@pytest.mark.parametrize("old,new,where,want", [
    ("96a763cf45a413de, new", "96a763cf45a413df, new", "notes", "entry train.py ref"),
    ("new aa07b64c7a4e3089", "new aa07b64c7a4e3088", "notes", "entry train.py new"),
    ("tree new train.py sha256 aa07b64c", "tree new train.py sha256 aa07b64d", "log", "log train.py new"),
])
def test_git_checks_fail_on_an_edited_copy(old, new, where, want):
    log, notes, _ = read()
    if where == "log":
        log = log.replace(old, new, 1)
    else:
        assert notes.count(old) == 1, old
        notes = notes.replace(old, new)
    assert want in git_problems(log, notes)


@pytest.mark.skipif(not HAS_GIT, reason="needs a git checkout")
def test_the_tree_check_sees_a_harness_change():
    assert "trees 93bc10b 77ab43f" in git_problems(*read()[:2], pairs=(("93bc10b", NEW),))


@pytest.mark.parametrize("old,new", [('"step",', '"step", "lr_sched",'), ('"data_state", ', ''),
                                     ('out[tag]["ck"]["rng_cuda"] = ck["rng_cuda"]', 'pass')])
def test_the_key_check_follows_default_parity(old, new):
    assert PARITY_SRC.count(old) == 1, old
    assert "checkpoint keys" in problems(*read(), parity_src=PARITY_SRC.replace(old, new))
