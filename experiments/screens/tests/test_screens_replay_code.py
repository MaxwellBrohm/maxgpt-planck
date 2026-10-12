"""AMENDMENT BASE-REPLAY's same-code guard (SCREENS.txt): while plans/CURRENT names stage2_replay, the tracked files under
harness/, data_prep/, tokenizer/ and corpus/ plus the 5 E2 files a SCREENS run reads must equal 93bc10b's (the commit
base_s101 ran from; d21df3d, 3227c60 and 27d8cb7 are equal there), by git blob id, so the replays run base_s101's and
the stage 2 pairs' code path (C4). Works in a git checkout (git ls-files) and in an index export (every file but
caches). Not in mutation_screens.py, whose scratch copies hold only part of these trees: its mutants were run by hand
on a full index export (the entry's Tests line).
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_replay_code.py
"""
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import screens_lib as L  # noqa: E402
import replay_lib as R  # noqa: E402


def current() -> str:
    return [ln.strip() for ln in open(os.path.join(L.HERE, "plans", "CURRENT")) if not ln.startswith("#")][0]


def git(*args) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if os.path.isdir("/Library/Developer/CommandLineTools"):
        env.setdefault("DEVELOPER_DIR", "/Library/Developer/CommandLineTools")
    return subprocess.run(["git", *args], cwd=L.ROOT, env=env, capture_output=True, text=True)


def test_code_paths_equal_93bc10b_while_the_replay_is_next():
    if current() not in R.REPLAY_PLANS:
        pytest.skip(f"plans/CURRENT names {current()}: the replay is not the next launch")
    n, digest = R.code_digest()
    assert (n, digest) == (R.CODE_N, R.CODE_DIGEST), (
        f"{n} files, digest {digest[:16]}: harness/, data_prep/, tokenizer/, corpus/ or the E2 run files differ from "
        "93bc10b, so stage2_replay would not replay base_s101's code (AMENDMENT BASE-REPLAY, SAME CODE)")


def test_the_pinned_digest_is_93bc10b_by_git():
    if not os.path.isdir(os.path.join(L.ROOT, ".git")) or git("cat-file", "-e", R.CODE_REF).returncode:
        pytest.skip("no git checkout holding 93bc10b (an index export)")
    out = git("ls-tree", "-r", R.CODE_REF, "--", *R.CODE_DIRS, *R.CODE_E2)
    rows = sorted((ln.split("\t", 1)[1], ln.split()[2]) for ln in out.stdout.splitlines())
    import hashlib
    txt = "".join(f"{p} {b}\n" for p, b in rows)
    assert (len(rows), hashlib.sha256(txt.encode()).hexdigest()) == (R.CODE_N, R.CODE_DIGEST)
    for f in ("harness/train.py", "data_prep/bpb.py", "tokenizer/v0/tok_v0_8k.json"):
        assert R.git_blob(os.path.join(L.ROOT, f)) == git("hash-object", f).stdout.strip(), f


def test_the_digest_sees_every_change(tmp_path):
    for rel in ("harness/a.py", "data_prep/b.py", "tokenizer/v0/t.json", "corpus/c.py", *R.CODE_E2,
                "experiments/E2_lr_transfer/notes.txt", "harness/__pycache__/a.cpython-314.pyc", "harness/z.pyc",
                "harness/.pytest_cache/README.md", "data_prep/__pycache__/x.txt"):
        os.makedirs(os.path.dirname(tmp_path / rel), exist_ok=True)
        (tmp_path / rel).write_text(rel)
    files = R.code_files(str(tmp_path))
    assert len(files) == 4 + len(R.CODE_E2) and "experiments/E2_lr_transfer/notes.txt" not in files
    assert not [f for f in files if "__pycache__" in f or f.endswith(".pyc")]
    ref = R.code_digest(str(tmp_path))
    (tmp_path / "corpus" / "c.py").write_text("corpus/c.py ")
    assert R.code_digest(str(tmp_path))[1] != ref[1]
    (tmp_path / "corpus" / "c.py").write_text("corpus/c.py")
    assert R.code_digest(str(tmp_path)) == ref
    (tmp_path / "tokenizer" / "new.py").write_text("x")
    assert R.code_digest(str(tmp_path))[0] == ref[0] + 1
    os.unlink(tmp_path / "tokenizer" / "new.py")
    shutil.rmtree(tmp_path / "data_prep")
    assert R.code_digest(str(tmp_path))[0] == ref[0] - 1
