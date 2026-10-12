"""AMENDMENT SEEDS-103-105's same-code guard (SCREENS.txt, CODE PATH AND C4's PAIRING): while plans/CURRENT names
robustness_k5, the tracked files under harness/, data_prep/, tokenizer/ and corpus/ plus the 5 E2 files a SCREENS run
reads must equal 77ab43f's by git blob id: the tree whose only change there against 27d8cb7 (where base_s103..105 ran;
equal to 93bc10b, where base_s101 and base_s102 ran) is harness/train.py's default-off data_seed, which E3 notes PART 3
CUDA PARITY passed on CUDA. The digest recipe is replay_lib's (tested there to see every change). Not in
mutation_screens.py, whose scratch copies hold only part of these trees: its mutants are run by hand on a full copy.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_robust_code.py
"""
import hashlib
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import replay_lib as R  # noqa: E402
import robust_lib as RB  # noqa: E402
import screens_lib as L  # noqa: E402
from test_screens_replay_code import current, git  # noqa: E402


def test_code_paths_equal_77ab43f_while_the_robustness_plan_is_next():
    if current() not in RB.ROBUST_PLANS:
        pytest.skip(f"plans/CURRENT names {current()}: the robustness plan is not the next launch")
    n, digest = R.code_digest()
    assert (n, digest) == (RB.CODE_N, RB.CODE_DIGEST), (
        f"{n} files, digest {digest[:16]}: harness/, data_prep/, tokenizer/, corpus/ or the E2 run files differ from "
        "77ab43f, the tree E3's PART 3 CUDA PARITY tested (AMENDMENT SEEDS-103-105, CODE PATH AND C4's PAIRING)")


def test_the_pinned_digest_is_77ab43f_by_git_and_differs_from_93bc10b_only_in_harness():
    if not os.path.isdir(os.path.join(L.ROOT, ".git")) or git("cat-file", "-e", RB.CODE_REF).returncode:
        pytest.skip("no git checkout holding 77ab43f (an index export)")
    rows = {}
    for ref in (RB.CODE_REF, R.CODE_REF):
        out = git("ls-tree", "-r", ref, "--", *R.CODE_DIRS, *R.CODE_E2)
        rows[ref] = dict(sorted((ln.split("\t", 1)[1], ln.split()[2]) for ln in out.stdout.splitlines()))
    txt = "".join(f"{p} {b}\n" for p, b in sorted(rows[RB.CODE_REF].items()))
    assert (len(rows[RB.CODE_REF]), hashlib.sha256(txt.encode()).hexdigest()) == (RB.CODE_N, RB.CODE_DIGEST)
    a, b = rows[R.CODE_REF], rows[RB.CODE_REF]
    changed = sorted(p for p in set(a) | set(b) if a.get(p) != b.get(p))
    assert changed == ["harness/configs/sample_tiny/config.yaml", "harness/mutants.py", "harness/mutants_dataseed.py",
                       "harness/notes.txt", "harness/test_data_seed.py", "harness/test_data_seed_more.py",
                       "harness/train.py"], changed
    assert git("diff", "--quiet", "93bc10b", "27d8cb7", "--", *R.CODE_DIRS, *R.CODE_E2).returncode == 0
