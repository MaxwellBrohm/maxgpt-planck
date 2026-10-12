"""Mutation check for SCREENS2's SEQUENTIAL RULE (screens2_seq.py, screens2_seqsim.py): batch 1's runner
(mutation_screens.py: scratch copies only, the repo is never edited) on MUTANTS_SEQ alone.
  python experiments/screens/tests/mutation_screens2_seq.py [--scratch DIR] [--only NAME,...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mutation_screens as MS  # noqa: E402
from mutants_screens2_seq import MUTANTS_SEQ  # noqa: E402

if __name__ == "__main__":
    MS.MUTANTS[:] = MUTANTS_SEQ
    sys.exit(MS.main())
