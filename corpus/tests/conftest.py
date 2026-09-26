"""pytest setup for the corpus tests: put corpus/ and corpus/tests/ on sys.path (flat imports,
as in harness/)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
