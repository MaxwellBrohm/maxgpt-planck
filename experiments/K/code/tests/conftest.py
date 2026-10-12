"""pytest setup for track K code tests (CPU only, tiny models; no language model is ever loaded)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)
import kcommon as K                                                   # noqa: E402

if K.HARNESS not in sys.path:
    sys.path.insert(1, K.HARNESS)

import torch                                                          # noqa: E402

torch.set_num_threads(int(os.environ.get("K_TEST_THREADS", "4")))
