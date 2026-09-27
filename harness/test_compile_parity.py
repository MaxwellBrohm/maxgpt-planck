"""compile_parity.pooled: the exchangeability check behind the compile parity verdict (CPU,
synthetic loss curves; the real curves come from the PC runs)."""
from __future__ import annotations

import itertools
import random
import statistics as st

from compile_parity import dist, pooled


def curves(seed: int = 0):
    rnd = random.Random(seed)
    base = [9.0 * 0.98 ** s + 0.5 for s in range(200)]

    def curve(shift: float = 0.0):
        return [b + shift + rnd.gauss(0, 0.01) for b in base]

    eager = [{"arm": "eager", "loss": curve()} for _ in range(4)]
    det = {"arm": "eager:det", "loss": curve()}
    cdet = {"arm": "default:det", "loss": curve()}
    same = [{"arm": "default", "loss": curve()} for _ in range(3)]
    far = [{"arm": "max-autotune-no-cudagraphs", "loss": curve(0.05)} for _ in range(3)]
    return eager + [det, dict(det), cdet, dict(cdet)] + same + far


def test_pooled_counts_det_duplicates_once_and_flags_an_offset():
    runs = curves()
    p = pooled(runs)
    d, m = p["default"], p["max-autotune-no-cudagraphs"]
    eager = [r["loss"] for r in runs[:5]]                   # the 4 eager + the one eager:det
    ee = st.median(dist(x, y)["mean_abs"] for x, y in itertools.combinations(eager, 2))
    assert d["mean_abs"]["eager_eager_median"] == m["mean_abs"]["eager_eager_median"] == ee
    assert (d["n_eager"], d["n_compiled"]) == (5, 4)          # bitwise :det repeats count once
    assert (m["n_eager"], m["n_compiled"]) == (5, 3)
    assert d["mean_abs"]["n_perm"] == 126 and m["mean_abs"]["n_perm"] == 56   # C(9,4), C(8,3)
    for key in ("mean_abs", "tail50"):
        # an offset of 5x the per-step noise: the observed labelling is the most extreme one
        assert m[key]["p"] == 1 / 56
        assert m[key]["compiled_eager_median"] > 3 * m[key]["eager_eager_max"]
        # same distribution as eager
        assert d[key]["compiled_eager_median"] < 2 * d[key]["eager_eager_max"]


def test_pooled_is_calibrated_when_compiled_runs_are_eager_like():
    """Under the null (compiled curves drawn like eager ones) p is about uniform: over 40
    independent draws the mean p is near 0.5 and p <= 0.05 about 5 % of the time."""
    ps = [pooled(curves(seed))["default"] for seed in range(40)]
    for key in ("mean_abs", "tail50"):
        p = [r[key]["p"] for r in ps]
        assert 0.4 < sum(p) / len(p) < 0.6, key
        assert sum(x <= 0.05 for x in p) <= 6, key


def test_pooled_needs_two_eager_runs():
    runs = [{"arm": "eager", "loss": [1.0] * 60}, {"arm": "default", "loss": [1.1] * 60}]
    assert pooled(runs) == {}
