"""Sidecar formats of neardedup.py: the SidecarWriter .mh.npy (add or extend) and the headerless,
appendable .mh.u32 that an append-only shard writer can use; both read the same in pass B."""
import numpy as np
import pytest

import neardedup as ND
import neardedup_lsh as NL
from nd_fixtures import Gen


def keep_of(stem):
    return NL.load_result(stem)["keep"].astype(bool)


def test_worker_sketch_then_extend_equals_add(tmp_path):
    g = Gen(4)
    texts, created = [g.text(80) for _ in range(5)], ["2019-01-02", None, "2020", "junk", 2001]
    a, b = ND.SidecarWriter(str(tmp_path / "a")), ND.SidecarWriter(str(tmp_path / "b"))
    for t, c in zip(texts, created):
        a.add(t, c)
    b.extend(ND.sketch(texts[:2], created[:2]))
    b.extend(ND.sketch(texts[2:], created[2:]))
    a.close(), b.close()
    ra, rb = ND.load_sidecar(str(tmp_path / "a")), ND.load_sidecar(str(tmp_path / "b"))
    assert np.array_equal(ra["sig"], rb["sig"]) and ra["day"].tolist() == rb["day"].tolist()
    assert len(ra) == 5 and ra["day"][1] == ra["day"][3] == ND.UNDATED
    assert np.array_equal(ra["sig"][0], ND.signature(texts[0]))
    assert ND.SidecarWriter(str(tmp_path / "e")).close() and len(ND.load_sidecar(
        str(tmp_path / "e"))) == 0
    with open(tmp_path / "r.mh.u32", "wb") as f:                # the appendable raw form
        for t, c in zip(texts, created):
            f.write(ND.sig_row(t, c).tobytes())
    rr = ND.load_sidecar(str(tmp_path / "r"))
    assert np.array_equal(rr["sig"], ra["sig"]) and rr["day"].tolist() == ra["day"].tolist()
    NL.find_near_dups([{"stem": str(tmp_path / "a")}, {"stem": str(tmp_path / "r")}],
                      str(tmp_path))
    assert keep_of(str(tmp_path / "r")).sum() == 0 and keep_of(str(tmp_path / "a")).all()
    with open(tmp_path / "r.mh.u32", "ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError, match="whole number of rows"):
        ND.load_sidecar(str(tmp_path / "r"))
