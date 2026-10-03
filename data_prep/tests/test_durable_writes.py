"""data_prep publishes shards, eval sets and manifests durably (incident 2026-09-29: a power cut left renamed files
at 0 bytes). os.fsync and os.replace are wrapped (in this process; pretokenize's pool workers write inside the
.partial dir, which the parent fsyncs whole) to record, in call order, the inode each fsync touched and each rename's
destination:
  write_json   prep_common.write_json fsyncs the data BEFORE the rename and the directory AFTER
  pretokenize  each <out>/<source>/ has every shard, .ids and manifest.json fsynced before its .partial rename and
               <out>/ fsynced after; the top manifest.json as write_json
  eval_sets    <out>/ has every file fsynced before its .partial rename and its parent fsynced after"""
from __future__ import annotations

import os

import prep_common as C
from conftest import PRETOK_ARGS


class Rec:
    def __init__(self, mp):
        self.ev = []
        real_fsync, real_replace = os.fsync, os.replace

        def fsync(fd):
            st = os.fstat(fd)
            self.ev.append(("fsync", (st.st_dev, st.st_ino)))
            return real_fsync(fd)

        def replace(src, dst, *a, **k):
            real_replace(src, dst, *a, **k)
            self.ev.append(("replace", os.path.abspath(dst)))

        mp.setattr(os, "fsync", fsync)
        mp.setattr(os, "replace", replace)

    def fsynced(self, path, lo=0, hi=None):
        st = os.stat(path)
        return ("fsync", (st.st_dev, st.st_ino)) in self.ev[lo:hi]

    def renamed(self, path):
        idx = [i for i, e in enumerate(self.ev) if e == ("replace", os.path.abspath(path))]
        assert idx, f"{path}: never renamed into place"
        return idx[-1]

    def check_file(self, path):
        i = self.renamed(path)
        assert self.fsynced(path, 0, i), f"{os.path.basename(path)}: data not fsynced before its rename"
        assert self.fsynced(os.path.dirname(os.path.abspath(path)), i + 1), \
            f"{os.path.basename(path)}: directory not fsynced after the rename"

    def check_dir(self, path):
        i = self.renamed(path)
        files = [os.path.join(d, n) for d, _, fs in os.walk(path) for n in fs]
        assert files, f"{path} is empty"
        for p in files:
            assert self.fsynced(p, 0, i), f"{os.path.relpath(p, path)}: not fsynced before the directory rename"
        assert self.fsynced(path, 0, i), f"{path}: directory not fsynced before its rename"
        assert self.fsynced(os.path.dirname(os.path.abspath(path)), i + 1), "parent not fsynced after the rename"


def test_write_json_order(monkeypatch, tmp_path):
    rec = Rec(monkeypatch)
    p = str(tmp_path / "manifest.json")
    C.write_json(p, {"a": 1})
    rec.check_file(p)
    assert C.read_json(p) == {"a": 1}


def test_pretokenize_publish_order(monkeypatch, corpus, tmp_path):
    import pretokenize
    out = str(tmp_path / "out")
    rec = Rec(monkeypatch)
    assert pretokenize.main([corpus["input"], out, "--heldout", corpus["heldout"], "--sources", "web,dolly",
                             *PRETOK_ARGS]) == 0
    for s in ("web", "dolly"):
        assert any(n.endswith((".bin", ".jsonl")) for n in os.listdir(os.path.join(out, s))), f"{s}: no shard"
        rec.check_dir(os.path.join(out, s))
    rec.check_file(os.path.join(out, "manifest.json"))


def test_eval_sets_publish_order(monkeypatch, corpus, tmp_path):
    import eval_sets
    out = str(tmp_path / "ev")
    rec = Rec(monkeypatch)
    assert eval_sets.main([corpus["heldout"], out, "--sources", "dolly"]) == 0
    rec.check_dir(out)

