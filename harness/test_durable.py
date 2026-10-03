"""durable.py and the harness writers that publish through it (incident 2026-09-29: a power cut left the last .pt
files of an E2 trunk at 0 bytes under a surviving latest.json and TRUNK_DONE).

os.fsync and os.replace are wrapped to record, in call order, which inode each fsync touched and each rename's
destination. A published file must be fsynced BEFORE the rename that publishes it and its directory AFTER; a published
directory must have every file and directory in it fsynced before its rename and its parent after. latest.json must
be renamed into place only after the checkpoint's own directory fsync. CPU only, no model."""
from __future__ import annotations

import errno
import json
import os
import stat

import pytest
import torch

import durable
import from_pipeline as FP
import pipeline_fixture as PF
import runio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


class Rec:
    """records ("fsync", (dev, inode)) and ("replace", abs dst) in order; the real calls still run."""

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
        return i

    def check_dir(self, path):
        i = self.renamed(path)
        for d, _, files in os.walk(path):
            for n in files:
                assert self.fsynced(os.path.join(d, n), 0, i), f"{n}: not fsynced before the directory rename"
            assert self.fsynced(d, 0, i), f"{d}: directory not fsynced before its rename"
        assert self.fsynced(os.path.dirname(os.path.abspath(path)), i + 1), "parent not fsynced after the rename"
        return i


@pytest.fixture
def rec(monkeypatch):
    return Rec(monkeypatch)


def test_atomic_write_order(rec, tmp_path):
    p = str(tmp_path / "a.json")
    with durable.atomic_write(p) as f:
        f.write('{"x": 1}')
    rec.check_file(p)
    assert open(p).read() == '{"x": 1}' and not os.path.exists(p + ".tmp")


def test_atomic_write_exception_publishes_nothing(rec, tmp_path):
    p = str(tmp_path / "a.json")
    with pytest.raises(RuntimeError):
        with durable.atomic_write(p) as f:
            f.write("half")
            raise RuntimeError("crash")
    assert not os.path.exists(p) and not rec.ev


def test_publish_file_order(rec, tmp_path):
    tmp = str(tmp_path / "x.pt.tmp")
    open(tmp, "wb").write(b"\x01" * 1000)
    durable.publish_file(tmp, str(tmp_path / "x.pt"))
    rec.check_file(str(tmp_path / "x.pt"))


def test_publish_dir_order(rec, tmp_path):
    part = tmp_path / "run.partial"
    (part / "sub").mkdir(parents=True)
    for n in ("a.jsonl", "DONE", "sub/b.bin"):
        (part / n).write_bytes(b"data")
    durable.publish_dir(str(part), str(tmp_path / "run"))
    rec.check_dir(str(tmp_path / "run"))


def _fail_on(kind, code):
    """an os.fsync that raises `code` for directories ("dir") or regular files ("file") and is real otherwise."""
    real = os.fsync

    def f(fd):
        is_dir = stat.S_ISDIR(os.fstat(fd).st_mode)
        if is_dir == (kind == "dir"):
            raise OSError(code, os.strerror(code))
        return real(fd)
    return f


REFUSALS = [errno.EINVAL, errno.EACCES, errno.EPERM, errno.EIO, errno.ENOSYS, errno.EBADF]


@pytest.mark.parametrize("platform", ["linux", "darwin"])
@pytest.mark.parametrize("where", ["open", "fsync"])
def test_fsync_dir_refused_never_raises(monkeypatch, capsys, tmp_path, platform, where):
    """a filesystem that refuses a directory fsync (drvfs /mnt/d, FAT, a network fs; errno unknown) or even the
    directory open: no raise on any platform, the fd opened read-only and closed, one stderr warning per errno."""
    monkeypatch.setattr(durable, "_WARNED", set())
    monkeypatch.setattr(durable.sys, "platform", platform)
    opened, closed, real_open, real_close = [], [], os.open, os.close

    def op(p, flags, *a, **k):
        if where == "open" and code is not None:
            raise OSError(code, os.strerror(code))
        fd = real_open(p, flags, *a, **k)
        opened.append((fd, flags))
        return fd

    monkeypatch.setattr(os, "open", op)
    monkeypatch.setattr(os, "close", lambda fd: (closed.append(fd), real_close(fd))[1])
    for code in REFUSALS:
        if where == "fsync":
            monkeypatch.setattr(os, "fsync", _fail_on("dir", code))
        durable.fsync_dir(str(tmp_path))
        durable.fsync_dir(str(tmp_path))
    err = capsys.readouterr().err
    assert [err.count(f"({errno.errorcode[c]})") for c in REFUSALS] == [1] * len(REFUSALS), "not warned once each"
    assert all(fl & (os.O_RDONLY | os.O_WRONLY | os.O_RDWR) == os.O_RDONLY for _, fl in opened), "not read-only"
    assert sorted(fd for fd, _ in opened) == sorted(closed), "a directory fd was left open"
    code = None
    durable.fsync_dir(str(tmp_path))                       # the real call on this machine: silent
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("code", [errno.EIO, errno.EACCES, errno.EINVAL])
def test_checkpoint_on_a_mount_refusing_dir_fsync(monkeypatch, capsys, tmp_path, code):
    """a run saving onto a /mnt/d-like mount, on Linux: directory fsync refused, the save completes, the checkpoint
    is whole and latest.json names it."""
    monkeypatch.setattr(durable, "_WARNED", set())
    monkeypatch.setattr(durable.sys, "platform", "linux")
    monkeypatch.setattr(os, "fsync", _fail_on("dir", code))
    out = str(tmp_path / "out")
    for step in (4, 8):
        runio.save_checkpoint(out, {"step": step, "w": torch.arange(10.0)}, step, tag="final" if step == 8 else None)
    ck = os.path.join(out, "final_00000008.pt")
    assert runio.latest_checkpoint(out) == ck and torch.load(ck)["step"] == 8
    assert capsys.readouterr().err.count("directory fsync refused") == 1


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_file_fsync_failure_raises_and_publishes_nothing(monkeypatch, tmp_path, platform):
    monkeypatch.setattr(durable.sys, "platform", platform)
    monkeypatch.setattr(os, "fsync", _fail_on("file", errno.EIO))
    tmp, path = str(tmp_path / "x.pt.tmp"), str(tmp_path / "x.pt")
    open(tmp, "wb").write(b"\x01" * 100)
    with pytest.raises(OSError):
        durable.publish_file(tmp, path)
    with pytest.raises(OSError):
        with durable.atomic_write(str(tmp_path / "a.json")) as f:
            f.write("{}")
    (part := tmp_path / "run.partial").mkdir()
    (part / "DONE").write_bytes(b"")
    with pytest.raises(OSError):
        durable.publish_dir(str(part), str(tmp_path / "run"))
    assert sorted(os.listdir(tmp_path)) == ["a.json.tmp", "run.partial", "x.pt.tmp"], "something was published"


def test_exdev_raises_without_a_copy(monkeypatch, tmp_path):
    """os.replace across filesystems raises EXDEV as it always did: a copy is not atomic, so none is attempted."""
    def xdev(src, dst, *a, **k):
        raise OSError(errno.EXDEV, os.strerror(errno.EXDEV))
    monkeypatch.setattr(os, "replace", xdev)
    tmp = str(tmp_path / "x.tmp")
    open(tmp, "w").write("x")
    (tmp_path / "d.partial").mkdir()
    for call in (lambda: durable.publish_file(tmp, str(tmp_path / "x")),
                 lambda: durable.publish_dir(str(tmp_path / "d.partial"), str(tmp_path / "d"))):
        with pytest.raises(OSError) as e:
            call()
        assert e.value.errno == errno.EXDEV
    assert sorted(os.listdir(tmp_path)) == ["d.partial", "x.tmp"]


def test_fsync_dir_reaches_fsync_here(rec, tmp_path):
    durable.fsync_dir(str(tmp_path))
    assert rec.fsynced(str(tmp_path)), "the directory fsync never reached os.fsync on this platform"


@pytest.mark.parametrize("tag", [None, "stable", "final"])
def test_save_checkpoint_order(rec, tmp_path, tag):
    out = str(tmp_path / "out")
    runio.save_checkpoint(out, {"step": 5, "w": torch.arange(10.0)}, 5, tag=tag)
    ck = os.path.join(out, f"{tag or 'ckpt'}_00000005.pt")
    i_ck = rec.check_file(ck)
    i_latest = rec.check_file(os.path.join(out, "latest.json"))
    k = ("fsync", (os.stat(out).st_dev, os.stat(out).st_ino))
    first_dir_sync = min(i for i in range(i_ck + 1, len(rec.ev)) if rec.ev[i] == k)
    assert first_dir_sync < i_latest, "latest.json was published before the checkpoint's rename was on disk"
    assert runio.latest_checkpoint(out) == ck and os.path.getsize(ck) > 0


def test_checkpoint_bytes_unchanged(tmp_path):
    """the old _atomic_save (torch.save to path.tmp, os.replace) and the new one write the same bytes."""
    payload = {"model": {"w": torch.randn(4, 4, generator=torch.Generator().manual_seed(0))}, "step": 3}
    for d in ("old", "new"):
        os.makedirs(tmp_path / d)
    p_old, p_new = str(tmp_path / "old" / "final_00000003.pt"), str(tmp_path / "new" / "final_00000003.pt")
    torch.save(payload, p_old + ".tmp")
    os.replace(p_old + ".tmp", p_old)
    runio._atomic_save(payload, p_new)
    assert open(p_old, "rb").read() == open(p_new, "rb").read()


def test_from_pipeline_publish_order(rec, tmp_path):
    out = str(tmp_path / "conv")
    FP.run([PF.fake_run_dir()], out, allow_nontrainable=True)
    assert any(n.startswith("chat-") for n in os.listdir(out)), "fixture wrote no shard"
    rec.check_dir(out)
    assert json.load(open(os.path.join(out, "manifest.json")))["counts"]["admitted"] > 0


@pytest.mark.parametrize("copy", ["pipeline/durable.py", "pc/wsl/durable.py"])
def test_copies_match(copy):
    assert open(os.path.join(ROOT, copy), "rb").read() == open(os.path.join(HERE, "durable.py"), "rb").read(), \
        f"{copy} drifted from harness/durable.py"
