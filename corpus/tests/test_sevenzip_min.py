"""Tests for the stdlib 7z reader (sevenzip_min.py, sevenzip_hdr.py)."""
import os
import shutil
import subprocess

import pytest

import sevenzip_min as Z
from fixtures_se import _num, write_7z
from sevenzip_hdr import Reader

FILES = {"Badges.xml": b"<badges>" + b"<row Id='1'/>" * 3000 + b"</badges>",
         "Comments.xml": bytes(range(256)) * 700,
         "Posts.xml": ("<posts>" + "<row Body='café and bread'/>\n" * 5000 + "</posts>").encode()}
SEVENZ = shutil.which("7zz") or shutil.which("7z") or shutil.which("7za")


def read_all(path):
    return {m.name: m.read() for m in Z.iter_members(path)}


@pytest.mark.parametrize("coder", ["copy", "lzma", "lzma2", "bzip2"])
@pytest.mark.parametrize("encode_header", [True, False])
def test_roundtrip(tmp_path, coder, encode_header):
    p = str(tmp_path / "a.7z")
    write_7z(p, FILES, coder=coder, encode_header=encode_header)
    assert read_all(p) == FILES


def test_number_encoding_roundtrip():
    for v in [0, 1, 127, 128, 16383, 16384, 2**21 - 1, 2**21, 2**35 + 7, 2**56 - 1, 2**56,
              2**64 - 1]:
        assert Reader(_num(v)).num() == v, v


def test_partial_read_then_skip(tmp_path):
    p = str(tmp_path / "a.7z")
    write_7z(p, FILES)
    got = {}
    for m in Z.iter_members(p):
        got[m.name] = m.read(10) if m.name != "Posts.xml" else m.read()
    assert got["Badges.xml"] == FILES["Badges.xml"][:10]
    assert got["Posts.xml"] == FILES["Posts.xml"]
    wanted = dict((n, f.read()) for n, f in Z.open_members(p, {"Posts.xml"}))
    assert wanted == {"Posts.xml": FILES["Posts.xml"]}


def test_small_reads_cross_chunk_boundaries(tmp_path):
    p = str(tmp_path / "a.7z")
    write_7z(p, FILES)
    for m in Z.iter_members(p):
        parts = []
        while b := m.read(4093):
            parts.append(b)
        assert b"".join(parts) == FILES[m.name]


def test_crc_mismatch_is_detected(tmp_path):
    p = str(tmp_path / "a.7z")
    write_7z(p, FILES, coder="copy")
    with open(p, "r+b") as f:
        f.seek(32 + len(FILES["Badges.xml"]) + 5)       # inside Comments.xml
        b = f.read(1)
        f.seek(-1, 1)
        f.write(bytes([b[0] ^ 0xFF]))
    with pytest.raises(Z.Bad7z, match="CRC"):
        read_all(p)


def test_header_crc_is_checked(tmp_path):
    p = str(tmp_path / "a.7z")
    write_7z(p, FILES, coder="copy", encode_header=False)
    with open(p, "r+b") as f:
        f.seek(-3, 2)
        f.write(b"\xAA")
    with pytest.raises(Z.Bad7z):
        read_all(p)


def test_not_a_7z(tmp_path):
    p = tmp_path / "x.7z"
    p.write_bytes(b"PK\x03\x04 not a 7z at all" * 4)
    with pytest.raises(Z.Bad7z):
        read_all(str(p))


def test_unsupported_coder_raises_or_falls_back(tmp_path, monkeypatch):
    p = str(tmp_path / "ppmd.7z")
    write_7z(p, FILES, coder="copy", cid=b"\x03\x04\x01")         # PPMd id, not supported
    with pytest.raises(Z.Unsupported7z):
        read_all(p)
    monkeypatch.setattr(Z.shutil, "which", lambda name: None)
    with pytest.raises(Z.Unsupported7z):
        list(Z.open_members(p, {"Posts.xml"}))


@pytest.mark.skipif(not SEVENZ, reason="no 7z binary on PATH (the PC has none)")
@pytest.mark.parametrize("args", [["-m0=lzma"], ["-m0=lzma2"], ["-m0=copy"], ["-m0=deflate"],
                                  ["-m0=bzip2"], ["-m0=lzma2", "-mhc=off"],
                                  ["-m0=lzma2", "-ms=off"], ["-m0=lzma2", "-mx=9", "-mmt=4"]])
def test_real_7zip_archives(tmp_path, args):
    src = tmp_path / "src"
    src.mkdir()
    for n, b in FILES.items():
        (src / n).write_bytes(b)
    p = str(tmp_path / "real.7z")
    subprocess.run([SEVENZ, "a", "-bd", *args, p] + [str(src / n) for n in FILES], check=True,
                   stdout=subprocess.DEVNULL)
    assert read_all(p) == FILES


@pytest.mark.skipif(not SEVENZ, reason="no 7z binary on PATH (the PC has none)")
def test_real_ppmd_falls_back_to_binary(tmp_path):
    src = tmp_path / "Posts.xml"
    src.write_bytes(FILES["Posts.xml"])
    p = str(tmp_path / "ppmd.7z")
    subprocess.run([SEVENZ, "a", "-bd", "-m0=ppmd", p, str(src)], check=True,
                   stdout=subprocess.DEVNULL)
    with pytest.raises(Z.Unsupported7z):
        read_all(p)
    assert dict((n, f.read()) for n, f in Z.open_members(p, {"Posts.xml"})) == {
        "Posts.xml": FILES["Posts.xml"]}
    assert os.path.getsize(p) > 0
