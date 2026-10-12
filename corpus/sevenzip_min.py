"""Minimal read-only 7z reader on the standard library (lzma, zlib, bz2), for the Stack Exchange dumps.

The PC has no 7z binary and no py7zr (checked 2026-09-26), and py7zr is not pure Python, so this reads
the archives directly. It streams: iter_members() yields one member at a time as a file-like object
that decompresses on demand, so memory stays bounded whatever the member size.

Supported: 7z format 0.x; plain or encoded (compressed) headers; any number of folders (solid blocks),
each with exactly ONE coder out of Copy, LZMA, LZMA2, Deflate and BZip2. Substream CRC32s are checked
as each member is read to its end (a skipped member is still decompressed and checked). Anything else
(AES, PPMd, BCJ/BCJ2 chains, external or multi-coder folders) raises Unsupported7z; open_members()
then falls back to a 7z binary (7zz, 7z or 7za) if one is on PATH.
Format reference: 7-Zip's DOC/7zFormat.txt.
"""
import bz2
import lzma
import shutil
import struct
import subprocess
import zlib

from sevenzip_hdr import SIG, Bad7z, Unsupported7z, Reader, parse_header, parse_streams, unpack_size

CHUNK = 1 << 20
LZMA_DICT_MAX = (1 << 30) + (1 << 29)        # liblzma's largest dictionary


def _decoder(cid, props):
    if cid == b"\x00":
        return None
    if cid == b"\x21":
        p = props[0]
        d = LZMA_DICT_MAX if p >= 40 else (2 | (p & 1)) << (p // 2 + 11)
        return lzma.LZMADecompressor(lzma.FORMAT_RAW, filters=[
            {"id": lzma.FILTER_LZMA2, "dict_size": min(d, LZMA_DICT_MAX)}])
    if cid == b"\x03\x01\x01":
        d, size = props[0], int.from_bytes(props[1:5], "little")
        return lzma.LZMADecompressor(lzma.FORMAT_RAW, filters=[
            {"id": lzma.FILTER_LZMA1, "lc": d % 9, "lp": d // 9 % 5, "pb": d // 45,
             "dict_size": max(4096, min(size, LZMA_DICT_MAX))}])
    if cid == b"\x04\x01\x08":
        return zlib.decompressobj(-15)
    if cid == b"\x04\x02\x02":
        return bz2.BZ2Decompressor()
    raise Unsupported7z(f"coder {cid.hex()}")


def _chunks(fh, offset, packsize, f):
    """Decompressed chunks of one folder."""
    if len(f["coders"]) != 1 or f["npacked"] != 1:
        raise Unsupported7z(f"folder with {len(f['coders'])} coders")
    dec, left = _decoder(*f["coders"][0][:2]), packsize
    fh.seek(offset)

    def raw():
        nonlocal left
        b = fh.read(min(CHUNK, left))
        if not b and left:
            raise Bad7z("packed stream truncated")
        left -= len(b)
        return b
    if dec is None:
        while left:
            yield raw()
    elif isinstance(dec, (lzma.LZMADecompressor, bz2.BZ2Decompressor)):
        while True:
            if dec.eof:
                if not isinstance(dec, bz2.BZ2Decompressor) or not (dec.unused_data or left):
                    break
                data, dec = dec.unused_data, bz2.BZ2Decompressor()   # concatenated bzip2 streams
            elif dec.needs_input:
                if not left:
                    break
                data = raw()
            else:
                data = b""
            out = dec.decompress(data, max_length=CHUNK)
            if out:
                yield out
    else:
        while left:
            out = dec.decompress(raw(), CHUNK)
            while out:
                yield out
                out = dec.decompress(dec.unconsumed_tail, CHUNK) if dec.unconsumed_tail else b""
        yield dec.flush()


class Member:
    """One archive member, read sequentially: read(n) returns b'' at its end."""

    def __init__(self, name, size, crc, feed):
        self.name, self.size, self.crc, self._feed = name, size, crc, feed
        self._left, self._crc = size, 0

    def read(self, n=-1):
        n = self._left if n is None or n < 0 else min(n, self._left)
        parts, got = [], 0
        while got < n:
            parts.append(self._feed(n - got))
            got += len(parts[-1])
        out = b"".join(parts)
        self._left -= len(out)
        self._crc = zlib.crc32(out, self._crc)
        if not self._left and self.crc is not None and self._crc != self.crc:
            raise Bad7z(f"{self.name}: CRC mismatch")
        return out

    def drain(self):
        while self.read(CHUNK):
            pass


def _load_header(fh):
    head = fh.read(32)
    if head[:6] != SIG:
        raise Bad7z("not a 7z archive")
    off, size, crc = struct.unpack("<QQI", head[12:32])
    fh.seek(32 + off)
    hdr = fh.read(size)
    if len(hdr) != size or zlib.crc32(hdr) != crc:
        raise Bad7z("next header truncated or CRC mismatch")
    while hdr[:1] == b"\x17":                                  # encoded header
        (pos, psz), folders, _ = parse_streams(Reader(memoryview(hdr)[1:]))
        hdr = b"".join(_chunks(fh, 32 + pos, psz[0], folders[0]))[:unpack_size(folders[0])]
    return parse_header(hdr)


def iter_members(path):
    """Yield Member objects in archive order (empty entries such as directories are skipped).
    Consume each before advancing; whatever is left unread is drained (and CRC-checked)."""
    with open(path, "rb") as fh:
        main, (names, empty) = _load_header(fh)
        if main is None:
            return
        (pos, psizes), folders, (sizes, crcs) = main
        names = iter(n for n, e in zip(names, empty) if not e)
        offset = 32 + pos
        for fi, f in enumerate(folders):
            gen, pending = _chunks(fh, offset, psizes[fi], f), [b""]

            def feed(n, gen=gen, pending=pending):
                while not pending[0]:
                    pending[0] = next(gen, None)
                    if pending[0] is None:
                        raise Bad7z("folder stream ends early")
                b, pending[0] = pending[0][:n], pending[0][n:]
                return b
            for size, crc in zip(sizes[fi], crcs[fi]):
                m = Member(next(names), size, crc, feed)
                yield m
                m.drain()
            offset += psizes[fi]


def open_members(path, wanted):
    """Yield (name, file-like) for each member in `wanted`, in archive order. Stdlib reader first;
    on Unsupported7z, falls back to a 7z binary on PATH (one extraction pass per member)."""
    try:
        with open(path, "rb") as fh:
            main, _ = _load_header(fh)
        for f in (main[1] if main else []):
            if len(f["coders"]) != 1 or f["npacked"] != 1:
                raise Unsupported7z(f"folder with {len(f['coders'])} coders")
            _decoder(*f["coders"][0][:2])
    except Unsupported7z:
        exe = shutil.which("7zz") or shutil.which("7z") or shutil.which("7za")
        if not exe:
            raise
    else:
        for m in iter_members(path):
            if m.name in wanted:
                yield m.name, m
        return
    for name in sorted(wanted):
        p = subprocess.Popen([exe, "e", "-so", path, name], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL)
        yield name, p.stdout
        p.stdout.close()
        if p.wait() != 0:
            raise Bad7z(f"{exe} failed on {name}")
