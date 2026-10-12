"""7z header parsing for sevenzip_min.py (split out to keep each file short). Pure functions over the
decoded header bytes; nothing here touches the packed data. Reference: 7-Zip's DOC/7zFormat.txt.
parse_streams() -> ((pack_pos, [packed sizes]), [folder], ([[substream sizes]], [[substream crc]]));
a folder is {"coders": [(codec id, props, n_out)], "n_out", "bound", "npacked", "sizes", "crc"}.
parse_header() -> (main streams or None, (names, empty flags)).
"""
import struct

SIG = b"7z\xbc\xaf\x27\x1c"


class Unsupported7z(Exception):
    pass


class Bad7z(Exception):
    pass


class Reader:
    def __init__(self, b):
        self.b, self.i = b, 0

    def byte(self):
        if self.i >= len(self.b):
            raise Bad7z("header ends early")
        self.i += 1
        return self.b[self.i - 1]

    def take(self, n):
        v = self.b[self.i:self.i + n]
        if len(v) < n:
            raise Bad7z("header ends early")
        self.i += n
        return v

    def num(self):
        first, mask, val = self.byte(), 0x80, 0
        for i in range(8):
            if not first & mask:
                return val | ((first & (mask - 1)) << (8 * i))
            val |= self.byte() << (8 * i)
            mask >>= 1
        return val

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def bits(self, n):
        out, mask, cur = [], 0, 0
        for _ in range(n):
            if not mask:
                cur, mask = self.byte(), 0x80
            out.append(bool(cur & mask))
            mask >>= 1
        return out

    def defined(self, n):
        return [True] * n if self.byte() else self.bits(n)

    def expect(self, v):
        got = self.byte()
        if got != v:
            raise Bad7z(f"expected property {v:#x}, got {got:#x}")


def _pack_info(r):
    pos, n, sizes = r.num(), r.num(), []
    while (t := r.byte()) != 0:
        if t == 0x09:
            sizes = [r.num() for _ in range(n)]
        elif t == 0x0A:
            for d in r.defined(n):
                if d:
                    r.u32()
        else:
            raise Bad7z(f"pack info property {t:#x}")
    return pos, sizes


def _folder(r):
    coders, n_in = [], 0
    for _ in range(r.num()):
        flag = r.byte()
        if flag & 0x80:
            raise Unsupported7z("alternative coder methods")
        cid = bytes(r.take(flag & 0x0F))
        nin, nout = (r.num(), r.num()) if flag & 0x10 else (1, 1)
        props = bytes(r.take(r.num())) if flag & 0x20 else b""
        coders.append((cid, props, nout))
        n_in += nin
    n_out = sum(c[2] for c in coders)
    bound = [(r.num(), r.num())[1] for _ in range(n_out - 1)]
    npacked = n_in - len(bound)
    if npacked > 1:
        for _ in range(npacked):
            r.num()
    return {"coders": coders, "n_out": n_out, "bound": bound, "npacked": npacked}


def unpack_size(f):
    main = [i for i in range(f["n_out"]) if i not in f["bound"]]
    return f["sizes"][main[0]]


def _unpack_info(r):
    r.expect(0x0B)
    folders = [None] * r.num()
    if r.byte():
        raise Unsupported7z("external folder records")
    folders = [_folder(r) for _ in folders]
    r.expect(0x0C)
    for f in folders:
        f["sizes"] = [r.num() for _ in range(f["n_out"])]
    while (t := r.byte()) != 0:
        if t != 0x0A:
            raise Bad7z(f"unpack info property {t:#x}")
        for f, d in zip(folders, r.defined(len(folders))):
            f["crc"] = r.u32() if d else None
    return folders


def _substreams(r, folders):
    nums, t = [1] * len(folders), r.byte()
    if t == 0x0D:
        nums, t = [r.num() for _ in folders], r.byte()
    sizes = []
    for f, n in zip(folders, nums):
        s = [r.num() for _ in range(n - 1)] if t == 0x09 and n else []
        sizes.append(s + [unpack_size(f) - sum(s)] if n else [])
    if t == 0x09:
        t = r.byte()
    crcs = [[None] * n for n in nums]
    if t == 0x0A:
        slots = [(i, j) for i, (f, n) in enumerate(zip(folders, nums))
                 for j in range(n) if not (n == 1 and f.get("crc") is not None)]
        for (i, j), d in zip(slots, r.defined(len(slots))):
            crcs[i][j] = r.u32() if d else None
        t = r.byte()
    for i, (f, n) in enumerate(zip(folders, nums)):
        if n == 1 and f.get("crc") is not None:
            crcs[i][0] = f["crc"]
    if t != 0:
        raise Bad7z(f"substreams property {t:#x}")
    return sizes, crcs


def parse_streams(r):
    pack = folders = sub = None
    while (t := r.byte()) != 0:
        if t == 0x06:
            pack = _pack_info(r)
        elif t == 0x07:
            folders = _unpack_info(r)
        elif t == 0x08:
            sub = _substreams(r, folders)
        else:
            raise Bad7z(f"streams property {t:#x}")
    if sub is None:
        sub = ([[unpack_size(f)] for f in folders], [[f.get("crc")] for f in folders])
    return pack, folders, sub


def _files(r):
    n, names, empty = r.num(), [], None
    while (t := r.byte()) != 0:
        data = bytes(r.take(r.num()))
        if t == 0x0E:
            empty = Reader(data).bits(n)
        elif t == 0x11:
            if data[0]:
                raise Unsupported7z("external file names")
            names = data[1:].decode("utf-16-le").split("\x00")[:n]
    return names, empty or [False] * n


def parse_header(hdr):
    """The plain (kHeader) header -> (main streams or None, (names, empty-stream flags))."""
    r = Reader(hdr)
    r.expect(0x01)
    main, files = None, ([], [])
    while (t := r.byte()) != 0:
        if t == 0x02:
            while r.byte():
                r.take(r.num())
        elif t == 0x03:
            parse_streams(r)
        elif t == 0x04:
            main = parse_streams(r)
        elif t == 0x05:
            files = _files(r)
        else:
            raise Bad7z(f"header property {t:#x}")
    return main, files
