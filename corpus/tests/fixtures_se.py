"""Hand-made Stack Exchange dump fixture (Posts.xml, Comments.xml) and a tiny stdlib 7z writer.

Every planted case that must NOT reach the output carries a *_MARKER string. write_7z() writes a
minimal but valid 7z (one folder, one coder, substream sizes and CRCs, optional LZMA-encoded header),
so the reader is tested on the PC too, where there is no 7z binary. Nothing here is written into the
repo: tests write to a pytest tmp dir.
"""
import bz2
import lzma
import os
import struct
import xml.etree.ElementTree as ET
import zlib

from sevenzip_hdr import SIG

LIC3, LIC4 = "CC BY-SA 3.0", "CC BY-SA 4.0"
BREAD = ("<p>I bake a loaf of bread every Sunday, but by Wednesday it is dry and hard. "
         "What is the best way to keep it fresh for the whole week at home?</p>")
GOOD = "<p>Keep it in a cloth bag in a cool place, and slice it only when you need it.</p>"


def post(i, kind, created, body, score=1, lic=LIC3, **kw):
    a = {"Id": str(i), "PostTypeId": str(kind), "CreationDate": created, "Score": str(score),
         "Body": body, "OwnerUserId": str(100 + i)}
    if lic:
        a["ContentLicense"] = lic
    a.update({k: str(v) for k, v in kw.items()})
    return a


def comment(i, pid, created, text, lic=LIC3, score=0):
    a = {"Id": str(i), "PostId": str(pid), "Score": str(score), "Text": text,
         "CreationDate": created, "UserId": str(500 + i)}
    if lic:
        a["ContentLicense"] = lic
    return a


def posts():
    p = [post(1, 1, "2012-03-04T10:00:00.123", BREAD, score=7, Title="How do I keep bread fresh?",
              AcceptedAnswerId=3),
         post(2, 2, "2012-03-04T11:00:00.000", GOOD + "<p>A bread box &amp; a tea towel also work "
              "well for most people I know.</p>", score=5, ParentId=1),
         post(3, 2, "2013-01-02T09:00:00.000", "<p>Freeze half of the loaf on the day you bake it, "
              "then thaw slices in the toaster when you want them.</p>", score=2, ParentId=1),
         post(4, 2, "2023-01-05T00:00:00.000", "<p>POST_ANSWER_MARKER I wrote this later.</p>",
              ParentId=1),
         post(5, 2, "2012-05-05T00:00:00.000", "<p>NEG_SCORE_MARKER Throw it away.</p>", score=-2,
              ParentId=1),
         post(6, 2, "2015-05-05T00:00:00.000", "<p>EDITED_LATE_MARKER Use a plastic bag.</p>",
              ParentId=1, LastEditDate="2023-02-01T00:00:00.000"),
         post(7, 2, "2020-05-05T00:00:00.000", "<p>AIISM_ANSWER_MARKER As an AI language model, I "
              "cannot taste bread.</p>", ParentId=1),
         post(8, 2, "2016-05-05T00:00:00.000", "<p>NC_LICENSE_MARKER Wrap it well.</p>",
              lic="CC BY-NC-SA 4.0", ParentId=1),
         post(9, 2, "2010-06-01T00:00:00.000", "<p>An old answer: a clay pot keeps the crust "
              "crisp and the crumb soft for days.</p>", score=1, lic=None, ParentId=1),
         post(10, 1, "2023-03-01T00:00:00.000", "<p>POST_QUESTION_MARKER " + BREAD[3:],
              Title="A question asked too late"),
         post(11, 2, "2023-03-02T00:00:00.000", "<p>LATE_QUESTION_ANSWER_MARKER " + GOOD[3:],
              ParentId=10),
         post(12, 1, "2014-02-02T00:00:00.000", "<p>NO_ANSWER_MARKER " + BREAD[3:] + GOOD,
              Title="Nobody answered this one"),
         post(13, 1, "2019-02-02T00:00:00.000", "<p>AIISM_QUESTION_MARKER Is chat gpt right "
              "about sourdough starters and how long they keep in the fridge?</p>",
              Title="A question about starters"),
         post(14, 2, "2019-02-03T00:00:00.000", "<p>ORPHANED_BY_QUESTION_MARKER " + GOOD[3:],
              ParentId=13),
         post(15, 5, "2012-01-01T00:00:00.000", "<p>TAG_WIKI_MARKER Bread is baked dough.</p>"),
         post(16, 2, "2012-01-01T00:00:00.000", "<p>ORPHAN_MARKER " + GOOD[3:], ParentId=999),
         post(17, 1, "2016-01-01T00:00:00.000", "<p>GERMAN_SE_MARKER Wir sind gestern mit dem Zug "
              "nach Hamburg gefahren und haben dort Freunde besucht, das Wetter war schlecht.</p>",
              Title="Warum ist das Brot hart"),
         post(18, 2, "2016-01-02T00:00:00.000", "<p>Das Essen im kleinen Restaurant war sehr gut "
              "und wir kommen wieder, wenn das Wetter besser ist.</p>", ParentId=17),
         post(19, 1, "2016-01-01T00:00:00.000", "<p>SHORT_MARKER Why?</p>", Title="Hm"),
         post(20, 2, "2016-01-02T00:00:00.000", "<p>Because.</p>", ParentId=19),
         post(21, 1, "2021-06-01T00:00:00.000",
              "<p>My recipe says:</p><blockquote><p>Knead for ten minutes.</p></blockquote>"
              "<p>Steps I follow:</p><ol><li>Mix the flour &amp; water</li><li>Rest it</li></ol>"
              "<pre><code>flour  500 g\n  water 350 g\n</code></pre><p>Is ten minutes "
              "really <em>necessary</em> for a <a href=\"https://example.com/x\">basic</a> "
              "loaf?</p>",
              Title="How long should I knead bread dough?", lic=LIC4),
         post(22, 2, "2022-11-30T23:59:59.990", "<p>Ten minutes by hand is about right, less with "
              "a mixer, and the dough should feel smooth and springy.</p>", ParentId=21, lic=LIC4),
         post(23, 2, "2022-12-01T00:00:00.000", "<p>BOUNDARY_ANSWER_MARKER Knead less.</p>",
              ParentId=21, lic=LIC4)]
    return p


def comments():
    return [comment(1, 1, "2012-03-04T10:30:00.000", "Did you try a bread box?"),
            comment(2, 2, "2023-05-05T00:00:00.000", "POST_COMMENT_MARKER nice"),
            comment(3, 1, "2012-03-05T00:00:00.000", "Possible duplicate of [Keeping bread]"
                    "(https://cooking.stackexchange.com/q/9) TEMPLATE_COMMENT_MARKER"),
            comment(4, 3, "2013-01-03T00:00:00.000", "AIISM_COMMENT_MARKER As an AI language "
                    "model I agree"),
            comment(5, 5, "2012-05-06T00:00:00.000", "PARENT_DROPPED_MARKER agreed"),
            comment(6, 2, "2012-03-06T00:00:00.000", "Thanks, that worked! See "
                    "[this page](https://example.org/bread) for more."),
            comment(7, 3, "2013-01-04T00:00:00.000", "NC_COMMENT_MARKER hmm", lic="CC BY-NC 4.0"),
            comment(8, 3, "2013-01-03T12:00:00.000", "Freezing works for rye too.")]


def xml_bytes(tag, rows):
    root = ET.Element(tag)
    for a in rows:
        ET.SubElement(root, "row", a)
    return b'<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="utf-8")


def write_dump_dir(d):
    os.makedirs(d, exist_ok=True)
    files = {"Badges.xml": xml_bytes("badges", [{"Id": "1", "Name": "Teacher"}]),
             "Comments.xml": xml_bytes("comments", comments()),
             "Posts.xml": xml_bytes("posts", posts())}
    for name, b in files.items():
        with open(os.path.join(d, name), "wb") as f:
            f.write(b)
    return files


def _num(v):
    for n in range(8):
        if v < 1 << (8 * n + 7 - n):
            first = ((0xFF << (8 - n)) & 0xFF) | (v >> (8 * n))
            return bytes([first]) + (v & ((1 << 8 * n) - 1)).to_bytes(n, "little")
    return b"\xff" + v.to_bytes(8, "little")


def _encode(coder, data):
    if coder == "copy":
        return b"\x00", b"", data
    if coder == "bzip2":                        # two concatenated streams, as a parallel encoder may
        h = len(data) // 2
        return b"\x04\x02\x02", b"", bz2.compress(data[:h]) + bz2.compress(data[h:])
    if coder == "lzma2":
        return b"\x21", bytes([16]), lzma.compress(data, format=lzma.FORMAT_RAW, filters=[
            {"id": lzma.FILTER_LZMA2, "dict_size": 1 << 20}])
    props = bytes([(2 * 5 + 0) * 9 + 3]) + (1 << 20).to_bytes(4, "little")
    return b"\x03\x01\x01", props, lzma.compress(data, format=lzma.FORMAT_RAW, filters=[
        {"id": lzma.FILTER_LZMA1, "dict_size": 1 << 20, "lc": 3, "lp": 0, "pb": 2}])


def _streams(pack_pos, packed, cid, props, unpack, subsizes=None, crcs=None):
    b = b"\x06" + _num(pack_pos) + _num(1) + b"\x09" + _num(packed) + b"\x00"
    b += b"\x07\x0b" + _num(1) + b"\x00" + _num(1) + bytes([len(cid) | (0x20 if props else 0)])
    b += cid + (_num(len(props)) + props if props else b"") + b"\x0c" + _num(unpack) + b"\x00"
    if subsizes is not None:
        b += b"\x08\x0d" + _num(len(subsizes))
        b += b"\x09" + b"".join(map(_num, subsizes[:-1])) if len(subsizes) > 1 else b""
        b += b"\x0a\x01" + b"".join(struct.pack("<I", c) for c in crcs) + b"\x00"
    return b + b"\x00"


def write_7z(path, files, coder="lzma2", encode_header=True, cid=None):
    """files: {name: bytes}, in archive order. One solid folder with one coder. cid overrides the
    coder id written into the header (to plant an unsupported coder)."""
    names = list(files)
    data = b"".join(files[n] for n in names)
    real_cid, props, packed = _encode(coder, data)
    cid = cid or real_cid
    hdr = b"\x01\x04" + _streams(0, len(packed), cid, props, len(data),
                                 [len(files[n]) for n in names],
                                 [zlib.crc32(files[n]) for n in names])
    nm = b"\x00" + b"".join(n.encode("utf-16-le") + b"\x00\x00" for n in names)
    hdr += b"\x05" + _num(len(names)) + b"\x11" + _num(len(nm)) + nm + b"\x00\x00"
    body = packed
    if encode_header:
        hcid, hprops, hpacked = _encode("lzma", hdr)
        body += hpacked
        hdr = b"\x17" + _streams(len(packed), len(hpacked), hcid, hprops, len(hdr))
    tail = struct.pack("<QQI", len(body), len(hdr), zlib.crc32(hdr))
    with open(path, "wb") as f:
        f.write(SIG + b"\x00\x04" + struct.pack("<I", zlib.crc32(tail)) + tail + body + hdr)
