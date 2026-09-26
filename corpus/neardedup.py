"""MinHash near-duplicate detection, the per-document half (CORPUS 3.2 step 2, pass A).

Every stage1 shard gets a sidecar <stem>.mh.npy with one row per document, in shard order:
    dtype [('day', '<u4'), ('sig', '<u4', (126,))]
'day' is the document's date as a proleptic ordinal (datetime.date.toordinal) for the keep rule,
UNDATED (0xFFFFFFFF) when it has none. 'sig' is its MinHash signature. The global half (LSH
banding, union-find, the keep rule, drop lists) is neardedup_lsh.py; paragraph dedup is
neardedup_para.py.

Parameters (fixed; the sidecar is only valid with these):
- Shingles: word 5-grams of the normalized text. Normalization: lowercase; NFKD with combining
  marks removed (cafe == café); every digit run becomes '0' (dates, counters and page numbers do
  not make copies look different); words are maximal runs of letters and digits, so punctuation,
  whitespace and '_' only separate words. A text of 1-4 words is one shingle.
- Shingle ids: crc32 per word, splitmix64, then a polynomial over the 5 words and splitmix64
  again (64-bit ids, order-sensitive).
- Signature: NPERM = 126 minima of h_k(x) = ((a_k*x + b_k) mod 2^64) >> 32 (multiply-shift,
  a_k odd), a_k, b_k from numpy's PCG64 seeded with SEED. Stored as uint32.
- LSH: BANDS = 14 bands x ROWS = 9 rows (CORPUS 3.2, Essential-Web's setting). A pair with
  shingle Jaccard J becomes a candidate with probability 1-(1-J^9)^14: 0.13 at J=0.6, 0.44 at
  0.7, 0.66 at 0.75, 0.87 at 0.8, 0.97 at 0.85, 0.999 at 0.9 (S-curve midpoint (1/14)^(1/9) =
  0.746).
- Verification (VERIFY = 0.7, CORPUS's nominal Jaccard): a candidate edge counts only when the
  signature estimate (share of the 126 positions that agree) is at least 0.7. Combined detection
  probability: 0.001 at J=0.6, 0.21 at 0.7, 0.59 at 0.75, 0.86 at 0.8, 0.975 at 0.85, 0.999 at
  0.9. So the working threshold is about J=0.75-0.8, and pairs under 0.65 essentially never merge,
  which limits union-find chaining through template-heavy pages.
A text with no words (only punctuation or symbols) gets a signature drawn from its bytes, so it
matches only an identical text. tests/test_neardedup.py checks detection per Jaccard bin against
these curves, and the signature values are pinned (identical on the Mac and the PC).

Use (pass A, in the worker that already holds the normalized text of each kept document):
    w = SidecarWriter(stem); w.add(text, meta.get("created")) per doc in shard order; w.close()
or, when a worker sketches and a parent commits: rows = sketch(texts, created) in the worker,
then SidecarWriter(stem).extend(rows[committed]) in shard order. An append-only shard writer
(stream_io's frames) can instead append sig_row(text, created).tobytes() per committed line to
<stem>.mh.u32 (MH_RAW_SUFFIX; crash recovery truncates it to 508 x committed lines, like the keys
file); load_sidecar reads either form. Cost: about 16 MB/s per core on
the Mac (M5), 5 KB web docs; the PC figure is in the scale log.
"""
import datetime
import hashlib
import os
import re
import unicodedata
import zlib

import numpy as np

NGRAM = 5
BANDS, ROWS = 14, 9
NPERM = BANDS * ROWS
VERIFY = 0.7
SEED = 20260926
BLOCK = 4096                   # shingles per matrix block: memory per doc stays about 4 MB
UNDATED = 0xFFFFFFFF
MH_SUFFIX = ".mh.npy"
MH_RAW_SUFFIX = ".mh.u32"      # the same rows, headerless and appendable (508 bytes per doc)
SIG_DTYPE = np.dtype([("day", "<u4"), ("sig", "<u4", (NPERM,))])
ND_SUFFIX = ".nd.npy"          # pass B result per shard (neardedup_lsh.py)
ND_DTYPE = np.dtype([("keep", "u1"), ("csize", "<u4"), ("lshard", "<i4"), ("lrow", "<u4")])
PARAMS = dict(ngram=NGRAM, bands=BANDS, rows=ROWS, nperm=NPERM, verify=VERIFY, seed=SEED,
              hash="multiply-shift 64->32 on splitmix64 5-gram ids")

_U64 = np.uint64
_rng = np.random.Generator(np.random.PCG64(SEED))
_A = (_rng.integers(0, 2**64 - 1, NPERM, dtype=np.uint64, endpoint=True) | _U64(1))
_B = _rng.integers(0, 2**64 - 1, NPERM, dtype=np.uint64, endpoint=True)
_POLY = _U64(0x9E3779B97F4A7C15)
_BAND_SEEDS = _rng.integers(0, 2**64 - 1, BANDS, dtype=np.uint64, endpoint=True)
_WORD = re.compile(r"[^\W_]+")
_DIGITS = re.compile(r"\d+")
_MARKS = re.compile("[̀-ͯ᪰-᫿᷀-᷿⃐-⃿︠-︯]")
_ISO = re.compile(r"^\s*(\d{4})(?:-(\d{2})-(\d{2}))?")


def mix64(z: np.ndarray) -> np.ndarray:
    """splitmix64 finalizer, elementwise on a uint64 array (wraps mod 2^64)."""
    z = (z ^ (z >> _U64(30))) * _U64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> _U64(27))) * _U64(0x94D049BB133111EB)
    return z ^ (z >> _U64(31))


def normalize_for_dedup(text: str) -> str:
    t = text.lower()
    if not t.isascii():
        t = _MARKS.sub("", unicodedata.normalize("NFKD", t))
    return _DIGITS.sub("0", t)


def words(text: str) -> list:
    return _WORD.findall(normalize_for_dedup(text))


def shingles(text: str, n: int = NGRAM) -> np.ndarray:
    """Sorted unique 64-bit ids of the text's word n-grams (empty for a text with no words)."""
    ws = words(text)
    if not ws:
        return np.zeros(0, dtype=np.uint64)
    wh = mix64(np.array([zlib.crc32(w.encode("utf-8")) for w in ws], dtype=np.uint64))
    m, k = max(1, len(ws) - n + 1), min(n, len(ws))
    acc = wh[:m].copy()
    for j in range(1, k):
        acc = acc * _POLY + wh[j:j + m]
    return np.unique(mix64(acc))


def minhash(sh: np.ndarray) -> np.ndarray:
    """uint32[NPERM] signature of a non-empty shingle id array."""
    out = np.full(NPERM, 0xFFFFFFFF, dtype=np.uint64)
    for s in range(0, len(sh), BLOCK):
        v = (sh[s:s + BLOCK, None] * _A + _B) >> _U64(32)
        np.minimum(out, v.min(axis=0), out=out)
    return out.astype(np.uint32)


def signature(text: str) -> np.ndarray:
    sh = shingles(text)
    if sh.size:
        return minhash(sh)
    raw = hashlib.shake_128(text.encode("utf-8")).digest(4 * NPERM)
    return np.frombuffer(raw, dtype="<u4").astype(np.uint32)


def est_jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signature estimate of Jaccard: share of equal positions (rows of a vs rows of b)."""
    return (np.asarray(a) == np.asarray(b)).mean(axis=-1)


def band_hashes(sigs: np.ndarray) -> np.ndarray:
    """(n, NPERM) uint32 signatures -> (BANDS, n) uint64 band keys (band-major, so one band of a
    shard is contiguous). Band b hashes its 9 values with splitmix64 from a per-band seed."""
    sigs = np.asarray(sigs, dtype=np.uint32).reshape(-1, NPERM)
    out = np.empty((BANDS, len(sigs)), dtype=np.uint64)
    for b in range(BANDS):
        h = np.full(len(sigs), _BAND_SEEDS[b], dtype=np.uint64)
        for j in range(b * ROWS, (b + 1) * ROWS):
            h = mix64(h ^ sigs[:, j].astype(np.uint64))
        out[b] = h
    return out


def day_number(created) -> int:
    """'2019-03-04...' or '2019' or 2019 -> date ordinal; anything else (None, '', junk) ->
    UNDATED. Only the keep rule uses it; the date gate is hygiene.date_status."""
    if created is None or isinstance(created, bool):
        return UNDATED
    if isinstance(created, (int, np.integer)):
        created = f"{int(created):04d}"
    m = _ISO.match(str(created))
    if not m:
        return UNDATED
    try:
        y, mo, d = int(m.group(1)), int(m.group(2) or 1), int(m.group(3) or 1)
        return datetime.date(y, mo, d).toordinal() if y >= 1 else UNDATED
    except ValueError:
        return UNDATED


def sig_row(text: str, created=None) -> np.ndarray:
    """One document's sidecar row (a 0-d SIG_DTYPE array; .tobytes() is 508 bytes)."""
    row = np.zeros((), dtype=SIG_DTYPE)
    row["sig"], row["day"] = signature(text), day_number(created)
    return row


def sketch(texts, created=None) -> np.ndarray:
    """-> SIG_DTYPE array for a list of texts (created: parallel list, or None for undated)."""
    texts = list(texts)
    out = np.zeros(len(texts), dtype=SIG_DTYPE)
    for i, t in enumerate(texts):
        out["sig"][i] = signature(t)
        out["day"][i] = day_number(created[i] if created is not None else None)
    return out


def save_atomic(path: str, arr: np.ndarray):
    """np.save to path via a temp file, fsync and rename (a crash never leaves a short file)."""
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class SidecarWriter:
    """Pass A: add(text, created) once per kept document, in shard order; close() writes
    <stem>.mh.npy and returns its path. Memory: 508 bytes per document."""

    def __init__(self, stem: str):
        self.path, self.rows = stem + MH_SUFFIX, []

    def add(self, text: str, created=None):
        self.rows.append(sig_row(text, created))

    def extend(self, rows: np.ndarray):
        """Append rows already sketched elsewhere (e.g. by a worker, via sketch())."""
        self.rows.extend(np.asarray(rows, dtype=SIG_DTYPE))

    def close(self) -> str:
        arr = np.array(self.rows, dtype=SIG_DTYPE) if self.rows else np.zeros(0, SIG_DTYPE)
        save_atomic(self.path, arr)
        return self.path


def load_sidecar(stem: str, mmap: bool = True) -> np.ndarray:
    """<stem>.mh.npy, else the headerless <stem>.mh.u32."""
    if not os.path.exists(stem + MH_SUFFIX) and os.path.exists(stem + MH_RAW_SUFFIX):
        path, size = stem + MH_RAW_SUFFIX, os.path.getsize(stem + MH_RAW_SUFFIX)
        if size % SIG_DTYPE.itemsize:
            raise ValueError(f"{path}: {size} bytes is not a whole number of rows")
        if size == 0:
            return np.zeros(0, dtype=SIG_DTYPE)
        return np.memmap(path, dtype=SIG_DTYPE, mode="r") if mmap else np.fromfile(path, SIG_DTYPE)
    arr = np.load(stem + MH_SUFFIX, mmap_mode="r" if mmap else None)
    if arr.dtype != SIG_DTYPE:
        raise ValueError(f"{stem}{MH_SUFFIX}: dtype {arr.dtype}, expected {SIG_DTYPE}")
    return arr
