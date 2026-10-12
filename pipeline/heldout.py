"""Held-out gate (SPEC section 9). Imports the E004 held-out lists read-only (never copies or edits them) and
builds three checks: HELDOUT_VOCAB (words and phrases that must never appear in training text, plus every digit),
HELDOUT_ECHO (word 5-grams and whole-sentence normalized frames shared with the E004 eval, dev and probe texts)
and HELDOUT_STRUCT (skeleton-level limits that keep E004's H1-H5 axes unseen).
RC-12 lists are loaded from pipeline/heldout_ext/ when the sealed session has exported them; until then
RC12_STATUS is "pending" and admit.py must refuse to mark anything trainable."""
import hashlib
import json
import os
import re
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
E004_CODE = os.path.normpath(os.path.join(HERE, "..", "experiments", "E004_general_updating", "code"))
if E004_CODE not in sys.path:
    sys.path.append(E004_CODE)

import heldout_e004 as H  # noqa: E402  (read-only import)
import text_e004 as TX  # noqa: E402

EXT_DIR = os.path.join(HERE, "heldout_ext")
MAX_CORRECTIONS = 3
MAX_D = 10
MAX_SLOTS_PER_TYPE = 2

# ---- vocabulary ----------------------------------------------------------------------------------------------
# markers: the eval marker strings, widened to their stems where the stem itself is unmistakably a marker
MARKER_TERMS = ["hold on", "correction:", "one more change", "oops", "on second thought", "one more thing"]
HONORIFIC_RE = re.compile(r"(?<![A-Za-z])(?:mr|mrs|ms|dr)(?:\.|(?![A-Za-z]))", re.I)
DIGIT_RE = re.compile(r"[0-9٠-٩۰-۹⁰-⁹₀-₉０-９]")


def _phrase_re(p):
    """word-bounded, case-insensitive, optional plural; a plural phrase also bans its singular."""
    stem = p[:-1] if p.endswith("s") and not p.endswith("ss") else p
    return r"(?<![A-Za-z0-9])" + re.escape(stem) + r"(?:s|es)?(?![A-Za-z0-9])"


def _load_ext_vocab():
    path = os.path.join(EXT_DIR, "vocab.txt")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def _load_ext_13grams():
    path = os.path.join(EXT_DIR, "ngram13_sha256.txt")
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip()}


HELDOUT_PHRASES = sorted(set(H.all_heldout_objects()) | set(H.SPORT))
EXT_VOCAB, EXT_13, RC12_STATUS, _VOCAB_TERMS, _VOCAB_RE, _HASH = [], set(), "pending", [], None, None


def load_ext(ext_dir=None):
    """(re)load the RC-12 exports from ext_dir (default pipeline/heldout_ext/) and rebuild the vocabulary gate."""
    global EXT_DIR, EXT_VOCAB, EXT_13, RC12_STATUS, _VOCAB_TERMS, _VOCAB_RE, _HASH
    EXT_DIR = ext_dir or os.path.join(HERE, "heldout_ext")
    EXT_VOCAB, EXT_13 = _load_ext_vocab(), _load_ext_13grams()
    RC12_STATUS = "loaded" if (EXT_VOCAB or EXT_13) else "pending"
    _VOCAB_TERMS = HELDOUT_PHRASES + MARKER_TERMS + list(H.ALIAS_SURNAMES) + EXT_VOCAB
    _VOCAB_RE = re.compile("|".join("(" + _phrase_re(t) + ")" for t in _VOCAB_TERMS), re.I)
    _HASH = None


load_ext()

# head nouns of held-out object pairs: banned from POOLS and KEY NOUNS (stricter than the text gate, which uses
# the phrases): the program never picks a held-out object word as a slot value or an object noun.
HELDOUT_HEADS = sorted({h for pairs in H.EVAL_OBJECTS.values() for _, h in pairs}
                       | {h for pairs in H.H6_OBJECT_PAIRS.values() for _, h in pairs} | set(H.E001_OBJECTS))
_HEADS_RE = re.compile("|".join(_phrase_re(t) for t in HELDOUT_HEADS), re.I)


def vocab_hits(text):
    """held-out terms found in text (HELDOUT_VOCAB), honorifics, and 'digit' if any digit character occurs."""
    out = [m.group(0).lower() for m in _VOCAB_RE.finditer(text)]
    out += [m.group(0) for m in HONORIFIC_RE.finditer(text)]
    if DIGIT_RE.search(text):
        out.append("digit")
    return out


def pool_ok(value):
    """a pool value or object noun is usable: no held-out term, no held-out head noun, no digit, and no word 5-gram
    shared with the E004 texts (a multi-word item such as a topic is training text too; step 3 found one)."""
    return not vocab_hits(value) and not _HEADS_RE.search(value) and not echo5(value)


def echo5(text):
    """word 5-grams of text shared with the E004 eval, dev and probe texts."""
    grams5 = echo_sets()[0]
    w = TX.words(text)
    return [" ".join(g) for g in (tuple(w[i:i + 5]) for i in range(len(w) - 4)) if g in grams5]


# ---- E004 text echo --------------------------------------------------------------------------------------------
_ECHO = None


def _collect_e004():
    """E004 eval, dev and probe texts collected the way ngram_overlap.eval_texts does (no train modules)."""
    import items_e004 as I
    texts, frames = set(), set()
    for name in I.DRAWS:
        for items in I.draw(name).values():
            for it in items:
                terms = [t for ph, h in it["objects"] for t in (ph, h)] + ([it["alias"]] if it["alias"] else [])
                for u, a in it["turns"]:
                    texts |= {u, a}
                    frames |= {tuple(TX.normalize(u, terms, it["values"])), tuple(TX.normalize(a, terms, it["values"]))}
                texts |= {it["question"], it["prefix"] + " " + it["gold"]}
                frames |= {tuple(TX.normalize(it["question"], terms, it["values"])),
                           tuple(TX.normalize(it["prefix"] + " {v}", terms, it["values"]))}
    grams5 = set()
    for t in texts:
        w = TX.words(t)
        grams5 |= {tuple(w[i:i + 5]) for i in range(len(w) - 4)}
    frames = {f for f in frames if len(f) >= 3}
    return grams5, frames


def echo_sets():
    global _ECHO
    if _ECHO is None:
        _ECHO = _collect_e004()
    return _ECHO


_SENT = re.compile(r"[^.!?]+[.!?]?")


def echo_hits(text, obj_terms=(), values=()):
    """HELDOUT_ECHO: shared word 5-grams with E004 texts, and whole-turn or whole-sentence frames equal to an E004
    frame after normalization (objects -> <o>, values -> <v>)."""
    grams5, frames = echo_sets()
    w = TX.words(text)
    out = ["5gram:" + " ".join(g) for g in (tuple(w[i:i + 5]) for i in range(len(w) - 4)) if g in grams5]
    pieces = {text} | {s.strip() for s in _SENT.findall(text) if s.strip()}
    hit = {tuple(TX.normalize(p, obj_terms, values)) for p in pieces} & frames
    out += sorted("frame:" + " ".join(f) for f in hit)
    if EXT_13:
        for i in range(len(w) - 12):
            h = hashlib.sha256(" ".join(w[i:i + 13]).encode()).hexdigest()
            if h in EXT_13:
                out.append("rc12_13gram")
    return out


def frame_echo(a, b, obj_terms=(), values=()):
    """E004 frame echo (text_e004.echo_runs) between two texts, used between a query and a statement."""
    return TX.echo_runs(TX.normalize(a, obj_terms, values), TX.normalize(b, obj_terms, values))


# ---- gate hash ---------------------------------------------------------------------------------------------------
def gate_hash():
    global _HASH
    if _HASH:
        return _HASH
    grams5, frames = echo_sets()
    blob = json.dumps({"vocab": sorted(_VOCAB_TERMS), "heads": HELDOUT_HEADS, "marker": MARKER_TERMS,
                       "g5": sorted(" ".join(g) for g in grams5), "frames": sorted(" ".join(f) for f in frames),
                       "ext13": sorted(EXT_13), "limits": [MAX_CORRECTIONS, MAX_D, MAX_SLOTS_PER_TYPE]})
    _HASH = hashlib.sha256(blob.encode()).hexdigest()[:16]
    return _HASH
