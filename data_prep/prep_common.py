"""Shared pieces of data_prep: import paths, tokenizer loading, the special-string rule, doc keys and
split keys, file hashing and atomic writes.

Flat imports, as in harness/ and corpus/: this module puts harness/, corpus/ and tokenizer/ on sys.path
(appended, so a data_prep module name always wins). Nothing here imports torch.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (os.path.join(ROOT, "harness"), os.path.join(ROOT, "corpus"), os.path.join(ROOT, "tokenizer")):
    if _p not in sys.path:
        sys.path.append(_p)

from oodh import in_oodh_reserve                   # noqa: E402  the ONE OOD-H rule
from sample_plan import SPECIAL_RE, SPECIAL_STRINGS  # noqa: E402  the tokenizer-sample rule

FORMAT_VERSION = "planck-pretok-v1"
TOK_DIR = os.path.join(ROOT, "tokenizer", "v0")
CHAT_ROLES = ("system", "user", "assistant", "tool")


class Refusal(ValueError):
    """Input that must stop a build (OOD-H tree, overlap with training, bad role); exit code 3."""


# ---------------------------------------------------------------- tokenizer
def tokenizer_path(vocab: int | None = None, path: str | None = None) -> str:
    """--tokenizer wins; otherwise the v0 family member of that size (the sizes are nested, P-098)."""
    if path:
        return os.path.abspath(path)
    import spec
    assert vocab in spec.NESTED_SIZES, f"v0 sizes are {spec.NESTED_SIZES}, not {vocab}"
    return os.path.join(TOK_DIR, spec.file_name(int(vocab)))


def load_tokenizer(path: str):
    """Exactly harness/data.py load_tokenizer (a test checks they agree): a control-token string typed in
    text becomes plain byte pieces, never a control id. Copied, not imported, because harness/data.py
    imports torch and every pretokenize worker would pay for it."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(path)
    tok.encode_special_tokens = True
    return tok


def tokenizer_info(path: str) -> dict:
    """File hash, size and the control ids the shards depend on. A file inside tokenizer/v0 must match
    tok_v0_manifest.json, so a stale copy is refused."""
    import spec
    tok = load_tokenizer(path)
    sha = sha256_file(path)
    base = os.path.basename(path)
    man = os.path.join(TOK_DIR, "tok_v0_manifest.json")
    if os.path.dirname(os.path.abspath(path)) == TOK_DIR and os.path.exists(man):
        with open(man, encoding="utf-8") as f:
            files = json.load(f)["files"]
        if base in files:
            assert files[base]["sha256"] == sha, f"{base} differs from tok_v0_manifest.json"
    ids = {s: tok.token_to_id(s) for s in spec.CONTROL_TOKENS}
    assert all(v is not None for v in ids.values()), f"{base} lacks control tokens: {ids}"
    return {"file": base, "sha256": sha, "vocab": tok.get_vocab_size(),
            "pad_id": ids["<|pad|>"], "eot_id": ids["<|endoftext|>"], "end_id": ids["<|end|>"],
            "role_ids": {"system": ids["<|system|>"], "user": ids["<|user|>"],
                         "assistant": ids["<|assistant|>"], "tool": ids["<|tool|>"]},
            "control_ids": sorted(ids.values()), "n_special": len(spec.SPECIALS)}


def encoder(tok):
    return lambda s: tok.encode(s, add_special_tokens=False).ids   # as harness/data.py build_loader


# ---------------------------------------------------------------- records
def clean_text(s: str) -> tuple[str, bool]:
    """The tokenizer-sample contract (corpus/sample_plan.py): every control or markup token string in
    corpus text becomes one space. Returns (text, changed)."""
    if not s:
        return s or "", False
    t = SPECIAL_RE.sub(" ", s)
    return t, t != s


def is_chat(rec: dict) -> bool:
    return isinstance(rec.get("turns"), list) and len(rec["turns"]) > 0


def doc_text(rec: dict) -> str:
    """The record's whole text: "text", or for a chat record without it, the turns joined by one newline
    (corpus/readers_chat.render_turns)."""
    if rec.get("text") is not None:
        return rec["text"]
    return "\n".join(t.get("text") or "" for t in rec.get("turns") or [])


def record_keys(rec: dict) -> list[str]:
    """Every key that ties a record to its split side: its id, meta.split_key (Dolly rows sharing a
    context) and meta.tree_id (OASST2 tree). Held-out exclusion and eval-set disjointness use all three."""
    meta = rec.get("meta") or {}
    keys = [rec["id"]]
    for k in ("split_key", "tree_id"):
        v = meta.get(k)
        if isinstance(v, str) and v and v not in keys:
            keys.append(v)
    return keys


def check_oodh(rec: dict) -> None:
    """Refuse any OASST2 record from the OOD-H reserve (corpus/oodh.py). An OASST2 record without a tree
    id is refused too: the rule cannot be applied to it."""
    meta = rec.get("meta") or {}
    tid = meta.get("tree_id")
    oasst = rec.get("source") == "oasst2" or meta.get("dataset") == "OpenAssistant/oasst2"
    if oasst and not tid:
        raise Refusal(f"OASST2 record without meta.tree_id: {rec.get('id')}")
    if tid and in_oodh_reserve(tid):
        raise Refusal(f"OOD-H reserved tree in the input: {tid} ({rec.get('id')})")


def text_hash(text: str) -> str:
    """16 hex chars of sha1 over the cleaned text, for exact-duplicate checks across splits."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def doc_key(seed: int, source: str, doc_id: str) -> int:
    """64-bit key: bucket = key % n_shards and the order inside a shard. It depends only on (seed,
    source, id), so the output does not depend on input order, chunking or worker count."""
    h = hashlib.sha256(f"{FORMAT_VERSION}:{seed}:{source}:{doc_id}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


# ---------------------------------------------------------------- files
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def code_hashes(names) -> dict:
    """sha256 of the code a manifest depends on (paths relative to the repo root)."""
    return {n: sha256_file(os.path.join(ROOT, n)) for n in names}


def home_rel(path: str) -> str:
    """A path for a manifest: under the home directory it is written as ~/..., so no user name lands in
    a file that may be committed."""
    p = os.path.abspath(path)
    home = os.path.expanduser("~")
    return "~" + p[len(home):] if p == home or p.startswith(home + os.sep) else p
