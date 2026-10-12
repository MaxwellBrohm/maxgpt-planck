"""OOD-H reserve: the ONE definition of which OASST2 trees are held back for OOD-H.

prereg/RC-12.draft.txt s14: the 150 OOD-H Part 1 threads are removed by thread ID from every
training corpus, the tokenizer's included. They are not selected yet, so a deterministic pool is
reserved now: every OASST2 tree whose

    int(sha256("planck-oodh-v1:" + message_tree_id)) % 8 == 0

is in the reserve (about 1 tree in 8). The reserve is excluded from the tokenizer sample and from
every held-out set used to tune anything. OOD-H Part 1 must later draw its 150 threads only from
this pool. Every corpus script calls this function; nothing re-implements the rule.

Removing a tree by id does not remove its text from elsewhere: some OASST2 prompts were submitted
more than once (in other trees) and some also appear as Dolly instructions. leaked_reserve_trees()
lists the reserved trees whose user turns occur verbatim (after prompt_key normalization) outside
the reserve; OOD-H Part 1 must not draw those, and the 150 threads it draws still get the CORPUS
3.2 step 6 13-gram decontamination.
"""
import hashlib
import unicodedata

SALT = "planck-oodh-v1:"
MOD = 8


def in_oodh_reserve(message_tree_id: str) -> bool:
    """True when this OASST2 tree belongs to the OOD-H reserve pool."""
    if not isinstance(message_tree_id, str) or not message_tree_id:
        raise ValueError(f"bad message_tree_id: {message_tree_id!r}")
    digest = hashlib.sha256((SALT + message_tree_id).encode("utf-8")).digest()
    return int.from_bytes(digest, "big") % MOD == 0


def prompt_key(text: str) -> str:
    """A user turn's text for exact-duplicate checks: NFC, casefolded, whitespace collapsed."""
    return " ".join(unicodedata.normalize("NFC", text or "").casefold().split())


def _user_turns(m):
    if m.get("role") == "prompter":
        yield m.get("text") or ""
    for r in m.get("replies") or []:
        yield from _user_turns(r)


LEAK_MIN_CHARS = 40     # shorter turns ("thanks!", "hi") repeat by chance and are not a leak


def leaked_reserve_trees(trees, other_texts=(), min_chars=LEAK_MIN_CHARS) -> set:
    """trees: OASST2 tree dicts (message_tree_id, prompt with nested replies); other_texts: more
    training texts to check against, e.g. every Dolly instruction. -> ids of reserved trees any of
    whose user turns (min_chars or longer after prompt_key) also occurs, as a whole turn, in a
    non-reserved tree or in other_texts."""
    reserved, outside = {}, set()
    for t in trees:
        keys = {k for k in map(prompt_key, _user_turns(t["prompt"])) if len(k) >= min_chars}
        if in_oodh_reserve(t["message_tree_id"]):
            reserved[t["message_tree_id"]] = keys
        else:
            outside |= keys
    outside |= {k for k in map(prompt_key, other_texts) if len(k) >= min_chars}
    return {tid for tid, keys in reserved.items() if keys & outside}

