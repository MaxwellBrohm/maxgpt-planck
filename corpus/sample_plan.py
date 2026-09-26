"""Planning arithmetic for make_tok_sample.py: groups, byte targets and deterministic selection.

Each document gets u = hash(seed, source, key) in [0, 1), where key is the record's
meta.split_key (the OASST2 tree id; for Dolly a hash of the shared Wikipedia context) or else the
doc id, so every doc of one tree or one context shares one u and one fate. Within a source, docs
are ordered by u; the held-out set is the lowest-u prefix, and the training sample is the next
prefix. The two are disjoint by construction, by doc id and by OASST2 tree id.

Groups (target shares of the training bytes the trainer reads, repeats counted; flags in
make_tok_sample.py). The research asks for the sample "weighted as in training"
(research/followup/tokenizer.md), so the defaults follow CORPUS 2.1 at 3-30M:
    chat  = oasst2 + dolly + irc   37%   S 25% + RH 8% + HD 4%, all chat-shaped
    web   = cccc                   28%   \
    se    = stackexchange          15%    | P (plain prose, 50-55%) and N (narrative, 8-13%);
    books = gutenberg              10%    | public-domain books stand in for N's fiction
    wiki  = wikimedia              10%   /
Inside chat: the chat pilot (S, RH) does not exist yet, so the assistant-style human chat (OASST2,
Dolly) stands in for it. Both are taken whole and repeated up to CHAT_MAX_REPEAT = 16 times
(CORPUS 2.1's repeat limit for a chat-shaped HD set), one integer repeat count for both, written to
the manifest's "weights" (tokenizer/sample_io.py honors it). IRC is one HD source and is capped at
IRC_MAX_SHARE = 4% of the total (all of HD is 4% at 3-30M); it is never repeated. Whole chat
covers the chat target first, up to the cap; IRC fills what is left up to its own cap.

Short supply: 'scale' (default) shrinks the total so every share still holds, so the total flag is
a ceiling; 'fill' keeps the total and hands the missing bytes to the other groups in proportion to
their shares. The chat capacity is 16 x (OASST2 + Dolly) + min(IRC, 4% of the total).
"""
import hashlib
import re

import numpy as np

# Control and markup token strings never enter the tokenizer sample (they would teach merges such
# as '<|'); each occurrence becomes one space. The role tokens and END_TOKEN match
# harness/chat_template.py (a test checks it); pad, endoftext and the tags follow PLAN Phase 1.
SPECIAL_STRINGS = ["<|pad|>", "<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>",
                   "<|end|>", "<|tool|>", "<note>", "</note>", "<think>", "</think>", "<lookup>",
                   "</lookup>", "<result>", "</result>"]
SPECIAL_RE = re.compile("|".join(re.escape(s) for s in sorted(SPECIAL_STRINGS, key=len,
                                                               reverse=True)))

GROUPS = {"chat": ["oasst2", "dolly", "irc"], "web": ["cccc"], "se": ["stackexchange"],
          "books": ["gutenberg"], "wiki": ["wikimedia"]}
DEFAULT_SHARES = {"chat": 0.37, "web": 0.28, "se": 0.15, "books": 0.10, "wiki": 0.10}
WHOLE_CHAT = ("oasst2", "dolly")
CHAT_MAX_REPEAT = 16
IRC_MAX_SHARE = 0.04
ARCHAIC_MAX_OF_PROSE = 0.25          # CORPUS 3.5: archaic public-domain prose <= 25% of prose
SOURCE_GROUP = {s: g for g, ss in GROUPS.items() for s in ss}


def unit_hash(seed, source: str, key: str) -> float:
    d = hashlib.sha256(f"{seed}\x1f{source}\x1f{key}".encode("utf-8")).digest()
    return int.from_bytes(d[:8], "big") / 2.0 ** 64


def check_shares(shares, allow_archaic=False):
    if set(shares) != set(GROUPS):
        raise ValueError(f"shares must name exactly {sorted(GROUPS)}")
    if any(v < 0 for v in shares.values()) or abs(sum(shares.values()) - 1) > 1e-6:
        raise ValueError(f"shares must be >= 0 and sum to 1: {shares}")
    prose = 1 - shares["chat"]
    if not allow_archaic and shares["books"] > ARCHAIC_MAX_OF_PROSE * prose + 1e-9:
        raise ValueError(f"books share {shares['books']} exceeds the CORPUS 3.5 archaic cap "
                         f"(25% of the prose share {prose:.2f}); pass --allow-archaic-over")


def take_prefix(b: np.ndarray, target: float) -> int:
    """How many leading docs (bytes b) to take so the sum lands closest to target."""
    if target <= 0 or len(b) == 0:
        return 0
    cum = np.cumsum(b)
    k = int(np.searchsorted(cum, target, side="left"))
    if k >= len(b):
        return len(b)
    before = cum[k - 1] if k else 0
    return k + 1 if cum[k] - target < target - before else k


def extend_ties(u_sorted: np.ndarray, n: int) -> int:
    """Grow a prefix of length n so it never splits docs that share one u (one OASST2 tree)."""
    while 0 < n < len(u_sorted) and u_sorted[n] == u_sorted[n - 1]:
        n += 1
    return n


def group_targets(avail, shares, total, mode="scale"):
    """avail: group -> bytes available for training. -> group -> target bytes."""
    live = {g: s for g, s in shares.items() if s > 0}
    if mode == "scale":
        f = min([1.0] + [avail[g] / (s * total) for g, s in live.items()])
        return {g: shares[g] * total * f for g in shares}
    if mode != "fill":
        raise ValueError(mode)
    tgt, remaining = {g: 0.0 for g in shares}, float(total)
    while live:
        norm = sum(live.values())
        short = [g for g, s in live.items() if avail[g] < remaining * s / norm]
        if not short:
            tgt.update({g: remaining * s / norm for g, s in live.items()})
            break
        for g in short:
            tgt[g] = float(avail[g])
            remaining -= avail[g]
            del live[g]
    return tgt


def chat_capacity(avail_s, total, max_repeat=CHAT_MAX_REPEAT, irc_share=IRC_MAX_SHARE):
    whole = sum(avail_s.get(s, 0) for s in WHOLE_CHAT)
    return max_repeat * whole + min(avail_s.get("irc", 0), irc_share * total)


def plan(avail_s, shares, total, mode="scale", max_repeat=CHAT_MAX_REPEAT,
         irc_share=IRC_MAX_SHARE):
    """-> (group targets, the planned total). In 'scale' mode the IRC cap shrinks with the total,
    so the total is found by fixed-point iteration (it contracts by irc_share / chat share)."""
    avail_g = {g: sum(avail_s.get(s, 0) for s in ss) for g, ss in GROUPS.items()}
    t = float(total)
    for _ in range(200):
        avail_g["chat"] = chat_capacity(avail_s, t, max_repeat, irc_share)
        gt = group_targets(avail_g, shares, total, mode)
        new = total if mode == "fill" else sum(gt.values())
        if abs(new - t) < 0.5:
            break
        t = new
    return gt, t


def source_targets(gt, avail_s, total, max_repeat=CHAT_MAX_REPEAT, irc_share=IRC_MAX_SHARE):
    """Split group targets over sources. -> (unique-byte target per source, chat repeat count)."""
    out = {srcs[0]: gt[g] for g, srcs in GROUPS.items() if len(srcs) == 1}
    chat, whole = gt["chat"], sum(avail_s.get(s, 0) for s in WHOLE_CHAT)
    irc_cap = min(avail_s.get("irc", 0), irc_share * total, chat)
    need, w = chat - irc_cap, 1
    if whole <= 0:
        out.update({s: 0.0 for s in WHOLE_CHAT})
    elif need <= whole:
        out.update({s: avail_s.get(s, 0) * need / whole for s in WHOLE_CHAT})
    else:
        w = min(max_repeat, int(need // whole))
        if need - w * whole > 0.5 * whole and w < max_repeat:
            w += 1
        out.update({s: float(avail_s.get(s, 0)) for s in WHOLE_CHAT})
    out["irc"] = max(0.0, min(irc_cap, chat - w * min(whole, need)))
    return out, w


V0_NOTES = [
    "D8: CCCC files carry no per-document license (the card's domain review kept 537 domains); "
    "used only with --allow-unrecorded-license, pending Max's ruling.",
    "Date gate: StackExchange is gated on the question date; threads with a post-cutoff marker "
    "(AI-ism or a year 2023-2029) are dropped, later answers without a marker are not.",
    "Date gate: Wikimedia is gated on the last revision; dump-time template text is caught only by "
    "the post-cutoff marker check.",
    "OASST2 and Dolly were written in 2023 (exempt from the date gate, CORPUS 2.2); the AI-ism "
    "filter (hygiene.aiism) removes turns and rows with AI self-reference or chat-model names, "
    "not all model-influenced text.",
    "Hygiene: CORPUS 3.2 steps 2 (MinHash near-dedup), 4 (quality heuristics) and 6 (13-gram "
    "decontamination) are not implemented; step 3 is a line-level stand-in on CCCC and Wikimedia.",
    "IRC nicknames are replaced for people who speak in the same channel-day; mentions of absent "
    "people stay.",
    "No chat pilot yet: OASST2 and Dolly stand in for the S, RH and HD share, repeated up to 16x.",
]
