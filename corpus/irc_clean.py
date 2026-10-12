"""Ubuntu IRC log cleanup, including the nickname scrub of CORPUS 3.2 step 5 ("IRC nicknames are
mapped to stable speaker labels, not deleted").

Raw lines (Common Pile ubuntu_irc) look like '[HH:MM] <nick> message', '[HH:MM]  * nick action' and
'=== nick is now known as other' (join, part and rename noise). clean_irc():
1. drops the [HH:MM] stamps and the '===' system lines;
2. gives every speaker (a '<nick>' or '* nick' line) a label P1, P2, ... in order of first
   appearance in the document (one channel-day). IRC nicknames are case-insensitive, so 'LinDol'
   and 'lindol' share a label. The label is stable within a document and means nothing across
   documents;
3. writes '<nick> msg' as 'P1: msg' and '* nick waves' as '* P1 waves';
4. replaces mentions of a speaker's nickname inside any line ('P2, try this', 'ask P2') with that
   speaker's label, matching whole nickname-shaped tokens case-insensitively. A nickname that is
   an ordinary English word (common_words(), built by make_common_words.py) keeps its mentions,
   so a speaker called 'help' does not erase the word 'help' from the log; its speaker label is
   still replaced; so do a nickname under 3 characters and one used as a word far more often than
   its owner speaks (said more than 3 x lines spoken + 5 times in the document: the live-CD default
   nick 'ubuntu' in #ubuntu, or 'sudo'). Nicknames of people who never speak in the document are
   not known here and stay.
"""
import os
import re
from collections import Counter

_TS = re.compile(r"^\[\d{1,2}:\d{2}(?::\d{2})?\]\s?")
_NICK = re.compile(r"^<([^>\s]{1,40})>\s?")
_ACTION = re.compile(r"^\*\s+([^\s*]{1,40})(?:\s|$)")
_SPLIT = re.compile(r"([A-Za-z0-9_\[\]\\`^{|}-]+)")    # runs of the characters a nickname may use
MIN_MENTION = 3        # 1-2 character nicknames ("a", "jo") keep their mentions
WORDLIKE_RATIO, WORDLIKE_SLACK = 3, 5
WORDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "common_words_en.txt")
_CACHE = {}


def common_words(path=WORDS_PATH) -> frozenset:
    """The committed common-word list (lowercase). A missing file is an error, never an empty
    guard: without it every speaker mention in the log would be treated as a nickname."""
    if path not in _CACHE:
        with open(path, encoding="utf-8") as f:
            words = frozenset(w.strip() for w in f if w.strip() and not w.startswith("#"))
        if len(words) < 1000:
            raise ValueError(f"{path}: only {len(words)} words; is it the right file?")
        _CACHE[path] = words
    return _CACHE[path]


def clean_irc(text: str, common=None) -> str:
    common = common_words() if common is None else common
    rows, labels, spoke = [], {}, Counter()
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = _TS.sub("", line).strip()
        if not line or line.startswith("==="):
            continue
        m = _NICK.match(line) or _ACTION.match(line)
        key = m.group(1).lower() if m else None
        if key is not None:
            labels.setdefault(key, f"P{len(labels) + 1}")
            spoke[key] += 1
        kind = "text" if m is None else ("msg" if line.startswith("<") else "act")
        rows.append((kind, key, _SPLIT.split(line[m.end():].strip() if m else line)))
    said = Counter(t.lower() for _, _, parts in rows for t in parts[1::2] if t.lower() in labels)
    mention = {k: v for k, v in labels.items() if len(k) >= MIN_MENTION and k not in common
               and said[k] <= WORDLIKE_RATIO * spoke[k] + WORDLIKE_SLACK}
    out = []
    for kind, key, parts in rows:
        if mention:
            parts[1::2] = [mention.get(t.lower(), t) for t in parts[1::2]]
        body = "".join(parts)
        if kind == "msg":
            out.append(f"{labels[key]}: {body}".rstrip())
        elif kind == "act":
            out.append(f"* {labels[key]} {body}".rstrip())
        else:
            out.append(body)
    return "\n".join(out)
