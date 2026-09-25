"""Teacher text -> turns (SPEC section 5, "Parse"). The render prompt asks for one line per planned label, in order,
then END. Labels are global: role letter (U user, A assistant, T tool) + turn index + 1, so every label is unique.

Tolerated drift (recorded in "drift", never a reject): CRLF, blank lines, trailing spaces, one leading EMPTY thought
block, one code fence around the whole output, lowercase labels, spaces around the colon, a copied bracket note after
the label ("U3 [copy exactly]: ..."), bold around the label only ("**U3:** ..."), one pair of quotes around a whole
line, "END." or "end", and two planned labels merged on one line in the planned order (split back).
Everything else is a FORMAT_* or THOUGHT_TAG code: FORMAT_LINES (labels missing, extra, out of order, duplicated,
or no END), FORMAT_EXTRA (text before the first label or after END), FORMAT_WRAP (an unlabeled line between labeled
lines), THOUGHT_TAG (a non-empty thought block, or channel/turn tags left in the text)."""
import re
import unicodedata

LABEL_RE = re.compile(r"^\s*(?:\*\*)?\s*([UATuat])\s*(\d+)\s*(?:\*\*)?\s*(?:\[[^\]\n]*\])?\s*(?:\*\*)?\s*:\s*(?:\*\*)?"
                      r"\s?(.*)$")
END_RE = re.compile(r"^\s*(?:\*\*)?\s*END\s*\.?\s*(?:\*\*)?\s*$", re.I)
MERGED_RE = re.compile(r"\s+([UAT])(\d+)\s*:\s")
THOUGHT_OPEN = re.compile(r"^\s*<\|channel>\s*thought\s*(.*?)<channel\|>", re.S)
TAGS = re.compile(r"<\|?(?:turn|channel)\|?>|<\|channel>|<channel\|>")
QUOTES = {('"', '"'), ("\u201c", "\u201d"), ("'", "'"), ("\u2018", "\u2019")}
ROLE_LETTER = {"user": "U", "assistant": "A", "tool": "T"}


def plan(skel):
    """[(label, turn index, role)] in order."""
    return [(ROLE_LETTER[t["role"]] + str(t["i"] + 1), t["i"], t["role"]) for t in skel["turns"]]


def normalize(s):
    """NFKC, straight quotes, collapsed whitespace: the EXACT_MISMATCH comparison form."""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return " ".join(s.split())


def _unquote(text, drift):
    t = text.strip()
    if len(t) >= 2 and (t[0], t[-1]) in QUOTES and t.count(t[0]) + (t.count(t[-1]) if t[-1] != t[0] else 0) == 2:
        drift.append("quoted")
        return t[1:-1].strip()
    return t


def _strip_wrappers(raw, codes, drift):
    s = raw.replace("\r\n", "\n").replace("\r", "\n")
    m = THOUGHT_OPEN.match(s)
    if m:
        if m.group(1).strip():
            codes.append(("THOUGHT_TAG", "non-empty thought block"))
        else:
            drift.append("empty_thought_block")
        s = s[m.end():]
    st = s.strip()
    if st.startswith("```") and st.endswith("```") and st.count("```") == 2:
        drift.append("fence")
        st = st[3:-3]
        st = st.split("\n", 1)[1] if "\n" in st and not LABEL_RE.match(st.split("\n", 1)[0]) else st
    return st


def _split_merged(label_num, text, expected_next, drift):
    """split 'text A4: more' when A4 is the next planned label; returns [(label, text)]."""
    out, cur_label, rest = [], label_num, text
    while True:
        m = MERGED_RE.search(rest)
        nxt = expected_next(cur_label)
        if not m or nxt is None or m.group(1) + m.group(2) != nxt:
            out.append((cur_label, rest))
            return out
        drift.append("merged")
        out.append((cur_label, rest[:m.start()]))
        cur_label, rest = nxt, rest[m.end():]


def parse(raw, planned):
    """raw teacher text + plan(skel) -> {"turns": {turn index: text}, "codes": [(code, detail)], "drift": [str],
    "labels": [label seen, in order]}."""
    codes, drift = [], []
    body = _strip_wrappers(raw or "", codes, drift)
    order = [lab for lab, _, _ in planned]
    pos = {lab: n for n, lab in enumerate(order)}

    def expected_next(lab):
        n = pos.get(lab)
        return order[n + 1] if n is not None and n + 1 < len(order) else None

    seen, texts, ended, started = [], {}, False, False
    for line in body.split("\n"):
        if not line.strip():
            continue
        if ended:
            codes.append(("FORMAT_EXTRA", "after END: " + line.strip()[:40]))
            continue
        if END_RE.match(line):
            ended = True
            continue
        m = LABEL_RE.match(line)
        if not m:
            codes.append(("FORMAT_WRAP" if started else "FORMAT_EXTRA", line.strip()[:40]))
            continue
        started = True
        lab = m.group(1).upper() + m.group(2)
        for sub_lab, sub_text in _split_merged(lab, m.group(3), expected_next, drift):
            seen.append(sub_lab)
            if sub_lab in texts:
                codes.append(("FORMAT_LINES", "duplicate " + sub_lab))
                continue
            texts[sub_lab] = _unquote(sub_text.rstrip(), drift)
    if not ended:
        codes.append(("FORMAT_LINES", "no END"))
    if seen != order:
        missing = [x for x in order if x not in seen]
        extra = [x for x in seen if x not in pos]
        detail = f"missing {missing[:3]}" if missing else (f"extra {extra[:3]}" if extra else "order")
        codes.append(("FORMAT_LINES", detail))
    turns = {i: texts[lab] for lab, i, _ in planned if lab in texts}
    if re.search(r"^\s*[uat]\d", body, re.M):
        drift.append("lowercase_label")
    return {"turns": turns, "codes": codes, "drift": sorted(set(drift)), "labels": seen}


def serialize(skel, texts):
    """turn texts {i: text} -> the canonical teacher output (used by the fake teacher and the tests)."""
    lines = [f"{lab}: {texts.get(i, '')}" for lab, i, _ in plan(skel)]
    return "\n".join(lines + ["END"])
