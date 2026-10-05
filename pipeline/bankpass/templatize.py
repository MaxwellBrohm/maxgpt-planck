"""Value-filled lines -> templates (BANKPASS s2a). The teacher never sees a hole: a call gives concrete values (a pet
kind and name, a relation and job, a topic text, a number word) and the program replaces the exact value strings with
holes afterwards. A line whose value is not found exactly once is dropped; the drop rate is reported per bank.

Holes: v, av (value with its article), old, o (object noun), t (topic), items (a list join), L (list name), A
(assistant name), N (number word), X (call-me name), W (avoid word), n (user name), lab, e, pred (a lookup predicate,
W3), ord; p (a leading subject pronoun, pronoun corrections only); M (marker prefix, added in front of every
correction line)."""
import re

PRONOUNS = ("he", "she", "it", "they")
ORDER = ("items", "av", "old", "v", "o", "t", "L", "A", "N", "X", "W", "n", "lab", "e", "pred", "ord")
OPTIONAL = {"old"}      # a correction may leave out the value it replaces (FAKE head and pronoun forms do)
DROP_MISSING, DROP_REPEATED, DROP_FORM, DROP_JOIN, DROP_PRONOUN = (
    "TPL_VALUE_MISSING", "TPL_VALUE_REPEATED", "TPL_FORM", "TPL_ITEMS_JOIN", "TPL_PRONOUN")


def value_re(v):
    """word-bounded; a capitalized value matches case-sensitively (the month May, not the modal), else any case."""
    flags = 0 if v[:1].isupper() else re.I
    return re.compile(r"(?<![A-Za-z0-9'])" + re.escape(v) + r"(?![A-Za-z0-9])", flags)


def join_forms(items):
    """the program's join forms: banks.join_items ("a, b and c") and the Oxford comma ("a, b, and c")."""
    if len(items) == 1:
        return [items[0]]
    plain = ", ".join(items[:-1]) + " and " + items[-1]
    oxford = ", ".join(items[:-1]) + ", and " + items[-1]
    return [plain] if len(items) == 2 else [oxford, plain]


def _replace_once(text, rx, hole):
    hits = list(rx.finditer(text))
    if not hits:
        return None, DROP_MISSING
    if len(hits) > 1:
        return None, DROP_REPEATED
    m = hits[0]
    return text[:m.start()] + "{" + hole + "}" + text[m.end():], None


def templatize(line, fill, form=None, correction=False):
    """line: a teacher line; fill: {hole: exact value} as given to the teacher ("av" is built from "v" and the
    article in fill["article"], if any). form (key queries and corrections): full, head, pronoun, ellipsis.
    -> (template, None) or (None, drop code)."""
    t = line.strip()
    fill = dict(fill)
    art = fill.pop("article", None)
    if form in ("pronoun", "ellipsis"):
        fill.pop("o", None)          # these forms do not name the object noun
    if fill.get("items") is not None:
        items = fill.pop("items")
        hit = None
        for j in join_forms(items):
            if len(value_re(j).findall(t)) == 1:
                hit = j
                break
        if hit is None:
            return None, DROP_JOIN
        t = value_re(hit).sub("{items}", t, count=1)
    if "v" in fill and art:
        rx = value_re(f"{art} {fill['v']}")
        if rx.search(t):
            t, why = _replace_once(t, rx, "av")
            if why:
                return None, why
            fill.pop("v")
    for hole in ORDER:
        v = fill.get(hole)
        if v is None or hole in ("items", "av"):
            continue
        if hole in OPTIONAL and not value_re(v).search(t):
            continue
        t, why = _replace_once(t, value_re(v), hole)
        if why:
            return None, why
    if form is not None:
        t, why = _form(t, fill, form)
        if why:
            return None, why
    if correction:
        t = _marker_prefix(t)
    return t, None


def _form(t, fill, form):
    """the referring form a key line was asked for must be the one it uses (s3 K: full, head, pronoun, ellipsis)."""
    low = t.lower()
    if form == "full":
        ok = "my {o}" in low or ("o" not in fill and "my " in low)
    elif form == "head":
        ok = "the {o}" in low and "my {o}" not in low
    elif form == "pronoun":
        m = re.match(r"^(\W*)(he|she|it|they)(?=\W)", t, re.I)
        if not m or "{o}" in t:
            return None, DROP_PRONOUN
        t = t[:m.start(2)] + "{p}" + t[m.end(2):]
        ok = True
    elif form == "ellipsis":
        ok = "{o}" not in t and not re.match(r"^\W*(he|she|it|they)\b", t, re.I)
    else:
        raise ValueError(form)
    return (t, None) if ok else (None, DROP_FORM)


def _marker_prefix(t):
    """corrections get the program's marker: '{M}' in front, the first letter lowered unless it is a hole or 'I'."""
    if t.startswith("{"):
        return "{M}" + t
    first = re.match(r"[A-Za-z']+", t)
    if first and (first.group(0) == "I" or first.group(0).startswith("I'")):
        return "{M}" + t
    return "{M}" + t[:1].lower() + t[1:]


def holes_of(template):
    return sorted(set(re.findall(r"\{(\w+)\}", template)))
