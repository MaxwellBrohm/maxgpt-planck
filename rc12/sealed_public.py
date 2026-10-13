"""RC-12 sealed-session kit: what may leave the session (prereg s13 S4: only gate tables, never item text).
Stdlib, no model. A gate table row and a draw-log line are published; a line is published only when it has a plain
summary's characters and none of the sealed content's own words (words in the dev content are public already)."""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
WORD = r"[a-z]{3,}"


def content_words(work):
    """the lowercase words (3+ letters) of the installed content files that dev's content does not hold."""
    import sealed_kit as K
    import sealed_overlap as SO
    import split_content as SPC
    new = set()
    for m in K.CONTENT_MODULES:
        path = os.path.join(work, "rc12", f"content_{m}.py")
        if os.path.isfile(path):
            new |= {w for s in SO.leaves(SPC.load_literals(path)) for w in re.findall(WORD, s.lower())}
    return new - {w for s in SO.leaves(SO.dev_content()) for w in re.findall(WORD, s.lower())}


def redact(line, words):
    """the line with every sealed content word and anything quoted withheld."""
    line = re.sub(r"'[^']*'|\"[^\"]*\"", "'(withheld)'", line)
    return re.sub(WORD, lambda m: "(withheld)" if m.group(0).lower() in words else m.group(0), line, flags=re.I)


def public_line(name, stdout, ok, words=frozenset()):
    """a gate's last output line for the public table: the build's own output is the items, so it is summarized; a
    line with other characters than a plain summary's, or with a sealed content word, is withheld."""
    if name == "build_stdout":
        return "the build reproduces the items" if ok else "the build does not reproduce the items"
    last = (stdout.strip().split("\n") or [""])[-1][:100]
    plain = re.fullmatch(r"[\w .,:;()=<>/%+-]*", last) and redact(last, words) == last
    return last if plain else "(last line withheld: not a plain summary)"
