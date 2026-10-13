"""RC-12 lock dry run: a synthetic content set with every value renamed (rc12/notes.txt STEP 15). Stdlib, no model.

The sealed session writes new content; a dry run cannot. But a dry run on DEV content cannot catch a gate, fixture,
fake or mutant that names a dev value ("teal", "groovy", "Nova") instead of reading the live pools, which would fail
on any real sealed content. This writes the eleven content_<module>.py files with every value of the S1 (a) lists (pools
except the closed classes, holders, PERSIST rule words) replaced by a made-up word, the same replacement applied
whole-word to every content string (templates naming a value, PROJECTS keys, COMPOSE nouns, K answers), so the
content keeps its shape and its internal consistency. Templates keep their dev wording, so S1 (b) to (d) still
fail on it; the point is the S4 gates.
  python3 -B rc12/dryrun_content.py OUT_DIR [--templates]   (--templates also rewords every sentence)"""
import hashlib
import os
import pprint
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sealed_kit as K                      # noqa: E402
import sealed_overlap as SO                 # noqa: E402

CONS, VOWS = "bdfgklmnprtvz", "aeiou"


def pseudo(word, taken):
    """a made-up lowercase word for one dev word, stable across runs and unique; it ends in s exactly when the dev
    word does (so the inflection rule's plural-value branch still has a value to work on, as dev's "drums")."""
    for salt in range(100):
        h = hashlib.md5(f"rc12-dryrun:{word}:{salt}".encode()).digest()
        w = "".join(CONS[h[i] % len(CONS)] + VOWS[h[i + 1] % len(VOWS)] for i in range(0, 6, 2))
        w += "s" if word.endswith("s") else "x"[:h[7] % 2]
        if w not in taken:
            taken.add(w)
            return w
    raise SystemExit(f"no pseudo word for {word}")


def mapping(dev):
    """dev value (any case) -> made-up value with the same capitalization and word count."""
    taken, out = set(), {}
    values = sorted({v for (m, n) in SO.DISJOINT for v in SO.leaves(dev[m][n])}, key=lambda v: (-len(v), v))
    for v in values:
        parts = [pseudo(p.lower(), taken) for p in v.split()]
        out[v] = " ".join(p.capitalize() if o[:1].isupper() else p for p, o in zip(parts, v.split()))
    return out


def reword(s):
    """--templates: a sentence of 3+ words gets "just" before its closing mark, so a gate that keys on dev
    wording sees new wording. "just" is a stopword of the content-word checks (text_e004.STOPWORDS) and no grader
    cue; a content word added to every sentence would make a U-diff question share a word with its correction."""
    m = re.fullmatch(r"(.*\S)([.?!])", s, re.S)
    return f"{m.group(1)} just{m.group(2)}" if m and len(s.split()) >= 3 else s


def rename(x, rx, table, words=False):
    if isinstance(x, str):
        x = rx.sub(lambda m: table[m.group(0)], x)
        return reword(x) if words else x
    if isinstance(x, dict):
        return {rename(k, rx, table): rename(v, rx, table, words) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(rename(v, rx, table, words) for v in x)
    return x


def main(out_dir, words=False):
    dev = SO.dev_content()
    table = mapping(dev)
    rx = re.compile(r"(?<![A-Za-z])(" + "|".join(re.escape(v) for v in table) + r")(?![A-Za-z])")
    os.makedirs(out_dir, exist_ok=True)
    for m, consts in dev.items():
        lines = [f'"""Dry-run content for {m}.py: dev content with every S1 (a) value renamed (dryrun_content.py)."""',
                 ""]
        for n, v in consts.items():
            lines.append(f"{n} = " + pprint.pformat(rename(v, rx, table, words), width=116, sort_dicts=False))
        with open(os.path.join(out_dir, f"content_{m}.py"), "w") as f:
            f.write("\n".join(lines) + "\n")
    print(f"wrote {len(K.CONTENT_MODULES)} content files to {out_dir}; {len(table)} values renamed")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3) or sys.argv[2:] not in ([], ["--templates"]):
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:] == ["--templates"])
