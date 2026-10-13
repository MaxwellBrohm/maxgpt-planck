"""RC-12 sealed-session kit, part 3: content skeletons and the s13 S1 separation check. Stdlib, no model.

  python3 -B rc12/sealed_overlap.py skeleton OUT_DIR    one content_<module>.py per content module, holding every
                                                        CONTENT name at its DEV value (the shape to rewrite)
  python3 -B rc12/sealed_kit.py overlap WORK            (calls main below) S1 against this repository's dev split

S1 as checked (prereg s13 S1, operational form): (a) every value pool, the holder lists and the PERSIST rule words
are disjoint from dev's, case-insensitive, except the closed natural classes WEEKDAY, MONTH and LANGUAGE (there are
only 7 weekdays; dev already uses 11 months; K answers are world knowledge), which a sealed split cannot avoid
sharing; (b) 0 word 5-grams of user text shared with dev's user text (every turn of every record); (c) no content
string of 5+ words identical to a dev one (templates, IDEAL wording); (d) no short content string (1 to 4 words:
objects, aliases, topics, nouns, short frames and questions such as "What's my name?") identical to a dev one,
identifier fields, the closed classes and the slots of D_SKIP and D_SKIP_NAMES aside (filler labels, BIND's
name-probe object words, the avoid lists). Prints counts and at most 10 examples per check; examples are
sealed text and stay in the session; OVERLAP.txt gets the counts only."""
import importlib
import json
import os
import pprint
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sealed_kit as K                      # noqa: E402
import split_content as SPC                 # noqa: E402

CLOSED = {"WEEKDAY", "MONTH", "LANGUAGE"}
DISJOINT = {("pools_vals", n) for n in ("COLOUR", "PETNAME", "PERSON", "ANAME", "CITY", "TOWN", "JOB", "INSTRUMENT",
                                         "FOOD", "PROJECT", "BOAT", "PUPPY", "BAND", "QUIZTEAM", "CAPITAL",
                                         "HOLDER_F", "HOLDER_M", "HOLDER_N")} | \
           {("fam_persist", n) for n in ("FIRST_WORDS", "CLOSINGS", "WORDS")}


def dev_content():
    """{module: {name: dev value}} read from this repository's modules on the dev split."""
    if os.environ.get("RC12_SPLIT", "dev") != "dev":
        sys.exit("run the skeleton and the overlap check without RC12_SPLIT (they read dev)")
    out = {}
    for m in K.CONTENT_MODULES:
        mod = importlib.import_module(m)
        out[m] = {n.rstrip("*"): getattr(mod, n.rstrip("*")) for n in mod.CONTENT}
    return out


def skeleton(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for m, consts in dev_content().items():
        lines = [f'"""Sealed content for {m}.py (prereg s13 S1, S3): rewrite every value below. Keep each name, '
                 f'type, dict key,\nlist length and {{placeholder}} (split_content.shape_diff checks them); '
                 f'change the wording and values."""', ""]
        for n, v in consts.items():
            lines.append(f"{n} = " + pprint.pformat(v, width=116, sort_dicts=False))
        with open(os.path.join(out_dir, f"content_{m}.py"), "w") as f:
            f.write("\n".join(lines) + "\n")
    print(f"wrote {len(K.CONTENT_MODULES)} skeletons to {out_dir}")


def leaves(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(k)
            yield from leaves(v)
    elif isinstance(x, (list, tuple, set, frozenset)):
        for v in x:
            yield from leaves(v)


def words(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def grams(texts, n=5):
    out = set()
    for t in texts:
        w = words(t)
        out.update(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    return out


def user_texts(path):
    with open(path, encoding="utf-8") as f:
        return [t["text"] for line in f for t in json.loads(line)["turns"]]


def report(name, bad, summary):
    summary.append(f"S1 {name}: {len(bad)} {'PASS' if not bad else 'FAIL'}")
    print(summary[-1])
    for b in sorted(bad, key=str)[:10]:          # examples are sealed text: the screen only, never OVERLAP.txt
        print("   ", b)
    return len(bad)


# short strings code fixes, so a sealed split must keep them: the filler labels, the avoid lists (filters that hold
# the fixed object words, such as OWN's pool names), and BIND's name-probe object words
D_SKIP = {"pools_fill_a.FILLERS_A": (2,), "pools_fill_b.FILLERS_B": (2,), "fam_bind.BIND_NAME": ("objw",)}
D_SKIP_NAMES = {"fam_bind.BIND_AVOID", "fam_own_topic.OWN_AVOID", "fam_role_lookup.LOOKUP_AVOID"}


def short_values(content, free):
    """every short content string (1 to 4 words, no placeholder): list items, dict values, and the keys of dicts
    whose keys are free (OWN_LIST topics, PROJECTS); identifier fields (split_content.PINS), structural dict keys
    and the closed classes are left out, as are the slots code fixes (D_SKIP, D_SKIP_NAMES)."""
    closed = {v.lower() for m, n in (("pools_vals", c) for c in CLOSED) for v in content[m][n]}

    def walk(x, pins, keys):
        if isinstance(x, str):
            if "{" not in x and 1 <= len(words(x)) <= 4 and x.lower() not in closed:
                yield x.lower()
        elif isinstance(x, dict):
            for k, v in x.items():
                if keys and isinstance(k, str):
                    yield from walk(k, (), False)
                if k not in pins:
                    yield from walk(v, pins, False)
        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                if not (isinstance(x, tuple) and i in pins):
                    yield from walk(v, pins, False)
    out = set()
    for m, consts in content.items():
        for n, v in consts.items():
            if (m, n) not in DISJOINT and n not in CLOSED and f"{m}.{n}" not in D_SKIP_NAMES:
                skip = tuple(SPC.PINS.get(f"{m}.{n}", ())) + D_SKIP.get(f"{m}.{n}", ())
                out |= set(walk(v, skip, f"{m}.{n}" in free))
    return out


def findings(dev, new, dev_texts, new_texts, free=frozenset()):
    """the S1 lists: (a) shared pool values, (b) shared user-text 5-grams, (c) shared 5+ word strings, (d) shared
    short content strings outside the (a) lists (objects, aliases, topics, nouns, short frames)."""
    pools = []
    across = {v.lower() for (m, n) in DISJOINT for v in leaves(dev[m][n])}
    for m, n in sorted(DISJOINT):
        mine = set(map(str.lower, leaves(new[m][n])))
        pools += [f"{m}.{n}: {v}" for v in mine & set(map(str.lower, leaves(dev[m][n])))]
        pools += [f"{m}.{n}: {v} (a dev value of another pool)" for v in mine & across]
    shared = [" ".join(g) for g in grams(new_texts) & grams(dev_texts)]

    def long_strings(content):
        return {" ".join(words(s)) for consts in content.values() for v in consts.values() for s in leaves(v)
                if len(words(s)) >= 5}
    return (sorted(set(pools)), sorted(shared), sorted(long_strings(new) & long_strings(dev)),
            sorted(short_values(new, free) & short_values(dev, free)))


def main(work):
    if os.path.exists(os.path.join(os.path.dirname(HERE), K.MARKER)):
        sys.exit("run the overlap check from the repository, not from a work copy (it reads dev from here)")
    dev = dev_content()
    new = {m: SPC.load_literals(os.path.join(work, "rc12", f"content_{m}.py")) for m in K.CONTENT_MODULES}
    free = {f"{m}.{n.rstrip('*')}" for m in K.CONTENT_MODULES for n in importlib.import_module(m).CONTENT
            if n.endswith("*")}
    a, b, c, d = findings(dev, new, user_texts(os.path.join(HERE, "dev", "rc12_dev.jsonl")),
                          user_texts(os.path.join(work, K.ITEMS)), free)
    summary = [f"commit {K.commit_of(work)}; items sha256 {K.sha256(os.path.join(work, K.ITEMS))}"]
    fails = report(f"(a) value pools, holders and rule words disjoint from dev (closed classes {sorted(CLOSED)} "
                   "aside)", a, summary)
    fails += report("(b) word 5-grams of user text shared with dev", b, summary)
    fails += report("(c) content strings of 5+ words identical to dev", c, summary)
    fails += report("(d) short content strings (objects, aliases, topics, frames) shared with dev", d, summary)
    summary.append(f"S1 overall: {'PASS' if not fails else 'FAIL'} ({fails} findings)")
    print(summary[-1])
    with open(os.path.join(work, "OVERLAP.txt"), "w") as f:     # counts only; archived and hashed by pack (S5)
        f.write("\n".join(summary) + "\n")
    return 1 if fails else 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "skeleton":
        skeleton(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "overlap":
        sys.exit(main(os.path.abspath(sys.argv[2])))
    else:
        sys.exit(__doc__)
