"""E004 mutation test, part A (pure Python, no torch, no model): the free-generation grader (gen_grade.py) and
the likelihood scorer (metrics_ft.right, byte copy of E002). Part B (mutation_e004_b.py) covers the generation
stop logic with a scripted stub model.
Each mutant is a weaker grader or scorer. It is KILLED when at least one fixture's verdict changes (strict, or
the failing-clause report); a crash is NOT a kill. For every grader mutant the log says whether a strict verdict
flipped (a wrong reply scored right, or a right one scored wrong). The baseline must match every fixture.
usage: python3 -B mutation_e004.py   (writes ../logs/mutation_e004.txt; exit 0 only if everything is killed)"""
import contextlib, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_grade as G
import grade_fixtures as F
import metrics_ft as MF

LOG = os.path.join(os.path.dirname(HERE), "logs", "mutation_e004.txt")
E002_NEG = re.compile(r"\b(not|no longer|never|instead of|rather than|isn't|wasn't|aren't|won't)\b|n't\b", re.I)


@contextlib.contextmanager
def patched(mod, **attrs):
    old = {k: getattr(mod, k) for k in attrs}
    for k, v in attrs.items():
        setattr(mod, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(mod, k, v)


def checks(**repl):
    """G.CHECKS with some clauses replaced (a function) or dropped (None)."""
    out = dict(G.CHECKS)
    for k, fn in repl.items():
        n = int(k[1:])
        if fn is None:
            del out[n]
        else:
            out[n] = fn
    return out


def _c4_cands_only(c):
    return not any(G.value_hits(c["reply"], v) for v in c["cands"] if v != c["gold"])


def _c4_after_gold(c):
    g = G.value_hits(c["reply"], c["gold"])
    return not g or not any(h > g[0] for v in c["pool"] if v != c["gold"] for h in G.value_hits(c["reply"], v))


def _c5_e002_window(c):
    for p in G.value_hits(c["reply"], c["gold"]):
        if E002_NEG.search(" ".join(c["reply"][:p].split()[-4:])):
            return False
    return True


def _c5_no_interjection(c):
    return not ("n't" in c["low"] or any(G._cue(c["low"], p) for p in G.NEG_WORDS))


def _c5_substring(c):
    t = " " + c["low"] + " "
    return not any(x in t for x in G.NEG_WORDS + ["n't", "no,"])


def _c5_no_nt(c):
    t = c["low"]
    return not (any(G._cue(t, p) for p in G.NEG_WORDS) or re.search(r"(?<![a-z'])no\s*(?:[,.!;]|$)", t))


def _c7_window1(c):
    ws = c["ws"]
    if c["low"].lstrip().startswith(G.ROLE_TAGS):
        return False
    return not any(w == "my" and set(ws[i + 1:i + 2]) & c["obj"] for i, w in enumerate(ws))


def _hits_all_ci(text, v):
    out = []
    for f in G._forms(v):
        out += [m.start() for m in re.finditer(r"(?<![A-Za-z0-9])" + re.escape(f) + r"(?![A-Za-z0-9])", text, re.I)]
    return sorted(out)


_real_obj, _real_grade = G.obj_words_of, G.grade


def _grade_lenient_as_strict(*a, **k):
    g = _real_grade(*a, **k)
    return dict(g, strict=g["lenient"])


GRADER_MUTANTS = [
    ("c1 dropped", dict(CHECKS=checks(c1=None))),
    ("c1 ignores the cap", dict(CHECKS=checks(c1=lambda c: bool(c["reply"].strip())))),
    ("c1 ignores emptiness", dict(CHECKS=checks(c1=lambda c: c["stop"] != "cap"))),
    ("c2 dropped", dict(CHECKS=checks(c2=None))),
    ("c2 only a word 3x in a row", dict(CHECKS=checks(c2=lambda c: not any(
        c["ws"][i] == c["ws"][i + 1] == c["ws"][i + 2] for i in range(len(c["ws"]) - 2))))),
    ("c2 only repeated 3-grams", dict(CHECKS=checks(c2=lambda c: len({tuple(c["ws"][i:i + 3]) for i in range(
        len(c["ws"]) - 2)}) == max(0, len(c["ws"]) - 2)))),
    ("c3 dropped", dict(CHECKS=checks(c3=None))),
    ("c3 gold as a substring", dict(CHECKS=checks(c3=lambda c: c["gold"].lower() in c["low"]))),
    ("c3 case-sensitive gold", dict(CHECKS=checks(c3=lambda c: re.search(
        r"(?<![A-Za-z0-9])" + re.escape(c["gold"]) + r"(?![A-Za-z0-9])", c["reply"]) is not None))),
    ("c4 dropped", dict(CHECKS=checks(c4=None))),
    ("c4 in-context candidates only", dict(CHECKS=checks(c4=_c4_cands_only))),
    ("c4 only values after the gold", dict(CHECKS=checks(c4=_c4_after_gold))),
    ("c5 dropped", dict(CHECKS=checks(c5=None))),
    ("c5 E002 window (4 words before gold)", dict(CHECKS=checks(c5=_c5_e002_window))),
    ("c5 without interjection no", dict(CHECKS=checks(c5=_c5_no_interjection))),
    ("c5 without n't", dict(CHECKS=checks(c5=_c5_no_nt))),
    ("c5 cues as substrings", dict(CHECKS=checks(c5=_c5_substring))),
    ("c6 dropped", dict(CHECKS=checks(c6=None))),
    ("c6 no question-mark check", dict(CHECKS=checks(c6=lambda c: not any(G._cue(c["low"], h) for h in G.HEDGES)))),
    ("c6 no hedge check", dict(CHECKS=checks(c6=lambda c: "?" not in c["reply"]))),
    ("c6 pre-registered hedges only", dict(HEDGES=G.HEDGES[:8])),
    ("c7 dropped", dict(CHECKS=checks(c7=None))),
    ("c7 role tags only", dict(CHECKS=checks(c7=lambda c: not c["low"].lstrip().startswith(G.ROLE_TAGS)))),
    ("c7 'my' only", dict(CHECKS=checks(c7=lambda c: not any(
        w == "my" and set(c["ws"][i + 1:i + 4]) & c["obj"] for i, w in enumerate(c["ws"]))))),
    ("c7 window 1 word", dict(CHECKS=checks(c7=_c7_window1))),
    ("c7 alias ignored", dict(obj_words_of=lambda p, h, a=None: _real_obj(p, h, None))),
    ("May matched case-insensitively", dict(value_hits=_hits_all_ci)),
    ("number words not accepted", dict(NUM_WORDS={})),
    ("grey not accepted", dict(SPELLINGS={})),
    ("strict = lenient", dict(grade=_grade_lenient_as_strict)),
]

NAN = float("nan")
SCORE_FIX = [({"gold": -1.0, "a": -2.0}, True), ({"gold": -1.0, "a": -2.0, "b": -3.0}, True),
             ({"gold": -1.0, "a": -1.0}, False), ({"gold": -1.0}, False), ({"gold": NAN, "a": -2.0}, False),
             ({"gold": -1.0, "a": NAN}, False), ({"a": -1.0, "b": -2.0}, False),
             ({"gold": -1.0, "a": -0.5, "b": -3.0}, False), ({"gold": None, "a": -1.0}, False),
             ({"gold": -1.0, "a": float("inf")}, False)]


def _ok(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


SCORER_MUTANTS = [
    ("tie counts as right", lambda s: "gold" in s and len(s) > 1 and all(_ok(v) for v in s.values())
     and all(s["gold"] >= v for k, v in s.items() if k != "gold")),
    ("gold-only set counts", lambda s: "gold" in s and all(_ok(v) for v in s.values())
     and all(s["gold"] > v for k, v in s.items() if k != "gold")),
    ("non-finite foils skipped", lambda s: "gold" in s and len(s) > 1 and _ok(s["gold"])
     and all(s["gold"] > v for k, v in s.items() if k != "gold" and _ok(v))),
    ("gold beats the mean foil", lambda s: "gold" in s and len(s) > 1 and all(_ok(v) for v in s.values())
     and s["gold"] > sum(v for k, v in s.items() if k != "gold") / (len(s) - 1)),
]


def score_mismatch(fn):
    bad = []
    for s, want in SCORE_FIX:
        try:
            got = bool(fn(s))
        except Exception as e:  # a crash is not a kill
            return None, repr(e)
        if got != want:
            bad.append((s, want, got))
    return bad, None


def grader_mutant(name, attrs):
    """-> (killed, strict_flip, detail); a crash returns killed False."""
    try:
        with patched(G, **attrs):
            bad = F.run(G)
    except Exception as e:
        return False, False, f"CRASH {e!r}"
    flip = [b for b in bad if b[3] != b[5]]
    show = (flip or bad)[:2]
    return bool(bad), bool(flip), "; ".join(f"{b[1]!r} want {b[3]} {b[4]} got {b[5]} {b[6]}" for b in show)


def main():
    lines, ok = [], True
    base = F.run(G)
    lines.append(f"baseline grader: {len(F.FIX)} fixtures ({len(F.GOOD)} correct natural answers, {len(F.BAD)} bad "
                 f"replies), mismatches {len(base)}")
    for b in base:
        lines.append(f"  BASELINE MISMATCH {b}")
    ok &= not base
    sb, err = score_mismatch(MF.right)
    lines.append(f"baseline scorer metrics_ft.right: {len(SCORE_FIX)} fixtures, mismatches "
                 f"{len(sb) if sb is not None else 'CRASH ' + err}")
    ok &= sb == []
    killed = 0
    for name, attrs in GRADER_MUTANTS:
        k, flip, detail = grader_mutant(name, attrs)
        killed += k
        ok &= k
        tag = "KILLED" if k else "SURVIVED"
        how = "strict verdict flips" if flip else "clause report only"
        lines.append(f"  {tag:8s} grader | {name:38s} | {how if k else ''} | {detail}")
    for name, fn in SCORER_MUTANTS:
        bad, err = score_mismatch(fn)
        k = bool(bad) and err is None
        killed += k
        ok &= k
        lines.append(f"  {'KILLED' if k else 'SURVIVED':8s} scorer | {name:38s} | "
                     f"{err or '; '.join(f'{s} want {w} got {g}' for s, w, g in (bad or [])[:1])}")
    n = len(GRADER_MUTANTS) + len(SCORER_MUTANTS)
    lines.append(f"mutants killed {killed}/{n} (none by a crash); {'ALL PASS' if ok else 'FAIL'}")
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
