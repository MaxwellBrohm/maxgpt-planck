"""Porter stemmer (M. F. Porter, "An algorithm for suffix stripping", Program 14(3), 1980), the paper's five steps
plus the two changes of Porter's own reference code (step 2: "bli" -> "ble" in place of "abli" -> "able", and
"logi" -> "log"). Written from the published algorithm for the OFFTOPIC overlap (check_base.stem); no dependency.

Why (2026-09-28, v1 quality review): the round 1 suffix stripper merged unrelated words (ready/reading, busy/bus,
care/car, cater/cat, letting/letter, plane/plan). Porter keeps those apart through its measure and cvc conditions
(care, plane, cater and letter keep their endings; busy -> busi, bus -> bu). It also no longer merges the y-adjective
with its noun (noisy/noise, rainy/rain, shaky/shaking) or the agent noun with its verb (speaker/speak): topic_forms()
adds those forms on the topic side only, from the part of speech the topic word set gives (dp2 recheck, 2026-09-28)."""
from functools import lru_cache

VOWELS = set("aeiou")


def _cons(w, i):
    """w[i] is a consonant: not a vowel, and y only after a vowel (or at the start)."""
    c = w[i]
    if c in VOWELS:
        return False
    if c == "y":
        return i == 0 or not _cons(w, i - 1)
    return True


def _m(stem):
    """the measure: the number of VC sequences in [C](VC)^m[V]."""
    n, i, L = 0, 0, len(stem)
    while i < L and _cons(stem, i):
        i += 1
    while i < L:
        while i < L and not _cons(stem, i):
            i += 1
        if i >= L:
            break
        n += 1
        while i < L and _cons(stem, i):
            i += 1
    return n


def _vowel_in(stem):
    return any(not _cons(stem, i) for i in range(len(stem)))


def _double(w):
    return len(w) >= 2 and w[-1] == w[-2] and _cons(w, len(w) - 1)


def _cvc(w):
    """*o: ends consonant-vowel-consonant, the last consonant not w, x or y."""
    return (len(w) >= 3 and _cons(w, len(w) - 3) and not _cons(w, len(w) - 2) and _cons(w, len(w) - 1)
            and w[-1] not in "wxy")


def _step1ab(w):
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("ss"):
        pass
    elif w.endswith("s"):
        w = w[:-1]
    if w.endswith("eed"):
        if _m(w[:-3]) > 0:
            w = w[:-1]
        return w
    for suf in ("ed", "ing"):
        if w.endswith(suf) and _vowel_in(w[:-len(suf)]):
            w = w[:-len(suf)]
            if w.endswith(("at", "bl", "iz")):
                return w + "e"
            if _double(w) and w[-1] not in "lsz":
                return w[:-1]
            if _m(w) == 1 and _cvc(w):
                return w + "e"
            return w
    return w


def _step1c(w):
    return w[:-1] + "i" if w.endswith("y") and _vowel_in(w[:-1]) else w


STEP2 = [("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"), ("izer", "ize"), ("bli", "ble"),
         ("alli", "al"), ("entli", "ent"), ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
         ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"),
         ("iviti", "ive"), ("biliti", "ble"), ("logi", "log")]
STEP3 = [("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"), ("ical", "ic"), ("ful", ""), ("ness", "")]
STEP4 = ["al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ion", "ou", "ism", "ate",
         "iti", "ous", "ive", "ize"]


def _replace(w, rules, min_m):
    """the longest matching suffix decides; it is replaced when the stem's measure exceeds min_m."""
    for suf, rep in sorted(rules, key=lambda r: -len(r[0])):
        if w.endswith(suf):
            stem = w[:-len(suf)]
            return stem + rep if _m(stem) > min_m else w
    return w


def _step4(w):
    for suf in sorted(STEP4, key=len, reverse=True):
        if w.endswith(suf):
            stem = w[:-len(suf)]
            if suf == "ion" and not stem.endswith(("s", "t")):
                return w
            return stem if _m(stem) > 1 else w
    return w


def _step5(w):
    if w.endswith("e"):
        stem = w[:-1]
        m = _m(stem)
        if m > 1 or (m == 1 and not _cvc(stem)):
            w = stem
    if _m(w) > 1 and _double(w) and w.endswith("l"):
        w = w[:-1]
    return w


@lru_cache(maxsize=100000)
def porter(word):
    """the Porter stem of one lowercase word; words of one or two letters are returned as they are."""
    w = word
    if len(w) <= 2:
        return w
    w = _step1ab(w)
    w = _step1c(w)
    w = _replace(w, STEP2, 0)
    w = _replace(w, STEP3, 0)
    w = _step4(w)
    return _step5(w)


def comparatives(adj):
    """comparative and superlative of an adjective (strong -> stronger, strongest; nice -> nicer; big -> bigger;
    easy -> easier), which the Porter stem keeps apart from the adjective (dp2 recheck: stronger, louder, slower)."""
    if adj.endswith("e"):
        return [adj + "r", adj + "st"]
    if adj.endswith("y") and len(adj) > 2 and adj[-2] not in VOWELS:
        return [adj[:-1] + "ier", adj[:-1] + "iest"]
    base = adj + adj[-1] if _cvc(adj) and _m(adj) == 1 and adj[-1] in "bdgmnprt" else adj
    return [base + "er", base + "est"]


def topic_forms(word, part):
    """forms of a topic word that Porter does not bring to its stem, for the TOPIC side of OFFTOPIC only (the text side
    stays plain Porter, so the v1 review's collisions stay apart): an adjective's comparatives and -ly adverb, the base
    under a -y or -ly adjective (noisy -> noise, sunny -> sun, monthly -> month), and the -er agent noun of a verb
    (speak -> speaker, drive -> driver, plan -> planner). No base or agent of three letters except through a
    doubled consonant: busy -> bus, cat -> cater and car -> carer are not made. dp2 recheck, 2026-09-28: Porter alone
    rejected on-topic lines through noise/noisy, leaks/leaky, sun/sunny, taste/tasty, speakers, drivers, monthly."""
    w, out = word.lower(), []
    if not w.isalpha():
        return out
    if part == "adj":
        out += comparatives(w)
        out.append(w[:-1] + "ily" if w.endswith("y") else w[:-1] + "y" if w.endswith("le") else w + "ly")
        if w.endswith("ly") and len(w) >= 6:
            out.append(w[:-2])
        elif w.endswith("y") and len(w) >= 5:
            b = w[:-1]
            out += [b, b + "e"]
            if b[-1] == b[-2] and b[-1] not in VOWELS:
                out.append(b[:-1])
        return out
    if part == "verb" and len(w) >= 4:
        out.append(w + "r" if w.endswith("e") else w + w[-1] + "er" if _cvc(w) and _m(w) == 1 else w + "er")
    return out
