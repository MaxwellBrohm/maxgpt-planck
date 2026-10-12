"""Controlled word list, step 2 (BANKPASS s4): eligibility, score, POS proxy, families, drops and the RS / RM / RL
lists, from wordlist.py's count tables. Cheap (seconds); runs on the PC or the Mac.

    python wordstats.py OUT_DIR counts_A.tsv.gz [counts_B.tsv.gz]    (the CLI; two samples: also the stability Jaccard)

Eligible: DF >= max(5, 5e-4 x docs) in >= 3 of the 6 voting groups. Score: geometric mean over the 6 groups of the DF
share, a zero count floored at 0.5 doc (so a modern word absent from the books sample is ranked low, not removed).
POS proxy: the context counts, each divided by its total over all words, argmax; "unk" under 3 contexts or when the
contexts are under 1% of the word's uses (closed-class words and past tenses: after, through, came, took). Teacher
labels (2 of 3 agreeing with the proxy, else "amb") need a teacher run (W3), so every row says pos_status pending.
Function words (lexicons STOPWORDS and FUNCTION_WORDS) get pos "func" and never join a family, nor do their forms
(making, does: a form with a function-word base stays its own headword instead of joining "mak" or "doe").
Families: -s/-es/-ies (base noun or verb, form not adj), -ed/-ing with e-drop and doubling (base verb), -er/-est (base
adj), bases of 3+ letters, only when both forms are eligible and the base is used at least a tenth as often as the
form (water is not wat+er, number not numb+er); a form joins its best-scored unflagged base. Irregular
forms and -ly adverbs need teacher labels: not joined here. Drops (flags, logged): proper (capitalized share > 0.5 of
5+ uses right after an all-lowercase word; never i or i-contractions), heldout_vocab (heldout.vocab_hits: held-out
text), safety (lexicons SAFETY_RE, the FAKE list until LDNOOBW is downloaded), letter (single letters except a and i).
Kept but marked (required_ok false): heldout_head (heldout.pool_ok false: an E004 held-out object head noun may be
said, but is never a slot value, an object noun or a required word).
Lists: RS = top 2,000 families, RM = top 5,000, RL = top 20,000, ranked by family score, dropped families excluded."""
import hashlib
import json
import math
import os
import re

from bankpass import wordtable as WT

ELIG_GROUPS, LISTS = 3, (("RS", 2000), ("RM", 5000), ("RL", 20000))
POS = ("noun", "verb", "adj")
PROPER_MAX, PROPER_MIN_USES, POS_MIN, POS_SHARE_MIN, BASE_TF_RATIO = 0.5, 5, 3, 0.01, 10
DROP_FLAGS = ("proper", "heldout_vocab", "safety", "letter")


_CONS = "bcdfghjklmnpqrstvwxz"


def func_words():
    import lexicons as L
    return L.STOPWORDS | L.FUNCTION_WORDS


def eligible(rows, docs):
    fl = {g: WT.floor(docs[g]) for g in WT.VOTING}
    return {w for w, r in rows.items() if sum(r[f"df_{g}"] >= fl[g] for g in WT.VOTING) >= ELIG_GROUPS}


def score(r, docs):
    return math.exp(sum(math.log(max(r[f"df_{g}"], 0.5) / docs[g]) for g in WT.VOTING) / len(WT.VOTING))


def pos_proxy(rows, words):
    tot = {p: sum(rows[w][p] for w in rows) or 1 for p in POS}
    out, func = {}, func_words()
    for w in words:
        r = rows[w]
        if w in func:
            out[w] = ("func", 1.0)
            continue
        if sum(r[p] for p in POS) < max(POS_MIN, POS_SHARE_MIN * r["tf"]):
            out[w] = ("unk", 0.0)
            continue
        rate = {p: r[p] / tot[p] for p in POS}
        best = max(POS, key=lambda p: rate[p])
        out[w] = (best, round(rate[best] / sum(rate.values()), 3))
    return out


def base_candidates(w):
    """(base, rule) pairs a word could be an inflection of, by the rubric rules."""
    out = []
    if w.endswith("ies") and len(w) > 4 and w[-4] in _CONS:
        out.append((w[:-3] + "y", "s"))
    if w.endswith("es") and re.search(r"(?:s|x|z|ch|sh|o)es$", w):
        out.append((w[:-2], "s"))
    if w.endswith("s") and not w.endswith("ss"):
        out.append((w[:-1], "s"))
    for suf, rule in (("ed", "ed"), ("ing", "ing"), ("er", "er"), ("est", "est")):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            stem = w[:-len(suf)]
            out.append((stem, rule))
            out.append((stem + "e", rule))
            if len(stem) >= 4 and stem[-1] == stem[-2] and stem[-1] in _CONS:
                out.append((stem[:-1], rule))
            if suf in ("ed", "er", "est") and stem.endswith("i") and stem[-2:-1] in tuple(_CONS):
                out.append((stem[:-1] + "y", rule))
    return out


RULE_OK = {"s": lambda bp, fp: bp in ("noun", "verb") and fp != "adj", "ed": lambda bp, fp: bp == "verb",
           "ing": lambda bp, fp: bp == "verb", "er": lambda bp, fp: bp == "adj", "est": lambda bp, fp: bp == "adj"}


def families(elig, sc, pos, flags=None, tf=None):
    """{headword: {"pos": p, "forms": {rule: [forms]}}} and {word: headword}. A flagged base takes no forms."""
    form_of, flags, func = {}, flags or {}, func_words()
    for w in sorted(elig, key=lambda x: -sc[x]):
        best = None
        if w in func:
            continue
        cands = [(b, rule) for b, rule in base_candidates(w) if b != w and b in elig and len(b) >= 3]
        if any(b in func for b, _ in cands):
            continue
        for b, rule in cands:
            if set(flags.get(b, ())) & set(DROP_FLAGS) or (tf and tf[b] * BASE_TF_RATIO < tf[w]):
                continue
            if RULE_OK[rule](pos[b][0], pos[w][0]) and (best is None or sc[b] > sc[best[0]]):
                best = (b, rule)
        if best:
            form_of[w] = best
    # a base that is itself a form joins its own base's family (walked -> walk, not walked -> walke)
    fam = {}
    for w in elig:
        if w in form_of:
            continue
        fam[w] = {"pos": pos[w][0], "forms": {}}
    for w, (b, rule) in form_of.items():
        while b in form_of:
            b = form_of[b][0]
        fam.setdefault(b, {"pos": pos[b][0], "forms": {}})["forms"].setdefault(rule, []).append(w)
    return fam, {w: (form_of[w][0] if w in form_of else w) for w in elig}


def flags_of(w, r):
    import heldout
    import lexicons as L
    out = []
    if r["mid"] >= PROPER_MIN_USES and r["capmid"] / r["mid"] > PROPER_MAX and w != "i" and not w.startswith("i'"):
        out.append("proper")
    if heldout.vocab_hits(w):
        out.append("heldout_vocab")
    elif not heldout.pool_ok(w):
        out.append("heldout_head")
    if L.SAFETY_RE.search(w):
        out.append("safety")
    if len(w) == 1 and w not in ("a", "i"):
        out.append("letter")
    return out


def article(r, w):
    if r["a"] + r["an"] >= 3:
        return ("an" if r["an"] > r["a"] else "a"), "counted"
    return ("an" if w[:1] in "aeiou" else "a"), "vowel_rule"


def build(path):
    head, rows = WT.read(path)
    docs = head["docs"]
    elig = eligible(rows, docs)
    sc = {w: score(rows[w], docs) for w in elig}
    pos = pos_proxy(rows, elig)
    fl = {w: flags_of(w, rows[w]) for w in elig}
    fam, head_of = families(elig, sc, pos, fl, {w: rows[w]["tf"] for w in elig})
    order = sorted(fam, key=lambda h: (-max(sc[x] for x in [h] + sum(fam[h]["forms"].values(), [])), h))
    ranked, rank = [h for h in order if not set(fl[h]) & set(DROP_FLAGS)], {}
    for i, h in enumerate(ranked, 1):
        rank[h] = i
    rule_of = {f: rule for h in fam for rule, fs in fam[h]["forms"].items() for f in fs}
    return {"head": head, "rows": rows, "elig": elig, "score": sc, "pos": pos, "fam": fam, "head_of": head_of,
            "flags": fl, "ranked": ranked, "rank": rank, "rule_of": rule_of}


def value_features(res, value):
    """the article and number of a pool value from core v0 counts (s3 P), replacing pools.features' vowel rule:
    article from "a"/"an" before the first word; plural when the last word is a joined -s form; mass (no article)
    when "much X" outnumbers "a/an X" (3+ uses). Unknown words fall back to the vowel rule, marked."""
    ws = value.lower().split()
    first, last = res["rows"].get(ws[0]), res["rows"].get(ws[-1])
    art, basis = article(first, ws[0]) if first else (("an" if ws[0][:1] in "aeiou" else "a"), "vowel_rule")
    plural = res["rule_of"].get(ws[-1]) == "s"
    mass = bool(last) and last["mass"] >= 3 and last["mass"] > last["a"] + last["an"]
    return {"article": "" if plural or mass else art, "number": "pl" if plural else "sg", "mass": mass,
            "basis": basis}


def required_ok(res, h):
    """may be a required word (proxy POS until teacher labels): ranked, a content POS, not a held-out head."""
    return h in res["rank"] and res["pos"][h][0] in POS and not res["flags"][h]


def write(res, out_dir, label):
    os.makedirs(out_dir, exist_ok=True)
    rows, fam = res["rows"], res["fam"]
    path = os.path.join(out_dir, f"wordlist_v0.{label}.tsv")
    code = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# controlled word list v0 ({label}): pipeline/bankpass/wordfam.py over wordlist.py counts\n")
        f.write(f"# input manifest sha256 {res['head']['input_manifest_sha256']}; wordlist.py sha256 "
                f"{res['head']['code_sha256']}; wordfam.py sha256 {code}\n")
        f.write("# docs per group " + json.dumps(res["head"]["docs"], sort_keys=True) + "\n")
        f.write("# pos = proxy only; pos_status pending (teacher labels, W3); dropped families carry flags, no rank\n")
        f.write("\t".join(["headword", "rank", "lists", "pos_proxy", "pos_conf", "pos_status", "forms", "score"]
                          + [f"df_{g}" for g in WT.GROUPS] + ["tf", "article", "article_basis", "a", "an", "pl_ctx",
                                                              "mass_ctx", "proper_share", "flags", "required_ok"]) + "\n")
        for h in sorted(fam, key=lambda x: (res["rank"].get(x, 10 ** 9), x)):
            r, k = rows[h], res["rank"].get(h)
            lists = ",".join(n for n, cap in LISTS if k and k <= cap)
            forms = ";".join(f"{rule}:{'|'.join(sorted(fs))}" for rule, fs in sorted(fam[h]["forms"].items()))
            art, basis = article(r, h)
            prop = round(r["capmid"] / r["mid"], 3) if r["mid"] else ""
            f.write("\t".join(map(str, [h, k or "", lists, res["pos"][h][0], res["pos"][h][1], "pending", forms,
                                        f"{res['score'][h]:.3e}"] + [r[f"df_{g}"] for g in WT.GROUPS]
                                  + [r["tf"], art, basis, r["a"], r["an"], r["pl"], r["mass"], prop,
                                     ",".join(res["flags"][h]), int(required_ok(res, h))])) + "\n")
    return path

