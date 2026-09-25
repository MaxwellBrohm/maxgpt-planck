"""RC-12 G-FMT (PERSIST) and G-DYN (OWN) graders (SPEC s4). Pure Python, no model.

G-FMT clauses: f1_degen (loop rule), f2_echo (reply copies the user turn: >= 80% of its word 4-grams),
  f3_short (fewer than 4 words: a bare marker is not a reply; notes.txt), f4_rule (the rule in force not followed),
  f5_old (override cell: the replaced rule still followed). Freshness (E001 battery.fresh) = f1 + f2.
  Rules: caps (>= 90% of >= 5 letters upper case), lowercase (no capital at all, >= 5 letters), first_word (the
  first word is the given word, any case), closing (the last words are the given phrase, trailing punctuation and
  quotes ignored), one_sentence (exactly one sentence of 2+ words; battery.n_sentences), question_end (ends with
  "?"), word (the given word present as a whole word, any case, and not 3+ of the include-words: stuffing, step 5
  audit), brackets (starts with "[" and ends with "]").
G-DYN clauses. pick: d0_source (the model's reply at the Q turn does not assert exactly ONE of the offered options:
  the conversation scores 0 and is counted as a source failure), then every G-VAL clause with the parsed gold.
  list2: d0_source (the Q reply is not a numbered/bulleted list of exactly 3 items, or the 2nd item has no content
  word of its own), d1_degen, d2_gold (every content word of the 2nd item, up to 4, present; 5-letter stems),
  d3_other (a content word unique to another item, or to a lure item the user mentioned, present),
  d4_assert (the first gold word sits in a negated / questioned / hedged / hypothetical sentence), d5_unsure,
  d6_voice (the list given to the user: "your second idea", "you listed"). No echo clause: the question never
  holds the gold item's words (L2), so d2_gold fails every echo; an echo also says "your list" (d6)."""
import re

import fam_persist as FPW
import grade_loop as L
import grade_text as T
import graders as G

FP = dict(min_words=4, caps=0.9, stuff=3)
STUFF_POOL = set(FPW.WORDS)       # the PERSIST include-word pool (dev); the sealed split must add its own pool
CONTENT_STOP = {"a", "an", "the", "of", "for", "with", "to", "and", "or", "your", "my", "our", "in", "on", "at",
                "some", "one", "idea", "ideas", "maybe", "try", "do", "have", "go"}
LIST_VOICE = T.rx(r"\byour\s+(?:\w+\s+){0,2}(?:idea|list|suggestion|pick)|\byou\s+(?:\w+\s+)?(?:listed|suggested|picked)\b")


def n_sentences(t):
    t = (t or "").strip()
    t = re.sub(r"\b(e\.g|i\.e|etc|dr|mr|mrs|ms|st|vs|approx)\.", lambda m: m.group(0).replace(".", ""), t, flags=re.I)
    t = re.sub(r"(\d)\.(\d)", r"\1\2", t)
    return len([p for p in re.split(r"(?<=[.!?])\s+|\n+", t) if len(re.findall(r"[A-Za-z]+", p)) >= 2])


def tail(t):
    t = re.sub(r"[\s\"'*_)\]]+$", "", (t or "").rstrip())
    return re.sub(r"[^\x00-\x7F]+$", "", t).rstrip()


def rule_ok(rule, arg, text):
    letters = [c for c in text if c.isalpha()]
    if rule == "caps":
        return len(letters) >= 5 and sum(c.isupper() for c in letters) / len(letters) >= FP["caps"]
    if rule == "lowercase":
        return len(letters) >= 5 and not any(c.isupper() for c in letters)
    if rule == "first_word":
        ws = T.lwords(text)
        return bool(ws) and ws[0] == arg.lower()
    if rule == "closing":
        want = T.lwords(arg)
        return T.lwords(tail(text))[-len(want):] == want and bool(re.search(r"[A-Za-z]\W*$", tail(text)))
    if rule == "one_sentence":
        return n_sentences(text) == 1
    if rule == "question_end":
        return tail(text).endswith("?")
    if rule == "word":
        if "stuff" not in T.OFF and len(set(T.lwords(text)) & STUFF_POOL) >= FP["stuff"]:
            return False          # step 5 audit: 3+ of the include-words = stuffing, not following the instruction
        return arg.lower() in T.lwords(text)
    if rule == "brackets":
        s = text.strip()
        return s.startswith("[") and s.endswith("]")
    raise ValueError(rule)


def g_fmt(reply, stop, prior, rec, probe):
    text = T.norm(reply)
    f = probe["fmt"]
    fails = []
    if L.degenerate(reply, stop, prior):
        fails.append("f1_degen")
    if T.echo(text, probe["question"]):
        fails.append("f2_echo")
    if "short" not in T.OFF and len(T.toks(text)) < FP["min_words"]:
        fails.append("f3_short")
    if "rule" not in T.OFF and not rule_ok(f["rule"], f["arg"], text):
        fails.append("f4_rule")
    if "old" not in T.OFF and f.get("absent") and rule_ok(f["absent"]["rule"], f["absent"]["arg"], text):
        fails.append("f5_old")
    return dict(ok=not fails, fails=fails, lenient=rule_ok(f["rule"], f["arg"], text))


def parse_list(t):
    """items of a numbered / bulleted list, cut to their head words (E001 battery.parse_list)."""
    items = []
    for line in (t or "").splitlines():
        m = re.match(r"^\s*(?:\d+[.)]|[-*\u2022])\s+(.*\S)", line)
        if m:
            s = re.sub(r"[*_`\"]", "", m.group(1))
            s = re.split(r"\s[-\u2013\u2014:(]|[:(,]", s)[0].strip()
            if s:
                items.append(s)
    return items


def stems(t):
    return [w[:5] for w in T.lwords(t) if len(w) >= 3 and w not in CONTENT_STOP]


def g_pick(reply, stop, prior, rec, probe):
    gf = probe["gold_fn"]
    src = T.norm(prior[gf["src_turn"] - 1])
    picked = [o for o in gf["options"] if T.asserted_hits(src, o, gf["options"])]
    if "source" not in T.OFF and len(picked) != 1:
        return dict(ok=False, fails=["d0_source"], lenient=False, source_fail=True, parsed=picked)
    res = G.g_val(reply, stop, prior, rec, probe, gold=picked[0] if picked else gf["options"][0])
    res.update(source_fail=False, parsed=picked)
    return res


def g_list(reply, stop, prior, rec, probe):
    gf = probe["gold_fn"]
    items = parse_list(T.norm(prior[gf["src_turn"] - 1]))
    shape_ok = len(items) == 3
    gold_st = stems(items[1])[:4] if shape_ok else []
    other_st = {s for j, it in enumerate(items) if j != 1 for s in stems(it)}
    other_st |= {s for it in probe.get("lure_items") or [] for s in stems(it)}
    other_st -= set(gold_st)
    if "source" not in T.OFF and (not shape_ok or not gold_st or not set(gold_st) - set(stems(items[0]) +
                                                                                           stems(items[2]))):
        return dict(ok=False, fails=["d0_source"], lenient=False, source_fail=True, parsed=items)
    text = T.norm(reply)
    have = stems(text)
    fails = []
    if L.degenerate(reply, stop, prior):
        fails.append("d1_degen")
    if not gold_st or not all(s in have for s in gold_st):
        fails.append("d2_gold")
    if "other" not in T.OFF and any(s in have for s in other_st):
        fails.append("d3_other")
    first = None
    for m in re.finditer(r"[A-Za-z']+", text):
        if gold_st and m.group(0).lower()[:5] == gold_st[0]:
            first = m
            break
    if first is not None and "assert" not in T.OFF and T.why_not_asserted(text, first) is not None:
        fails.append("d4_assert")
    if T.unsure(text):
        fails.append("d5_unsure")
    if "listvoice" not in T.OFF and LIST_VOICE.search(text):
        fails.append("d6_voice")
    return dict(ok=not fails, fails=fails, lenient="d2_gold" not in fails, source_fail=False, parsed=items)


def g_dyn(reply, stop, prior, rec, probe):
    return (g_pick if probe["gold_fn"]["type"] == "pick" else g_list)(reply, stop, prior, rec, probe)
