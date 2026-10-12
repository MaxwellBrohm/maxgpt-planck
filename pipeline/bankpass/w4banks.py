"""W4 bank builders (BANKPASS s2f, s2g, s3): the W3 judged view -> one frozen record list per bank. Every item the
stage holds for a bank stays a record (s2h): kept, or dropped with the first code that removed it (W3's own codes,
JUDGE_* from the votes, the W4 gates, then trim's DUP_EXACT / DUP_NEAR / FRAME_CAP / AUTHOR_TRIM, then the per-bank
choices below). Item text never leaves the PC; only the manifest (counts, shares, codes, hashes) goes to the repo.

  line banks   W3 kept (gates + both judges) -> W4 gates -> trim.run (dedup, near dedup, 5% frame cap with confirmed
               POS from wordlist_v1, author thirds) -> p_exact = 0.7 x kept / target, at most 0.7 (Max: p_exact on)
  pools        W3 kept -> PROGRAM_COLLISION (a closed-list value), WORD_COLLISION (assistant names only: s3 P "not a
               word-list word") -> W4 gates -> exact dedup -> author thirds; article and number from the word list
               (wordload.features, s3 P); relation: kept only when 2 of 3 label it "one", sex F/M/U by 2 of 3
               (REL_PLURAL, replacing FEMALE / MALE); entity kinds: kept with 2+ kept typed attributes and their
               predicates (ENTITY_FEW_ATTRS, s3 U), carried as features
  avoid_word   computed: the 80 best-ranked RM families with a confirmed adj or adv POS, not function words, pool_ok
  topic        the 999 topics W3 selected (others NOT_SELECTED); the coverage group as a feature
  topicwords   one record per (topic, author): that author's words; features carry the topic's req words: families
               listed under the same part by 2+ teachers, in RS or RM with that confirmed POS, required_ok, not
               generic (in 5% or more of all topic sets), not topic_words.NOT_REQUIRED
  labels, list names, paraphrase checks, relation labels, entity attributes: w4feats.py"""
import collections

import lexicons as L
import heldout
import topic_words as TW
from bankpass import store, trim, w4gates, w3state as WS

AVOID_N, REQ_MIN_AUTHORS, GENERIC_SHARE, MIN_ATTRS = 80, 2, 0.05, 2
PROGRAM = None


def program_values():
    global PROGRAM
    if PROGRAM is None:
        import fake_data as F
        PROGRAM = {v.lower() for vals in (F.WEEKDAYS, F.MONTHS, F.COLOURS, F.NUMBER_WORDS, F.ORDINALS, F.TIMES)
                   for v in vals}
    return PROGRAM


def as_record(it):
    """the W3 outcome as a record status: judge drops and pending items become dropped with their code."""
    r = {k: v for k, v in it.items() if k not in ("w3", "drop_w3")}
    if it.get("status") == "kept" and it.get("w3") != "kept":
        r.update(status="dropped", drop=it.get("drop_w3") or ("JUDGE_MISSING" if it.get("w3") == "pending"
                                                               else "W3_DROPPED"))
    return r


def gate_kept(recs, g, value=False):
    """the W4 gates on every kept record; kept ones get gates.w4 = [] (ran, clean)."""
    for r in recs:
        if r["status"] != "kept":
            continue
        hits = g.text_hits(r["text"]) if value else g.template_hits(r["text"], r["bank"])
        if hits:
            w4gates.apply(r, hits)
        else:
            r["gates"] = dict(r.get("gates") or {}, w4=[])
    return recs


def pos_fn(wl):
    def pos_of(w):
        d = wl.families.get(wl.head(w))
        return d["pos"] if d and d["pos_status"] == "confirmed" else None
    return pos_of


def line_bank(items, target, g, wl, seed="bankpass-w4"):
    recs = gate_kept([as_record(i) for i in items], g)
    rep = trim.run(recs, target, pos_fn(wl), seed)
    return recs, {"target": target, "p_exact": rep["p_exact"], "frames": rep["frames"],
                  "top_frame_share": rep["top_frame_share"], "distinct_2": rep["distinct_2"]}


def _drop(r, code):
    r.update(status="dropped", drop=code)


def pool_bank(items, vt, g, wl, relfeat=None, attrs=None, seed="bankpass-w4"):
    recs = [as_record(i) for i in items]
    for r in recs:
        if r["status"] != "kept":
            continue
        if r["text"].lower() in program_values():
            _drop(r, "PROGRAM_COLLISION")
        elif vt == "assistant_name" and wl.families.get(wl.head(r["text"])) and " " not in r["text"] \
                and "proper" not in wl.families[wl.head(r["text"])]["flags"]:
            _drop(r, "WORD_COLLISION")
    gate_kept(recs, g, value=True)
    for r in recs:
        if r["status"] != "kept":
            continue
        feats = dict(r.get("features") or {})
        if vt not in ("pet_name", "assistant_name"):
            feats.update({k: v for k, v in wl.features(r["text"]).items() if k in ("article", "number", "mass",
                                                                                 "basis")})
        if vt == "relation":
            sex, num = (relfeat or {}).get(r["text"], (None, None))
            feats.update(sex={"woman": "F", "man": "M"}.get(sex, "U"), number=num)
            if num != "one":
                _drop(r, "REL_PLURAL")
        if vt == "entity_kind":
            a = (attrs or {}).get(r["id"], [])
            feats["attrs"] = a
            if len(a) < MIN_ATTRS:
                _drop(r, "ENTITY_FEW_ATTRS")
        r["features"] = feats
    trim.dedup(recs)
    trim.author_thirds(recs, seed)
    return recs, {}


def avoid_bank(wl, author, g):
    out, n = [], 0
    for h in wl.ranked("RM"):
        d = wl.families[h]
        if d["pos"] not in ("adj", "adv") or d["pos_status"] != "confirmed" or h in L.FUNCTION_WORDS \
                or h in L.STOPWORDS or d["flags"]:
            continue
        hits = w4gates.gates.pool_checks(h, "avoid_word") + g.text_hits(h)
        out.append(store.make_item("P", "pool.avoid_word", len(out), h, author, "dropped" if hits else "kept",
                                   hits[0][0] if hits else None, holes=[], features={"pos": d["pos"],
                                                                                     "rank": d["rank"]},
                                   gates={"gate_hash": heldout.gate_hash(), "rc12": heldout.RC12_STATUS,
                                          "hits": [list(x) for x in hits], "w4": []}))
        n += not hits
        if n >= AVOID_N:
            break
    return out, {}


def req_bank(wl, part, author, g):
    """pool.req_<part> (computed): wordlist_v1's RM families confirmed as this POS and required_ok, held-out and W4
    gates passed; the skeleton's fallback when a topic's required words are used up."""
    out = []
    for h in wl.required(part, "RM"):
        hits = w4gates.gates.pool_checks(h, "req_" + part) + g.text_hits(h)
        out.append(store.make_item("P", "pool.req_" + part, len(out), h, author, "dropped" if hits else "kept",
                                   hits[0][0] if hits else None, holes=[], features={"pos": part,
                                                                                     "rank": wl.families[h]["rank"]},
                                   gates={"gate_hash": heldout.gate_hash(), "rc12": heldout.RC12_STATUS,
                                          "hits": [list(x) for x in hits], "w4": []}))
    return out, {}


def topic_bank(items, selected, groups, g):
    recs = [as_record(i) for i in items]
    for r in recs:
        if r["status"] == "kept" and r["id"] not in selected:
            _drop(r, "NOT_SELECTED")
        elif r["status"] == "kept":
            r["features"] = dict(r.get("features") or {}, group=groups.get(r["id"]))
    return gate_kept(recs, g, value=True), {}


def families_of(words, wl):
    return {wl.head(w) for w in words}


def req_words(sets, wl, generic):
    """{part: [families]} for one topic from {author: {part: [words]}}."""
    out = {}
    for part, key in (("noun", "nouns"), ("verb", "verbs"), ("adj", "adjs")):
        c = collections.Counter(h for s in sets.values() for h in families_of(s.get(key, []), wl))
        keep = []
        for h, n in sorted(c.items()):
            d = wl.families.get(h)
            if n < REQ_MIN_AUTHORS or not d or not d["lists"] & {"RS", "RM"} or d["pos"] != part \
                    or d["pos_status"] != "confirmed" or not d["required_ok"] or h in generic \
                    or h in TW.NOT_REQUIRED or not heldout.pool_ok(h):
                continue
            keep.append(h)
        out[part] = keep
    return out


def generic_families(topic_sets, wl):
    """families present in GENERIC_SHARE or more of all topic sets (s5: computed, replacing Claude's GENERIC)."""
    c = collections.Counter()
    for sets in topic_sets.values():
        c.update({h for s in sets.values() for k in ("nouns", "verbs", "adjs") for h in families_of(s.get(k, []), wl)})
    n = max(1, len(topic_sets))
    return {h for h, k in c.items() if k / n >= GENERIC_SHARE}


def topicwords_bank(items, selected, wl, g):
    recs = [as_record(i) for i in items]
    sets = collections.defaultdict(dict)
    for r in recs:
        tid = (r.get("features") or {}).get("topic_id")
        if r["status"] == "kept" and tid not in selected:
            _drop(r, "TOPIC_NOT_SELECTED")
        if r["status"] != "kept":
            continue
        f = r["features"]
        for k in ("nouns", "verbs", "adjs"):
            ok = [w for w in f.get(k, []) if not g.text_hits(w)]
            f.setdefault("w4_dropped", []).extend(w for w in f.get(k, []) if w not in ok)
            f[k] = ok
        sets[tid][r["author"]["model"]] = f
    generic = generic_families(sets, wl)
    for r in recs:
        if r["status"] == "kept":
            f = r["features"]
            f["req"] = req_words(sets[f["topic_id"]], wl, generic)
            r["text"] = f["topic"] + ": " + "; ".join(f"{k}: {', '.join(f[k])}" for k in ("nouns", "verbs", "adjs"))
            r["gates"] = dict(r.get("gates") or {}, w4=[])
    return exact_dedup(recs), {"generic_families": sorted(generic), "topics": len(sets)}


def exact_dedup(recs):
    """per-topic banks (word sets, intents, topic openings): a later kept record whose dedup key repeats an earlier
    one (id order) is DUP_EXACT; near duplicates stay (a frame shared across topics is the point of an intent)."""
    seen = {}
    for r in sorted((r for r in recs if r["status"] == "kept"), key=lambda r: r["id"]):
        k = store.norm_key(r["text"])
        if k in seen:
            r.update(status="dropped", drop="DUP_EXACT", dedup={"cluster": seen[k]})
        else:
            seen[k] = r["id"]
    return recs


def by_bank(its):
    return WS.by_bank(its)
