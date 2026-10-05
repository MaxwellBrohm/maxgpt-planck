"""W3 derived calls (BANKPASS s3, s6): calls whose inputs are other calls' outputs, built from the stage view. Every
call id names its source, so a restart re-derives the same calls and skips the done ones.

  attr      an entity kind's author lists 3 typed attributes (judged later by the other two)
  pred      the kind's author writes a predicate per attribute ("opens on <value>"), templatized to {v}
  relfeat   all three teachers label every relation value: woman, man or either; one or more (2 of 3 decide)
  verbs     past tense and participle for the words a teacher itself labelled verb (irregular forms, s4)
  vote      each teacher picks the best of the pooled label candidates per key, and of the note-frame words
  topic selection (1,000: 334 Qwen, 333 Ministral, 333 Gemma, in hold order, no topic within content-family Jaccard
            0.5 of an earlier one, at most half 'common'), then per selected topic:
  wordset   all three teachers, 20 nouns, 10 verbs, 10 adjectives
  intent    8 per topic, authors rotating per intent (3, 3, 2 a teacher)
  topen     3 topic-specific openings by the topic's own author (so both other teachers can judge them; see notes)"""
import math

import lexicons as L
import pools as P
from bankpass import plan, prompts3 as P3, store, w3state as WS

SHORT, ORDER = store.SHORT, plan.TEACHER_ORDER
TOPIC_QUOTA = {"qwen3.5-9b": 334, "ministral-3-8b": 333, "gemma-4-12b": 333}
TOPIC_J, INTENTS, TOPEN_K, REL_BATCH, VERB_BATCH = 0.5, 8, 3, 20, 40
VTYPE_OF = {"day of the week": "weekday", "city": "city", "month": "month", "colour": "colour"}


def _cid(s):
    return int(store.sha256_text(s)[:8], 16)


def _families(text, wl):
    return {wl.head(w) for w in text.lower().replace("'", " ").split() if w.isalpha()
            and w not in L.FUNCTION_WORDS and w not in L.STOPWORDS}


def select_topics(its, recs, wl, plan_calls):
    """[(topic item, author)] in selection order; an author's topics are selected only once all its planned topic
    calls are finished, and only after every earlier author's (the hold order)."""
    chosen, fams = [], []
    for t in ORDER:
        cands = [it for it in its.values() if it["bank"] == "topic" and it["author"]["model"] == t
                 and it["status"] == "kept"]
        if not cands or not topic_calls_done(recs, t, plan_calls):
            break
        common, mine = 0, []
        rest = []
        for it in cands:
            f = _families(it["text"], wl)
            if any(f and g and len(f & g) / len(f | g) >= TOPIC_J for g in fams):
                continue
            if it["features"].get("likelihood") == "common" and common >= TOPIC_QUOTA[t] // 2:
                rest.append((it, f))
                continue
            common += it["features"].get("likelihood") == "common"
            mine.append((it, f))
            fams.append(f)
            if len(mine) == TOPIC_QUOTA[t]:
                break
        for it, f in rest:
            if len(mine) >= TOPIC_QUOTA[t]:
                break
            if not any(f and g and len(f & g) / len(f | g) >= TOPIC_J for g in fams):
                mine.append((it, f))
                fams.append(f)
        chosen += [(it, t) for it, _ in mine]
    return chosen


def topic_calls_done(recs, teacher, plan_calls):
    want = {c["call_id"] for c in plan_calls if c["kind"] == "topic" and c["teacher"] == teacher}
    return bool(want) and want <= WS.done_ids(recs)


def intent_k(topic_id, teacher):
    off = _cid("intent-author:" + topic_id) % 3
    return sum(ORDER[(off + i) % 3] == teacher for i in range(INTENTS))


def topic_calls(teacher, chosen):
    out = []
    for it, author in chosen:
        tid, text = it["id"], it["text"]
        out.append({"call_id": f"wordset.{SHORT[teacher]}.{tid}", "kind": "wordset", "bank": "topicwords.raw",
                    "class": "W", "teacher": teacher, "n": 3, "topic": text, "topic_id": tid,
                    "seed": _cid("wordset:" + teacher + tid)})
        k = intent_k(tid, teacher)
        out.append({"call_id": f"intent.{SHORT[teacher]}.{tid}", "kind": "intent", "bank": "intent", "class": "I",
                    "teacher": teacher, "n": k, "topic": text, "topic_id": tid, "seed": _cid("intent:" + teacher + tid)})
    for it in (it for it, a in chosen if a == teacher):     # amendment 3: one topic per call (topen2.*)
        out.append({"call_id": f"topen2.{SHORT[teacher]}.{it['id']}", "kind": "topen", "bank": "open.topic_spec",
                    "class": "O", "teacher": teacher, "n": TOPEN_K, "k": TOPEN_K, "topics": [it["text"]],
                    "topic_ids": [it["id"]], "seed": _cid(f"topen2:{teacher}:{it['id']}")})
    return out


def feature_calls(teacher, its, recs):
    out = []
    kinds = [it for it in its.values() if it["bank"] == "pool.entity_kind" and it["status"] == "kept"
             and it["author"]["model"] == teacher]
    for it in kinds:
        out.append({"call_id": f"attr.{it['id']}", "kind": "attr", "bank": "attr", "class": "U", "teacher": teacher,
                    "n": 3, "value": it["text"], "src_id": it["id"], "seed": _cid("attr:" + it["id"])})
    attrs = {}
    for it in its.values():
        if it["bank"] == "attr" and it["status"] == "kept" and it["author"]["model"] == teacher:
            attrs.setdefault(it["features"]["kind_id"], []).append(it)
    for kid, rows in sorted(attrs.items()):
        rng = P.seeded("bankpass-pred", kid)
        name = f"{P.nonce(rng)} {rows[0]['features']['kind'].title()}"
        out.append({"call_id": f"pred.{kid}", "kind": "pred", "bank": "pred", "class": "U", "teacher": teacher,
                    "n": len(rows), "attr_ids": [r["id"] for r in rows], "seed": _cid("pred:" + kid),
                    "rows": [[name, r["text"], rng.choice(P.pool(r["features"]["vtype"]).values)] for r in rows]})
    rels = [it for it in its.values() if it["bank"] == "pool.relation" and it["status"] == "kept"]
    by_call = {}
    for it in rels:
        by_call.setdefault(it["call"]["call_id"], []).append(it)
    for src, rows in sorted(by_call.items()):
        for b in range(math.ceil(len(rows) / REL_BATCH)):
            part = rows[b * REL_BATCH:(b + 1) * REL_BATCH]
            out.append({"call_id": f"relfeat.{SHORT[teacher]}.{src}.{b}", "kind": "relfeat", "bank": "pool.relation",
                        "class": "P", "teacher": teacher, "n": len(part), "rows": [r["text"] for r in part],
                        "item_ids": [r["id"] for r in part], "seed": _cid(f"relfeat:{teacher}:{src}:{b}")})
    for r in recs:
        if r["kind"] == "pos" and r["author"]["model"] == teacher and not r.get("problem") and not r.get("error") \
                and not r["call_id"].startswith("pos."):
            verbs = [ln.split(":", 1)[0].strip() for ln in r["lines"] if ln.rstrip().endswith(": verb")]
            for b in range(math.ceil(len(verbs) / VERB_BATCH)):
                part = verbs[b * VERB_BATCH:(b + 1) * VERB_BATCH]
                out.append({"call_id": f"verbs.{r['call_id']}.{b}", "kind": "verbs", "bank": "wordlabel.verbs",
                            "class": "V", "teacher": teacher, "n": len(part), "rows": part,
                            "seed": _cid(f"verbs:{r['call_id']}:{b}")})
    return out


VOTE_WHAT = {"label.notewas": "the word or phrase a note puts before an older value it replaced"}


def vote_calls(teacher, its, recs, plan_calls):
    """one vote per label bank, once every teacher's candidate call for it is finished."""
    done = WS.done_ids(recs)
    out = []
    for bank in sorted({c["bank"] for c in plan_calls if c["kind"] in ("label", "notewas")}):
        if not all(c["call_id"] in done for c in plan_calls if c["bank"] == bank):
            continue
        seen, cands = set(), []
        for it in its.values():
            if it["bank"] == bank and it["status"] == "kept" and store.norm_key(it["text"]) not in seen:
                seen.add(store.norm_key(it["text"]))
                cands.append(it)
        if len(cands) < 2:
            continue
        cands = cands[:15]
        key = bank.split(".", 1)[1]
        o = cands[0]["features"].get("o")
        what = VOTE_WHAT.get(bank) or P3.KEY_FACT[key].replace("{o}", o or "")
        out.append({"call_id": f"vote.{bank}.{SHORT[teacher]}", "kind": "vote", "bank": bank, "class": "N",
                    "teacher": teacher, "n": 1, "what": what, "seed": _cid(f"vote:{bank}:{teacher}"),
                    "cands": [c["text"].replace("{o}", o or "") for c in cands], "cand_ids": [c["id"] for c in cands]})
    return out
