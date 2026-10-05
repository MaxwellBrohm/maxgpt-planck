"""W4 features from the stage's derived calls (BANKPASS s3 N, L, P, U, Q): label votes, list-name choice, the
paraphrase checklist verdicts, relation sex and number, and entity attributes with their predicates. PC CPU.

  labels       label.<key>: the candidate 2 of 3 votes picked (LABEL_NOT_CHOSEN for the rest); list names: one per
               list type, the first kept in a seeded order (LISTNAME_NOT_CHOSEN; LIST_STATE and golds take one name)
  paraphrases  kept only when the checking teacher answered yes to every checklist point and no to "added"
  relations    sex (woman, man, either) and number (one, more) by 2 of 3 teacher labels
  entities     a kind's attributes both judges kept, each with a predicate both judges kept"""
import collections
import re

from bankpass import store, w4gates
from bankpass.w4banks import as_record, gate_kept, _drop


def vote_winners(recs, kind="vote"):
    """{bank: winning item id} where 2 of 3 votes agree (vote records: the 1-based index into cand_ids). kind "vote2"
    reads the W4 re-vote (w4vote.py, amendment 7) instead of the stage's votes."""
    votes = collections.defaultdict(list)
    for r in recs:
        if r["kind"] == kind and not r.get("problem") and not r.get("error") and r.get("lines"):
            ids, x = r["plan"]["cand_ids"], r["lines"][0].strip()
            votes[r["bank"]].append(ids[int(x) - 1] if x.isdigit() and 0 < int(x) <= len(ids) else None)
    out = {}
    for bank, vs in votes.items():
        c = collections.Counter(v for v in vs if v)
        if c and c.most_common(1)[0][1] >= 2:
            out[bank] = c.most_common(1)[0][0]
    return out


def chosen_bank(items, winner, code, g):
    """label and list-name banks: one kept item (the winner), every other kept one dropped with code."""
    recs = gate_kept([as_record(i) for i in items], g, value=True)
    for r in recs:
        if r["status"] == "kept" and r["id"] != winner:
            _drop(r, code)
    return recs, {}


def first_kept(items, seed):
    kept = sorted((i for i in items if i.get("w3") == "kept"),
                  key=lambda i: store.sha256_text(f"{seed}:{i['id']}"))
    return kept[0]["id"] if kept else None


def para_checks(recs):
    """{para item id: True when its checklist check passed (every point yes, added no)}."""
    out = {}
    for r in recs:
        if r["kind"] != "check" or r.get("problem") or r.get("error"):
            continue
        ans = dict(ln.split(": ", 1) for ln in r["lines"] if ": " in ln)
        ok = all(v == "yes" for k, v in ans.items() if k != "added") and ans.get("added") == "no"
        out.setdefault(r["plan"]["para_id"], ok)
    return out


_SEX = re.compile(r"^(.*): (woman|man|either), (one|more)$")


def relation_features(recs):
    votes = collections.defaultdict(lambda: ([], []))
    for r in recs:
        if r["kind"] == "relfeat" and not r.get("problem") and not r.get("error"):
            for ln in r["lines"]:
                m = _SEX.match(ln.strip())
                if m:
                    votes[m.group(1)][0].append(m.group(2))
                    votes[m.group(1)][1].append(m.group(3))

    def maj(xs):
        c = collections.Counter(xs).most_common(1)
        return c[0][0] if c and c[0][1] >= 2 else None
    return {v: (maj(s), maj(n)) for v, (s, n) in votes.items()}


def entity_attrs(its, g):
    """{entity kind item id: [[attr label, vtype, predicate template, attr id, pred id]]}: attributes both judges
    kept, with a predicate both judges kept, clean under the W4 gates (label and bare predicate), one per label in a
    kind, and a label typed the same in every kind (the first kind in id order sets it: ATTR_TYPE_CONFLICT)."""
    preds = {i["features"]["attr_id"]: i for i in its.values() if i["bank"] == "pred" and i.get("w3") == "kept"}
    out, typed, drops = collections.defaultdict(list), {}, collections.Counter()
    for i in sorted((i for i in its.values() if i["bank"] == "attr" and i.get("w3") == "kept"), key=lambda i: i["id"]):
        p = preds.get(i["id"])
        lab, vt = i["text"], i["features"]["vtype"]
        if p is None:
            drops["NO_PRED"] += 1
        elif g.text_hits(lab) or g.text_hits(w4gates.gates.bare(p["text"])):
            drops["W4_GATE"] += 1
        elif typed.setdefault(lab, vt) != vt:
            drops["ATTR_TYPE_CONFLICT"] += 1
        elif any(a[0] == lab for a in out[i["features"]["kind_id"]]):
            drops["ATTR_REPEATED"] += 1
        else:
            out[i["features"]["kind_id"]].append([lab, vt, p["text"], i["id"], p["id"]])
    return out, dict(drops)


def topic_text_check(items, topic_text):
    """W4 fix (stage P amendment 5 reused topic item ids: topic2.q.3 and topic.q.3 both made T.topic.3_0, so a word
    set, intent or topic opening derived before the amendment names a topic text its id no longer holds): a derived
    item whose topic text differs from the kept topic item's text for its id is dropped TOPIC_TEXT_MISMATCH."""
    n = 0
    for i in items:
        f = i.get("features") or {}
        tid = f.get("topic_id")
        if tid in topic_text and f.get("topic") != topic_text[tid] and i.get("w3") == "kept":
            i.update(w3="dropped", drop_w3="TOPIC_TEXT_MISMATCH")
            n += 1
    return n


def wordset_topics(items, min_authors=2):
    """topic ids with word sets from min_authors or more teachers still kept after the text check."""
    by = collections.defaultdict(set)
    for i in items:
        if i.get("w3") == "kept" and i.get("status") == "kept":
            by[i["features"]["topic_id"]].add(i["author"]["model"])
    return {t for t, a in by.items() if len(a) >= min_authors}


def topic_groups(recs, text):
    """{topic id: coverage group} from group calls whose topic text is the kept topic's text (metadata only)."""
    return {r["plan"]["topic_id"]: r["lines"][0] for r in recs if r["kind"] == "group" and r.get("lines")
            and not r.get("problem") and not r.get("error") and r["plan"]["topic"] == text.get(r["plan"]["topic_id"])}


def drop_without_sets(topic_recs, with_sets):
    """a kept topic with word sets from under 2 teachers for its own text is dropped TOPIC_NO_WORDSETS."""
    for r in topic_recs:
        if r["status"] == "kept" and r["id"] not in with_sets:
            r.update(status="dropped", drop="TOPIC_NO_WORDSETS")
    return topic_recs
