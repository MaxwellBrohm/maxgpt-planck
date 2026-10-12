"""Round 2 fixtures, part three (2026-09-28): boundary cases the full mutation run asked for, one per design limit of
the round 2 rules: a declarative clause after a question clause states a plant; a user reply sharing two words with
the assistant's line passes even below half of its own words; a -y adjective's base is on topic; "busy" does not make
"bus" a topic word. Same contract as fixtures_r2. FAKE fixture text written by Claude."""
import topic_words as TW
from check_base import content, stem
from check_behav import topic_set
from defects_text import _copy, filler_user
from fixtures_bound import _fillers, _topic_w
from fixtures_r2 import _far, _sty
from stemmer import topic_forms

# real words among the bases stemmer.topic_forms makes from the FAKE sets' -y adjectives (the fixture text must read)
Y_BASES = {"noise", "leak", "sun", "taste", "sand", "salt", "mess", "fluff", "trick", "crust", "fuss", "breeze",
           "smell", "shade", "leaf", "mud", "health", "thrift", "droop", "sleep", "catch"}


def r2c_limits(skel, texts, built):
    out, topics = [], list(skel["topic_text"].values())
    all_topic = set().union(*(topic_set(x) for x in topics))
    plain = set().union(*(content(x) | content(" ".join(TW.related(x))) for x in topics))
    for e in skel["events"]:            # clause split: "Can I share something, my dog is called Rex." states it
        for op in e["params"].get("ops", []):
            i, s = e["turns"].get(op["turn"]), skel["slots"].get(op.get("slot") or "")
            if op["op"] == "plant" and s and s["key"] in ("pet_name", "person_name") and \
                    skel["turns"][i]["mode"] == "guided" and skel["turns"][i]["max_w"] >= 10:
                out.append(("plant_after_question_clause_pass", skel, _copy(texts, i, _sty(
                    skel, f"Can I share something, my {s['noun']} is called {s['value']}.")), "ok", {}))
                break
    fill = {t["i"] for t in _fillers(skel, texts)}
    for u in filler_user(skel):         # two shared words, under half of the line's own: passes on the minimum
        tw = topic_set(skel["topic_text"][u["intent"].split(":")[1]])
        prev = f"{_topic_w(skel).capitalize()} takes some patience and practice."
        line = _sty(skel, "Patience and practice sound fine, yet evenings feel crowded lately.")
        shared = content(line) & content(prev)
        if u["i"] - 1 in fill and u["max_w"] >= 10 and not content(line) & tw and len(shared) == 2 \
                and len(shared) < 0.5 * len(content(line)):
            out.append(("offtopic_prev_two_low_share_pass", skel, _copy(_copy(texts, u["i"] - 1, prev), u["i"], line),
                        "ok", {}))
            break
    f = [t for t in _fillers(skel, texts) if t["max_w"] >= 10 and t["i"] - 1 in texts]
    if f:
        prev = content(texts[f[0]["i"] - 1])
        bases = [x for tx in topics for a in TW.words(tx, "adj") for x in topic_forms(a, "adj") if x in Y_BASES]
        base = next((x for x in bases if stem(x) not in plain | prev and stem(x) in all_topic), None)
        line = f"The {base} part is where it gets hard." if base else ""
        if base and not (content(line) - {stem(base)}) & (all_topic | prev):
            out.append(("offtopic_y_base_pass", skel, _copy(texts, f[0]["i"], line), "ok", {}))
        far = _far(skel, all_topic | prev, 1)
        bad = f"The bus and the {far[0]} matter most." if far else ""
        if any("busy" in TW.words(tx, "adj") for tx in topics) and stem("bus") not in all_topic | prev and far \
                and not content(bad) & (all_topic | prev):
            out.append(("offtopic_bus_from_busy_fire", skel, _copy(texts, f[0]["i"], bad), {"OFFTOPIC"}, {}))
    return out


BUILDERS = [r2c_limits]
