"""Skeleton-level held-out gate (SPEC section 9) and cheap hygiene on program-inserted text.
check(skel) returns [(code, detail)]: HELDOUT_VOCAB on every natural-language string the skeleton carries (exact
texts, must_include/must_exclude strings, slot values, nouns, system text, required words, golds, intents),
HELDOUT_ECHO on exact texts, HELDOUT_STRUCT on the structure, DASH on exact texts, and LEAK when an exact turn
states a value inside another event's distance window. An empty list means the skeleton may go to the teacher."""
import re

import heldout
from banks_keys import KEYS

DASH_RE = re.compile("[\\u2012\\u2013\\u2014\\u2015\\u2212]|--| - ")
STATE_OPS = {"set", "fix"}


def text_fields(skel):
    """(where, string) for every natural-language string in the skeleton."""
    out = []
    for sid, s in skel["slots"].items():
        out.append((f"slot {sid}", s["value"]))
        if s.get("noun"):
            out.append((f"slot {sid} noun", s["noun"]))
    for pid, p in skel["persons"].items():
        out.append((f"person {pid}", p["relation"]))
    for t in skel["turns"]:
        if t.get("text"):
            out.append((f"turn {t['i']} text", t["text"]))
        for f in ("must_include", "must_exclude"):
            out += [(f"turn {t['i']} {f}", x) for x in t.get(f, [])]
        if t.get("intent") and not t["intent"].startswith("topic:"):
            out.append((f"turn {t['i']} intent", t["intent"]))
    if skel["assistant"].get("system_text"):
        out.append(("system", skel["assistant"]["system_text"]))
    rw = skel["required_words"]
    out += [(f"required {k}", rw[k]) for k in ("noun", "verb", "adj")]
    for e in skel["events"]:
        for k, v in e["gold"].items():
            if k in ("slot", "act", "need", "result", "query", "perspective"):
                continue
            if isinstance(v, str):
                out.append((f"{e['id']} gold {k}", v))
            elif isinstance(v, list):
                out += [(f"{e['id']} gold {k}", x) for x in v if isinstance(x, str)]
    out += [(f"topic {t}", x) for t, x in skel.get("topic_text", {}).items()]
    return out


def _obj_terms(skel):
    """object terms for E004 normalization: slot nouns, and the labels of singleton keys ("favourite food")."""
    terms = {s["noun"] for s in skel["slots"].values() if s.get("noun")}
    terms |= {KEYS[s["key"]]["label"] for s in skel["slots"].values()
              if s["key"] in KEYS and not KEYS[s["key"]]["noun"]}
    return sorted(terms, key=len, reverse=True)


def _values(skel):
    return sorted({s["value"] for s in skel["slots"].values()}, key=len, reverse=True)


def struct_violations(skel):
    out = []
    for e in skel["events"]:
        p = e["params"]
        if "d" in p and not (0 <= p["d"] <= heldout.MAX_D):
            out.append(("HELDOUT_STRUCT", f"{e['id']} d={p['d']}"))
        if "alias" in p or p.get("ref_form") == "alias":
            out.append(("HELDOUT_STRUCT", f"{e['id']} alias"))
        per_slot = {}
        for op in p.get("ops", []):
            if op.get("op") in STATE_OPS:
                per_slot[op["slot"]] = per_slot.get(op["slot"], 0) + 1
        for sid, n in per_slot.items():
            if n > heldout.MAX_CORRECTIONS:
                out.append(("HELDOUT_STRUCT", f"{sid} corrected {n} times"))
    counts = {}
    for sid, s in skel["slots"].items():
        if s.get("counted"):
            counts[s["type"]] = counts.get(s["type"], 0) + 1
    out += [("HELDOUT_STRUCT", f"{t}: {n} slots") for t, n in counts.items() if n > heldout.MAX_SLOTS_PER_TYPE]
    out += _query_echo(skel)
    return out


def _query_echo(skel):
    """no E004 frame echo between an exact query and any exact statement of the queried slot."""
    out, terms, vals = [], _obj_terms(skel), _values(skel)
    by_i = {t["i"]: t for t in skel["turns"]}
    for e in skel["events"]:
        q = e["turns"].get("query")
        slot = e["gold"].get("slot")
        if q is None or slot is None or by_i[q]["mode"] != "exact":
            continue
        for op in e["params"].get("ops", []):
            if op.get("slot") != slot:
                continue
            st = by_i[e["turns"][op["turn"]]]
            if st.get("mode") == "exact" and st["role"] == "user":
                runs = heldout.frame_echo(by_i[q]["text"], st["text"], terms, vals)
                if runs:
                    out.append(("HELDOUT_STRUCT", f"{e['id']} frame echo {runs[0]}"))
    return out


def check(skel):
    out = []
    for where, s in text_fields(skel):
        hits = heldout.vocab_hits(s)
        if hits:
            out.append(("HELDOUT_VOCAB", f"{where}: {hits}"))
    terms, vals = _obj_terms(skel), _values(skel)
    for t in skel["turns"]:
        if t["mode"] == "exact" and t["role"] == "user":
            hits = heldout.echo_hits(t["text"], terms, vals)
            if hits:
                out.append(("HELDOUT_ECHO", f"turn {t['i']}: {hits[0]}"))
        if t.get("text") and DASH_RE.search(t["text"]):
            out.append(("DASH", f"turn {t['i']}"))
    for tid, text in skel.get("topic_text", {}).items():
        hits = heldout.echo_hits(text, terms, vals)
        if hits:
            out.append(("HELDOUT_ECHO", f"topic {tid}: {hits[0]}"))
    out += struct_violations(skel)
    return out
