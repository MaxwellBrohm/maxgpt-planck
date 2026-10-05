"""W3 call record -> item records (BANKPASS s2a, s2c, s2d, s2h), for every kind. Line, pool and topic calls go
through itemize.py unchanged; this adds the QUERY_VALUE check (a query or bait line holding a value of its key's type:
the decode-time value ban was not sent, see notes) and the items of the W3 kinds: label candidates, note-frame words,
list names, paraphrases, entity attributes, predicates, intents, topic openings and topic word sets. Each record keeps
its call provenance (teacher, prompt sha256, decode sha256, raw output sha256, seed, preset)."""
import re

import banks_keys as BK
import heldout
import pools as P
from bankpass import gates, itemize as IT, prompts3 as P3, prompts4 as P4, specs, store, templatize as T

VERSION = "w3items-v5"
SHORT = store.SHORT
OPEN_SPEC = dict(specs.line_specs()["open.greet"], min_w=2, max_w=25)
INTENT_W = (3, 16)
CORR_MARKER = re.compile(r"^\W*(?:actually|sorry|oops|wait|no|nope|oh|hmm|correction|scratch that|my bad|i mean|"
                         r"rather|well|whoops|ah|uh|um|erm|let me|to correct|update)\b", re.I)
PARA_COPY_J = 0.6


def base(rec):
    return {"prompt_id": rec["prompt_id"], "prompt_sha256": rec["prompt_sha256"], "call_id": rec["call_id"],
            "decode_sha256": rec["decode_sha256"], "seed": rec["seed"], "raw_sha256": rec["raw_sha256"],
            "sampling": rec.get("preset"), "time": rec["time"]}


def _item(cls, bank, n, text, rec, hits, features=None, seeds=None, holes=None):
    return store.make_item(cls, bank, n, text, rec["author"], "dropped" if hits else "kept",
                           hits[0][0] if hits else None, holes=holes if holes is not None else None,
                           features=features or {}, call=base(rec), seeds=seeds or {}, gates=IT.gate_block(hits))


def _value_res(vtype):
    """capitalized values match case-sensitively (the month May, not the modal), others in any case."""
    out = []
    for cap in (True, False):
        vals = sorted((v for v in P.pool(vtype).values if v[:1].isupper() == cap), key=len, reverse=True)
        if vals:
            out.append(re.compile(r"(?<![A-Za-z'])(?:" + "|".join(re.escape(v) for v in vals) + r")(?![A-Za-z])",
                                  0 if cap else re.I))
    return out


_VRE = {}


def query_value(tpl, key):
    vt = BK.KEYS[key]["vtype"]
    if vt not in _VRE:
        _VRE[vt] = _value_res(vt)
    for rx in _VRE[vt]:
        m = rx.search(gates.bare(tpl))
        if m:
            return [("QUERY_VALUE", m.group(0))]
    return []


def line_items(rec):
    c = rec["plan"]
    mined = [c["mined"]["text"]] if c.get("mined") else ()
    out = IT.line_items(rec, c, prompt_text=rec["prompt"], mined=mined)
    spec = specs.line_specs()[rec["bank"]]
    if spec.get("role") == "corr":          # amendment 2: the prompt says the marker is added separately
        for it in out:
            m = CORR_MARKER.match(it["text"].replace("{M}", "", 1)) if it["status"] == "kept" else None
            if m:
                it.update(status="dropped", drop="CORR_MARKER")
                it["gates"]["hits"] = [["CORR_MARKER", m.group(0)]]
    if spec.get("role") in ("query", "bait"):
        for it in out:
            hits = query_value(it["text"], spec["key"]) if it["status"] == "kept" else []
            if hits:
                it.update(status="dropped", drop=hits[0][0])
                it["gates"]["hits"] = [list(h) for h in hits]
    for it in out:
        it["call"]["sampling"] = rec.get("preset")
        if c.get("voice"):
            it["seeds"]["persona_id"] = c["voice"].get("persona_id")
        if c.get("mined"):
            it["seeds"]["mined_doc_ids"] = [c["mined"]["doc_id"]]
    return out


def word_checks(text, lo=1, hi=4):
    hits = gates.pool_checks(text.replace("{o}", "thing"), "label")
    hits = [h for h in hits if h[0] != "LEN_ITEM"]
    if not lo <= len(gates.words(text.replace("{o}", "x"))) <= hi:
        hits.append(("LEN_ITEM", text))
    return hits


def label_items(rec):
    c, out = rec["plan"], []
    for i, line in enumerate(rec["lines"]):
        txt = store.straight(line).strip(" .").lower()
        if c["kind"] == "label" and c.get("o"):
            tpl, why = T.templatize(txt, {"o": c["o"]})
            txt, hits = (tpl, []) if not why else (txt, [(why, txt)])
        else:
            hits = []
        lo, hi = (1, 3) if c["kind"] == "notewas" else (1, 4)
        hits += word_checks(txt, lo, hi)
        feats = {"key": c.get("key"), "vtype": c.get("vtype"), "o": c.get("o")}
        out.append(_item("N", rec["bank"], f"{SHORT[c['teacher']]}_{i}", txt, rec, hits, feats))
    return out


_PART = re.compile(r"^\W*part (?:one|two|1|2)\W*", re.I)


def para_items(rec):
    head, tail = (_PART.sub("", store.straight(x)).strip() for x in rec["lines"])
    bh, bt = P4.base_block()
    hits = []
    for code, rx in (("DASH", gates.G.DASH_RE), ("MARKDOWN", gates.L.MARKDOWN_RE), ("EMOJI", gates.L.EMOJI_RE)):
        if rx.search(head + " " + tail):
            hits.append((code, rx.search(head + " " + tail).group(0)))
    v = [h for h in heldout.vocab_hits(head + " " + tail) if h != "digit"]
    hits += [("HELDOUT_VOCAB", v[0])] if v else []
    hits += [("HELDOUT_ECHO", e) for e in heldout.echo5(head + " " + tail)[:1]]
    from bankpass import trim
    j = trim.jaccard(trim.grams3(head + " " + tail), trim.grams3(bh + " " + bt))
    if j >= PARA_COPY_J:
        hits.append(("PARA_COPY", f"jaccard {j:.2f}"))
    n = rec["call_id"].split(".", 1)[1]
    return [_item("Q", "instr.para", n, head + "\n" + tail, rec, hits, {"head": head, "tail": tail, "copy_j": round(j, 3)},
                  holes=[])]


def attr_items(rec):
    c, out = rec["plan"], []
    for i, line in enumerate(rec["lines"]):
        lab, _, typ = store.straight(line).rpartition(": ")
        lab = lab.strip().lower()
        hits = word_checks(lab, 2, 4) if typ in P3.ATTR_TYPES else [("ATTR_FORM", line)]
        vt = {"day of the week": "weekday", "city": "city", "month": "month", "colour": "colour"}.get(typ)
        out.append(_item("U", "attr", f"{c['src_id']}~{i}", lab, rec, hits,
                         {"kind": c["value"], "kind_id": c["src_id"], "vtype": vt}, holes=[]))
    return out


def pred_items(rec):
    c, out = rec["plan"], []
    for i, (line, row) in enumerate(zip(rec["lines"], c["rows"])):
        name, attr, value = row
        txt = store.straight(line).rstrip(" .")
        tpl, why = T.templatize(txt, {"v": value})
        hits = [(why, txt)] if why else []
        if not why:
            filled = f"the {name} {txt}"
            hits += [(code, rx.search(filled).group(0)) for code, rx in
                     (("DASH", gates.G.DASH_RE), ("DIGIT", heldout.DIGIT_RE)) if rx.search(filled)]
            hits += [("HELDOUT_ECHO", e) for e in heldout.echo_hits(filled)[:1]]
            hits += [("LEN_ITEM", txt)] if not 2 <= len(gates.words(txt)) <= 8 else []
        out.append(_item("U", "pred", f"{c['attr_ids'][i]}", tpl or txt, rec, hits,
                         {"attr_id": c["attr_ids"][i], "attr": attr, "probe_name": name, "probe_value": value}))
    return out


def intent_items(rec):
    c, out = rec["plan"], []
    for i, line in enumerate(rec["lines"]):
        txt = store.straight(line).rstrip(".")
        txt = txt if re.match(r"I\b", txt) else txt[:1].lower() + txt[1:]
        hits = gates.item_checks(txt, "open.greet", dict(OPEN_SPEC, side="prompt", min_w=INTENT_W[0],
                                                          max_w=INTENT_W[1]), rec["prompt"].replace(c["topic"], " "))
        v = [h for h in heldout.vocab_hits(txt) if h != "digit"]
        hits += [("HELDOUT_VOCAB", v[0])] if v else []
        for t in (txt, txt + " " + c["topic"]):
            hits += [("HELDOUT_ECHO", e) for e in heldout.echo5(t)[:1]]
        out.append(_item("I", "intent", f"{c['topic_id']}~{SHORT[c['teacher']]}_{i}", txt, rec, hits,
                         {"topic_id": c["topic_id"], "topic": c["topic"]}, holes=[]))
    return out


def topen_items(rec):
    c, out = rec["plan"], []
    for i, line in enumerate(rec["lines"]):
        tid, topic = c["topic_ids"][i // c["k"]], c["topics"][i // c["k"]]
        txt = store.straight(line)
        prompt = rec["prompt"]
        for t in c["topics"]:
            prompt = prompt.replace(t, " ")
        hits = gates.item_checks(txt, "open.greet", OPEN_SPEC, prompt)
        hits += gates.heldout_checks(txt, "open.greet", OPEN_SPEC, n_fills=0)
        if topic.lower() in txt.lower():
            hits.append(("TOPIC_COPY", topic))     # O(2) openings paraphrase their topic; a verbatim one is open.topic
        out.append(_item("O", "open.topic_spec", f"{tid}~{i % c['k']}", txt, rec, hits,
                         {"topic_id": tid, "topic": topic}, holes=[]))
    return out


def wordset_items(rec):
    c, feats, drops = rec["plan"], {"topic_id": rec["plan"]["topic_id"], "topic": rec["plan"]["topic"]}, []
    for line, part in zip(rec["lines"], ("nouns", "verbs", "adjs")):
        words, seen = [], set()
        for w in store.straight(line).split(":", 1)[1].split(","):
            w = w.strip().lower()
            if not w or w in seen:
                continue
            seen.add(w)
            bad = gates.pool_checks(w, "word") + ([("FORM", w)] if not re.fullmatch(r"[a-z]+(?:'[a-z]+)?", w) else [])
            (drops.append([w, bad[0][0]]) if bad else words.append(w))
        feats[part] = words
    feats["dropped_words"] = drops
    return [_item("W", "topicwords.raw", f"{c['topic_id']}~{SHORT[c['teacher']]}", c["topic"], rec, [], feats,
                  holes=[])]


KIND_ITEMS = {"label": label_items, "notewas": label_items, "listname": label_items, "para": para_items,
              "attr": attr_items, "pred": pred_items, "intent": intent_items, "topen": topen_items,
              "wordset": wordset_items}


PERSONAL = re.compile(r"\b(?:i|me|my|mine|we|us|our|you|your)\b", re.I)


def topic_checks(it):
    """amendment 3: a topic is a noun phrase about a subject: no name (a capital after the first word), no first or
    second person, and a first word that is a real word (Qwen wrote "na ..." in 4 of 490)."""
    w = it["text"].split()
    hits = [("TOPIC_NAME", x) for x in w[1:] if x[:1].isupper()][:1]
    hits += [("TOPIC_PERSONAL", m.group(0)) for m in [PERSONAL.search(it["text"])] if m]
    hits += [("TOPIC_FORM", w[0])] if w and w[0].lower() in ("na", "n", "a.", "the.") else []
    hits += [("TOPIC_FORM", "?")] if "?" in it["text"] else []
    if hits and it["status"] == "kept":
        it.update(status="dropped", drop=hits[0][0])
        it["gates"]["hits"] = [list(h) for h in hits]
    return it


def items_for(rec):
    """the item records of one author call record ([] for a call with a parse problem or a non-item kind)."""
    if rec.get("problem") or rec.get("error"):
        return []
    k = rec["kind"]
    if k == "line":
        return line_items(rec)
    if k in ("pool", "topic"):
        out = IT.value_items(rec, rec["plan"])
        return [topic_checks(it) for it in out] if k == "topic" else out
    f = KIND_ITEMS.get(k)
    return f(rec) if f else []
