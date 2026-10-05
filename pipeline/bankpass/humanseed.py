"""Prompt-side persona seeds (Nemotron-Personas-USA, BANKPASS s3 Q) and the safety rubric list (LDNOOBW English, s3 Z),
stored as items with provenance (W2). Neither is training text: a persona is the voice seed of a bank call
(prompts.voice) and of a render prompt (PERSONA_LEAK unchanged); the safety list is checker rubric.

Persona seed text: the record's one-line "persona" field with the person's name taken off ("Mary Alberti blends X"
-> "someone who blends X"; "Mary Alberti, a calm cook who X" -> "a calm cook who X"), so no persona name reaches a
prompt. The name is the capitalized run that opens the line (particles such as "de la" stay in it), confirmed by
another persona field opening with it. Drops: NO_LEAD (the line does not start with that name, or the name is
possessive), NAME_AGAIN (the first or last name occurs again), MINOR (age under 18: chat users are adults; the card
says minors are excluded, but about one record in five is under 18), DUP, and gates.seed_checks (held-out terms,
digits, E004 5-grams, FAKE safety list, role labels, accented letters, 4 to 80 words). Selection: a seeded hash order over every uuid in
the eleven shards; the first candidates that pass, up to the target. Reading parquet needs pyarrow (PC:
~/venv/bin/python)."""
import collections
import hashlib
import os
import re

from bankpass import gates, sources, store

PERSONA_TARGET, CANDIDATE_FACTOR = 5000, 3
FIELDS = ("persona", "professional_persona", "sports_persona", "arts_persona", "travel_persona", "culinary_persona")
META = ("uuid", "age", "sex", "occupation", "state")
PARTICLES = {"de", "la", "del", "da", "di", "van", "von", "der", "du", "le", "bin", "al", "el", "dos", "das", "y"}
_DASHES = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2212": "-"})
_LONG = re.compile(r"\s*[\u2013\u2014]\s*")


def clean(text):
    """straight quotes, NFKC, hyphen variants to '-', en and em dashes to ', ' (seed text is prompt side, and no
    dash is written into a prompt), single spaces."""
    t = store.straight(text).translate(_DASHES)
    return re.sub(r"\s+", " ", _LONG.sub(", ", t)).strip()


def person_name(row):
    """the full name: the run of capitalized words (particles such as "de la" inside it) that starts the persona line,
    ending at a word with a comma or a possessive; it must open another persona field too (each field starts with
    the full name), else [] (the line opens some other way: "At 38, ...")."""
    ws = [clean(row[f] or "").split() for f in FIELDS]
    name = []
    for t in ws[0]:
        b = _tok(t)
        if not b or not (b[:1].isupper() or (name and b.lower() in PARTICLES)):
            break
        name.append(t)
        if t.endswith((",", "'s")):
            break
    while name and _tok(name[-1]).lower() in PARTICLES:
        name.pop()
    shared = max(_shared(ws[0], w) for w in ws[1:])
    return name if name and shared >= len(name) else []


def _tok(word):
    return word.rstrip(",.;:").removesuffix("'s")


def _shared(a, b):
    k = 0
    while k < len(a) and k < len(b) and _tok(a[k]) == _tok(b[k]):
        k += 1
    return k


def seed_text(row):
    """-> (text, None) or (None, drop code). "Name verb ..." -> "someone who verb ..."; "Name, a ... who ..." -> the
    appositive as it stands; a possessive lead ("Name's soundtrack ...") is NO_LEAD."""
    name = person_name(row)
    words = clean(row["persona"] or "").split()
    if not name or words[:len(name)] != name or name[-1].endswith("'s"):
        return None, "NO_LEAD"
    rest = words[len(name):]
    own = {_bare(w) for w in name if w[:1].isupper()}
    if any(_bare(w) in own for w in rest):
        return None, "NAME_AGAIN"
    if not rest or not rest[0][:1].islower():
        return None, "NO_LEAD"
    body = " ".join(rest).rstrip(".")
    return (body if name[-1].endswith(",") else "someone who " + body), None


def _bare(word):
    return word.strip(".,;:!?\"()").lower().removesuffix("'s")


def age_band(age):
    return "80s+" if age >= 80 else f"{age // 10 * 10}s"


def hash_key(seed, uuid):
    return hashlib.sha256(f"{seed}|{uuid}".encode()).hexdigest()


def read_candidates(rec, n_cand, seed="bankpass-personas-v0"):
    """[(row dict, shard rel, row index)] for the first n_cand uuids in seeded hash order over all shards."""
    import pyarrow.parquet as pq
    shards = sorted(r for r in rec["files"] if r.endswith(".parquet"))
    keys = []
    for rel in shards:
        uu = pq.read_table(os.path.join(rec["_dir"], rel), columns=["uuid"]).column("uuid").to_pylist()
        keys += [(hash_key(seed, u), rel, i) for i, u in enumerate(uu)]
    keys.sort()
    want = collections.defaultdict(dict)
    for k, rel, i in keys[:n_cand]:
        want[rel][i] = k
    out = []
    for rel in shards:
        if not want[rel]:
            continue
        idx = sorted(want[rel])
        t = pq.read_table(os.path.join(rec["_dir"], rel), columns=list(FIELDS) + list(META))
        rows = t.take(idx).to_pylist()
        out += [(want[rel][i], row, rel, i) for i, row in zip(idx, rows)]
    return [(r, rel, i) for _, r, rel, i in sorted(out, key=lambda x: x[0])], len(keys)


def build_personas(rec, ctx_gate, target=PERSONA_TARGET, candidates=None):
    """-> (items, stats). candidates: [(row, rel, i)] (tests); read from the shards when None."""
    universe = None
    if candidates is None:
        candidates, universe = read_candidates(rec, target * CANDIDATE_FACTOR)
    items, kept, seen = [], 0, set()
    for row, rel, i in candidates:
        if kept >= target:
            break
        text, drop = seed_text(row)
        hits = []
        if not drop and int(row["age"]) < 18:
            drop = "MINOR"
        if not drop:
            hits = [list(h) for h in gates.seed_checks(text)]
            drop = hits[0][0] if hits else "DUP" if store.norm_key(text) in seen else None
        if not drop:
            seen.add(store.norm_key(text))
        src = sources.source_block(rec, rel, f"{rel} row {i} uuid {row['uuid']}")
        src["generated"] = "synthetic persona text (dataset card: NeMo Data Designer with gpt-oss-120b)"
        feats = {"uuid": row["uuid"], "age": int(row["age"]), "age_band": age_band(int(row["age"])),
                 "sex": row["sex"], "occupation": row["occupation"], "state": row["state"]}
        items.append(store.make_item("Q", "seed.persona", len(items), text or "", {"kind": "human", "source": src},
                                     status="dropped" if drop else "kept", drop=drop, features=feats,
                                     gates=dict(ctx_gate, hits=hits or ([drop] if drop else []))))
        kept += not drop
    return items, {"universe": universe, "candidates": len(candidates), "kept": kept}


def build_safety(rec, ctx_gate, rel="en"):
    """LDNOOBW English, one item per line (rubric only: no held-out or safety gate applies to it)."""
    with open(os.path.join(rec["_dir"], rel), encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    items, seen = [], set()
    for i, ln in enumerate(lines, 1):
        t = store.straight(ln)
        if not t:
            continue
        drop = "DUP" if t.lower() in seen else None
        seen.add(t.lower())
        items.append(store.make_item("Z", "rubric.safety", len(items), t,
                                     {"kind": "human", "source": sources.source_block(rec, rel, f"line {i}")},
                                     status="dropped" if drop else "kept", drop=drop,
                                     gates=dict(ctx_gate, hits=[drop] if drop else [],
                                                applied="none: rubric list, never training text")))
    return items, {"lines": len(lines), "kept": sum(it["status"] == "kept" for it in items)}


def safety_terms(items):
    return [it["text"] for it in items if it["status"] == "kept"]
