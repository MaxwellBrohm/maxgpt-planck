"""Hand-written validator for the skeleton JSON of SPEC section 2 (no jsonschema dependency).
validate(skel) -> list of error strings; empty means valid. Checks types, enums, id references, turn order
(user, then one assistant reply or assistant call + tool + assistant answer), event turn references and ranges."""

KINDS = {"S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"}
TOP = {"skel_id": str, "seed": (int, str), "attempt": list, "gen_version": str, "register": str, "slice": str,
       "topic_path": list, "topic_text": dict, "opening": str, "closing": str, "user": dict, "assistant": dict,
       "slots": dict, "persons": dict, "events": list, "turns": list, "required_words": dict,
       "heldout_gate": str, "provenance": dict, "rc12": str, "estimate": dict}
SLOT = {"type": str, "value": str, "nonce": bool, "owner": str, "key": str, "features": dict, "counted": bool}
TURN = {"i": int, "role": str, "mode": str, "min_w": int, "max_w": int, "must_include": list,
        "must_exclude": list, "events": list, "mask": int}
ENUMS = {"register": {"RS", "RM", "RL"}, "slice": {"core", "lookup"}, "closing": {"goodbye", "open", "abrupt"}}
USER_ENUMS = {"age_band": {"teen", "20s", "30s", "40s", "50s", "60s", "70s"},
              "style": {"terse", "chatty", "typos", "lowercase"}}


def _types(obj, spec, where, errs):
    for k, t in spec.items():
        if k not in obj:
            errs.append(f"{where}: missing {k}")
        elif not isinstance(obj[k], t) or (t is int and isinstance(obj[k], bool)):
            errs.append(f"{where}: {k} has type {type(obj[k]).__name__}")


def _turns(skel, errs):
    turns = skel["turns"]
    for n, t in enumerate(turns):
        _types(t, TURN, f"turn {n}", errs)
        if t.get("i") != n:
            errs.append(f"turn {n}: index {t.get('i')}")
        if t.get("role") not in ("user", "assistant", "tool") or t.get("mode") not in ("exact", "guided", "free"):
            errs.append(f"turn {n}: role or mode")
        if t.get("mode") == "exact" and not t.get("text"):
            errs.append(f"turn {n}: exact without text")
        if t.get("mode") == "guided" and (t.get("text") or not t.get("intent")):
            errs.append(f"turn {n}: guided needs intent and no text")
        if not 0 <= t.get("min_w", 0) <= t.get("max_w", 0):
            errs.append(f"turn {n}: word bounds")
        if t.get("mask") not in (0, 1) or (t.get("mask") == 1 and t.get("role") != "assistant"):
            errs.append(f"turn {n}: mask")
        if set(t.get("must_include", [])) & set(t.get("must_exclude", [])):
            errs.append(f"turn {n}: include and exclude overlap")
    roles = "".join({"user": "U", "assistant": "A", "tool": "T"}.get(t.get("role"), "?") for t in turns)
    import re
    if not re.fullmatch(r"(U(A|ATA))+", roles):
        errs.append("turn order " + roles)
    n_user = roles.count("U")
    if not 4 <= n_user <= 12:
        errs.append(f"{n_user} user turns")
    if roles.count("T") and skel.get("slice") != "lookup":
        errs.append("tool turn outside the lookup slice")


def _events(skel, errs):
    ids = [e.get("id") for e in skel["events"]]
    if len(set(ids)) != len(ids):
        errs.append("duplicate event ids")
    if not 2 <= len(ids) <= 4:
        errs.append(f"{len(ids)} events")
    n_turns = len(skel["turns"])
    for e in skel["events"]:
        if e.get("kind") not in KINDS:
            errs.append(f"{e.get('id')}: kind")
        for k in ("params", "gold", "turns"):
            if not isinstance(e.get(k), dict):
                errs.append(f"{e.get('id')}: {k}")
        for role, i in e.get("turns", {}).items():
            if not isinstance(i, int) or not 0 <= i < n_turns:
                errs.append(f"{e['id']}: turn {role}={i}")
            elif e["id"] not in skel["turns"][i]["events"]:
                errs.append(f"{e['id']}: turn {i} does not list the event")
        for sid in [e["params"].get(k) for k in ("slot", "slot_b", "queried", "distractor")] + \
                [op.get("slot") for op in e["params"].get("ops", [])]:
            if sid is not None and sid not in skel["slots"]:
                errs.append(f"{e['id']}: unknown slot {sid}")
    if "S9" in {e.get("kind") for e in skel["events"]} and skel["slice"] != "lookup":
        errs.append("S9 outside the lookup slice")


def validate(skel):
    errs = []
    _types(skel, TOP, "skeleton", errs)
    if errs:
        return errs
    for k, allowed in ENUMS.items():
        if skel[k] not in allowed:
            errs.append(f"{k}={skel[k]}")
    for k, allowed in USER_ENUMS.items():
        if skel["user"].get(k) not in allowed:
            errs.append(f"user {k}")
    if skel["user"].get("name") is not None and skel["user"]["name"] not in skel["slots"]:
        errs.append("user name slot")
    if skel["assistant"].get("name") not in skel["slots"]:
        errs.append("assistant name slot")
    if not 1 <= len(skel["topic_path"]) <= 3 or set(skel["topic_path"]) != set(skel["topic_text"]):
        errs.append("topic path")
    for sid, s in skel["slots"].items():
        _types(s, SLOT, f"slot {sid}", errs)
        if s.get("owner") not in ("user", "assistant") and s.get("owner") not in skel["persons"]:
            errs.append(f"slot {sid}: owner {s.get('owner')}")
    for pid, p in skel["persons"].items():
        if p.get("name") is not None and p["name"] not in skel["slots"]:
            errs.append(f"person {pid}: name slot")
    rw = skel["required_words"]
    if not all(isinstance(rw.get(k), str) for k in ("noun", "verb", "adj")) \
            or not isinstance(rw.get("turn_hint"), list):
        errs.append("required words")
    _turns(skel, errs)
    _events(skel, errs)
    return errs
