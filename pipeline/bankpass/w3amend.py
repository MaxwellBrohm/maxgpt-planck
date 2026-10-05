"""W3 plan amendments (2026-10-04, after reading Qwen's first hold; see notes). Amendment 1 (BANKPASS s3 P deviation):
version 2 of the
teacher pool calls (no seed words; prompts4.POOL_ASK2) and of the part-of-speech calls (prompts4.POS2), replacing
the version 1 calls of plan.json one for one (same bank, teacher rotation and size; ids pool2.* and pos2.*). Version 1
records stay on disk; their items are marked SUPERSEDED_V1 and their POS lines are not used. Topic calls keep their
seed words (their outputs were fine). Amendment 2: the note-frame word call gets a clearer prompt (notewas2.*, all
three teachers; Qwen read version 1 as kinds of facts), and every author call cut off at max_tokens (finish
"length", a parse problem) gets one retry with the same inputs and the next seed under the larger token budget
(id + "r1"; with the same seed Qwen's retries ran away again in 7 of 8).
Amendment 3: topic-specific openings are one call per topic (topen2.*; Qwen interleaved the 5 topics of a version 1
call, so its lines could not be mapped to topics), and the verbs calls built on version 1 POS lines are superseded.
Amendment 4: key queries and baits must end in "?" at decoding, and rule word counts, swap labels, lookup facts and
list joins are forced literals; every teacher runs those banks' calls as <id>v4 and any earlier calls of them (Qwen's,
and Ministral's first hold, which started on the earlier code) are superseded, so every teacher's lines in a bank come
from one prompt and one decode.
Amendment 6: amendment 4's decode is taken back out (a forced "?" or a multi-word literal anywhere in a line made
the teachers run on to max_tokens when their line did not hold it: Qwen 166 of 387 query calls, Ministral 212 of 268,
list joins and lookup facts near all); those banks run again for every teacher as <id>v6 with the original decode,
and the v4 calls are superseded. Amendment 4's CPU parts (the noun hole in twins, swap.q and PERSPECTIVE) stay.
Amendment 5: topics get a stricter prompt (topic2.*, all three teachers, same seed words): Ministral's version 1 topics
were questions and requests of 9 to 13 words. Topic selection and the label votes read the amended call list."""
SUPERSEDED = ("pool.", "pos.", "notewas.", "topen.", "verbs.pos.", "topic.")


def amend4_bank(bank):
    """banks whose decode changed in amendment 4: key queries and baits end in "?", and the forced literal of rule
    word counts, swap labels, lookup facts and list joins."""
    return (bank.startswith("key.") and bank.endswith((".query", ".bait"))) or bank in (
        "rule.max_words", "swap.q", "lookup.cf", "lookup.ctx") or bank.startswith("list.init.")


def current6(call_id):
    """the current calls of an amendment 4 bank: amendment 6 (<id>v6, the original decode)."""
    return call_id.endswith(("v6", "v6r1"))


def superseded_item(it):
    """an item from a superseded call: the version 1 prefixes, or a pre-amendment-4 call of an amendment 4 bank."""
    cid = it["call"]["call_id"]
    return superseded(cid) or (amend4_bank(it["bank"]) and not current6(cid))
RETRY_KINDS = ("line", "pool", "topic", "label", "notewas", "listname", "attr", "pred", "intent", "topen", "wordset",
               "para", "pos", "verbs", "ly", "relfeat")


def superseded(call_id):
    return call_id.startswith(SUPERSEDED)


def calls(plan):
    out = []
    for c in plan["calls"]:
        if c["kind"] == "pool":
            out.append(dict(c, call_id="pool2." + c["call_id"].split(".", 1)[1], v=2, seed_words=None))
        elif c["kind"] == "pos":
            out.append(dict(c, call_id="pos2." + c["call_id"].split(".", 1)[1], v=2))
        elif c["kind"] == "topic":
            out.append(dict(c, call_id="topic2." + c["call_id"].split(".", 1)[1], v=2))
        elif c["kind"] == "notewas":
            out.append(dict(c, call_id="notewas2." + c["call_id"].split(".", 1)[1], v=2))
        elif c["kind"] == "line" and amend4_bank(c["bank"]):
            out.append(dict(c, call_id=c["call_id"] + "v6"))
    return out


def retries(recs, teacher):
    """one retry per author call of this teacher that was cut off at max_tokens."""
    have = {r["call_id"] for r in recs}
    out = []
    for r in recs:
        if r.get("kind") == "line" and amend4_bank(r["bank"]) and not current6(r["call_id"]):
            continue                         # a superseded call is never retried
        if (r["author"]["model"] == teacher and r.get("kind") in RETRY_KINDS and r.get("problem")
                and r.get("finish") == "length" and not r["call_id"].endswith("r1") and r["call_id"] + "r1" not in have):
            out.append(dict(r["plan"], call_id=r["call_id"] + "r1", retry_of=r["call_id"],
                            seed=r["plan"].get("seed", 0) + 1))
    return out


def base_calls(plan):
    """the plan's base calls with amendment 1 applied."""
    return [c for c in plan["calls"] if not superseded(c["call_id"])
            and not (c["kind"] == "line" and amend4_bank(c["bank"]))] + calls(plan)
