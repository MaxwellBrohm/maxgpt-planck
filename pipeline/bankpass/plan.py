"""Stage P call plan (BANKPASS s1, s2i, s6): a seeded, deterministic list of call ids. Targets come from MEASURED use
rates: skeletons are built with the current sampler and every exact bank line, marker prefix, slot and system text is
counted; target lines per bank = max(30, rate per 1k x 2.5) (SPEC 4's 50 uses per 100k needs rate x 2 lines, plus
25% for lines lost to later gates). Authorship rotates over the three pinned teachers inside every bank, so each
teacher writes a third of each bank (s0). Fills are drawn here, at plan time, from the loaded pools, so a restart
re-issues byte-identical calls. Nothing here calls a model."""
import collections
import json
import math

import banks as B
import banks_keys as BK
import pools as P
from bankpass import specs, store

TEACHER_ORDER = ("qwen3.5-9b", "ministral-3-8b", "gemma-4-12b")
LINES_PER_CALL = 10
MARKER_TARGETS = {"marker.fix": 60, "marker.err": 20}
# s3 P: teacher pool targets (items kept), listed 20 per call, 1.5x for dedup and judge losses
POOL_TARGETS = {"job": 400, "hobby": 300, "food": 400, "grocery": 400, "chore": 200, "plan": 400, "object": 400,
                "pet_kind": 40, "pet_name": 800, "assistant_name": 300, "relation": 30, "entity_kind": 60}
POOL_PER_CALL, POOL_OVERGEN = 20, 1.5
TOPICS_STAGE_P, TOPICS_PER_CALL = 1000, 10
# dry pilot 2 measured output rates (notes.txt), tok/s at bulk concurrency; estimates exclude load and lock waits
TOK_PER_S = {"qwen3.5-9b": 441, "ministral-3-8b": 692, "gemma-4-12b": 161}
TOK_PER_LINE = 22
HOLD_MIN = 33


def measure_rates(n=1000, shard="bankpass-rates", register="RM"):
    """exact bank line uses, marker prefixes, slot types and system texts over n shard skeletons (per n)."""
    import skeleton_shard as SS
    lines, slots, system, markers = (collections.Counter() for _ in range(4))
    for sk in SS.shard(shard, n, register):
        for t in sk["turns"]:
            ref = t.get("bank_ref")
            if not ref:
                continue
            lines[ref.rsplit(".", 1)[0]] += 1
            for m in B.MARKERS:
                markers["marker.fix"] += t["text"].startswith(m)
            for m in B.ERR_FIX_MARKERS:
                markers["marker.err"] += t["text"].startswith(m)
        for s in sk["slots"].values():
            slots[s["type"]] += 1
        sref = sk["assistant"].get("system_ref")
        if sref:
            system[sref.rsplit(".", 1)[0]] += 1
    lines.update(system)
    lines.update(markers)
    return {"n": n, "shard": shard, "register": register, "lines": dict(lines), "slots": dict(slots)}


def targets(rates):
    """{bank: target lines}: max(30, rate per 1k x 2.5); markers exactly the s3 M counts (60 + 20), because the use
    cap counts the whole corrected line, not its marker."""
    per_k = 1000.0 / rates["n"]
    out = {}
    for bank in specs.line_specs():
        r = rates["lines"].get(bank, 0) * per_k
        out[bank] = max(30, math.ceil(r * 2.5))
    out.update(MARKER_TARGETS)
    return out


def _rotation(bank, n_calls):
    """author per call: an even rotation, started at a seeded offset per bank so no teacher always goes first."""
    off = int(store.sha256_text("bankpass-author:" + bank)[:8], 16) % 3
    return [TEACHER_ORDER[(off + i) % 3] for i in range(n_calls)]


def _key_fill(key, rng, form):
    d = BK.KEYS[key]
    used = set()
    nouns = P.pool(d["noun"]).values if d["noun"] else [None]
    if form == "pronoun":
        nouns = [x for x in nouns if BK.pronouns(key, x)[0]]
    noun = rng.choice(nouns)
    v, _ = P.draw(rng, d["vtype"], used, allow_nonce=False)
    old, _ = P.draw(rng, d["vtype"], used, allow_nonce=False)
    f = {"v": v, "old": old, "o": noun, "article": P.features(d["vtype"], v)["article"] or None}
    if form == "pronoun":
        f["p"] = BK.pronouns(key, noun)[0]
    return f


def ready(bank, spec):
    """a prompt exists for this bank (prompts.py); rule, list, lookup and swap lines are written at W3 build time."""
    from bankpass import prompts
    return spec["class"] == "K" or bank in prompts.CLASS_ASK


def line_calls(tgt, n_per=LINES_PER_CALL):
    """the call list for every line bank: ceil(target / n_per) calls, forms round robin for key queries and
    corrections (a pronoun form only where the key has pronouns), fills drawn now."""
    sp, calls = specs.line_specs(), []
    for bank in sorted(tgt):
        s = sp[bank]
        n_calls = math.ceil(tgt[bank] / n_per)
        forms = s["forms"] or [None]
        for i, teacher in enumerate(_rotation(bank, n_calls)):
            rng = P.seeded("bankpass-plan", bank, i)
            form = forms[i % len(forms)]
            if s["class"] == "K":
                fills = [_key_fill(s["key"], rng, form) for _ in range(n_per)]
            elif s["holes_required"] in (["t"], ["A"]):
                h = s["holes_required"][0]
                fills = [{h: rng.choice(P.pool("topic" if h == "t" else "assistant_name").values)}
                         for _ in range(n_per)]
            else:
                fills = []
            calls.append({"call_id": f"{bank}.{i}", "bank": bank, "class": s["class"], "teacher": teacher,
                          "n": n_per, "form": form, "fills": fills, "seed": int(store.sha256_text(f"{bank}:{i}")[:8], 16),
                          "verbalized": s["verbalized"], "ready": ready(bank, s)})
    return calls


def pool_calls(seeds=None):
    """seeds: the word list's seed nouns (wordload.seed_nouns); the FAKE req_noun pool when None."""
    calls, seeds = [], list(seeds or P.pool("req_noun").values)
    for vt, target in sorted(POOL_TARGETS.items()):
        n_calls = math.ceil(target * POOL_OVERGEN / POOL_PER_CALL)
        for i, teacher in enumerate(_rotation("pool." + vt, n_calls)):
            rng = P.seeded("bankpass-plan", "pool", vt, i)
            calls.append({"call_id": f"pool.{vt}.{i}", "bank": "pool." + vt, "class": "P", "teacher": teacher,
                          "n": POOL_PER_CALL, "seed_words": rng.sample(seeds, 3), "ready": True,
                          "seed": int(store.sha256_text(f"pool.{vt}:{i}")[:8], 16)})
    n_calls = math.ceil(TOPICS_STAGE_P * POOL_OVERGEN / TOPICS_PER_CALL)
    for i, teacher in enumerate(_rotation("topic", n_calls)):
        rng = P.seeded("bankpass-plan", "topic", i)
        calls.append({"call_id": f"topic.{i}", "bank": "topic", "class": "T", "teacher": teacher,
                      "n": TOPICS_PER_CALL, "seed_words": rng.sample(seeds, 3), "ready": True,
                      "seed": int(store.sha256_text(f"topic:{i}")[:8], 16)})
    return calls


def estimate(calls):
    """{teacher: {calls, lines, out_tokens, minutes, holds}} at the measured rates (estimates, s6)."""
    out = {}
    for c in calls:
        e = out.setdefault(c["teacher"], {"calls": 0, "lines": 0})
        e["calls"] += 1
        e["lines"] += c["n"]
    for t, e in out.items():
        e["out_tokens"] = e["lines"] * TOK_PER_LINE
        e["minutes"] = round(e["out_tokens"] / TOK_PER_S[t] / 60, 1)
        e["holds"] = max(1, math.ceil(e["minutes"] / HOLD_MIN))
    return out


def build(n_rates=1000, wordlist=None):
    rates = measure_rates(n_rates)
    tgt = targets(rates)
    calls = line_calls(tgt) + pool_calls(wordlist.seed_nouns() if wordlist else None)
    body = json.dumps(calls, sort_keys=True)
    return {"version": "bp-plan-v0", "rates": rates, "targets": tgt, "calls": calls,
            "seed_ref": wordlist.ref if wordlist else P.pool("req_noun").ref,
            "sha256": store.sha256_text(body), "estimate": estimate(calls)}
