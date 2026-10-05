"""Call records -> item records (BANKPASS s2a, s2c, s2d, s2h): each parsed line is templatized with the fill its call
gave, then the item checks and the held-out gates run; every line becomes a record, kept or dropped with its first
code, all hits listed. Judges, dedup and the author thirds come after (judge.py, trim.py)."""
import heldout
from bankpass import gates, specs, store, templatize as T

ROLE_HOLES = {"plant": ("v", "o"), "corr": ("v", "old", "o"), "query": ("o",), "bait": ("o",), "twin": ("v", "o")}


def fill_for(call, i):
    """the {hole: value} a line was written with, restricted to what its role may say."""
    if call["class"] != "K":
        return dict((call.get("fills") or [{}] * call["n"])[i] or {})
    f = call["fills"][i]
    role = call["bank"].split(".")[2]
    out = {h: f[h] for h in ROLE_HOLES[role] if f.get(h) is not None}
    if role in ("plant", "corr", "twin") and f.get("article"):   # W3 fix: job corrections and twins hold {av} too
        out["article"] = f["article"]
    return out


def gate_block(hits):
    return {"gate_hash": heldout.gate_hash(), "rc12": heldout.RC12_STATUS, "hits": [list(h) for h in hits]}


def line_items(call, plan_call, prompt_text=None, mined=(), n_fills=gates.N_FILLS):
    """item records for one line-bank call record."""
    spec = specs.line_specs()[call["bank"]]
    role = spec.get("role")
    base = {"prompt_id": call["prompt_id"], "prompt_sha256": call["prompt_sha256"], "call_id": call["call_id"],
            "decode_sha256": call["decode_sha256"], "seed": call["seed"], "raw_sha256": call["raw_sha256"],
            "sampling": call.get("sampling"), "time": call["time"]}
    out = []
    if call["problem"]:
        return out
    for i, line in enumerate(call["lines"]):
        fill = fill_for(plan_call, i)
        n = f"{call['call_id'].rsplit('.', 1)[1]}_{i}"
        tpl, why = T.templatize(line, fill, form=plan_call.get("form") if role in ("query", "corr") else None,
                                correction=role == "corr")
        feats = {"likelihood": call["likelihoods"][i], "form": plan_call.get("form"), "role": role}
        if why == T.DROP_MISSING and call["bank"] == "open.topic":
            why = "TPL_TOPIC_SPECIFIC"      # s3 O(2): an opening that paraphrases its topic is keyed to the topic id
        if why:
            out.append(store.make_item(spec["class"], call["bank"], n, line, call["author"], "dropped", why,
                                       holes=[], features=feats, call=base, seeds={"fill": fill}))
            continue
        hits = gates.item_checks(tpl, call["bank"], spec, prompt_text, mined)
        hits += gates.heldout_checks(tpl, call["bank"], spec, n_fills=n_fills)
        out.append(store.make_item(spec["class"], call["bank"], n, tpl, call["author"],
                                   "dropped" if hits else "kept", hits[0][0] if hits else None, features=feats,
                                   call=base, seeds={"fill": fill}, gates=gate_block(hits)))
    return out


def value_items(call, plan_call):
    """item records for one pool or topic call: values, no holes; pool_ok, the form checks and the E004 echo."""
    vtype = call["bank"].split(".", 1)[1] if call["class"] == "P" else "topic"
    base = {"prompt_id": call["prompt_id"], "prompt_sha256": call["prompt_sha256"], "call_id": call["call_id"],
            "seed": call["seed"], "raw_sha256": call["raw_sha256"], "time": call["time"]}
    out = []
    if call["problem"]:
        return out
    for i, v in enumerate(call["lines"]):
        val = store.straight(v)
        if vtype not in ("name", "surname", "city", "assistant_name", "pet_name"):
            val = val[:1].lower() + val[1:] if not val[:2].isupper() else val
        hits = gates.pool_checks(val, vtype)
        if vtype == "topic":       # W3 fix: topics are 4 to 8 word phrases (s3 T), not 1 to 4 word pool values
            lo, hi = specs.pool_specs()["topic"]["min_w"], specs.pool_specs()["topic"]["max_w"]
            hits = [h for h in hits if h[0] != "LEN_ITEM"] + ([] if lo <= len(val.split()) <= hi
                                                              else [("LEN_ITEM", val)])
            hits += [("HELDOUT_ECHO", e) for e in heldout.echo_hits(val)]
        n = f"{call['call_id'].rsplit('.', 1)[1]}_{i}"
        out.append(store.make_item(call["class"], call["bank"], n, val, call["author"], "dropped" if hits else "kept",
                                   hits[0][0] if hits else None, holes=[],
                                   features={"likelihood": call["likelihoods"][i], "vtype": vtype}, call=base,
                                   seeds={"seed_words": plan_call.get("seed_words")}, gates=gate_block(hits)))
    return out


def items_for(call, plan_call, **kw):
    return value_items(call, plan_call) if call["class"] in ("P", "T") else line_items(call, plan_call, **kw)
