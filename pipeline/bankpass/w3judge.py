"""W3 judge calls (BANKPASS s2e, s3): what one teacher still has to judge, built from the stage view.
  judge   every gated item of a judged bank that another teacher wrote: line banks on a FRESH fill (judge3), topic
          openings, hobby / plan / object pool values, list names, entity attributes and predicates. Both non-author
          teachers judge every such item (no pruning after a first drop, so the per author-pair drop rates compare).
  group   a topic's coverage group (metadata only), by the next teacher in hold order (q -> m -> g -> q)
  check   an instruction paraphrase against the base block's checklist, by the next teacher in hold order"""
import banks as B
from bankpass import judge3 as J3, plan, prompts3 as P3, specs, store

SHORT, ORDER = store.SHORT, plan.TEACHER_ORDER
NEXT = {ORDER[i]: ORDER[(i + 1) % 3] for i in range(3)}
TYPE_NP = {"weekday": "a day of the week", "city": "a city", "month": "a month", "colour": "a colour"}


def _cid(s):
    return int(store.sha256_text(s)[:8], 16)


def judged_by(it, teacher):
    return any(j["model"] == teacher for j in it.get("judges") or [])


def _ser(qs):
    return [[q, t, e, list(k) if isinstance(k, tuple) else k] for q, t, e, k in qs]


def judge_input(it, teacher, its):
    """(text, qs, fill) the judge reads for one item."""
    bank = it["bank"]
    if bank == "open.topic_spec":
        f = {"t": it["features"]["topic"]}
        return it["text"], J3.line_questions("open.topic", f), f
    if bank in specs.line_specs():
        f = J3.fresh_fill(it, teacher)
        return J3.fill_text(it["text"], f), J3.line_questions(bank, f), f
    if bank.startswith("pool."):
        d = {"vtype": bank.split(".", 1)[1], "value": it["text"]}
        return it["text"], J3.value_questions("pool", d), d
    if bank.startswith("listname."):
        d = {"value": it["text"], "desc": "a list of " + P3.LIST_KIND[bank.split(".", 1)[1]][1]}
        return it["text"], J3.value_questions("listname", d), d
    if bank == "attr":
        d = {"attr": it["text"], "kind": it["features"]["kind"], "type_np": TYPE_NP[it["features"]["vtype"]]}
        return f"{d['kind']}: {d['attr']}", J3.value_questions("attr", d), d
    if bank == "pred":
        f = it["features"]
        attr = its[f["attr_id"]]["text"] if f["attr_id"] in its else f["attr"]
        d = {"name": f["probe_name"], "pred": B.fill(it["text"], v=f["probe_value"]), "attr": attr,
             "value": f["probe_value"]}
        return f"the {d['name']} {d['pred']}", J3.value_questions("pred", d), d
    raise KeyError(bank)


def judge_calls(teacher, its, selected_ids=None):
    out = []
    for iid, it in its.items():
        if it["status"] != "kept" or it["author"]["model"] == teacher or judged_by(it, teacher):
            continue
        b = it["bank"]
        if not (b in specs.line_specs() or b in ("open.topic_spec", "attr", "pred") or b.startswith("listname.")
                or b in ("pool.hobby", "pool.plan", "pool.object")):
            continue
        if b == "open.topic_spec" and selected_ids is not None and it["features"]["topic_id"] not in selected_ids:
            continue
        text, qs, fill = judge_input(it, teacher, its)
        if not qs:
            continue
        out.append({"call_id": f"j.{SHORT[teacher]}.{iid}", "kind": "judge", "bank": b, "class": "J",
                    "teacher": teacher, "item_id": iid, "text": text, "qs": _ser(qs), "fill": fill,
                    "n": len(qs), "seed": _cid(f"j:{teacher}:{iid}")})
    return out


def group_calls(teacher, chosen, recs_done):
    out = []
    for it, author in chosen:
        if NEXT[author] != teacher:
            continue
        cid = f"group.{SHORT[teacher]}.{it['id']}"
        if cid not in recs_done:
            out.append({"call_id": cid, "kind": "group", "bank": "topic", "class": "T", "teacher": teacher, "n": 1,
                        "topic": it["text"], "topic_id": it["id"], "seed": _cid(cid)})
    return out


def check_calls(teacher, its, recs_done):
    out = []
    for iid, it in its.items():
        if it["bank"] != "instr.para" or it["status"] != "kept" or NEXT[it["author"]["model"]] != teacher:
            continue
        cid = f"check.{SHORT[teacher]}.{iid}"
        if cid not in recs_done:
            out.append({"call_id": cid, "kind": "check", "bank": "instr.para", "class": "Q", "teacher": teacher,
                        "n": 18, "head": it["features"]["head"], "tail": it["features"]["tail"], "para_id": iid,
                        "seed": _cid(cid)})
    return out
