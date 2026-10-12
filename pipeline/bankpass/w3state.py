"""W3 stage state (PC CPU, outside or inside a hold): the stage directory's call records -> item records (cached per
call id) -> the judged view every later step reads. Deterministic: the same records always give the same view.

  <root>/plan.json                 the base plan (w3plan.build)
  <root>/calls/<q|m|g>.jsonl       one teacher's call records, every kind (author, derived, judge), fsync per line
  <root>/state/items.<q|m|g>.jsonl the item cache: {call_id, v, items} per itemized call record

Order of authors: TEACHER_ORDER (Qwen, Ministral, Gemma, the hold order), then id. Exact duplicates inside a bank
(store.norm_key) are dropped before judging (DUP_EXACT_PREJUDGE): only the first in that order is judged, which saves
GPU time and changes no outcome, since dedup keeps one of them anyway."""
import os

from bankpass import judge as J, judge3 as J3, plan, store, w3amend, w3items as WI

SHORT = store.SHORT
ORDER = plan.TEACHER_ORDER
JUDGED_BANK_PREFIX = ("key.", "marker.", "open.", "close.", "social.", "identity.", "recap", "return", "swap.",
                      "rule.", "list.", "lookup.", "system.", "listname.", "attr", "pred")
POOL_JUDGED = {"pool.hobby", "pool.plan", "pool.object"}


_HUMAN = {}


def install_human(path=None):
    """install the W2 human pools (BANKPASS_HUMAN, e.g. human_v0: GeoNames cities) once per process, so fills,
    fresh judge fills, the QUERY_VALUE check and the held-out fills draw the same pools the plan drew."""
    path = path or os.environ.get("BANKPASS_HUMAN")
    if path and path not in _HUMAN:
        from bankpass import load
        _HUMAN[path] = load.install(load.from_dir(path))
    return path


def calls_path(root, teacher):
    return os.path.join(root, "calls", SHORT[teacher] + ".jsonl")


def cache_path(root, teacher):
    return os.path.join(root, "state", "items." + SHORT[teacher] + ".jsonl")


def records(root, teacher=None):
    out = []
    for t in ([teacher] if teacher else ORDER):
        out += store.read_jsonl(calls_path(root, t))
    return out


def done_ids(recs, max_errors=2):
    """call ids that are finished (a record without an error), and ids that failed max_errors times."""
    ok, errs = set(), {}
    for r in recs:
        if r.get("error"):
            errs[r["call_id"]] = errs.get(r["call_id"], 0) + 1
        else:
            ok.add(r["call_id"])
    return ok | {c for c, n in errs.items() if n >= max_errors}


def refresh(root, teachers=ORDER, kinds=None):
    """itemize every author call record not yet in the cache (optionally only some kinds). Returns n new."""
    new = 0
    for t in teachers:
        cp = cache_path(root, t)
        have = {r["call_id"] for r in store.read_jsonl(cp) if r.get("v") == WI.VERSION}
        for rec in store.read_jsonl(calls_path(root, t)):
            if rec["call_id"] in have or rec.get("error") or rec["kind"] in ITEMLESS:
                continue
            if kinds and rec["kind"] not in kinds:
                continue
            store.append_jsonl(cp, {"call_id": rec["call_id"], "v": WI.VERSION, "items": WI.items_for(rec)})
            have.add(rec["call_id"])
            new += 1
    return new


ITEMLESS = {"judge", "vote", "relfeat", "check", "group", "pos", "verbs", "ly"}


def items(root):
    """{item id: item} from the cache, in author order then id."""
    out = {}
    for t in ORDER:
        for r in store.read_jsonl(cache_path(root, t)):
            if r.get("v") != WI.VERSION:
                continue
            for it in r["items"]:
                out[it["id"]] = it
    return dict(sorted(out.items(), key=lambda kv: (ORDER.index(kv[1]["author"]["model"]), kv[0])))


def needs_judges(it):
    b = it["bank"]
    return b.startswith(JUDGED_BANK_PREFIX) or b in POOL_JUDGED


def prejudge_dedup(its):
    """mark exact duplicates inside a bank DUP_EXACT_PREJUDGE (the first in author order stays)."""
    seen = {}
    for it in its.values():
        if it["status"] != "kept" or not needs_judges(it) or it["bank"].startswith(("pool.", "attr", "pred")):
            continue
        k = (it["bank"], store.norm_key(it["text"]))
        if k in seen:
            it.update(status="dropped", drop="DUP_EXACT_PREJUDGE", dedup={"cluster": seen[k]})
        else:
            seen[k] = it["id"]


def judge_blocks(recs):
    """{item id: [judge block]} from judge call records (one per judge and item; a repeat keeps the first)."""
    out = {}
    for r in recs:
        if r["kind"] != "judge" or r.get("error"):
            continue
        p = r["plan"]
        qs = [tuple(q[:3]) + ((tuple(q[3]) if isinstance(q[3], list) else q[3]),) for q in p["qs"]]
        ans = J3.parse(r.get("raw"), qs) if not r.get("problem") else {q[0]: None for q in qs}
        blk = J3.record(r["author"]["model"], ans, qs, p["fill"], {"call_id": r["call_id"],
                                                                   "prompt_sha256": r["prompt_sha256"],
                                                                   "raw_sha256": r["raw_sha256"]})
        lst = out.setdefault(p["item_id"], [])
        if all(b["model"] != blk["model"] for b in lst):
            lst.append(blk)
    return out


def view(root, recs=None):
    """the judged view: items with judges attached and a W3 status. Judged items need two non-author keeps
    (judge.vote); a teacher item still waiting for a judge has w3 "pending"."""
    recs = records(root) if recs is None else recs
    its = items(root)
    for it in its.values():      # amendments: superseded calls, and attributes / predicates built on them (first,
        src = it["features"].get("kind_id") if it["bank"] == "attr" else it["features"].get("attr_id") \
            if it["bank"] == "pred" else None       # so no superseded item is a dedup representative)
        if it["status"] == "kept" and (w3amend.superseded_item(it)
                                       or (src in its and its[src].get("drop") == "SUPERSEDED_V1")):
            it.update(status="dropped", drop="SUPERSEDED_V1")
    prejudge_dedup(its)
    jb = judge_blocks(recs)
    for iid, it in its.items():
        it["judges"] = jb.get(iid, [])
        if it["status"] != "kept":
            it["w3"] = "dropped"
        elif not needs_judges(it):
            it["w3"] = "kept"
        else:
            v = J.vote(it)
            it["w3"] = "kept" if v is None else ("pending" if v == "JUDGE_MISSING" and
                                                 all(j["verdict"] == "keep" for j in it["judges"]) else "dropped")
            if it["w3"] == "dropped":
                it["drop_w3"] = v
    return its


def by_bank(its):
    out = {}
    for it in its.values():
        out.setdefault(it["bank"], []).append(it)
    return out
