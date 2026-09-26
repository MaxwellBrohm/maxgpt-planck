"""OOD-H Part 1, step 3: merge the Claude-written probe batches into RC-12 records (oodh/DESIGN.txt s4-s6).

Loads no model. Reads sealed/oodh/candidates.jsonl (the threads) and the batch files (probes, written by hand in three
slightly different shapes), writes sealed/oodh/oodh_part1.jsonl (item content: sealed only) and oodh/HASHES.txt
(public: hashes and counts only). Every record plays in rc12/runner.py through oodh/play.FixedHistory and grades with
rc12/graders.grade_conv unchanged.

Record (the rc12/common.py format; id oodh-NNN in tree-id order, split "oodh", family "OODH", cell = the sorted
probe-kind mix, seed null, knowledge false, meta.unit "mean", plus tree_id and message_ids):
  thread user turns 1..n: text = the OASST2 user turn; ideal = history_reply = the OASST2 assistant turn after it
    (stripped); msg_id; kind C if it holds an H-CORR gold, else S if it holds a VAL gold or an H-CORR stale value,
    else L if it mentions a candidate or pool value of a VAL probe (a same-type lure: the RC-12 cheaters read every
    non-D turn, so they see it), else D with the batch's hand asks label.
  probe turns n+1..: kind X, the last P; text = the question; ideal = the IDEAL reply.
  probes: the batch fields mapped to the RC-12 names (probe_kind H-FACT | H-ASK | H-CORR | H-ABS, source_msg_id,
    abstain); src is recomputed as EVERY thread user turn mentioning the gold (the batch's src must be among them),
    d = turn - max(src), dc = turn - max(src) on H-CORR; accepted always starts with the gold.
Selection (exactly --n threads, deterministic): every usable thread is eligible; threads with an H-CORR probe first
(DESIGN s5: used wherever a thread has one), then greedily the thread that brings the pooled probe-kind shares
closest (L1) to TARGET (H-ABS 1 in 5 by DESIGN s5, H-FACT and H-ASK equal), ties by sha256(PICK_SALT + tree id).
Fewer eligible than --n: nothing is chosen; exit 3, or with --allow-short all are written and HASHES.txt says SHORT.
Run: python3 -B oodh/build.py [--allow-short]"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
import select  # noqa: E402,F401  the stdlib module, cached first: oodh/ holds a select.py of its own
REPO = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(REPO, "rc12"), HERE]
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from collections import Counter  # noqa: E402

import grade_text as T  # noqa: E402
import pools_vals as V  # noqa: E402
import hashes as H  # noqa: E402

SEALED = os.path.join(REPO, "sealed", "oodh")
BATCHES = ["batch_1.jsonl", "batch_2.jsonl", "batch_3.jsonl"]
ASKS_FILES = {"batch_1.jsonl": "batch_1_asks.json"}   # hand asks labels (tree id -> per turn) where records have none
TARGET = {"H-FACT": 0.40, "H-ASK": 0.40, "H-ABS": 0.20}
PICK_SALT = "planck-oodh-pick-v1:"
N = 150


class BuildError(Exception):
    pass


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def mentions(text, v):
    return bool(v) and T.mentioned(T.norm(text), v)


def norm_probe(p, message_ids):
    """one batch probe in the RC-12 probe shape (batch 1: kind H-*, turn_kind X|P, text; batches 2, 3: kind X|P,
    probe_kind H-*, question)."""
    hk = p["kind"].startswith("H-")
    pk, kind = (p["kind"], p["turn_kind"]) if hk else (p["probe_kind"], p["kind"])
    smid = p.get("source_msg_id") or p.get("source_message_id")
    if smid is None and p.get("source_msg") is not None:
        smid = message_ids[p["source_msg"]]
    gold = p.get("gold")
    acc = list(dict.fromkeys(([gold] if gold else []) + list(p.get("accepted") or [])))
    return dict(turn=p["turn"], kind=kind, grader=p["grader"], question=p.get("question") or p["text"], gold=gold,
                gold_fn=None, accepted=acc, candidates=list(p.get("candidates") or []), pool=None,
                pool_values=list(p.get("pool_values") or []), stale=list(p.get("stale") or []), holder=p["holder"],
                object_words=list(p.get("object_words") or []), pair_id=None, d=None, dc=None,
                src=list(p.get("src") or []), prefix=p.get("prefix"), ideal=p["ideal"], probe_kind=pk,
                source_msg_id=smid, abstain=p["grader"] == "ABS")


def load_items(sealed=SEALED, batches=BATCHES, asks_files=ASKS_FILES):
    """every usable thread of every batch, joined with its thread text from candidates.jsonl."""
    cands = read_jsonl(os.path.join(sealed, "candidates.jsonl"))
    items, seen = [], set()
    for name in batches:
        fn = os.path.join(sealed, asks_files.get(name, "-"))
        extra = json.load(open(fn)) if os.path.exists(fn) else {}
        for r in read_jsonl(os.path.join(sealed, name)):
            ci = r["cand_index"] if "cand_index" in r else r["meta"]["cand_index"]
            c = cands[ci]
            tid = r["tree_id"]
            if c["tree_id"] != tid or c["message_ids"] != r["message_ids"]:
                raise BuildError(f"{name}: cand_index {ci} does not match tree {tid[:8]} / its message ids")
            if tid in seen:
                raise BuildError(f"tree {tid[:8]} appears in two batches")
            seen.add(tid)
            roles = [t["role"] for t in c["turns"]]
            if roles != ["user", "assistant"] * (len(roles) // 2) or not roles:
                raise BuildError(f"tree {tid[:8]}: thread does not alternate user/assistant ending on assistant")
            users = [t["text"] for t in c["turns"][0::2]]
            replies = [t["text"].strip() for t in c["turns"][1::2]]
            if "turns" in r:
                asks = [t["asks"] for t in r["turns"][:len(users)]]
                if [t["text"] for t in r["turns"][:len(users)]] != users:
                    raise BuildError(f"tree {tid[:8]}: the batch's user turns differ from candidates.jsonl")
            elif "turn_asks" in r:
                asks = list(r["turn_asks"])
            else:
                asks = extra.get(tid)
            probes = [norm_probe(p, r["message_ids"]) for p in r["probes"]]
            items.append(dict(tree_id=tid, message_ids=r["message_ids"], cand_index=ci, batch=name,
                              path_rule=r.get("path_rule", c["path_rule"]), users=users, replies=replies,
                              asks=asks, probes=probes))
    return items


def kinds_of(item):
    return [p["probe_kind"] for p in item["probes"]]


def pick_key(item):
    return hashlib.sha256((PICK_SALT + item["tree_id"]).encode()).hexdigest()


def distance(counts):
    tot = sum(counts[k] for k in TARGET)
    return sum(abs(counts[k] / tot - share) for k, share in TARGET.items()) if tot else 0.0


def select(items, n=N):
    """exactly n threads by the rule of the module docstring, or None when fewer than n are eligible."""
    if len(items) < n:
        return None
    pool = sorted(items, key=pick_key)
    chosen = [it for it in pool if "H-CORR" in kinds_of(it)][:n]
    rest = [it for it in pool if "H-CORR" not in kinds_of(it)]
    counts = Counter(k for it in chosen for k in kinds_of(it))
    while len(chosen) < n:
        best = min(rest, key=lambda it: distance(counts + Counter(kinds_of(it))))   # first minimum: pool order
        rest.remove(best)
        chosen.append(best)
        counts += Counter(kinds_of(best))
    return chosen


def record(item, rid):
    users, n = item["users"], len(item["users"])
    probes = [dict(p) for p in item["probes"]]
    if [p["turn"] for p in probes] != list(range(n + 1, n + 1 + len(probes))):
        raise BuildError(f"{rid}: probe turns are not {n + 1}..")
    if [p["kind"] for p in probes] != ["X"] * (len(probes) - 1) + ["P"]:
        raise BuildError(f"{rid}: probe kinds are not X.. P")
    kind, facts = {}, {i: [] for i in range(1, n + 1)}
    for p in probes:
        if p["grader"] != "VAL":
            continue
        hit = [i for i in range(1, n + 1) if mentions(users[i - 1], p["gold"])]
        if not hit or not set(p["src"]) <= set(hit):
            raise BuildError(f"{rid} u{p['turn']}: gold not in its source user turn (G8)")
        p["src"], p["d"] = hit, p["turn"] - max(hit)
        obj = " ".join(p["object_words"])
        corr = p["probe_kind"] == "H-CORR"
        for i in hit:
            kind[i] = "C" if corr or kind.get(i) == "C" else "S"
            facts[i].append(dict(holder=p["holder"], object=obj, value=p["gold"], role="corr" if corr else "gold"))
        if corr:
            p["dc"] = p["d"]
            for s in p["stale"]:
                for i in range(1, max(hit) + 1):
                    if mentions(users[i - 1], s):
                        kind.setdefault(i, "S")
                        facts[i].append(dict(holder=p["holder"], object=obj, value=s, role="stale"))
    lure = {v for p in probes if p["grader"] == "VAL" for v in p["candidates"] + p["pool_values"]}
    turns = []
    for i in range(1, n + 1):
        k = kind.get(i) or ("L" if any(mentions(users[i - 1], v) for v in lure) else "D")
        t = dict(i=i, kind=k, text=users[i - 1], ideal=item["replies"][i - 1], facts=facts[i],
                 vals=V.scan(users[i - 1], V.ALL_VALUES), history_reply=item["replies"][i - 1],
                 msg_id=item["message_ids"][2 * (i - 1)])
        if k == "D":
            if not item["asks"] or type(item["asks"][i - 1]) is not bool:
                raise BuildError(f"{rid} (tree {item['tree_id'][:8]}) u{i}: D turn without a hand asks label")
            t["asks"] = item["asks"][i - 1]
        turns.append(t)
    for p in probes:
        turns.append(dict(i=p["turn"], kind=p["kind"], text=p["question"], ideal=p["ideal"], facts=[],
                          vals=V.scan(p["question"], V.ALL_VALUES)))
    cell = "+".join(sorted(set(kinds_of(item))))
    meta = dict(unit="mean", batch=item["batch"], cand_index=item["cand_index"], n_thread_user_turns=n,
                path_rule=item["path_rule"])
    return dict(id=rid, split="oodh", family="OODH", cell=cell, seed=None, knowledge=False, n_turns=len(turns),
                turns=turns, probes=probes, meta=meta, tree_id=item["tree_id"], message_ids=item["message_ids"])


def build(items, n=N, allow_short=False):
    chosen = select(items, n)
    short = chosen is None
    if short and not allow_short:
        raise BuildError(f"only {len(items)} usable threads, {n} needed (rerun with --allow-short to write them all)")
    chosen = sorted(items if short else chosen, key=lambda it: it["tree_id"])
    return [record(it, f"oodh-{k:03d}") for k, it in enumerate(chosen, 1)], short


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--allow-short", action="store_true")
    ap.add_argument("--sealed", default=SEALED)
    ap.add_argument("--hashes", default=os.path.join(HERE, "HASHES.txt"))
    a = ap.parse_args()
    if "sealed" not in os.path.abspath(a.sealed).split(os.sep):
        sys.exit(f"refusing --sealed outside a sealed/ directory: {a.sealed}")
    try:
        recs, short = build(load_items(a.sealed), a.n, a.allow_short)
    except BuildError as e:
        print(f"BUILD FAILED: {e}")
        return 3
    out = os.path.join(a.sealed, "oodh_part1.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    ins = [os.path.join(a.sealed, b) for b in ["candidates.jsonl"] + BATCHES + list(ASKS_FILES.values())]
    for line in H.write_hashes(recs, short, a.n, out, a.hashes, [f for f in ins if os.path.exists(f)]):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
