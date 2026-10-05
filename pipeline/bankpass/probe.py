"""W3 bank probe (BANKPASS s6: "60 skeletons per teacher on the stage P banks, dry, own dir, not pilot data").
DRY: nothing here is training data or pilot data; outputs go under <stage>/probe/dry/<teacher>/.

  build   (PC CPU, after the last judge hold) a provisional bank set from the W3 view: every line bank's kept items
          (gates and both judges), the teacher pools, relation sex and number by 2 of 3 labels, entity kinds with
          their kept typed attributes, key labels by vote, the 1,000 selected topics with the union of the three
          teachers' word sets; human_v0 cities; FAKE where stage P has nothing yet (first names: SSA is blocked;
          avoid words: computed at W4; lookup predicates per type, list names: W4 picks). Saved as bankset.json,
          then 60 skeletons built on it (shard bankprobe-1004, RM), each carrying the stage P intents of its topics.
  run     (inside a hold, against bankserve) teachers/drive.py on those 60 skeletons, one attempt each, with the
          bank set installed in the driver process and topic guidance taken from the stage P intents (forms 0 to 3
          and 5; form 4 "react briefly" and digressions keep their program wording)."""
import argparse
import collections
import json
import os
import sys

import banks as B
import banks_keys as BK
import fake_data as F
import pools as P
import topic_words as TW
from bankpass import load, store, w3deps as WD, w3state as WS

PRESET = {"qwen3.5-9b": "card", "ministral-3-8b": "shared", "gemma-4-12b": "card"}
CONC = {"qwen3.5-9b": 32, "ministral-3-8b": 16, "gemma-4-12b": 4}
TEACHER_POOLS = ("job", "hobby", "food", "grocery", "chore", "plan", "object", "pet_kind", "pet_name",
                 "assistant_name")
SHARD, N = "bankprobe-1004", 60


def _kept(its, bank):
    return [i for i in its.values() if i["bank"] == bank and i.get("w3") == "kept"]


def _majority(labels):
    c = collections.Counter(labels)
    v, n = c.most_common(1)[0] if c else (None, 0)
    return v if n >= 2 else None


def snapshot(root, wl):
    recs = WS.records(root)
    its = WS.view(root, recs)
    with open(os.path.join(root, "plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    snap = {"banks": {}, "keys": {}, "markers": {}, "pools": {}, "female": [], "male": [], "entity": {},
            "attr_types": {}, "topic_words": {}, "intents": {}, "labels": {}}
    for bank in B.BANKS:
        snap["banks"][bank] = [i["text"] for i in _kept(its, bank)]
    for m in ("marker.fix", "marker.err"):
        snap["markers"][m] = [i["text"].rstrip() + " " for i in _kept(its, m)]
    for key in BK.KEYS:
        snap["keys"][key] = {}
        for role in ("plant", "query", "bait", "corr", "twin"):
            rows = _kept(its, f"key.{key}.{role}")
            snap["keys"][key][role] = ([[i["features"]["form"], i["text"]] for i in rows] if role in ("query", "corr")
                                       else [i["text"] for i in rows])
    for vt in TEACHER_POOLS:
        snap["pools"][vt] = sorted({i["text"] for i in _kept(its, "pool." + vt)})
    rel = collections.defaultdict(lambda: {"sex": [], "num": []})
    for r in recs:
        if r["kind"] == "relfeat" and not r.get("problem") and not r.get("error"):
            for ln in r["lines"]:
                v, _, rest = ln.rpartition(": ")
                sex, _, num = rest.partition(", ")
                rel[v]["sex"].append(sex)
                rel[v]["num"].append(num)
    rels = []
    for i in _kept(its, "pool.relation"):
        sex, num = _majority(rel[i["text"]]["sex"]), _majority(rel[i["text"]]["num"])
        if num == "one":
            rels.append(i["text"])
            if sex in ("woman", "man"):
                snap["female" if sex == "woman" else "male"].append(i["text"])
    snap["pools"]["relation"] = sorted(set(rels))
    attrs = collections.defaultdict(list)
    for a in _kept(its, "attr"):
        attrs[a["features"]["kind"]].append((a["text"], a["features"]["vtype"]))
    for kind, rows in sorted(attrs.items()):
        ok = [(lab, vt) for lab, vt in rows if snap["attr_types"].get(lab, vt) == vt]
        if ok:
            snap["entity"][kind] = [lab for lab, _ in ok]
            snap["attr_types"].update(dict(ok))
    snap["pools"]["entity_kind"] = sorted(snap["entity"])
    from bankpass import w3amend
    chosen = WD.select_topics(its, recs, wl, w3amend.base_calls(plan))
    snap["pools"]["topic"] = [it["text"] for it, _ in chosen]
    sets = collections.defaultdict(lambda: ([], [], []))
    for w in _kept(its, "topicwords.raw"):
        for k, part in enumerate(("nouns", "verbs", "adjs")):
            sets[w["text"]][k].extend(x for x in w["features"][part] if x not in sets[w["text"]][k])
    snap["topic_words"] = {t: [" ".join(p) for p in sets[t]] for t in snap["pools"]["topic"] if t in sets}
    for i in _kept(its, "intent"):
        snap["intents"].setdefault(i["features"]["topic"], []).append(i["text"])
    for r in recs:
        if r["kind"] == "vote" and not r.get("problem") and r["lines"]:
            snap["labels"].setdefault(r["bank"], []).append(r["plan"]["cand_ids"][int(r["lines"][0]) - 1]
                                                           if r["lines"][0].isdigit() and 0 < int(r["lines"][0]) <=
                                                           len(r["plan"]["cand_ids"]) else None)
    for bank, votes in snap["labels"].items():
        win = _majority([v for v in votes if v])
        if win and bank.startswith("label.") and bank != "label.notewas":
            snap["keys"][bank.split(".", 1)[1]]["label"] = its[win]["text"]
    return snap


def install(snap):
    """the snapshot into the running modules (in place, like load.install); FAKE stays wherever it is empty."""
    bs = load.BankSet()
    for bank, texts in snap["banks"].items():
        if texts:
            bs.banks[bank] = texts
    if snap["markers"].get("marker.fix"):
        bs.markers = snap["markers"]["marker.fix"]
    if snap["markers"].get("marker.err"):
        bs.err_markers = snap["markers"]["marker.err"]
    for key, roles in snap["keys"].items():
        for role, rows in roles.items():
            if rows:
                bs.keys[key][role] = [tuple(r) for r in rows] if role in ("query", "corr") else rows
    for vt, vals in snap["pools"].items():
        if vals:
            bs.pools[vt] = (vals, "STAGEP_PROBE", "bank pass stage P (provisional)", "apache-2.0")
    bs.topic_words.update({t: tuple(v) for t, v in snap["topic_words"].items()})
    undo = load.install(bs)
    old = (set(BK.FEMALE), set(BK.MALE), dict(F.ENTITY_KINDS), dict(F.ATTR_TYPES))
    if snap["pools"].get("relation"):
        BK.FEMALE.clear(), BK.FEMALE.update(snap.get("female", []))
        BK.MALE.clear(), BK.MALE.update(snap.get("male", []))
    if snap["entity"]:
        F.ENTITY_KINDS.clear(), F.ENTITY_KINDS.update(snap["entity"])
        F.ATTR_TYPES.update(snap["attr_types"])

    def restore():
        undo()
        BK.FEMALE.clear(), BK.FEMALE.update(old[0])
        BK.MALE.clear(), BK.MALE.update(old[1])
        F.ENTITY_KINDS.clear(), F.ENTITY_KINDS.update(old[2])
        F.ATTR_TYPES.clear(), F.ATTR_TYPES.update(old[3])
    return restore


def patch_guidance(intents):
    import render_intents as RI
    orig = RI.user_guidance

    def user_guidance(skel, t):
        it = t.get("intent") or ""
        if it.startswith("topic:"):
            _, tid, form = it.split(":")
            text = skel["topic_text"].get(tid)
            opts = intents.get(text) or []
            if form not in ("digress", "4") and opts:
                k = int(store.sha256_text(f"{skel['skel_id']}:{t['i']}")[:8], 16) % len(opts)
                return opts[k]
        return orig(skel, t)
    RI.user_guidance = user_guidance
    return orig


def build(root, wl):
    import skeleton_shard as SS
    snap = snapshot(root, wl)
    d = os.path.join(root, "probe")
    os.makedirs(d, exist_ok=True)
    restore = install(snap)
    try:
        sks = SS.shard(SHARD, N, "RM")
    finally:
        restore()
    with open(os.path.join(d, "bankset.json"), "w", encoding="utf-8") as f:
        json.dump(snap, f, sort_keys=True)
    with open(os.path.join(d, "skels60.jsonl"), "w", encoding="utf-8") as f:
        for sk in sks:
            f.write(json.dumps(sk, sort_keys=True) + "\n")
    counts = {k: (len(v) if not isinstance(v, dict) else sum(bool(x) for x in v.values()))
              for k, v in snap.items()}
    return {"skeletons": len(sks), "bankset_sha256": store.sha256_file(os.path.join(d, "bankset.json")),
            "skels_sha256": store.sha256_file(os.path.join(d, "skels60.jsonl")), "counts": counts}


def run(root, teacher, port):
    d = os.path.join(root, "probe")
    with open(os.path.join(d, "bankset.json"), encoding="utf-8") as f:
        snap = json.load(f)
    install(snap)
    patch_guidance(snap["intents"])
    out = os.path.join(d, "dry", teacher)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "DRY.txt"), "w") as f:
        f.write("dry: bank pass W3 probe renders on provisional stage P banks (2026-10-04); not training or pilot "
                "data\n")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "teachers"))
    import drive
    return drive.main(["--serve", "--allow-real-teacher", "--endpoint", f"http://127.0.0.1:{port}", "--out",
                       os.path.join(out, "run"), "--skeletons", os.path.join(d, "skels60.jsonl"), "--n", str(N),
                       "--max-attempts", "1", "--structured", "labels_exact", "--ban-dashes", "--preset",
                       PRESET[teacher], "--concurrency", str(CONC[teacher]), "--timeout", "1800", "--http-tries", "3",
                       "--log-every", "20"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--wordlist", default=os.environ.get("BANKPASS_WORDLIST"))
    a = ap.parse_args(argv)
    WS.install_human()
    if a.cmd == "build":
        from bankpass import wordload
        print(json.dumps(build(a.root, wordload.read(a.wordlist))))
        return 0
    return run(a.root, a.teacher, a.port)


if __name__ == "__main__":
    sys.exit(main())
