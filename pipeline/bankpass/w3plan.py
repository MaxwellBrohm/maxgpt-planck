"""W3 stage P base plan (BANKPASS s1, s3, s6): every call a teacher authors without waiting for another teacher, as a
seeded, deterministic list. Line banks are overgenerated 3x their s1 targets (s6: "lines 3x overgenerated"), pools
and topics 1.5x (plan.py), and every bank rotates its author per call so each teacher writes a third. Labels: 5 key
label candidates per key and teacher, 4 note-frame words and 6 list names per list type per teacher; word labels:
part of speech for the top 8,000 ranked families and the -ly pairs, labelled by all three teachers; 150 instruction
paraphrases (100 kept after the checklist). Calls that depend on another call's output (features, topic-dependent
calls, votes, judges) are derived later from the stage records (w3deps.py, w3judge.py)."""
import json
import math

import banks_keys as BK
import pools as P
from bankpass import plan, store, w3fill

PLAN_VERSION = "bp-plan-v1"
LINE_OVERGEN = 3
POS_TOP, POS_BATCH, LY_BATCH = 8000, 50, 40
LABEL_N, NOTEWAS_N, LISTNAME_N = 5, 4, 6
PARA_CALLS = 150
SHORT = store.SHORT
# call order inside a teacher's authoring phase: the kinds other calls wait on go first
KIND_ORDER = ("topic", "pool", "pos", "label", "notewas", "listname", "line", "ly", "para")


def line_calls(tgt, inputs):
    calls = plan.line_calls({b: t * LINE_OVERGEN for b, t in tgt.items()})
    for c in calls:
        if c["class"] != "K":
            c["fills"] = w3fill.fills(c["bank"], c["call_id"], c["n"])
        c.update(kind="line", ready=True, voice=inputs.voice(c["call_id"], c["bank"]),
                 mined=inputs.example(c["call_id"], c["bank"]))
    return calls


def value_calls(inputs, seeds):
    calls = plan.pool_calls(seeds)
    for c in calls:
        c.update(kind="topic" if c["class"] == "T" else "pool", voice=inputs.voice(c["call_id"], c["bank"]))
    return calls


def _o_for(key):
    noun = BK.KEYS[key]["noun"]
    if not noun:
        return None
    vals = P.pool(noun).values
    if noun == "relation":
        vals = [v for v in vals if BK.pronouns(key, v)[0]]
    return P.seeded("bankpass-label-o", key).choice(vals)


def label_calls():
    out = []
    for key in sorted(BK.KEYS):
        o = _o_for(key)
        for t in plan.TEACHER_ORDER:
            out.append({"call_id": f"label.{key}.{SHORT[t]}", "kind": "label", "bank": f"label.{key}", "class": "N",
                        "teacher": t, "n": LABEL_N, "key": key, "o": o,
                        "seed": int(store.sha256_text(f"label:{key}:{t}")[:8], 16)})
    for t in plan.TEACHER_ORDER:
        out.append({"call_id": f"notewas.{SHORT[t]}", "kind": "notewas", "bank": "label.notewas", "class": "N",
                    "teacher": t, "n": NOTEWAS_N, "seed": int(store.sha256_text(f"notewas:{t}")[:8], 16)})
        for vt in sorted(w3fill.B.LIST_NAMES):
            out.append({"call_id": f"listname.{vt}.{SHORT[t]}", "kind": "listname", "bank": f"listname.{vt}",
                        "class": "N", "teacher": t, "n": LISTNAME_N, "vtype": vt,
                        "seed": int(store.sha256_text(f"listname:{vt}:{t}")[:8], 16)})
    return out


def ly_pairs(wl, top=POS_TOP):
    """(adjective, adverb) pairs: a ranked word ending in -ly whose base (-y to -ily, -le to -ly, -ic to -ically)
    is a ranked family too; the base is never a function word (the word list flags those func)."""
    heads = set(wl.ranked("RL")[:top])
    out = []
    for w in sorted(heads):
        if not w.endswith("ly") or len(w) < 5:
            continue
        for base in (w[:-2], w[:-3] + "y" if w.endswith("ily") else None, w[:-2] + "le" if w.endswith("ly") else None,
                     w[:-4] if w.endswith("ically") else None):
            if base and base in heads and wl.families[base]["pos"] not in ("func",):
                out.append((base, w))
                break
    return out


def word_calls(wl):
    words = wl.ranked("RL")[:POS_TOP]
    pairs = ly_pairs(wl)
    out = []
    for t in plan.TEACHER_ORDER:
        for i in range(math.ceil(len(words) / POS_BATCH)):
            rows = words[i * POS_BATCH:(i + 1) * POS_BATCH]
            out.append({"call_id": f"pos.{SHORT[t]}.{i}", "kind": "pos", "bank": "wordlabel.pos", "class": "V",
                        "teacher": t, "n": len(rows), "rows": rows, "seed": i})
        for i in range(math.ceil(len(pairs) / LY_BATCH)):
            rows = [list(p) for p in pairs[i * LY_BATCH:(i + 1) * LY_BATCH]]
            out.append({"call_id": f"ly.{SHORT[t]}.{i}", "kind": "ly", "bank": "wordlabel.ly", "class": "V",
                        "teacher": t, "n": len(rows), "rows": rows, "seed": i})
    return out


def para_calls():
    return [{"call_id": f"para.{i}", "kind": "para", "bank": "instr.para", "class": "Q", "teacher": t, "n": 2,
             "seed": int(store.sha256_text(f"para:{i}")[:8], 16)}
            for i, t in enumerate(plan._rotation("instr.para", PARA_CALLS))]


def build(inputs, wl, n_rates=1000):
    """the base plan; wl: a loaded word list (wordload.read), inputs: w3fill.Inputs."""
    rates = plan.measure_rates(n_rates)
    tgt = plan.targets(rates)
    calls = (line_calls(tgt, inputs) + value_calls(inputs, wl.seed_nouns()) + label_calls() + word_calls(wl)
             + para_calls())
    ids = [c["call_id"] for c in calls]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate call ids in the plan")
    body = json.dumps(calls, sort_keys=True)
    return {"version": PLAN_VERSION, "rates": rates, "targets": tgt, "line_overgen": LINE_OVERGEN,
            "pool_targets": plan.POOL_TARGETS, "topics_stage_p": plan.TOPICS_STAGE_P, "calls": calls,
            "inputs": {"persona_sha256": inputs.persona_sha, "mined_sha256": inputs.mined_sha,
                       "wordlist": wl.ref, "pools": {vt: P.pool(vt).ref for vt in sorted(P.POOLS)}},
            "sha256": store.sha256_text(body)}


def ordered(calls, teacher):
    """one teacher's base calls in KIND_ORDER, plan order inside a kind."""
    mine = [c for c in calls if c["teacher"] == teacher]
    return sorted(mine, key=lambda c: KIND_ORDER.index(c["kind"]))


def main(argv=None):
    """PC CPU: install the W2 human pools (GeoNames cities), load the persona seeds, mined examples and word list,
    build the base plan and write <stage>/plan.json (refuses to overwrite a different plan: a restart must reuse it)."""
    import argparse
    import os
    from bankpass import load, wordload
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--bankpass-root", required=True)
    ap.add_argument("--human", required=True, help="the W2 human bank dir (human_v0)")
    ap.add_argument("--wordlist", required=True)
    a = ap.parse_args(argv)
    undo = load.install(load.from_dir(a.human))
    try:
        p = build(w3fill.Inputs.from_root(a.bankpass_root), wordload.read(a.wordlist))
    finally:
        undo()
    p["human_bankset"] = os.path.abspath(a.human)
    out = os.path.join(a.stage, "plan.json")
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            old = json.load(f)
        if old["sha256"] != p["sha256"]:
            raise SystemExit(f"{out} exists with another plan ({old['sha256'][:12]} vs {p['sha256'][:12]})")
        print(json.dumps({"same_plan": p["sha256"]}))
        return 0
    os.makedirs(a.stage, exist_ok=True)
    with open(out + ".tmp", "w", encoding="utf-8") as f:
        json.dump(p, f, sort_keys=True)
    os.replace(out + ".tmp", out)
    est = {}
    for c in p["calls"]:
        e = est.setdefault(c["teacher"], {})
        e[c["kind"]] = e.get(c["kind"], 0) + 1
    print(json.dumps({"plan": p["sha256"], "calls": len(p["calls"]), "by_teacher_kind": est,
                      "targets_total": sum(p["targets"].values())}))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
