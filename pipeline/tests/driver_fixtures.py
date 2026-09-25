"""Fixtures for the step 4 driver tests: a skeleton list with planted duplicates and an infeasible skeleton, the stub
plan for it, and verify(), which compares a run directory with the plan outcome by outcome."""
import collections
import copy
import json
import os
import re
import sys

sys.dont_write_bytecode = True
PIPE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PIPE not in sys.path:
    sys.path.insert(0, PIPE)

import checker  # noqa: E402
import fake_teacher  # noqa: E402
import pools  # noqa: E402
import render_prompt as R  # noqa: E402
import shards  # noqa: E402
import skeleton  # noqa: E402
import stub_plan  # noqa: E402


def clone(sk, tag):
    c = copy.deepcopy(sk)
    c["skel_id"] = f"{sk['skel_id']}-{tag}"
    return c


def swap_value(sk, tag="near"):
    """a copy of sk with one planted name or city replaced everywhere by an unused pool value: a render that differs
    from sk's only in a slot value (DUP_NEAR after slot masking). Returns None if the copy does not render clean."""
    text = json.dumps(sk)
    for sid, s in sk["slots"].items():
        if s["type"] not in ("name", "city") or not s.get("counted"):
            continue
        for new in pools.pool(s["type"]).values:
            if re.search(r"\b" + re.escape(new) + r"\b", text):
                continue
            c = json.loads(re.sub(r"\b" + re.escape(s["value"]) + r"\b", new, text))
            c["skel_id"] = f"{sk['skel_id']}-{tag}"
            if not R.feasible(c) and checker.run(c, fake_teacher.raw(c))["ok"]:
                return c
            break
    return None


def make_infeasible(sk):
    c = clone(sk, "infeasible")
    v = c["slots"][c["assistant"]["name"]]["value"]
    t0 = c["topic_path"][0]
    c["topic_text"][t0] = c["topic_text"][t0] + " with " + v
    assert R.feasible(c)
    return c


def corpus(n=70, seed="e2e"):
    """-> (skeletons, groups). Base skeletons from iter_shard, then 2 exact-duplicate pairs, 1 near pair, and 1
    infeasible copy. One exact copy sits near its original (both in flight together), the other at the very end."""
    base = skeleton.shard(seed, n)
    feas = [s for s in base if not R.feasible(s)]
    near = None
    for s in feas[4:]:
        near = swap_value(s)
        if near:
            break
    groups = [("DUP_EXACT", [feas[0]["skel_id"], feas[0]["skel_id"] + "-dup"]),
              ("DUP_EXACT", [feas[1]["skel_id"], feas[1]["skel_id"] + "-dup"]),
              ("DUP_NEAR", [s["skel_id"], near["skel_id"]])]
    extra = [clone(feas[0], "dup"), near, make_infeasible(feas[2])]
    skels = list(base)
    for k, e in enumerate(extra):
        skels.insert(3 + 7 * k, e)
    skels.append(clone(feas[1], "dup"))      # last: after any stop or kill, so resume must rebuild the dedup index
    return skels, groups


def write_skeletons(skels, path):
    with open(path, "w", encoding="utf-8") as f:
        for s in skels:
            f.write(json.dumps(s) + "\n")


def load_run(out):
    acc = list(shards.read(os.path.join(out, "accepted"), "accepted", fix=False))
    rej = list(shards.read(os.path.join(out, "rejects"), "rejects", fix=False))
    with open(os.path.join(out, "yield.jsonl")) as f:
        snaps = [json.loads(x) for x in f if x.strip()]
    return acc, rej, snaps


def verify(tc, out, plan, skels):
    """assert the run in `out` matches plan["expected"] exactly. tc is the unittest.TestCase."""
    exp = plan["expected"]
    acc, rej, snaps = load_run(out)
    tc.assertEqual(len(acc), exp["accepted"], "accepted count")
    tc.assertEqual(len(rej), exp["rejected"], "rejected count")
    pairs = [(r["skel_id"], r["attempt"]) for r in acc + rej]
    tc.assertEqual(len(pairs), len(set(pairs)), "an attempt was recorded twice")
    by = collections.defaultdict(dict)
    for r in acc:
        by[r["skel_id"]][r["attempt"]] = ("accept", r)
    for r in rej:
        by[r["skel_id"]][r["attempt"]] = ("reject", r)
    want_primary = collections.Counter()
    for sid, p in exp["per_skel"].items():
        got = by.get(sid, {})
        tc.assertEqual(sorted(got), list(range(len(p["outcomes"]))), f"{sid} {p['scenario']} attempts")
        for a, (kind, code, pure, name) in enumerate(p["outcomes"]):
            gk, rec = got[a]
            tc.assertEqual(gk, kind, f"{sid} a{a} {p['scenario']} {name}: {rec.get('codes')}")
            if kind == "reject":
                tc.assertIn(code, rec["codes"], f"{sid} a{a} {name}")
                if pure:
                    tc.assertEqual(rec["primary"], code, f"{sid} a{a} {name}")
                want_primary[rec["primary"]] += 1
    for g in exp["groups"]:
        winners = [m for m in g["members"] if any(k == "accept" for k, _ in by.get(m, {}).values())]
        tc.assertEqual(len(winners), 1, f"group {g}")
        win_conv = f"{winners[0]}.a0"
        for m in g["members"]:
            if m == winners[0]:
                tc.assertEqual(sorted(by[m]), [0])
                continue
            tc.assertEqual(sorted(by[m]), list(range(stub_plan.MAX_ATTEMPTS)), f"group member {m}")
            for a, (kind, rec) in by[m].items():
                tc.assertEqual((kind, rec["primary"], rec["dup_of"]), ("reject", g["code"], win_conv), m)
                want_primary[g["code"]] += 1
    final = snaps[-1]
    tc.assertTrue(final["final"])
    tc.assertEqual((final["accepted"], final["rejected"]), (exp["accepted"], exp["rejected"]))
    tc.assertEqual(final["by_primary"], dict(want_primary), "yield by primary reason")
    sk_by = {s["skel_id"]: s for s in skels}
    for r in acc:
        sk = sk_by[r["skel_id"]]
        tc.assertEqual(r["blocked"], ["FAKE_PROVENANCE", "DECON_PENDING"])
        tc.assertFalse(r["trainable"])
        tc.assertEqual(len(r["turns"]), len(sk["turns"]))
        for t, st in zip(r["turns"], sk["turns"]):
            if st["mode"] == "exact":
                tc.assertEqual(t["text"], st["text"])
            for sp in t["spans"]:
                value = r["slots"][sp["slot_id"]]["value"].lower()
                tc.assertTrue(t["text"][sp["start"]:sp["end"]].lower().startswith(value), sp)
    return acc, rej, final
