"""oodh/build.py and oodh/hashes.py on the synthetic fixture (tests/fixture.py; no real item text). Loads no model.
Run: python3 -B oodh/tests/test_build.py"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
OODH = os.path.dirname(HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != OODH]
import select  # noqa: E402,F401  the stdlib module first: oodh/ holds a select.py of its own
sys.path[:0] = [OODH, HERE]

import hashlib  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402

import build as B  # noqa: E402
import fixture as FX  # noqa: E402
import graders as G  # noqa: E402  (rc12, on the path through build)
import hashes as H  # noqa: E402

TESTS = []


def test(f):
    TESTS.append(f)
    return f


def world(n=9):
    root = os.path.join(tempfile.mkdtemp(prefix="oodh_test_"), "sealed", "oodh")
    return root, FX.make_world(root, n)


def by_tree(recs):
    return {r["tree_id"]: r for r in recs}


@test
def load_three_shapes():
    root, tids = world()
    items = B.load_items(root)
    assert [it["tree_id"] for it in items] != [] and sorted(it["tree_id"] for it in items) == sorted(tids)
    assert {it["batch"] for it in items} == {"batch_1.jsonl", "batch_2.jsonl", "batch_3.jsonl"}
    for it in items:
        ps = it["probes"]
        assert [p["kind"] for p in ps] == ["X"] * (len(ps) - 1) + ["P"], ps
        assert all(p["probe_kind"].startswith("H-") and p["question"].endswith("?") for p in ps)
        for p in ps:
            if p["grader"] == "VAL":
                assert p["accepted"][0] == p["gold"] and p["source_msg_id"] in it["message_ids"][0::2], p
            else:
                assert p["source_msg_id"] is None and p["accepted"] == []
        assert all(type(a) is bool for a in it["asks"]), it["asks"]
    p = B.norm_probe(dict(kind="H-FACT", turn_kind="P", grader="VAL", turn=3, text="Which?", gold="x", accepted=["y"],
                          holder="user", ideal="It is x.", pool_values=[], source_msg=2), ["m0", "m1", "m2"])
    assert (p["accepted"], p["question"], p["probe_kind"], p["kind"], p["source_msg_id"]) == (
        ["x", "y"], "Which?", "H-FACT", "P", "m2"), p


@test
def record_kinds_facts_src():
    root, tids = world()
    recs = by_tree(B.build(B.load_items(root), 9)[0])
    a, b = recs[tids[0]], recs[tids[1]]                 # fixture: even k -> type A, odd k -> type B
    assert [t["kind"] for t in a["turns"]] == ["S", "C", "X", "X", "P"]
    assert [t["kind"] for t in b["turns"]] == ["S", "L", "D", "X", "P"]
    assert b["turns"][2]["asks"] is False and all("asks" not in t for t in a["turns"] + b["turns"][:2])
    roles = {f["role"] for f in a["turns"][0]["facts"]}
    assert roles == {"gold", "stale"} and [f["role"] for f in a["turns"][1]["facts"]] == ["corr"]
    corr, fact, ab = a["probes"]
    assert (fact["src"], fact["d"], corr["src"], corr["d"], corr["dc"], ab["src"]) == ([1], 3, [2], 1, 1, [])
    assert b["probes"][0]["d"] == 3 and a["cell"] == "H-ABS+H-CORR+H-FACT" and b["cell"] == "H-ASK+H-FACT"
    assert b["turns"][2]["history_reply"] == "You are welcome, have a lovely day."
    for r in (a, b):
        n = r["meta"]["n_thread_user_turns"]
        assert r["n_turns"] == len(r["turns"]) and r["id"].startswith("oodh-") and r["meta"]["unit"] == "mean"
        assert all(t["ideal"] == t["history_reply"] and t["msg_id"] == r["message_ids"][2 * (t["i"] - 1)]
                   for t in r["turns"][:n])
    items = B.load_items(root)
    it = [x for x in items if x["tree_id"] == tids[1]][0]
    it["users"][2] = "Thank you so much, the poem is perfect."          # the gold now in u1 and u3
    rec = B.record(it, "oodh-x")
    assert rec["probes"][0]["src"] == [1, 3] and rec["probes"][0]["d"] == 1 and rec["turns"][2]["kind"] == "S"


@test
def records_grade_with_rc12():
    root, _ = world()
    for r in B.build(B.load_items(root), 9)[0]:
        replies = [t["ideal"] for t in r["turns"]]
        g = G.grade_conv(r, replies, ["eos"] * len(replies))
        assert g["unit"] == 1.0, (r["id"], g["probes"])


def expect_error(root, mutate, what):
    items = B.load_items(root)
    try:
        mutate(items)
        B.build(items, len(items))
    except B.BuildError:
        return
    raise AssertionError(f"no BuildError for {what}")


@test
def build_errors():
    root, tids = world()
    expect_error(root, lambda its: its[0]["probes"][0].update(src=[1]), "a gold missing from its src turn")
    expect_error(root, lambda its: [it.update(asks=[None] * len(it["users"])) for it in its], "a D turn without asks")
    expect_error(root, lambda its: its[1]["probes"][0].update(turn=9), "probe turns out of order")
    expect_error(root, lambda its: its[1]["probes"][-1].update(kind="X"), "no P probe")
    cands = B.read_jsonl(os.path.join(root, "candidates.jsonl"))
    cands[0]["message_ids"] = cands[0]["message_ids"][::-1]
    with open(os.path.join(root, "candidates.jsonl"), "w") as fh:
        fh.write("".join(json.dumps(c) + "\n" for c in cands))
    try:
        B.load_items(root)
        raise AssertionError("no BuildError for mismatched message ids")
    except B.BuildError:
        pass
    root, tids = world()
    b2 = B.read_jsonl(os.path.join(root, "batch_2.jsonl"))
    b2[0]["turns"][0]["text"] += " (edited)"
    with open(os.path.join(root, "batch_2.jsonl"), "w") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in b2))
    try:
        B.load_items(root)
        raise AssertionError("no BuildError for a batch user turn that differs from candidates.jsonl")
    except B.BuildError:
        pass
    root, tids = world()
    b1 = B.read_jsonl(os.path.join(root, "batch_1.jsonl"))
    with open(os.path.join(root, "batch_3.jsonl"), "a") as fh:
        fh.write(json.dumps(dict(b1[0], turn_asks=[True, True])) + "\n")
    try:
        B.load_items(root)
        raise AssertionError("no BuildError for a tree in two batches")
    except B.BuildError:
        pass


def fake(tid, kinds):
    return dict(tree_id=tid, probes=[dict(probe_kind=k) for k in kinds])


@test
def select_rule():
    items = ([fake(f"f{i}", ["H-FACT"] * 3) for i in range(20)] + [fake(f"a{i}", ["H-ASK", "H-ASK", "H-ABS"])
             for i in range(5)] + [fake("corr", ["H-CORR", "H-FACT"])])
    got = B.select(items, 6)
    assert len(got) == 6 and got[0]["tree_id"] == "corr"
    shuffled = items[:]
    random.Random(7).shuffle(shuffled)
    assert [x["tree_id"] for x in B.select(shuffled, 6)] == [x["tree_id"] for x in got]
    naive = [items[-1]] + sorted(items[:-1], key=B.pick_key)[:5]
    count = lambda xs: B.Counter(k for x in xs for k in B.kinds_of(x))  # noqa: E731
    assert B.distance(count(got)) < B.distance(count(naive)), (B.distance(count(got)), B.distance(count(naive)))
    assert sum(x["tree_id"].startswith("a") for x in got) >= 3
    twins = [fake("t1", ["H-FACT"]), fake("t2", ["H-FACT"])]
    assert B.select(twins, 1)[0] == min(twins, key=B.pick_key)
    assert B.select(items, 27) is None and len(B.select(items, 26)) == 26
    once = [fake("c", ["H-CORR", "H-FACT", "H-FACT", "H-ASK", "H-ASK", "H-ABS"])] + [fake(f"g{i}", ["H-FACT"] * 3)
                                                                                    for i in range(4)]
    assert sorted(x["tree_id"] for x in B.select(once, 3))[0] == "c" and len({id(x) for x in B.select(once, 3)}) == 3


@test
def build_exact_and_short():
    root, tids = world()
    items = B.load_items(root)
    recs, short = B.build(items, 7)
    assert len(recs) == 7 and not short and sum("H-CORR" in r["cell"] for r in recs) == 5
    assert [r["tree_id"] for r in recs] == sorted(r["tree_id"] for r in recs)
    assert [r["id"] for r in recs] == [f"oodh-{k:03d}" for k in range(1, 8)]
    try:
        B.build(items, 10)
        raise AssertionError("a short build must fail without allow_short")
    except B.BuildError:
        pass
    recs, short = B.build(items, 10, allow_short=True)
    assert short and len(recs) == 9


@test
def hashes_file():
    root, tids = world()
    recs, short = B.build(B.load_items(root), 10, allow_short=True)
    data, hp = os.path.join(root, "oodh_part1.jsonl"), os.path.join(root, "HASHES.txt")
    with open(data, "w") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in recs))
    H.write_hashes(recs, short, 10, data, hp, [data])
    text = open(hp).read()
    assert hashlib.sha256(open(data, "rb").read()).hexdigest() in text
    assert hashlib.sha256("\n".join(sorted(tids)).encode()).hexdigest() in text
    assert H.ids_sha(sorted(tids, reverse=True)) == hashlib.sha256("\n".join(sorted(tids)).encode()).hexdigest()
    assert "status SHORT: 9 usable threads, target 10" in text and "H-CORR 5" in text
    assert "birthday" not in text and "Porto" not in text
    H.put_gate(hp, [H.GATE_HEAD + " one", "  x"])
    H.put_gate(hp, [H.GATE_HEAD + " two", "  y"])
    text2 = open(hp).read()
    assert text2.count(H.GATE_HEAD) == 1 and text2.startswith(text) and text2.endswith("two\n  y\n")


@test
def cli():
    root, tids = world()
    hp = os.path.join(root, "HASHES.txt")
    run = lambda *a: subprocess.run([sys.executable, "-B", os.path.join(OODH, "build.py"), "--sealed", root,  # noqa
                                     "--hashes", hp, *a], capture_output=True, text=True, timeout=100)
    r = run()
    assert r.returncode == 3 and "only 9 usable threads" in r.stdout, (r.returncode, r.stdout, r.stderr)
    r = run("--allow-short")
    assert r.returncode == 0 and "status SHORT" in open(hp).read(), r.stderr
    r = run("--n", "9")
    assert r.returncode == 0 and "status COMPLETE: 9 threads" in open(hp).read(), r.stderr
    assert len(B.read_jsonl(os.path.join(root, "oodh_part1.jsonl"))) == 9
    bad = subprocess.run([sys.executable, "-B", os.path.join(OODH, "build.py"), "--sealed", tempfile.mkdtemp()],
                         capture_output=True, text=True, timeout=100)
    assert bad.returncode != 0 and "refusing" in bad.stderr


def main():
    failed = 0
    for f in TESTS:
        try:
            f()
            print(f"ok   {f.__name__}")
        except Exception as e:                      # noqa: BLE001
            failed += 1
            print(f"FAIL {f.__name__}: {type(e).__name__}: {str(e)[:300]}")
    print(f"{len(TESTS) - failed} of {len(TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
