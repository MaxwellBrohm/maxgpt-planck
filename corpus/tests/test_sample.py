"""make_tok_sample tests: share targeting, disjointness, the reserve guard, determinism."""
import glob
import json
import os

import numpy as np
import pytest

import extract
import make_tok_sample as M
import sample_plan as P
from fixtures import SENTS, tree_ids, write_raw

KB = 1000
SUPPLY = {"oasst2": 8, "dolly": 4, "irc": 90, "cccc": 110, "stackexchange": 60,
          "gutenberg": 40, "wikimedia": 40}


def write_extracted(root, supply_kb=SUPPLY, seed=0, reserved_tree=None, unrecorded=False):
    """A fake extract output: shards of 150-500 byte docs, with an extract_stats.json. OASST2
    docs come two threads per tree for part of the set, to test that a tree never splits; Dolly
    docs share a split_key (one context) three at a time. unrecorded marks CCCC's license so."""
    rng = np.random.default_rng(seed)
    trees = tree_ids(False, 400)
    outputs = []
    for s, kb in supply_kb.items():
        os.makedirs(f"{root}/{s}", exist_ok=True)
        path, n, total = f"{s}/{s}-00000.jsonl", 0, 0
        with open(f"{root}/{path}", "w") as f:
            while total < kb * KB:
                k = int(rng.integers(2, 7))
                text = " ".join(SENTS[int(i)] for i in rng.integers(0, len(SENTS), k)) + f" #{n}"
                if n % 17 == 3:
                    text += " <|user|> planted <think>x</think><|end|>"
                meta = {}
                if s == "oasst2":
                    meta["tree_id"] = trees[n // 2] if n < 200 else trees[100 + n]
                    if reserved_tree and n == 5:
                        meta["tree_id"] = reserved_tree
                if s == "dolly":
                    meta["split_key"] = f"ctx:{n // 3}"
                if s == "cccc" and unrecorded:
                    meta.update(license=None, license_basis="unrecorded")
                rec = {"id": f"{s}:{n}", "source": s, "text": text, "meta": meta}
                if s in ("oasst2", "dolly"):
                    rec["turns"] = [{"role": "user", "text": text}]
                f.write(json.dumps(rec) + "\n")
                total += len(text.encode())
                n += 1
        outputs.append({"path": path, "source": s})
    with open(f"{root}/extract_stats.json", "w") as f:
        json.dump({"outputs": outputs}, f)
    return root


def read_split(out, split):
    recs = []
    for p in sorted(glob.glob(f"{out}/{split}/*.jsonl")):
        recs += [json.loads(line) for line in open(p)]
    return recs


@pytest.fixture(scope="module")
def ex(tmp_path_factory):
    return write_extracted(str(tmp_path_factory.mktemp("ex")))


def test_shares_within_two_points(ex, tmp_path):
    m = M.build_sample(ex, str(tmp_path), total_bytes=150 * KB, heldout_bytes=5 * KB)
    for g, want in P.DEFAULT_SHARES.items():
        assert abs(m["achieved"]["group_shares"][g] - want) <= 0.02, (g, m["achieved"])
    assert abs(m["achieved"]["train_total_weighted_bytes"] - 150 * KB) <= 4 * KB
    tr = m["achieved"]["train_text_bytes"]
    assert tr["oasst2"] > 0.7 * SUPPLY["oasst2"] * KB and tr["dolly"] > 0.7 * SUPPLY["dolly"] * KB
    w = m["weights"]
    assert set(w) == {"oasst2", "dolly"} and w["oasst2"] == w["dolly"] and 2 <= w["oasst2"] <= 16
    assert m["achieved"]["source_shares"]["irc"] <= P.IRC_MAX_SHARE + 0.005
    assert m["achieved"]["train_weighted_bytes"]["oasst2"] == tr["oasst2"] * w["oasst2"]


def test_chat_repeat_cap_and_irc_cap_bind(tmp_path):
    ex = write_extracted(str(tmp_path / "ex"), dict(SUPPLY, oasst2=2, dolly=1, irc=300))
    m = M.build_sample(ex, str(tmp_path / "a"), total_bytes=400 * KB, heldout_bytes=KB)
    assert m["weights"] == {"oasst2": 16, "dolly": 16}                 # repeat cap reached
    a = m["achieved"]
    assert a["train_total_weighted_bytes"] < 200 * KB                  # scale: the total shrank
    assert abs(a["group_shares"]["chat"] - 0.37) <= 0.02, a["group_shares"]
    assert abs(a["source_shares"]["irc"] - P.IRC_MAX_SHARE) <= 0.005   # IRC at its cap, no more
    m2 = M.build_sample(ex, str(tmp_path / "b"), total_bytes=400 * KB, heldout_bytes=KB,
                        max_repeat=4)
    assert m2["weights"] == {"oasst2": 4, "dolly": 4}
    assert m2["achieved"]["train_total_weighted_bytes"] < 0.5 * a["train_total_weighted_bytes"]


def test_weights_reach_the_tokenizer_reader(ex, tmp_path):
    import importlib
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tokenizer"))
    sio = importlib.import_module("sample_io")
    m = M.build_sample(ex, str(tmp_path), total_bytes=150 * KB, heldout_bytes=5 * KB)
    assert sio.load_weights(str(tmp_path)) == m["weights"]
    n = sum(len(t.encode()) for t in sio.iter_texts(str(tmp_path), "train"))
    assert n == m["achieved"]["train_total_weighted_bytes"]


def test_dolly_context_groups_never_split(ex, tmp_path):
    M.build_sample(ex, str(tmp_path), total_bytes=150 * KB, heldout_bytes=5 * KB)
    tr, ho = read_split(str(tmp_path), "train"), read_split(str(tmp_path), "heldout")
    kt = {r["meta"]["split_key"] for r in tr if r["source"] == "dolly"}
    kh = {r["meta"]["split_key"] for r in ho if r["source"] == "dolly"}
    assert kt and kh and not kt & kh


def test_unrecorded_license_needs_an_explicit_yes(tmp_path):
    ex = write_extracted(str(tmp_path / "ex"), unrecorded=True)
    with pytest.raises(SystemExit, match="allow-unrecorded-license cccc"):
        M.build_sample(ex, str(tmp_path / "a"), total_bytes=100 * KB, heldout_bytes=KB)
    m = M.build_sample(ex, str(tmp_path / "b"), total_bytes=100 * KB, heldout_bytes=KB,
                       allow_unrecorded=("cccc",))
    assert m["config"]["allow_unrecorded_license"] == ["cccc"]
    assert m["license_unrecorded_docs"]["cccc"] > 0 and m["achieved"]["train_text_bytes"]["cccc"]
    no_web = {"chat": 0.37, "web": 0.0, "se": 0.23, "books": 0.15, "wiki": 0.25}
    m2 = M.build_sample(ex, str(tmp_path / "c"), no_web, total_bytes=100 * KB, heldout_bytes=KB)
    assert m2["achieved"]["train_text_bytes"]["cccc"] == 0


def test_custom_shares_and_fill_mode(ex, tmp_path):
    shares = {"chat": 0.5, "web": 0.2, "se": 0.1, "books": 0.1, "wiki": 0.1}
    m = M.build_sample(ex, str(tmp_path / "a"), shares, total_bytes=120 * KB, heldout_bytes=4 * KB)
    for g, want in shares.items():
        assert abs(m["achieved"]["group_shares"][g] - want) <= 0.02, (g, m["achieved"])
    short = write_extracted(str(tmp_path / "short"), dict(SUPPLY, gutenberg=12))
    m2 = M.build_sample(short, str(tmp_path / "b"), total_bytes=150 * KB, heldout_bytes=4 * KB,
                        short_supply="fill")               # books short: total holds
    assert abs(m2["achieved"]["train_total_weighted_bytes"] - 150 * KB) <= 4 * KB
    assert m2["achieved"]["group_shares"]["books"] < 0.08
    m3 = M.build_sample(short, str(tmp_path / "c"), total_bytes=150 * KB, heldout_bytes=4 * KB)
    a3 = m3["achieved"]
    assert a3["train_total_weighted_bytes"] < 110 * KB   # scale: shares win
    whole = sum(a3["train_text_bytes"][s] for s in P.WHOLE_CHAT)
    for g, want in P.DEFAULT_SHARES.items():         # integer repeats: chat is off by <= whole / 2
        tol = 0.02 + (0.5 * whole / a3["train_total_weighted_bytes"] if g == "chat" else 0)
        assert abs(a3["group_shares"][g] - want) <= tol, (g, a3)


def test_train_heldout_disjoint_by_id_and_tree(ex, tmp_path):
    m = M.build_sample(ex, str(tmp_path), total_bytes=150 * KB, heldout_bytes=5 * KB)
    tr, ho = read_split(str(tmp_path), "train"), read_split(str(tmp_path), "heldout")
    assert tr and ho
    assert not {r["id"] for r in tr} & {r["id"] for r in ho}
    tt = {r["meta"]["tree_id"] for r in tr if r["source"] == "oasst2"}
    ht = {r["meta"]["tree_id"] for r in ho if r["source"] == "oasst2"}
    assert tt and ht and not tt & ht
    assert set(m["checks"].values()) == {0}
    assert {r["source"] for r in ho} == set(SUPPLY)
    for s in SUPPLY:
        hb = m["achieved"]["heldout_text_bytes"][s]
        assert hb <= 0.25 * SUPPLY[s] * KB * 1.05 + 600 and hb <= 5 * KB + 600


def test_manifest_hashes_and_determinism(ex, tmp_path):
    a = M.build_sample(ex, str(tmp_path / "a"), total_bytes=150 * KB, heldout_bytes=5 * KB)
    b = M.build_sample(ex, str(tmp_path / "b"), total_bytes=150 * KB, heldout_bytes=5 * KB)
    c = M.build_sample(ex, str(tmp_path / "c"), total_bytes=150 * KB, heldout_bytes=5 * KB, seed=2)
    assert [f["sha256"] for f in a["files"]] == [f["sha256"] for f in b["files"]]
    assert [f["sha256"] for f in a["files"]] != [f["sha256"] for f in c["files"]]
    import hashlib
    for f in a["files"]:
        data = open(tmp_path / "a" / f["path"], "rb").read()
        assert hashlib.sha256(data).hexdigest() == f["sha256"] and len(data) == f["bytes"]


def test_reserved_tree_in_input_is_refused(tmp_path):
    ex = write_extracted(str(tmp_path / "ex"), {"oasst2": 3, "dolly": 1, "irc": 1, "cccc": 1,
                                                "stackexchange": 1, "gutenberg": 1,
                                                "wikimedia": 1},
                         reserved_tree=tree_ids(True, 1)[0])
    with pytest.raises(SystemExit, match="OOD-H reserved tree"):
        M.build_sample(ex, str(tmp_path / "out"), total_bytes=5 * KB, heldout_bytes=KB)


def test_bad_shares_refused(ex, tmp_path):
    with pytest.raises(ValueError, match="sum to 1"):
        M.build_sample(ex, str(tmp_path), {"chat": .5, "web": .5, "se": .1, "books": 0, "wiki": 0})
    with pytest.raises(ValueError, match="archaic"):
        M.build_sample(ex, str(tmp_path), {"chat": .2, "web": .3, "se": .1, "books": .3, "wiki": .1})


def test_end_to_end_from_fixture_extract(tmp_path):
    raw = str(tmp_path / "raw")
    facts = write_raw(raw)
    extract.run_extract(raw, str(tmp_path / "ex"), gutenberg_cap=2000)
    with pytest.raises(SystemExit, match="no per-document license"):
        M.build_sample(str(tmp_path / "ex"), str(tmp_path / "s0"), total_bytes=12 * KB,
                       heldout_bytes=KB)
    m = M.build_sample(str(tmp_path / "ex"), str(tmp_path / "s"), total_bytes=12 * KB,
                       heldout_bytes=KB, allow_unrecorded=("cccc",))
    blob = "".join(open(p).read() for p in glob.glob(str(tmp_path / "s" / "*" / "*.jsonl")))
    assert blob and "OODH_RESERVED_MARKER" not in blob
    assert not any(t in blob for t in facts["reserved_trees"])
    assert set(m["checks"].values()) == {0}
    assert m["achieved"]["heldout_text_bytes"]["oasst2"] > 0


def test_special_strings_are_stripped(ex, tmp_path):
    m = M.build_sample(ex, str(tmp_path), total_bytes=150 * KB, heldout_bytes=5 * KB)
    recs = [r for sp in ("train", "heldout") for r in read_split(str(tmp_path), sp)]
    texts = [r["text"] for r in recs] + [t["text"] for r in recs for t in r.get("turns", [])]
    assert any("planted" in t for t in texts)
    assert not any(P.SPECIAL_RE.search(t) or "<|" in t or "<think>" in t for t in texts)
    assert m["docs_with_special_strings_stripped"] == sum("planted" in r["text"] for r in recs)
    assert any("planted" in t["text"] for r in recs for t in r.get("turns", []))


def test_special_strings_cover_the_harness_template():
    import importlib.util
    import sys
    path = os.path.join(os.path.dirname(__file__), "..", "..", "harness", "chat_template.py")
    spec = importlib.util.spec_from_file_location("chat_template_for_test", path)
    ct = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ct                       # dataclasses look the module up here
    spec.loader.exec_module(ct)
    assert set(ct.ROLE_TOKENS.values()) | {ct.END_TOKEN} <= set(P.SPECIAL_STRINGS)


def test_source_targets_never_exceed_the_repeat_cap():
    avail = {"oasst2": 10, "dolly": 5, "irc": 100}
    gt = {"chat": 10_000, "web": 0, "se": 0, "books": 0, "wiki": 0}
    st, w = P.source_targets(gt, avail, total=1000)
    assert w == P.CHAT_MAX_REPEAT and st["oasst2"] == 10 and st["dolly"] == 5
    assert st["irc"] == 0.04 * 1000                                    # IRC stays at its cap
    st2, w2 = P.source_targets(dict(gt, chat=12), avail, total=1000)  # below whole + cap: no repeat
    assert w2 == 1 and st2["irc"] == 12 and st2["oasst2"] == 0          # IRC covers it first


def test_oasst_tree_never_splits_across_seeds(tmp_path):
    ex = write_extracted(str(tmp_path / "ex"), dict(SUPPLY, oasst2=30))   # threads come in pairs
    for seed in range(8):
        out = str(tmp_path / f"s{seed}")
        M.build_sample(ex, out, total_bytes=150 * KB, heldout_bytes=5 * KB, seed=seed)
        tt = {r["meta"]["tree_id"] for r in read_split(out, "train") if r["source"] == "oasst2"}
        ht = {r["meta"]["tree_id"] for r in read_split(out, "heldout") if r["source"] == "oasst2"}
        assert tt and ht and not tt & ht, seed
