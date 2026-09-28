"""gen.py end to end at a small scale: determinism, the SPEC 5 check, and the harness loader reading K shards
(SPEC 11: two arms at one seed draw identical SKILL items at identical stream positions)."""
import json
import os

import numpy as np
import pytest

import gen
import kcommon as K

SMALL_FACT = ["--slots", "400000", "--n-high", "200", "--n-low", "50", "--e-bio", "20", "--e-qa", "4"]


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("kdata"))
    assert gen.main(["skill", d, "--slots", "2000000", "--workers", "2"]) == 0
    for arm in ("F0", "FL", "FH"):
        assert gen.main(["fact", d, "--arm", arm, *SMALL_FACT]) == 0
    for arm in ("L0", "L100"):
        assert gen.main(["k5", d, "--arm", arm, "--slots", "400000", "--n5", "20", "--e5", "6"]) == 0
    assert gen.main(["evals", d, "--small"]) == 0
    return d


def test_check_passes(root):
    assert gen.main(["check", root]) == 0
    rep = json.load(open(os.path.join(root, "check.json")))
    assert rep["pass"] and {"skill_train", "k_eval_id", "k_eval_s"} <= set(rep["sets"])
    assert "cue_gate" in rep        # K3: the fitted cue model ran on the stream and every eval set, and passed
    gate = rep["cue_gate"]
    assert gate["fails"] == [] and {"train", "k_eval_id", "k_eval_s"} <= set(gate["ceiling"]["F"])
    fm = gate["f_model"]            # K3 round 2: the F model ran too, with its closed column list
    assert {"train", "k_eval_id", "k_eval_s"} <= set(fm["ceiling"]["F"]) and {"N", "pre_intro"} <= set(fm["features"])
    assert not {c for c in fm["features"] if c.startswith(("ent_", "own_", "shares", "cent", "ans_"))}
    assert all(gate["ceiling"]["F"][s] >= fm["ceiling"]["F"][s] for s in fm["ceiling"]["F"])
    assert {"cent_umax", "tf", "maj0"} <= set(gate["features"])
    assert {"B-CENTER", "F-DEFX", "A1H"} <= set(rep["sets"]["skill_train"]["oracles"]["B"])
    assert {"F0", "FL", "FH", "L0", "L100"} <= set(rep["fact_purity"])


def test_fact_bytes_deterministic(root, tmp_path):
    assert gen.main(["fact", str(tmp_path), "--arm", "FH", *SMALL_FACT]) == 0
    for sub in ("fact_bio", "fact_qa"):
        m1 = json.load(open(os.path.join(root, "FH", sub, "manifest.json")))["files"]
        m2 = json.load(open(os.path.join(tmp_path, "FH", sub, "manifest.json")))["files"]
        assert m1 == m2


def test_eval_manifest_has_every_set(root):
    man = json.load(open(os.path.join(root, "evals", "manifest.json")))
    assert set(man["sets"]) == {"k_eval_id", "k_eval_s", "k_eval_ood", "fb_probe", "conflict", "k5", "k_dev",
                                "k_probe", "k2_a", "k2_b"}
    for name, v in man["sets"].items():
        assert K.sha256_file(os.path.join(root, "evals", f"{name}.jsonl")) == v["sha256"]
    fb = K.read_jsonl(os.path.join(root, "evals", "fb_probe.jsonl"))
    k2a = K.read_jsonl(os.path.join(root, "evals", "k2_a.jsonl"))
    ents = lambda rs: {r["ent"] for r in rs if r.get("kind") in ("bio", "qa")}          # noqa: E731
    assert not ents(fb) & ents(k2a)
    dev = [r for r in K.read_jsonl(os.path.join(root, "evals", "k_dev.jsonl")) if r.get("kind") == "bio"]
    assert len({r["ent"] for r in dev}) == len(dev) and len({r["attr"] for r in dev}) > 1


def _loader(root, arm, seed=201):
    from data import build_loader
    dcfg = gen.data_block(arm, root=root)
    dcfg["sources"] = [s for s in dcfg["sources"] if s["name"] in ("skill", "fact_bio", "fact_qa", "fact_k5")]
    ld = build_loader(dcfg, seq_len=256, micro_batch=4, base_dir=root, seed=seed)
    log = {n: [] for n in ld.names}
    for src, n in zip(ld.sources, ld.names):
        f = src.next_item

        def rec(f=f, n=n):
            ids, flags = f()
            log[n].append((ids.tolist(), flags.tolist()))
            return ids, flags
        src.next_item = rec
    for _ in range(40):
        ld.next_batch()
    return ld, log


def test_arms_draw_identical_skill_positions(root):
    _, fh = _loader(root, "FH")
    _, f0 = _loader(root, "F0")
    _, fl = _loader(root, "FL")
    assert fh["skill"] == f0["skill"] == fl["skill"] and len(fh["skill"]) > 100
    assert [len(x[0]) for x in fh["fact_bio"]] == [len(x[0]) for x in f0["fact_bio"]]
    assert fh["fact_bio"] != f0["fact_bio"]
    for ids, flags in fh["fact_bio"][:50]:
        assert len(ids) == K.BIO_LEN and ids[-1] == K.EOT and flags == [False] + [True] * 15
    for ids, flags in fh["skill"][:50]:          # only assistant content and its <|end|> are supervised
        exp, inside = [], False
        for t in ids:
            exp.append(inside and t != K.ASST)
            inside = True if t == K.ASST else (False if t == K.END else inside)
        assert flags == exp and sum(flags) >= 4


def test_k5_tool_turns_are_unsupervised(root):
    _, log = _loader(root, "L100")
    seen = 0
    for ids, flags in log["fact_k5"]:
        if ids[0] == K.TOOL:
            end = ids.index(K.END)
            assert not any(flags[:end + 1]) and len(ids) == 41
            seen += 1
    assert seen > 10


def test_data_blocks_sum_to_one():
    for arm in ("F0", "FL", "FH", "L0", "L50", "L100"):
        tot = sum(s["share"] for s in gen.data_block(arm)["sources"])
        assert abs(tot - 1.0) < 1e-4, (arm, tot)
    assert [s["name"] for s in gen.data_block("CAP")["sources"]] == ["fact_bio", "fact_qa"]
    assert {s["name"] for s in gen.data_block("CONT")["sources"]} & {"fact_bio", "fact_qa", "fact_k5"} == set()
    assert all(not s.get("shuffle", True) for s in gen.data_block("FH")["sources"][:3])
    assert np.isclose(gen.data_block("FH")["sources"][1]["share"], 0.5 * 16 / 19, atol=1e-6)


def test_fact_purity_catches_planted_leaks(tmp_path):
    import check_k
    import lookup as L
    import pools as P
    import world as Wd
    pl = P.get()
    w = Wd.make_world(pl)
    for arm, ents in (("F0", [0, -1]), ("FL", [K.N_LOW + 1, 0])):
        os.makedirs(tmp_path / arm / "fact_bio")
        Wd.bio_block(w, pl, np.array(ents), 0).tofile(str(tmp_path / arm / "fact_bio" / "x.bin"))
        os.makedirs(tmp_path / arm / "fact_qa")             # the same leaks planted in the QA stream only
        K.write_jsonl(str(tmp_path / arm / "fact_qa" / "x.jsonl"),
                      Wd.qa_block(w, pl, np.array(ents + [-1]), np.zeros(3, np.int8), 0, 0))
    os.makedirs(tmp_path / "L0" / "fact_k5")
    K.write_jsonl(str(tmp_path / "L0" / "fact_k5" / "x.jsonl"),
                  [L.k5_chat(w.names[3], w.attrs, w.values[3], 0, 1, True, list(range(6)), "x")])
    rep = check_k.fact_purity(str(tmp_path), pl)
    assert rep["F0"]["fb_name_in_F0"] == 1 and rep["FL"]["non_low_name_in_FL"] == 1
    assert rep["F0"]["qa_fb_name_in_F0"] == 1 and rep["FL"]["qa_non_low_name_in_FL"] == 1
    assert rep["F0"]["skill_or_heldout_token"] == 0 and rep["FL"]["skill_or_heldout_token"] == 0
    assert rep["L0"]["hit_in_L0"] == 1 and rep["L0"]["non_low_name"] == 0


def test_check_requires_b_groups_on_k_eval(tmp_path):
    """K3 round 4 (REVIEW 5b O-1): a K-EVAL B set must score 3n/16 cube groups (BD-26). One written without groups
    (every record a plain item, so purity finds no group and skips every group bar) fails gen.py check."""
    import check_k
    import evalsets as E
    import pools as P
    pl = P.get()
    os.makedirs(tmp_path / "evals")
    recs = E.skill_set(pl, P.Lex(pl), "eval", 7201, 32, fams="B", groups=False, tag="K-EVAL-ID")
    with open(tmp_path / "evals" / "k_eval_id.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    out = check_k.run(str(tmp_path), cues=False)
    assert not out["pass"] and "k_eval_id: B groups 0 of 32 records, not 3n/16" in out["fails"], out["fails"]
    assert out["sets"]["k_eval_id"]["fails"] == [], out["sets"]["k_eval_id"]["fails"]
