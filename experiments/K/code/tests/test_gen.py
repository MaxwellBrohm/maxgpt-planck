"""gen.py end to end at a small scale: determinism, the SPEC 5 check, and the harness loader reading K shards
(SPEC 11: two arms at one seed draw identical SKILL items at identical stream positions)."""
import json
import os

import numpy as np
import pytest

import gen
import kcommon as K

SIX_MARKS = ("O13e", "O13a", "O14e", "O14a", "SURFe", "SURFa")     # SPEC 9: F's one-kind rules, out of its ceiling
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
    import check_k                  # K3 round 15 (REVIEW 16b N-1): the check's own code, by sha256 (mutant
    assert rep.get("code_sha256") == {c: K.sha256_file(os.path.join(K.HERE, c)) for c in check_k.CHECK_CODE} and {
        "check_k.py", "cuegate.py", "cues.py"} <= set(check_k.CHECK_CODE), rep.get("code_sha256")   # r16_check_no_sha)
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
    ce = rep["ceilings"]            # K3 round 7 (REVIEW 8a G-1): SPEC 9's ceiling per family and set
    import oracles as O
    for fam in "RUBFAP":
        for sn, c in ce[fam].items():
            assert c["value"] == max(x for x in (c["gate"], c["ta"], c["rule_value"]) if x is not None), (fam, sn, c)
            assert c["value"] >= gate["ceiling"][fam][sn] and c["rule"] not in SIX_MARKS, (fam, sn, c)   # round 10
            assert c["rule"] is None or fam in O.TH[c["rule"]][0], (fam, sn, c)    # K3 round 8 (REVIEW 9b T-1)
    assert ce["F"]["train"]["rule_value"] >= rep["sets"]["skill_train"]["oracles"]["F"]["F-ANTESIZE"]
    assert "A" in gate["ta_model"]["ceiling"] and gate["ta_model"]["a_model"]["n_train"] > 0      # A's TA side
    import check_k                  # K3 round 11 (REVIEW 12b T-2): every family of K-EVAL-ID and K-EVAL-S is read
    assert all(set(rep["sets"][s]["oracles"]) == set("RUBFAP") for s in ("k_eval_id", "k_eval_s")), rep["sets"]
    import pools as P               # K3 round 11 (REVIEW 12b T-6): SURF's views on every eval set are the training
    import purity as U              # stream's fit, recomputed here; the set's own 2-fold fit differs on most rows
    lex, fit, gaps = P.Lex(P.get()), rep["sets"]["skill_train"]["surf_counts"], 0     # (mutant m12_eval_surf_self_fit)
    for sn in check_k.EVAL_SPLIT:
        facts = [U.item_facts(r, lex) for r in K.read_jsonl(os.path.join(root, "evals", f"{sn}.jsonl"))
                 if r.get("qs") and r.get("fam") in tuple("RUBFAP")]
        for fam, row in rep["sets"][sn]["oracles"].items():
            F = [f for f in facts if f["fam"] == fam]
            for v, (key, _) in O.VIEWS.items():
                assert row[v] == round(U.surf(F, fit[fam][v], key), 4), (sn, fam, v, row[v])
                gaps += row[v] != round(U.surf(F, None, key), 4)
    assert gaps >= 20, gaps


def test_ceilings_take_the_largest_reader():
    """K3 round 7 (REVIEW 8a G-1): U-TIGHTSPAN read U about 0.60 on gen.py's stream against the gate's 0.555, so SPEC
    9's ceiling is the largest of the gate's, the TA model's and the family's barred cheap rules' readings on the same
    set; F's one-kind rules (SURFe, SURFa, O13, O14) are skill marks, not cheap readers, and stay out. K3 round 8
    (REVIEW 9b T-1): only the rules barred on the family itself count. O4 (barred on F only, at 0.35) reads R and U
    1.00 (it is their skill) and O9 answers as O4 where the attribute occurs, so a ceiling over every family's rules
    read 1.000 on every R, U, B, A and P set (mutant ceilings_any_family); here O4 and O9 at 1.0 on U must not be
    taken. REVIEW 9a G-1: F-ANTESIZE, barred on F, is an F ceiling row. K3 round 10 (REVIEW 11b T-3): each of the six
    marks, alone at 0.99 over SURF's 0.631 on an F set, leaves the ceiling the gate's with SURF its rule (only O13a
    was held before; mutant m11_ceilings_o14a_in: O14a set F's K-EVAL-ID ceiling, 0.625 -> 0.627)."""
    import check_k
    out = {"cue_gate": {"ceiling": {"U": {"train": 0.555, "k_eval_id": 0.549}, "F": {"train": 0.633}},
                        "ta_model": {"ceiling": {"U": {"train": 0.547, "k_eval_id": 0.62}}}},
           "sets": {"skill_train": {"oracles": {"U": {"U-TIGHTSPAN": 0.601, "O2": 0.5, "IDEAL": 1.0, "O4": 1.0,
                                                      "O9": 1.0},
                                                "F": {"SURF": 0.631, "SURFe": 0.815, "O13a": 0.75, "IDEAL": 1.0,
                                                      "F-ANTESIZE": 0.625}}},
                    "k_eval_id": {"oracles": {"U": {"U-TIGHTSPAN": 0.594, "O2": 0.48, "O4": 1.0},
                                              "F": {"SURF": 0.62, "F-ANTESIZE": 0.64}}}}}
    out["cue_gate"]["ceiling"]["F"]["k_eval_id"] = 0.63
    ce = check_k.ceilings(out)
    assert (ce["U"]["train"]["value"], ce["U"]["train"]["from"]) == (0.601, "U-TIGHTSPAN"), ce["U"]
    assert (ce["U"]["k_eval_id"]["value"], ce["U"]["k_eval_id"]["from"]) == (0.62, "ta"), ce["U"]
    assert (ce["F"]["train"]["value"], ce["F"]["train"]["from"], ce["F"]["train"]["rule"]) == (0.633, "gate", "SURF")
    assert (ce["F"]["k_eval_id"]["value"], ce["F"]["k_eval_id"]["from"]) == (0.64, "F-ANTESIZE"), ce["F"]
    import oracles as O
    for m in SIX_MARKS:         # K3 round 10 (REVIEW 11b T-3): SPEC 9's six one-kind marks, each alone at the top of
        tab = {"SURF": 0.631, "F-ANTESIZE": 0.625, "IDEAL": 1.0, m: 0.99}      # F's barred rules, stay out
        one = {"cue_gate": {"ceiling": {"F": {"train": 0.633}}}, "sets": {"skill_train": {"oracles": {"F": tab}}}}
        c = check_k.ceilings(one)["F"]["train"]
        assert O.TH[m][0] == "F" and (c["value"], c["from"], c["rule"]) == (0.633, "gate", "SURF"), (m, c)
    assert sorted(check_k.MARKS) == sorted(SIX_MARKS), check_k.MARKS
    # K3 round 11 (REVIEW 12b T-8): A's ceiling takes A's attribute-blind side as every family's does (gen.py check's
    # A train cell: ta 0.5073 over the gate's 0.5025; mutant r12f_ceilings_a_no_ta: the gate's)
    a = {"cue_gate": {"ceiling": {"A": {"train": 0.5025}}, "ta_model": {"ceiling": {"A": {"train": 0.5073}}}},
         "sets": {"skill_train": {"oracles": {"A": {"O9|10": 0.5, "IDEAL": 1.0}}}}}
    c = check_k.ceilings(a)["A"]["train"]
    assert (c["value"], c["from"], c["gate"], c["ta"]) == (0.5073, "ta", 0.5025, 0.5073), c


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


def test_skill_stream_mixes_families(root):
    """K3 round 14 (REVIEW 15b T-1): write_skill's seeded permutation is the only thing that mixes the families in
    the SKILL stream (the harness reads a shard's lines in file order; gen.data_block sets shuffle False), so the
    written shards are read here in stream order: no family runs longer than 16 records, and in every window of 512
    records each family's share is within 0.5x to 1.5x of its share of the stream (this fixture: runs at most 7,
    window ratios 0.70 to 1.24). Unshuffled the stream goes family by family in blocks of 512 (mutant
    r15_skill_unshuffled), and training would cycle through the families."""
    import itertools
    from collections import Counter
    d = os.path.join(root, "skill")
    fams = [r["fam"] for f in sorted(os.listdir(d)) if f.endswith(".jsonl") for r in K.read_jsonl(os.path.join(d, f))]
    assert len(fams) >= 10240 and set(fams) == set("RUBFAP")
    run = max(len(list(g)) for _, g in itertools.groupby(fams))
    assert run <= 16, run
    tot = Counter(fams)
    for lo in range(0, len(fams) - 511, 512):
        c = Counter(fams[lo:lo + 512])
        assert all(0.5 <= (c[x] / 512) / (tot[x] / len(fams)) <= 1.5 for x in tot), (lo, c)


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
    lang = {n for n, _, _ in gen.LANG}      # K3 round 15 (REVIEW 16b): the LANG sources' pinned shares (sum 0.9999)
    for arm in ("FH", "L0", "CONT"):         # are normalized to the LANG share; 7 roundings at 6 decimals move it
        ls = [s["share"] for s in gen.data_block(arm)["sources"] if s["name"] in lang]   # under 4e-6 (mutant
        assert len(ls) == len(gen.LANG) and abs(sum(ls) - K.SHARES["lang"]) < 4e-6, (arm, sum(ls))  # r16_lang_
    #                                                                       share_unnormalized: 0.249975)


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


def _plant_s1(recs, pick, own):
    """recs with one training-name token (pool S1) in place of the first T1 token of a user turn of the first record
    pick takes (a name a K-EVAL-ID or K-DEV item may not carry); -> (recs, the planted record's id)."""
    s1 = min(t for t, c in own.items() if c == "S1")
    out = json.loads(json.dumps(recs))
    for r in out:
        for t in r["turns"] if pick(r) else []:
            j = next((i for i, x in enumerate(t["ids"]) if own.get(x) == "T1"), None) if t["role"] == "user" else None
            if j is not None:
                t["ids"][j] = s1
                return out, r["id"]
    raise AssertionError("no record to plant")


def test_check_reads_every_eval_family_and_split(root, tmp_path):
    """K3 round 11 (REVIEW 12b T-2, T-3, T-5): gen.py check must read every family of every eval set, with each set's
    own split, and report the OOD sets' fails. On a copy of root's K-EVAL-ID, K-DEV and K-EVAL-OOD, one training-name
    token (S1) is planted in a P record of K-EVAL-ID, in a K-DEV record and in an OOD-DFAR record: each set fails
    purity's skill_train_token, and the check fails. Mutants: m12g_check_evals_no_p (no P eval item read),
    r12e_kdev_split_evals (K-DEV read as K-EVAL-S's split, where S names are its own), m12g_ood_fails_dropped (the
    OOD sets' fails never reach the check's)."""
    import check_k
    import pools as P
    own = P.get().owner()
    os.makedirs(tmp_path / "evals")
    plants = {"k_eval_id": lambda r: r.get("fam") == "P", "k_dev": lambda r: bool(r.get("qs")),
              "k_eval_ood": lambda r: r.get("set") == "OOD-DFAR"}
    for name, pick in plants.items():
        recs, _ = _plant_s1(K.read_jsonl(os.path.join(root, "evals", f"{name}.jsonl")), pick, own)
        K.write_jsonl(str(tmp_path / "evals" / f"{name}.jsonl"), recs)
    out = check_k.run(str(tmp_path), cues=False)
    assert not out["pass"] and "k_eval_id: purity skill_train_token: 1" in out["fails"], out["fails"]   # T-2
    assert set(out["sets"]["k_eval_id"]["oracles"]) == set("RUBFAP"), out["sets"]["k_eval_id"]["oracles"].keys()
    assert "k_dev: purity skill_train_token: 1" in out["fails"], out["fails"]              # T-5
    assert out["sets"]["OOD-DFAR"]["purity"]["skill_train_token"] == 1, out["sets"]["OOD-DFAR"]["purity"]
    assert "OOD-DFAR: purity skill_train_token: 1" in out["fails"], out["fails"]           # T-3


def test_check_requires_b_groups_on_k_eval(tmp_path):
    """K3 round 4 (REVIEW 5b O-1): a K-EVAL B set must score 3n/16 cube groups (BD-26). One written without groups
    (every record a plain item, so purity finds no group and skips every group bar) fails gen.py check. K3 round 12
    (REVIEW 13b T-3): K-EVAL-S too, at its own split (mutant m13b_group_count_id_only: the rule on K-EVAL-ID only)."""
    import check_k
    import evalsets as E
    import pools as P
    pl = P.get()
    os.makedirs(tmp_path / "evals")
    for name, split, seed, tag in (("k_eval_id", "eval", 7201, "K-EVAL-ID"), ("k_eval_s", "evals", 7202, "K-EVAL-S")):
        recs = E.skill_set(pl, P.Lex(pl), split, seed, 32, fams="B", groups=False, tag=tag)
        with open(tmp_path / "evals" / f"{name}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)
    out = check_k.run(str(tmp_path), cues=False)
    assert not out["pass"] and "k_eval_id: B groups 0 of 32 records, not 3n/16" in out["fails"], out["fails"]
    assert "k_eval_s: B groups 0 of 32 records, not 3n/16" in out["fails"], out["fails"]          # 13b T-3
    for name in ("k_eval_id", "k_eval_s"):
        assert out["sets"][name]["fails"] == [] and out["sets"][name]["groups"] == {}, (name, out["sets"][name])


def test_check_fails_on_each_fact_leak(tmp_path):
    """K3 round 12 (REVIEW 13b T-1, T-2): a fact-stream leak must reach gen.py check's verdict, and three counters
    that test_fact_purity_catches_planted_leaks held only at 0 or never wrote must each count a plant of its own: one
    S1 token (a training name) in FH's filler bio doc, its QA stream absent (skill_or_heldout_token), a miss chat in
    L100 (miss_in_L100) and an L0 miss chat for entity N_LOW + 1, outside FB_low (non_low_name; a miss is L0's own
    line). Mutants: m13b_fact_bio_token_off, m13b_k5_miss_l100_off, m13b_k5_non_low_off (each counter off) and
    m13b_fact_fails_dropped (the counts never reach check.json's fails)."""
    import check_k
    import lookup as L
    import pools as P
    import world as Wd
    pl = P.get()
    w, own = Wd.make_world(pl), pl.owner()
    bio = Wd.bio_block(w, pl, np.array([K.N_LOW + 5, -1]), 0)
    assert not {own.get(int(t), "") for t in np.unique(bio)} & {
        "S1", "S2", "S3", "T1", "T2", "T3", "VH", "AH", "AL", "MK"}, "the bio block must be clean before the plant"
    bio[1, 5] = min(t for t, c in own.items() if c == "S1")
    os.makedirs(tmp_path / "FH" / "fact_bio")
    bio.tofile(str(tmp_path / "FH" / "fact_bio" / "x.bin"))
    for arm, e in (("L100", 3), ("L0", K.N_LOW + 1)):
        os.makedirs(tmp_path / arm / "fact_k5")
        K.write_jsonl(str(tmp_path / arm / "fact_k5" / "x.jsonl"),
                      [L.k5_chat(w.names[e], w.attrs, w.values[e], 0, 1, False, list(range(6)), arm)])
    rep = check_k.fact_purity(str(tmp_path), pl)
    assert rep["FH"]["skill_or_heldout_token"] == 1, rep["FH"]          # m13b_fact_bio_token_off
    assert rep["L100"]["miss_in_L100"] == 1, rep["L100"]                # m13b_k5_miss_l100_off
    assert rep["L0"]["non_low_name"] == 1, rep["L0"]                    # m13b_k5_non_low_off
    assert rep["L0"]["hit_in_L0"] == 0 and rep["L100"]["non_low_name"] == 0, rep
    assert gen.main(["check", str(tmp_path)]) == 1                      # m13b_fact_fails_dropped
    assert json.load(open(tmp_path / "check.json"))["fails"] == [
        "fact FH: skill_or_heldout_token 1", "fact L0: non_low_name 1", "fact L100: miss_in_L100 1"]


def test_check_reads_the_ood_p_question_range(tmp_path):
    """K3 round 12 (REVIEW 13b T-4): gen.py check holds OOD-PLONG's P items to 5-7 questions and every other OOD set's
    to SPEC 4's 2-4, both ways. 32 ordinary P items (2-4 questions) tagged OOD-PLONG and 32 PLONG items (5-7) tagged
    OOD-VALH each fail the count; under their own tags the same items pass. Mutants: m13c_ood_qrange_wide (every OOD
    set read against 2-7), plong_range_ignored (2-4 everywhere)."""
    import check_k
    import evalsets as E
    import pools as P
    pl = P.get()
    short = E.skill_set(pl, P.Lex(pl), "eval", 7203, 32, fams="P")
    long_ = E.skill_set(pl, P.Lex(pl), "eval", 7204, 32, fams="P", ood="PLONG")
    assert {len(r["qs"]) for r in short} == {2, 3, 4} and {len(r["qs"]) for r in long_} == {5, 6, 7}
    os.makedirs(tmp_path / "evals")
    for tags, want in ((("OOD-PLONG", "OOD-VALH"), ["OOD-PLONG", "OOD-VALH"]), (("OOD-VALH", "OOD-PLONG"), [])):
        K.write_jsonl(str(tmp_path / "evals" / "k_eval_ood.jsonl"),
                      [{**r, "set": t} for t, rs in zip(tags, (short, long_)) for r in rs])
        out = check_k.run(str(tmp_path), cues=False)
        assert out["fails"] == [f"{t}: P: 32 items with a question count outside SPEC 4's range" for t in want], (
            tags, out["fails"])


def test_check_fails_on_a_skill_token_in_fact_qa_and_k5(tmp_path):
    """K3 round 13 (REVIEW 14b T-5): the fact-purity counter skill_or_heldout_token has two readers that K3 round 12's
    planted FH bio doc does not reach: the fact QA records and the K5 chats. A clean FH QA stream (FB entities 0 and 1
    and a filler QA) and a clean L100 hit chat pass gen.py check; one S1 token (a training name) in place of the
    answer of the first QA record and of the hit chat fails it with one line each (mutants m14_fact_qa_token_off,
    m14_k5_token_off)."""
    import check_k
    import lookup as L
    import pools as P
    import world as Wd
    pl = P.get()
    w, own = Wd.make_world(pl), pl.owner()
    s1 = min(t for t, c in own.items() if c == "S1")
    forbidden = {"S1", "S2", "S3", "T1", "T2", "T3", "VH", "AH", "AL", "MK"}
    for planted in (False, True):
        root = tmp_path / ("planted" if planted else "clean")
        qa = Wd.qa_block(w, pl, np.array([0, 1, -1]), np.zeros(3, np.int8), 0, 0)
        k5 = [L.k5_chat(w.names[3], w.attrs, w.values[3], 0, 1, True, list(range(6)), "L100")]
        assert not {own.get(x, "") for r in qa + k5 for t in r["turns"] for x in t["ids"]} & forbidden
        assert k5[0]["turns"][0]["role"] == "tool" and K.NONE not in k5[0]["turns"][0]["ids"]     # a hit
        if planted:
            qa[0]["turns"][-1]["ids"][0] = s1
            k5[0]["turns"][-1]["ids"][0] = s1
        os.makedirs(root / "FH" / "fact_bio")          # fact_purity finds an arm by its fact_bio dir (no doc here)
        os.makedirs(root / "FH" / "fact_qa")
        K.write_jsonl(str(root / "FH" / "fact_qa" / "x.jsonl"), qa)
        os.makedirs(root / "L100" / "fact_k5")
        K.write_jsonl(str(root / "L100" / "fact_k5" / "x.jsonl"), k5)
        rep = check_k.fact_purity(str(root), pl)
        assert rep["FH"]["skill_or_heldout_token"] == rep["L100"]["skill_or_heldout_token"] == int(planted), rep
        assert gen.main(["check", str(root)]) == int(planted)
        assert json.load(open(root / "check.json"))["fails"] == ([
            "fact FH: skill_or_heldout_token 1", "fact L100: skill_or_heldout_token 1"] if planted else [])


def test_check_code_names_every_module_the_check_runs():
    """K3 round 16 (2026-11-22, REVIEW 17b; mutant r17_check_code_no_purity survived): check.json's code_sha256
    (REVIEW 16b N-1) covers check_k.CHECK_CODE, which must be every K module the check imports, read from the source:
    check_k's K-local imports and theirs, to closure (top-level or inside functions). It is exactly that set."""
    import ast
    import check_k
    local = {f[:-3] for f in os.listdir(K.HERE) if f.endswith(".py")}

    def deps(m):
        out = set()
        for n in ast.walk(ast.parse(open(os.path.join(K.HERE, m + ".py")).read())):
            if isinstance(n, ast.Import):
                out |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module:
                out.add(n.module.split(".")[0])
        return out & local
    seen, todo = set(), ["check_k"]
    while todo:
        m = todo.pop()
        if m not in seen:
            seen.add(m)
            todo += sorted(deps(m))
    assert sorted(check_k.CHECK_CODE) == sorted(m + ".py" for m in seen)


def test_spec_7_budget_constants():
    """K3 round 16 (2026-11-22, REVIEW 17b; mutant r17c_b_slots survived): SPEC 7's budget as kcommon holds it, which
    every stream length and FACT draw is written from: B = 7,630 x 32,768 slots (250.0M), the run configs' step count
    (kconfig) times 16 x 2048; the 3% write margin, FB exposures in the first 95%, the shares, the loads and E."""
    import kconfig as KC
    assert K.B_SLOTS == 7_630 * 32_768 == 250_019_840 == KC.STEPS * 16 * 2048
    assert (K.WRITE_MARGIN, K.FB_REGION) == (1.03, 0.95)
    assert K.SHARES == {"skill": 0.25, "fact": 0.50, "lang": 0.25}
    assert (K.N_HIGH, K.N_LOW, K.N_CAP, K.E_BIO, K.E_QA) == (62_000, 15_500, 124_000, 100, 10)
