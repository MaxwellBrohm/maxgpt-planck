"""The fitted cue-model gate (K3 round 1, 2026-09-27): cues.item_features must be blind to the question's name, the
learners must find a planted cue and score a blind one at chance, and on real blocks the gate must pass the fixed
design while failing the designs it replaced (A's own both-items, U's asymmetric counts)."""
import copy
import functools

import numpy as np
import pytest
import torch

import cuegate as G
import cues as C
import items as I
import oracles as O
import skill as S
from kfix import lex, old_questions, pools

Q_COLS = {"name_absent", "same_q_before", "ans_same_q", "ans_same_q_cur", "ans_attr_other_q",   # read the question
          *[f"qn_asked{i}" for i in range(3)], "qn_last", *[f"qn_attrs{i}" for i in range(3)]}  # (round 4: its name
#                                                                                    among earlier questions' names)


@functools.lru_cache(maxsize=None)
def recs(fams="RUBFAP", split="train", n=512, seed=0):
    g = S.Gen(pools(), split)
    return tuple(I.render(it, f"{f}.b{seed}.i{i}") for f in fams
                 for i, it in enumerate(S.block(g, f, n, (97, ord(f), seed))))


@functools.lru_cache(maxsize=None)
def feats(fams="RUBFAP", split="train", n=512, seed=0):
    return tuple(C.item_features(r, lex()) for r in recs(fams, split, n, seed))


@pytest.fixture
def threads():
    k = torch.get_num_threads()
    torch.set_num_threads(min(k, 4))
    yield
    torch.set_num_threads(k)


def _ask_other(rec):
    """The same record with the final question naming another entity of the item (None if it has one entity)."""
    st, qs, intro = O.parse(rec, lex())
    q = qs[-1]
    other = [n for n in intro if n != q["name"]]
    if not other:
        return None
    out = copy.deepcopy(rec)
    ids = out["turns"][q["turn"] - 1]["ids"]
    lo = 0 if q["form"] == 0 else 1
    ids[lo:lo + 3] = list(other[0])
    return out


def test_features_blind_to_the_question_name():
    """No column but the listed question readers changes when the final question names another entity: no feature
    binds a name (R, U, B's skill) or resolves a reference (F's), so the cue model cannot either."""
    n = 0
    for rec in recs(n=64):
        alt = _ask_other(rec)
        if alt is None:
            continue
        a, b = C.item_features(rec, lex()), C.item_features(alt, lex())
        keep = [i for i, c in enumerate(a["names"]) if c not in Q_COLS]
        assert a["X"].shape == b["X"].shape and np.array_equal(a["X"][:, keep], b["X"][:, keep]), rec["id"]
        n += 1
    assert n > 300


def test_candidates_and_gold():
    for f, rec in zip(feats(n=64), recs(n=64)):
        st, qs, _ = O.parse(rec, lex())
        assert len(f["X"]) == qs[-1]["n"] + 1 and f["X"][-1, f["names"].index("is_none")] == 1
        assert (f["gold"] == len(f["X"]) - 1) == (rec["fam"] == "A")


def test_blind_scores_are_chance_with_tie_credit(threads):
    F = list(feats("R", n=64))
    m = G.Choice(F[0]["X"].shape[1])
    torch.nn.init.zeros_(m.head.weight)
    torch.nn.init.zeros_(m.head.bias)
    hit, _ = G.predict(m, F)
    assert np.allclose(hit, [1 / len(f["X"]) for f in F])


def test_planted_cue_is_found_and_fails_the_bar(threads):
    """A column marking the gold in 90% of items (else a random other candidate): the learners must read it and the
    gate must fail R; without it R passes."""
    r = np.random.default_rng(0)

    def plant(F):
        out = []
        for f in F:
            col = np.zeros((len(f["X"]), 1), np.float32)
            col[f["gold"] if r.random() < 0.9 else int(r.integers(0, len(f["X"])))] = 1
            out.append({**f, "X": np.hstack([f["X"], col])})
        return out
    tr, ev = list(feats("R", n=512, seed=1)), list(feats("R", "eval", n=512, seed=2))
    L = {k: {**G.LEARNERS[k], "batch": 64} for k in ("logit", "mlp")}  # 512 items at batch 1,024 get one step
    rep = G.gate(plant(tr), {"ev": plant(ev)}, L)       # an epoch (the gate: 80); K3 round 2's 149 columns need more
    assert min(rep["ceiling"]["R"].values()) > 0.75 and any("cue model R train" in x for x in rep["fails"]), rep
    assert G.gate(tr, {"ev": ev}, L)["fails"] == []


def test_training_number_is_cross_fitted(threads):
    """Golds moved to a random candidate: nothing predicts them, so the cross-fitted training number sits near
    chance (an in-sample fit memorizes: 0.6 or more here)."""
    r = np.random.default_rng(3)
    F = [{**f, "gold": int(r.integers(0, len(f["X"]))), "fam": "R"} for f in feats("RUBFAP", n=128, seed=4)]
    chance = float(np.mean([1 / len(f["X"]) for f in F]))
    rep = G.gate(F, None, {"mlp": dict(hidden=64, epochs=40, lr=1e-2)})
    assert rep["table"]["R"]["mlp"]["train"] < chance + 0.05, (rep["table"], chance)


def test_ta_a_training_number_is_cross_fitted(monkeypatch, threads):
    """K3 round 7: A's attribute-blind side (cuegate.ta_a_run) scores its training number by the 2-fold cross-fit too.
    R, U and P golds moved to a random candidate (" none" among them): nothing predicts them, so R's training reading
    sits near chance, while an in-sample fit memorizes them (mutant ta_a_in_sample)."""
    r = np.random.default_rng(3)
    orig = C.ta_item_features

    def moved(rec, lx, **kw):
        f = orig(rec, lx, **kw)
        return f if f["fam"] == "A" else {**f, "gold": int(r.integers(0, len(f["X"])))}
    monkeypatch.setattr(C, "ta_item_features", moved)
    rep = G.ta_a_run(list(recs("RUPA", n=128, seed=4)), {}, lex(), learners={"mlp": dict(hidden=64, epochs=40,
                                                                                          lr=1e-2)})
    feats_r = [orig(x, lex(), none=True) for x in recs("R", n=128, seed=4)]
    chance = float(np.mean([1 / len(f["X"]) for f in feats_r]))
    assert rep["table"]["mlp"]["train"]["R"] < chance + 0.08, (rep["table"], chance)


def test_gate_bars():
    assert G.BAR == {"R": .65, "U": .65, "B": .65, "F": .65, "A": .50, "P": .65} and G.GROUP_BAR == .125
    # K3 round 15 (REVIEW 16b T-4): the attribute-blind model's families and T's bar (SPEC 5 C4), unpinned before;
    # gen.py's sample reads F's K-PROBE TA cell 0.6623, between 0.65 and 0.66 (mutant r16_ta_bar_066)
    assert (G.TA_FAMS, G.TA_BAR, G.TA_A_FAMS, G.FAM_MODELS) == ("RUFP", .65, "RUPA", "RUBFP")


@functools.lru_cache(maxsize=None)
def stream(fams, n, seed):
    """Training blocks at the stream's shares (skill.ROUND blocks per family), features."""
    g = S.Gen(pools(), "train")
    return tuple(C.item_features(I.render(it, f"{f}.b{b}.i{i}"), lex()) for f in fams for b in range(S.ROUND[f])
                 for i, it in enumerate(S.block(g, f, n, (97, ord(f), seed + b))))


def test_gate_passes_r_u_a_at_stream_shares(threads):
    """R, U and A at the stream's shares (5 : 5 : 2 blocks of 512), eval blocks of 256 scored by the training fit:
    the design passes and A stays at or under its absence half (0.50). A's both-items are cheap-indistinguishable
    from R and U items, so " none" is a prior there, and it loses only while values outnumber both-items: with one
    512-item block each of R, U and A (this test's first version, K3 round 1) the gate read A 0.963 (train) / 0.789
    (eval); 5 : 5 : 5 blocks of 256 read 0.791 / 0.520 (REVIEW 3b). With A's former both-items (one other entity
    stating the attribute) the gate read 0.98 on A, and with U's former counts 0.87 on U (code/notes.txt K3). K3
    round 7: blocks of 512 (256 before). Round 7's both-items (fams.fam_A) read A 0.531-0.555 cross-fitted at 256 on 4
    seeds (round 6's 0.520-0.529), 0.492-0.516 at 512 and 0.501 on 81,920 fresh items (K3 ROUND 7)."""
    ev = list(feats("RUA", "eval", n=256, seed=6))
    rep = G.gate(list(stream("RUA", 512, 50)), {"ev": ev})
    assert rep["fails"] == [], (rep["fails"], rep["ceiling"])
    assert max(rep["ceiling"]["A"].values()) <= 0.53 and max(rep["ceiling"]["U"].values()) < 0.65, rep["ceiling"]


F_ASKED = {"N", "O", "pre_intro", "pre_def", *[f"gap{i}" for i in range(6)], *[f"q_rank{i}" for i in range(5)],
           *[f"a_N{i}" for i in range(3)], *[f"a_O{i}" for i in range(3)],        # K3 round 3: the alias windows
           "y_xgap", *[f"y_xg{i}" for i in range(12)], "y_kwin", "y_owin", "y_owin_n", "y_near_is_k", "y_gaprank",
           "y_kdef", "y_kdef_ex", *[f"q_nattr{i}" for i in range(4)]}       # K3 round 8: the asked name's attributes


def test_f_model_reads_the_asked_name_only_through_its_columns():
    """K3 round 2 (REVIEW 3a GATE-F-BLIND): the F model may read the asked name, but only as the N or O form of the
    attribute's named statements, the references' place against its first mention and definition, and its intro
    rank. Every other column is unchanged when the final question names another entity of the item; in particular
    no column says whether an asked-name statement shares a reference's turn or sits near it (REVIEW 3a: with those,
    ellipsis items read 0.92-1.00)."""
    n = 0
    for rec in recs("F", n=64):
        alt = _ask_other(rec)
        a, b = C.f_item_features(rec, lex()), C.f_item_features(alt, lex())
        keep = [i for i, c in enumerate(a["names"]) if c not in F_ASKED]
        assert a["names"] == b["names"] and np.array_equal(a["X"][:, keep], b["X"][:, keep]), rec["id"]
        n += 1
    assert n == 64 and F_ASKED <= set(a["names"]) and len(a["names"]) == 124


def test_f_model_reads_the_antecedent_as_a_count():
    """K3 round 8 (REVIEW 9a G-1): the F model carries F9-ANTESIZE's class: per candidate the statements and holders of
    the attribute of the statement just before it (pv_cnt, pv_hold), per item the asked name's other attributes
    (q_nattr). On test_reference's F-ANTESIZE item: the key's ellipsis follows a statement of a2 (stated once, by the
    key), the foil's one of a1 (stated twice, by 2 names), and the asked name states a1 and a2 (mutants f_pv_off,
    f_pvhold_off, f_qnattr_with_asked). K3 round 9 (REVIEW 10b T-1): q_nattr is the asked name's count, F-ANTESIZE's
    nma, not every name's: without the key's a1 the asked name states a2 only (1; the foil states a1), and with the
    foil's antecedent a3 the asked name states 2 of the item's 3 other attributes; on an F block it equals the asked
    name's count on every item, which differs from every name's on some (mutant f_qnattr_all_names). K3 round 10
    (REVIEW 11b T-5): pv_hold counts holders, not statements: with the foil stating a1 once more, a1 has 3 statements
    and 2 holders before the foil's ellipsis (mutant r11_pvhold_counts_statements)."""
    from test_reference import _antesize_item
    rec, _ = _antesize_item()
    f = C.f_item_features(rec, lex())
    col = lambda c: [int(x) for x in f["X"][:, f["names"].index(c)]]                     # noqa: E731
    assert col("kind_ell") == [0, 1, 1] and col("pv_cnt1") == [0, 1, 0] and col("pv_cnt2") == [0, 0, 1], f["X"]
    assert col("pv_hold1") == [0, 1, 0] and col("pv_hold2") == [0, 0, 1] and col("q_nattr2") == [1, 1, 1]
    rec, _ = _antesize_item(foil_pre=int(pools()["A"][1]))      # K3 round 10 (REVIEW 11b T-5): a1 3 statements,
    f = C.f_item_features(rec, lex())                           # 2 holders (the foil states it twice)
    assert col("kind_ell") == [0, 1, 1] and col("pv_cnt3") == [0, 0, 1] and col("pv_hold2") == [0, 0, 1], f["X"]
    for kw, want in (({"key_a1": False}, 1), ({"foil_ante": int(pools()["A"][3])}, 2)):
        f = C.f_item_features(_antesize_item(**kw)[0], lex())
        assert [int(x) for x in f["X"][:, f["names"].index(f"q_nattr{want}")]] == [1] * len(f["X"]), (kw, f["X"])
    differ = 0
    for rec in recs("F", n=64):
        st, qs, _ = O.parse(rec, lex())
        q = qs[-1]
        S_ = st[:q["n"]]
        nma = len({s["attr"] for s in S_ if s["name"] == q["name"] and s["attr"] != q["attr"]})
        every = len({s["attr"] for s in S_ if s["name"] is not None and s["attr"] != q["attr"]})
        f = C.f_item_features(rec, lex())
        assert f["X"][0, f["names"].index(f"q_nattr{min(nma, 3)}")] == 1, (rec["id"], nma)
        differ += min(nma, 3) != min(every, 3)
    assert differ > 0, differ


def test_run_draws_a_seeded_sample_not_the_first_items(monkeypatch):
    """REVIEW 3b T-2: above CAP the gate fits a seeded uniform draw of the training stream, never its first items."""
    seen = []
    orig = C.item_features

    def spy(rec, lx):
        seen.append(rec["id"])
        return orig(rec, lx)
    monkeypatch.setattr(C, "item_features", spy)
    tr = list(recs("RF", n=64, seed=3))
    ev = {"ev": list(recs("R", "eval", n=16, seed=4))}
    rep = G.run(tr, ev, lex(), cap=48, learners={"logit": dict(hidden=0, epochs=1, lr=2e-2)})
    got = seen[:48]
    assert rep["n_train"] == 48 and len(set(got)) == 48 and set(got) <= {r["id"] for r in tr}
    assert got != [r["id"] for r in tr[:48]] and {r["fam"] for r in tr if r["id"] in got} == {"R", "F"}
    assert list(G.draw(128, 48)) == list(G.draw(128, 48)) and list(G.draw(40, 48)) == list(range(40))
    assert "f_model" in rep and rep["f_model"]["n_train"] == sum(r["id"] in got and r["fam"] == "F" for r in tr)


def test_set_learner_finds_an_odd_one_out(threads):
    """REVIEW 3b T-2: SPEC 5 lists the DeepSets learner for cues that compare a candidate with the item's others. A
    planted comparative cue: every candidate's column holds one value and the gold's the other (which value is the
    gold's flips per item), so no single candidate's columns say anything. The set learner must find it; logit and
    the per-candidate MLP must not."""
    r = np.random.default_rng(5)
    F = []
    for _ in range(3000):
        n = int(r.integers(3, 7))
        g = int(r.integers(0, n))
        hi = r.random() < 0.5
        x = np.full((n, 1), 1.0 if hi else 0.0, np.float32)
        x[g] = 0.0 if hi else 1.0
        F.append({"fam": "R", "grp": -1, "gold": g, "cheap_absent": True,
                  "X": np.hstack([x, r.normal(size=(n, 3)).astype(np.float32)])})
    rep = G.gate(F[:2000], {"ev": F[2000:]}, {**G.LEARNERS, "set": {**G.LEARNERS["set"], "epochs": 20}})
    t = rep["table"]["R"]
    assert t["set"]["ev"] > 0.9 and max(t["logit"]["ev"], t["mlp"]["ev"]) < 0.45, t


def test_gate_catches_the_old_b_design(monkeypatch, threads):
    """REVIEW 3a B-CENTER: with the build design's B (the asked name the unique centre of 4) the NAMES columns take
    the gate over B's bars; the cube design passes (train blocks, an eval block with cube groups, where the gate,
    blind to the question's name, gives a group's 4 questions one answer: no group right)."""
    import fams as Fm
    from test_skill import _old_fam_b

    def b_items(split, nb, seed, groups=False):
        g = S.Gen(pools(), split)
        return [C.item_features(I.render(it, f"B.b{b}.i{i}"), lex()) for b in range(nb)
                for i, it in enumerate(S.block(g, "B", 256, (95, 66, seed + b), groups=groups))]
    L = {k: G.LEARNERS[k] for k in ("logit", "mlp")}
    new = G.gate(b_items("train", 3, 0), {"ev": b_items("eval", 1, 9, True)}, L)
    assert new["fails"] == [] and max(new["ceiling"]["B"].values()) < 0.35, new["ceiling"]
    assert new["n_groups"]["ev"] == 48 and max(v["ev"] for v in new["groups"].values()) <= 1 / 16, new["groups"]
    monkeypatch.setattr(Fm, "fam_B", _old_fam_b)
    old = G.gate(b_items("train", 3, 0), {"ev": b_items("eval", 1, 9)}, L)
    assert min(old["ceiling"]["B"].values()) > 0.8, old["ceiling"]
    assert any("cue model B train" in x for x in old["fails"]) and any("cue model B ev" in x for x in old["fails"])


def test_group_tolerance_uses_the_sets_own_groups():
    """REVIEW 3a GATE-CODE, K3 round 3: B groups (grp 0-3, adjacent) score the product of their 4 credits and pass at
    C5's tolerance for the set's number of groups. The same group rate 0.1875 passes at 48 groups (limit 0.268) and
    fails at 2,048 (0.147)."""
    def items(ngroups, hit):
        F = []
        for i in range(ngroups):
            for j in range(4):
                x = np.zeros((2, 1), np.float32)
                x[0 if (i < hit or j) else 1] = 1.0         # a missed group misses its first question only
                F.append({"fam": "B", "grp": j, "gold": 0, "cheap_absent": True, "X": x})
        return F
    L = {"logit": dict(hidden=0, epochs=30, lr=5e-2)}
    small = G.gate(items(512, 400), {"ev": items(48, 9)}, L)
    big = G.gate(items(512, 400), {"ev": items(2048, 384)}, L)
    assert small["n_groups"]["ev"] == 48 and big["n_groups"]["ev"] == 2048
    assert small["groups"]["logit"]["ev"] == big["groups"]["logit"]["ev"] == 0.1875
    assert not any("groups ev" in x for x in small["fails"]) and any("groups ev" in x for x in big["fails"])


def test_run_reports_f_model_fails(monkeypatch, threads):
    """The F model's fails reach run()'s report (and gen.py check's): a planted F-model column marking the gold. K3
    round 8: 60 epochs at lr 0.05 (30 at 0.02 before): with BD-48's F items and 124 columns the 64-item halves'
    cross-fit read 0.875 at the old setting (0.961 now; the eval set 1.000 at both). K3 round 9 (REVIEW 10b T-2): the
    plant reaches the eval set, so its "ev" fail must reach run()'s too (mutant f_model_ev_fails_off)."""
    orig = C.f_item_features

    def planted(rec, lx):
        f = orig(rec, lx)
        col = np.zeros((len(f["X"]), 1), np.float32)
        col[f["gold"]] = 1
        return {**f, "X": np.hstack([f["X"], col]), "names": f["names"] + ["planted"]}
    monkeypatch.setattr(C, "f_item_features", planted)
    rep = G.run(list(recs("F", n=128, seed=6)), {"ev": list(recs("F", "eval", n=64, seed=7))}, lex(),
                learners={"logit": {**G.LEARNERS["logit"], "batch": 16, "epochs": 60, "lr": 5e-2}})
    assert rep["f_model"]["ceiling"]["F"]["train"] > 0.9 and rep["ceiling"]["F"]["train"] > 0.9, rep["ceiling"]
    assert any("cue model F (F model) train" in x for x in rep["fails"]), rep["fails"]
    assert any(x.startswith("cue model F (F model) ev:") for x in rep["fails"]), rep["fails"]


def test_question_columns_read_the_final_name_among_earlier_questions():
    """K3 round 4 (REVIEW 5a S-1, ruled cheap in SPEC 5): the final name among earlier questions' names (times asked,
    the latest, distinct attributes asked of it), the distinct names asked, and the names stating nothing of the
    attribute that no earlier question named, on a hand-built item in the pre-round-4 style (the final name asked
    twice); asking another name moves only the final-name columns."""
    p = pools()
    s1, s2, s3 = p.triple("S")
    n = [(int(s1[i]), int(s2[i]), int(s3[i])) for i in range(3)]
    a0, a1, a2 = (int(x) for x in p["A"][:3])
    v = [int(x) for x in p["V"][:6]]
    St, Qn = I.St, I.Qn
    ex = [("S", [St("full", 0, a1, v[0]), St("full", 1, a0, v[1])]), ("Q", Qn(0, a1, 0)),
          ("S", [St("reord", 2, a2, v[2]), St("full", 0, a0, v[3])]), ("Q", Qn(2, a2, 1)),
          ("S", [St("full", 0, a2, v[4])]), ("Q", Qn(0, a2, 0)), ("Q", Qn(0, a0, 0))]
    f = C.item_features(I.render(I.Item("R", n, ex), "R.b0.i0"), lex())
    col = dict(zip(f["names"], f["X"][-1]))
    assert col["qn_asked2"] == col["qn_last"] == col["qn_attrs2"] == col["q_names2"] == col["q_free_unasked0"] == 1
    ex[-1] = ("Q", Qn(1, a0, 0))
    g = dict(zip(f["names"], C.item_features(I.render(I.Item("R", n, ex), "R.b0.i0"), lex())["X"][-1]))
    assert g["qn_asked0"] == g["qn_attrs0"] == 1 and g["qn_last"] == 0 and g["q_names2"] == g["q_free_unasked0"] == 1


def test_gate_catches_the_old_question_draw(monkeypatch, threads):
    """K3 round 4 (REVIEW 5a S-1): with earlier questions drawn from every stated key (the pre-round-4 draw) the gate's
    question columns read the final name's size and A's asked entity: R, U, A, P blocks of 512 at the stream's shares
    (the mlp at batch 128) fail A and read R and P higher than on the bystander draw, which passes. (At gen.py's size
    the round-4 gate read REVIEW 5a's round-3 fresh sets at R 0.695-0.709, A 0.662-0.690, P 0.772-0.781.) K3 round 7:
    with A's round-7 both-items in the pooled mix the R and P lift here is 0.040 (R +0.003, P +0.037; round 6's A
    0.062), so the bound is 0.03 (0.05 before); the old draw still fails A (0.595 / 0.676)."""
    L = {"mlp": {**G.LEARNERS["mlp"], "batch": 128}}

    def run():
        g, ge = S.Gen(pools(), "train"), S.Gen(pools(), "eval")
        tr = [C.item_features(I.render(it, f"{f}.b{b}.i{i}"), lex()) for f in "RUAP" for b in range(S.ROUND[f])
              for i, it in enumerate(S.block(g, f, 512, (97, ord(f), 60 + b)))]
        ev = [C.item_features(I.render(it, f"{f}.b0.i{i}"), lex()) for f in "RUAP"
              for i, it in enumerate(S.block(ge, f, 512, (97, ord(f), 70)))]
        return G.gate(tr, {"ev": ev}, L)
    new = run()
    monkeypatch.setattr(S, "_questions", old_questions)
    old = run()
    assert new["fails"] == [] and any("cue model A" in x for x in old["fails"]), (new["fails"], old["fails"])
    lift = sum(old["ceiling"][f]["train"] - new["ceiling"][f]["train"] for f in "RP")
    assert lift > 0.03, (old["ceiling"], new["ceiling"])


def test_family_alone_models_join_the_ceiling(monkeypatch, threads):
    """K3 round 5 (REVIEW 6a G-1): the pooled fit is a finite-sample ceiling over a mix, and it missed REVIEW 6a's R cue,
    which the same columns read on 40,960 R items alone. run() fits each of R, U, B, F and P alone (never A: its gold
    is always " none") on up to fam_cap of its own items, a seeded draw, and a family's ceiling is the largest of its
    models'. Planted: a column marking the gold in 90% of R items and every candidate but the gold in U items, which
    outnumber R's 2 to 1, so the pooled weight takes U's sign; R alone reads it, fails with its tag, lifts R's ceiling.
    K3 round 8 (REVIEW 9b T-2): A's attribute-blind side's fails reach run()'s too. A column planted in
    ta_item_features(none=True), marking " none" on 9 of 10 A items (REVIEW 9b demo_m2.py: A's TA side 0.961, failed),
    must put "cue model A (TA) train" in run()'s fails (mutant ta_a_fails_dropped: the same ceiling, no fail).
    K3 round 9 (REVIEW 10b T-2): the eval set holds 64 A items too, which the plant marks as it does training's, so
    A's TA side must fail there as well (mutant ta_a_eval_fails_off), and R alone on the eval set's R items
    (mutant fam_alone_ev_fails_off)."""
    r = np.random.default_rng(11)
    orig, orig_ta = C.item_features, C.ta_item_features

    def planted_ta(rec, lx, none=False):
        f = orig_ta(rec, lx, none=none)
        if not none:
            return f
        col = np.zeros((len(f["X"]), 1), np.float32)
        if rec["fam"] == "A":
            col[f["gold"] if r.random() < 0.9 else int(r.integers(0, len(f["X"])))] = 1
        return {**f, "X": np.hstack([f["X"], col]), "names": f["names"] + ["planted"]}
    monkeypatch.setattr(C, "ta_item_features", planted_ta)

    def planted(rec, lx):
        f = orig(rec, lx)
        col = np.zeros((len(f["X"]), 1), np.float32)
        hit = r.random() < 0.9
        if rec["fam"] == "R":
            col[f["gold"] if hit else int(r.integers(0, len(f["X"])))] = 1
        elif rec["fam"] == "U":
            col[:] = 1
            col[f["gold"]] = 0
        return {**f, "X": np.hstack([f["X"], col]), "names": f["names"] + ["planted"]}
    monkeypatch.setattr(C, "item_features", planted)
    tr = list(recs("RA", n=256, seed=8)) + list(recs("U", n=512, seed=8))
    L = {"logit": {**G.LEARNERS["logit"], "batch": 32, "epochs": 20}}
    ev = list(recs("RU", "eval", n=64, seed=9)) + list(recs("A", "eval", n=64, seed=9))
    rep = G.run(tr, {"ev": ev}, lex(), cap=400, learners=L, fam_cap=200)
    fm = rep["fam_models"]      # REVIEW 7b T-2: the pooled draw (cap 400: about 102 R and 198 U) is not where the
    assert set(fm) == {"R", "U"} and fm["R"]["n_train"] == fm["U"]["n_train"] == 200, (     # family-alone fits draw
        fm.keys(), {f: v["n_train"] for f, v in fm.items()})        # from (mutant fam_alone_from_pooled_draw)
    assert rep["n_train"] == 400 and rep["ta_model"]["n_train"] == {"R": 200, "U": 200}, rep["ta_model"]["n_train"]
    assert rep["ta_model"]["a_model"]["n_train"] == 400 and "A" in rep["ta_model"]["ceiling"], (     # K3 round 7:
        rep["ta_model"].get("a_model", {}).get("n_train"), rep["ta_model"]["ceiling"].keys())      # A's TA side
    pooled, alone = rep["table"]["R"]["logit"]["train"], fm["R"]["ceiling"]["R"]["train"]
    assert alone > 0.78 and alone > pooled + 0.3 and rep["ceiling"]["R"]["train"] == alone, (pooled, alone)
    assert _ceilings_join(rep) == [], rep["ceiling"]       # K3 round 15 (REVIEW 16b T-1)
    assert any("cue model R (R alone) train" in x for x in rep["fails"]), rep["fails"]
    assert rep["ta_model"]["ceiling"]["A"]["train"] > 0.8, rep["ta_model"]["ceiling"]["A"]      # K3 round 8 (T-2)
    assert any(x.startswith("cue model A (TA) train") for x in rep["fails"]), rep["fails"]
    assert any(x.startswith("cue model A (TA) ev:") for x in rep["fails"]), rep["fails"]       # K3 round 9 (T-2)
    assert any(x.startswith("cue model R (R alone) ev:") for x in rep["fails"]), rep["fails"]
    assert list(G.draw(256, 200, 880)) != list(G.draw(256, 200, 881)) == list(G.draw(256, 200, 881))
    off = G.run(tr, {"ev": ev}, lex(), learners=L, fam_cap=0)
    assert off["fam_models"] == {} and not any("alone" in x for x in off["fails"]) and off["ta_model"]["n_train"] == {}
    assert "a_model" not in off["ta_model"]


def _ceilings_join(rep) -> list:
    """K3 round 15 (REVIEW 16b T-1): the cells of run()'s report whose ceiling is not the largest of the family's
    pooled learners and its family-alone model's (and, for F, the F model's)."""
    bad = []
    for fam, row in rep["ceiling"].items():
        for sn, c in row.items():
            parts = [max(v[sn] for v in rep["table"][fam].values() if sn in v)]
            parts += [rep["fam_models"][fam]["ceiling"][fam][sn]] if sn in rep["fam_models"].get(fam, {}).get(
                "ceiling", {}).get(fam, {}) else []
            parts += [rep["f_model"]["ceiling"]["F"][sn]] if fam == "F" and "f_model" in rep else []
            bad += [] if c == max(parts) else [(fam, sn, c, parts)]
    return bad


def test_a_family_alone_model_below_the_pooled_one_leaves_its_ceiling(monkeypatch, threads):
    """K3 round 15 (REVIEW 16b T-1): a family's ceiling is the LARGEST of its pooled and family-alone models' (K3 round
    5), so an alone model that reads less than the pooled one leaves the pooled ceiling standing. Only planted alone
    > pooled was tested (above), and a run() whose alone ceiling replaced the family's passed every test (mutant
    r16_fam_ceiling_replaces), which on gen.py's sample lowers 13 SPEC 9 ceiling cells (P K-PROBE 0.5196 to 0.4804).
    Here every family-alone fit is read as 0 (gate wrapped on its " alone" tag), so each R and U cell must keep the
    pooled model's ceiling, and every cell is the largest of its parts."""
    real = G.gate

    def low(F, E=None, learners=None, seed=0, tag=""):
        out = real(F, E, learners, seed, tag)
        if "alone" in tag:
            out["table"] = {f: {m: {sn: 0.0 for sn in v} for m, v in row.items()} for f, row in out["table"].items()}
            out["ceiling"] = {f: {sn: 0.0 for sn in v} for f, v in out["ceiling"].items()}
        return out
    monkeypatch.setattr(G, "gate", low)
    L = {"logit": {**G.LEARNERS["logit"], "batch": 32, "epochs": 4}}
    rep = G.run(list(recs("RU", n=128, seed=8)), {"ev": list(recs("RU", "eval", n=32, seed=9))}, lex(), cap=200,
                learners=L, fam_cap=100)
    assert set(rep["fam_models"]) == {"R", "U"}, rep["fam_models"].keys()
    for fam in "RU":
        for sn in ("train", "ev"):
            pooled = rep["table"][fam]["logit"][sn]
            assert rep["fam_models"][fam]["ceiling"][fam][sn] == 0.0 < pooled, (fam, sn, pooled)
            assert rep["ceiling"][fam][sn] == pooled, (fam, sn, pooled, rep["ceiling"][fam][sn])
    assert _ceilings_join(rep) == [], rep["ceiling"]


def _swap_final_attr(rec, a2):
    """The same record with the final question's attribute token replaced by a2."""
    st, qs, _ = O.parse(rec, lex())
    out = copy.deepcopy(rec)
    ids = out["turns"][qs[-1]["turn"] - 1]["ids"]
    ids[3 if qs[-1]["form"] == 0 else 0] = a2
    return out


def test_ta_features_blind_to_the_question_attribute():
    """K3 round 6 (REVIEW 7a's blind_check.py): the attribute-blind model's columns never read the final question's
    attribute. Swapped for every other attribute token, every TA row and the gold index stay; IDEAL (the control,
    which reads it) moves on every variant (REVIEW 7a's O4 does on R, U and P; on F it is -1 in 3/4 of items)."""
    n, moved = 0, 0
    attrs = [int(x) for x in pools()["A"]]
    for rec in recs("RUFP", n=32):
        a = C.ta_item_features(rec, lex())
        st, qs, _ = O.parse(rec, lex())
        o4 = O.answers(st, qs[-1], O.prompt_of(rec, qs[-1]["turn"]), lex())["IDEAL"]
        for a2 in attrs:
            if a2 == qs[-1]["attr"]:
                continue
            alt = _swap_final_attr(rec, a2)
            b = C.ta_item_features(alt, lex())
            assert a["gold"] == b["gold"] and (a["X"] is None) == (b["X"] is None), rec["id"]
            assert a["X"] is None or np.array_equal(a["X"], b["X"]), rec["id"]
            st2, qs2, _ = O.parse(alt, lex())
            moved += O.answers(st2, qs2[-1], O.prompt_of(alt, qs2[-1]["turn"]), lex())["IDEAL"] != o4
            n += 1
    assert n == 128 * 5 and moved == n, (n, moved)


def test_ta_features_read_u_lure_as_a_copy():
    """K3 round 6 final: U's lure asks the final key itself earlier (SPEC 4), so an attribute-blind reader may copy
    that earlier question's attribute (the rule TA-QN read U 0.51-0.52 on fresh sets): the TA model carries it as
    qn_at, an earlier question naming the final name asked this key, as the name-blind side carries ans_same_q. On U
    items whose final name was asked earlier, qn_at marks the gold's row alone (no other earlier question names the
    final entity in U, skill._questions); elsewhere it is 0 (mutant ta_qn_at_off)."""
    n_lure = 0
    for rec in recs("U", n=512, seed=3):
        f = C.ta_item_features(rec, lex())
        if f["X"] is None:
            continue
        col = f["X"][:, f["names"].index("qn_at")]
        st, qs, _ = O.parse(rec, lex())
        if any(p["name"] == qs[-1]["name"] for p in qs[:-1]):
            n_lure += 1
            assert f["gold"] >= 0 and col[f["gold"]] == 1 and col.sum() == 1, rec["id"]
        else:
            assert col.sum() == 0, rec["id"]
    assert n_lure > 30, n_lure


def _xor(f, r):
    """REVIEW 8b demo_mut.py's plant: two random bits per candidate that agree everywhere but on the gold, so no one
    column says anything and a linear score reads neither (logit stays near chance); a per-candidate MLP reads both."""
    p = (r.random(len(f["X"])) < 0.5).astype(np.float32)
    q = p.copy()
    q[f["gold"]] = 1.0 - q[f["gold"]]
    return {**f, "X": np.hstack([f["X"], p[:, None], q[:, None]]), "names": f["names"] + ["p", "q"]}


def test_ceilings_are_the_best_learners_reading(monkeypatch, threads):
    """K3 round 7 (REVIEW 8b T-2): a family's ceiling is the largest of the learners' readings, in the pooled gate and
    in the TA model. With the XOR plant on 4 R blocks (eval 1, the gate's 3 learners at batch 64) logit reads near 1/n
    and mlp and set 1.000: both models fail R on every set and their ceilings are that largest reading (mutants
    ta_ceiling_first_learner and gate_ceiling_first_learner read logit's, under the bar, and fail nothing)."""
    g, ge = S.Gen(pools(), "train"), S.Gen(pools(), "eval")
    tr = [I.render(it, f"R.b{b}.i{i}") for b in range(4)
          for i, it in enumerate(S.block(g, "R", 512, (98, ord("R"), b)))]
    ev = [I.render(it, f"R.b0.i{i}") for i, it in enumerate(S.block(ge, "R", 512, (98, ord("R"), 50)))]
    L = {k: {**v, "batch": 64} for k, v in G.LEARNERS.items()}
    r = np.random.default_rng(5)
    rep = G.gate([_xor(C.item_features(x, lex()), r) for x in tr], {"ev": [_xor(C.item_features(x, lex()), r)
                                                                          for x in ev]}, L)
    orig = C.ta_item_features
    monkeypatch.setattr(C, "ta_item_features", lambda rec, lx, **kw: _xor(orig(rec, lx, **kw), r))
    ta = G.ta_run(tr, {"ev": ev}, lex(), learners=L)
    for name, t, c, tag in (("gate", rep["table"]["R"], rep["ceiling"]["R"], ""), ("TA", ta["table"]["R"],
                                                                                  ta["ceiling"]["R"], " (TA)")):
        assert max(t["logit"].values()) < 0.5 and min(min(t[m].values()) for m in ("mlp", "set")) > 0.95, (name, t)
        assert c == {sn: max(t[m][sn] for m in t) for sn in ("train", "ev")}, (name, c, t)
        fails = rep["fails"] if name == "gate" else ta["fails"]
        assert {f"cue model R{tag} train", f"cue model R{tag} ev"} <= {x.split(":")[0] for x in fails}, fails


def test_ta_a_model_reads_a_from_the_asked_names_side(monkeypatch, threads):
    """K3 round 7 (REVIEW 8a S-1): the TA model skips A (its gold is never a key), so a reader of the asked name's side
    that says " none" where A's both-items gather was never fitted (A-ONEKEY3: A 0.56-0.57 with a mix gain). ta_a_run
    fits the TA columns with " none" as a last candidate on R, U, P and A at the stream's shares; an A item counts
    right on " none" or O9|10's cue, so A reads 0.50 plus half the both-items' " none" rate. On 20 R and U and 8 P and
    A blocks (eval 2 each) round 7's both-items pass (0.504 / 0.505 here) and round 6's draw fails (0.547 cross-fitted
    here at round 7's 3/8 R contexts; 0.549-0.580 on 57,344 fresh items at its own 1/4)."""
    import fams as Fm
    import oldfams
    nb = {"R": 20, "U": 20, "P": 8, "A": 8}

    def items():
        g, ge = S.Gen(pools(), "train"), S.Gen(pools(), "eval")
        tr = [I.render(it, f"{f}.b{b}.i{i}") for f in "RUPA" for b in range(nb[f])
              for i, it in enumerate(S.block(g, f, 512, (93, ord(f), b)))]
        ev = [I.render(it, f"{f}.b{b}.i{i}") for f in "RUPA" for b in range(2)
              for i, it in enumerate(S.block(ge, f, 512, (93, ord(f), 60 + b)))]
        return G.ta_a_run(tr, {"ev": ev}, lex())
    new = items()
    assert new["fails"] == [] and new["n_train"] == 28_672 and new["n"]["A.train"] == 4096, (new["ceiling"],
                                                                                            new["fails"])
    for m, t in new["table"].items():
        for sn, row in t.items():
            assert abs(row["A"] - (0.5 + 0.5 * new["none"][m][sn]["both"])) < 1e-3, (m, sn, row, new["none"][m][sn])
    assert new["ceiling"] == {sn: max(t[sn]["A"] for t in new["table"].values()) for sn in ("train", "ev")}
    monkeypatch.setattr(Fm, "fam_A", oldfams.fam_A_r6)
    old = items()
    assert any(x.startswith("cue model A (TA) train") for x in old["fails"]), old["ceiling"]


def test_ta_features_join_an_ellipsis():
    """K3 round 7 (REVIEW 8b T-1): ta_item_features takes the asked entity's keys as T does, by name or by ellipsis
    adjacency. On test_skill's _ell_item the candidates are a1, a0 (the ellipsis, the gold) and a2, in that order,
    the ellipsis flagged on a0 alone and a0's 3 holders counted. On an F block the gold is a candidate exactly where
    the key's latest statement is named (the late quarter) or an ellipsis (an alias gold is no key of T's): with the
    keys by name only (mutant ta_ellipsis_dropped) F's TA ceiling read 0.24 against 0.62 (REVIEW 8b, 8 blocks)."""
    from test_skill import _ell_item
    rec, v = _ell_item()
    f = C.ta_item_features(rec, lex())
    col = lambda c: list(f["X"][:, f["names"].index(c)])                   # noqa: E731
    assert f["gold"] == 1 and len(f["X"]) == 3 and col("ell") == [0, 1, 0] and col("share2") == [0, 1, 0], (
        f["gold"], None if f["X"] is None else len(f["X"]))
    n = {"late": 0, "ell": 0, "alias": 0}
    for r in recs("F", n=512, seed=4):
        m = r["meta"]
        kind = "late" if m["klate"] else m["kref"]
        n[kind] += 1
        assert (C.ta_item_features(r, lex())["gold"] >= 0) == (kind != "alias"), (r["id"], kind)
    assert n["late"] == 128 and n["ell"] + n["alias"] == 384 and min(n["ell"], n["alias"]) > 150, n


def test_ta_model_fails_round5_and_passes_round6(monkeypatch, threads):
    """K3 round 6 (REVIEW 7a S-1): cuegate.run fits the attribute-blind model (cues.ta_item_features) per family of R,
    U, F and P alone with T's bar, its fails tagged "(TA)". On round 6's R it reads near 1/n of the asked entity's
    keys (0.38-0.42 here), on U that plus its lure's copy (qn_at: 0.49-0.55 here), and F passes with its alias golds
    (outside the candidates) counted as misses; on round 5's designs (tests/oldfams.py) it finds the asked key by its
    holders (R) and its statement count (U) and fails both, through run()'s report. The family-alone draw: K3 round
    6's probe ta_from_pooled_draw is test_family_alone_models_join_the_ceiling's (its n_train)."""
    import oldfams
    L = {"logit": {**G.LEARNERS["logit"], "batch": 64}}

    def items():
        g, ge = S.Gen(pools(), "train"), S.Gen(pools(), "eval")
        tr = [I.render(it, f"{f}.b{b}.i{i}") for f in "RUF" for b in range(2)
              for i, it in enumerate(S.block(g, f, 512, (99, ord(f), b)))]
        ev = [I.render(it, f"{f}.b0.i{i}") for f in "RUF" for i, it in enumerate(S.block(ge, f, 512, (99, ord(f), 9)))]
        return tr, ev
    tr, ev = items()
    new = G.run(tr, {"ev": ev}, lex(), learners=L, fam_cap=1024)
    ta = new["ta_model"]
    assert set(ta["ceiling"]) == set("RUF") and ta["n_train"] == {f: 1024 for f in "RUF"}, ta["n_train"]
    assert not any("(TA)" in x for x in new["fails"]) and max(ta["ceiling"]["R"].values()) < 0.48 and max(
        ta["ceiling"]["U"].values()) < 0.62, ta["ceiling"]
    oldfams.use(monkeypatch, "RU")
    tr, ev = items()
    old = G.run(tr, {"ev": ev}, lex(), learners=L, fam_cap=1024)
    assert any("cue model R (TA) train" in x for x in old["fails"]), old["ta_model"]["ceiling"]
    assert any("cue model U (TA) ev" in x for x in old["fails"]), old["ta_model"]["ceiling"]


def test_ta_model_fits_p(monkeypatch, threads):
    """K3 round 10 (REVIEW 11b T-1): SPEC 5 and BD-43 put T's bar on P's attribute-blind model too, but no test ran
    ta_run on P (R, U, F; R, U and A's side). REVIEW 11b's tap_probe.py plant: a column in ta_item_features marking
    the gold key on 9 of 10 P items (else a random key). ta_run fits R, U, F and P alone, P's TA side reads it and
    fails on training and the eval set, R, U and F pass (mutant m11_ta_fams_no_p: TA_FAMS "RUF", no P row, no fail)."""
    r = np.random.default_rng(31)
    orig = C.ta_item_features

    def planted(rec, lx, none=False):
        f = orig(rec, lx, none=none)
        if none or f["X"] is None:
            return f
        col = np.zeros((len(f["X"]), 1), np.float32)
        if rec["fam"] == "P" and f["gold"] >= 0:
            col[f["gold"] if r.random() < 0.9 else int(r.integers(0, len(f["X"])))] = 1
        return {**f, "X": np.hstack([f["X"], col]), "names": f["names"] + ["planted"]}
    monkeypatch.setattr(C, "ta_item_features", planted)
    tr, ev = list(recs("RUFP", n=256, seed=12)), list(recs("RUFP", "eval", n=64, seed=13))
    ta = G.ta_run(tr, {"ev": ev}, lex(), learners={"logit": {**G.LEARNERS["logit"], "batch": 32, "epochs": 20}})
    assert sorted(x.split(":")[0] for x in ta["fails"]) == ["cue model P (TA) ev", "cue model P (TA) train"], (
        ta["fails"], ta["ceiling"])
    assert ta["n_train"] == {f: 256 for f in "RUFP"} and min(ta["ceiling"]["P"].values()) > 0.8, ta["ceiling"]


def _a_side_item():
    """K3 round 10 (REVIEW 11b T-5): an A both-item by hand. N0 states a1 twice and a2, N1 a1 and a0, N2 a0 and a2;
    the final question asks N0 a0 (gold " none"). Every attribute has two named holders and no reference statement."""
    p = pools()
    s1, s2, s3 = p.triple("S")
    nm = [(int(s1[i]), int(s2[i]), int(s3[i])) for i in range(3)]
    a0, a1, a2 = (int(x) for x in p["A"][:3])
    v = [int(x) for x in p["V"][:7]]
    St = I.St
    sts = [St("full", 0, a1, v[0]), St("full", 1, a1, v[1]), St("full", 1, a0, v[2]), St("full", 2, a0, v[3]),
           St("full", 0, a2, v[4]), St("full", 2, a2, v[5]), St("reord", 0, a1, v[6])]
    return I.render(I.Item("A", nm, [("S", [x]) for x in sts] + [("Q", I.Qn(0, a0, 0))]), "A.b0.i0")


def test_ta_a_side_reads_the_asked_names_side():
    """K3 round 10 (REVIEW 11b T-5): A's attribute-blind side carries the asked name's side as _ta_side's 23 columns
    on every row (cuegate.ta_a_run), which no test read. On _a_side_item: the item's 3 attributes (aq2), N0's 3
    statements (e_n2), the 1 attribute it lacks (e_miss1), U's structure (ulike: two named holders each), its least
    and most stated keys 1 and 2 (e_min0, e_max1); candidates a1, a2 and " none" (the gold), an A both-item (mutant
    r11_ta_side_blank: every side column 0)."""
    f = C.ta_item_features(_a_side_item(), lex(), none=True)
    side = [c for c in f["names"] if c.startswith(("aq", "e_n", "e_miss", "ulike", "e_min", "e_max"))]
    on = {"aq2", "e_n2", "e_miss1", "ulike", "e_min0", "e_max1"}
    assert len(side) == 23 and f["gold"] == 2 and len(f["X"]) == 3 and not f["cheap_absent"], (side, f["gold"])
    got = [{c for c in side if f["X"][j, f["names"].index(c)] == 1} for j in range(3)]
    assert got == [on] * 3 and all(f["X"][j, f["names"].index(c)] == 0 for j in range(3) for c in side
                                   if c not in on), got


def test_ta_vrank_on_a_handmade_item():
    """K3 round 11 (REVIEW 12b T-1): vrank, the rank of a key's latest value id among the latest values of the asked
    entity's keys (0 the smallest, 1 the largest), is the TA model's only reader of a value-order cue, and no test read
    it. By hand (an R item): N0 states a1, a0 and a2, N1 a0 and a2, N0 a1 again; the question asks N0 a1. Candidates
    a1, a0, a2 (first statement order), latest values v6, v0, v3 of 7 ascending ids: vrank 1, 0, 1/2, the gold a1's
    row first (mutant r12f_ta_vrank_off: vrank all 0)."""
    p = pools()
    s1, s2, s3 = p.triple("S")
    nm = [(int(s1[i]), int(s2[i]), int(s3[i])) for i in range(2)]
    a0, a1, a2 = (int(x) for x in p["A"][:3])
    v = sorted(int(x) for x in p["V"][:7])
    St = I.St
    sts = [St("full", 0, a1, v[2]), St("full", 1, a0, v[4]), St("full", 0, a0, v[0]), St("full", 0, a2, v[3]),
           St("full", 1, a2, v[5]), St("reord", 0, a1, v[6])]
    rec = I.render(I.Item("R", nm, [("S", [x]) for x in sts] + [("Q", I.Qn(0, a1, 0))]), "R.b0.i0")
    f = C.ta_item_features(rec, lex())
    assert f["gold"] == 0 and len(f["X"]) == 3, (f["gold"], None if f["X"] is None else len(f["X"]))
    assert list(f["X"][:, f["names"].index("vrank")]) == [1.0, 0.0, 0.5], list(f["X"][:, f["names"].index("vrank")])


def _value_order_plant(rec, r):
    """REVIEW 12b's vrank_plant.py: on 9 of 10 items whose asked entity has 2+ keys (T's join), the gold value id and
    the smallest latest value of the entity's keys swap everywhere in the item (a relabeling: the world rule, purity
    and every rule blind to value ids are unchanged), so the gold is the smallest of the keys' latest values."""
    st, qs, _ = O.parse(rec, lex())
    q = qs[-1]
    S_ = st[:q["n"]]
    E = [i for i, s in enumerate(S_) if s["tent"] == q["name"]]
    X = []
    for i in E:
        X += [S_[i]["attr"]] if S_[i]["attr"] not in X else []
    if len(X) < 2 or r.random() >= 0.9:
        return rec
    vals = [S_[[i for i in E if S_[i]["attr"] == x][-1]]["val"] for x in X]
    g, v = q["gold"], min(vals)
    if g not in vals or g == v:
        return rec
    out = copy.deepcopy(rec)
    for t in out["turns"]:
        t["ids"] = [v if x == g else g if x == v else x for x in t["ids"]]
    return out


def test_ta_model_reads_a_value_order_plant(threads):
    """K3 round 11 (REVIEW 12b T-1): a planted value-order cue (_value_order_plant) on R, U and P blocks of 256 (eval
    64 each) fails the attribute-blind model on each family, train and eval, through vrank (REVIEW 12b at 4,096 per
    family: TA 0.93-0.95 with it; without it 0.40-0.54 and no fail, every other model under C5)."""
    r = np.random.default_rng(1411)
    tr = [_value_order_plant(x, r) for x in recs("RUP", n=256, seed=14)]
    ev = [_value_order_plant(x, r) for x in recs("RUP", "eval", n=64, seed=15)]
    ta = G.ta_run(tr, {"ev": ev}, lex(), learners={"logit": {**G.LEARNERS["logit"], "batch": 32, "epochs": 20}})
    assert sorted(x.split(":")[0] for x in ta["fails"]) == sorted(f"cue model {f} (TA) {s}" for f in "RUP"
                                                                  for s in ("train", "ev")), (ta["fails"], ta["ceiling"])



def test_ta_osize_on_a_handmade_item():
    """G-2 (2026-10-11, REVIEW 12b D-1): osize, SPEC 5's "the holders' mean size" (the statement counts of the key's
    attribute's other holders, an alias statement's own holder left out, / 6), read 0 on every item until G-2 (the
    code dropped every holder, a name being a tuple as a ("ref", i) marker is). By hand (an R item): N0 states a1 and
    a0, N1 a1, a2 and a3, the question asks N0 a1. Key a1's other holder N1 states 3: osize 0.5; key a0 has no other
    holder: 0 (mutants m12g_ta_osize_off, g2_osize_dead)."""
    p = pools()
    s1, s2, s3 = p.triple("S")
    nm = [(int(s1[i]), int(s2[i]), int(s3[i])) for i in range(2)]
    a0, a1, a2, a3 = (int(x) for x in p["A"][:4])
    v = sorted(int(x) for x in p["V"][:5])
    St = I.St
    sts = [St("full", 0, a1, v[0]), St("full", 1, a1, v[1]), St("full", 1, a2, v[2]), St("full", 0, a0, v[3]),
           St("full", 1, a3, v[4])]
    rec = I.render(I.Item("R", nm, [("S", [x]) for x in sts] + [("Q", I.Qn(0, a1, 0))]), "R.b0.i0")
    f = C.ta_item_features(rec, lex())
    assert f["gold"] == 0 and len(f["X"]) == 2, f["gold"]
    assert list(f["X"][:, f["names"].index("osize")]) == [0.5, 0.0], list(f["X"][:, f["names"].index("osize")])


def _osize_items(n, seed, tag):
    """G-2's osize plant: R-shaped items in which the asked entity N0 states a0 and a1, N1 holds a0 and N2 a1, and
    the holder of the asked key also states two attributes N0 does not (size 3 against 1). Every other TA column is
    alike for the two keys (2 holders, 2 statements, one by N0; the six statements in a uniform order), so only osize
    tells the asked key."""
    p, r = pools(), np.random.default_rng([7, 9300, seed])
    s1, s2, s3 = p.triple("S")
    A, V = [int(x) for x in p["A"]], [int(x) for x in p["V"]]
    out = []
    for i in range(n):
        ix = r.choice(len(s1), 3, replace=False)
        nm = [(int(s1[j]), int(s2[j]), int(s3[j])) for j in ix]
        a = [A[j] for j in r.permutation(len(A))[:4]]
        v = [V[j] for j in r.choice(len(V), 6, replace=False)]
        big = int(r.integers(0, 2))
        sts = [I.St("full", 0, a[0], v[0]), I.St("full", 0, a[1], v[1]), I.St("full", 1, a[0], v[2]),
               I.St("full", 2, a[1], v[3]), I.St("full", 1 + big, a[2], v[4]), I.St("full", 1 + big, a[3], v[5])]
        sts = [sts[j] for j in r.permutation(len(sts))]
        out.append(I.render(I.Item("R", nm, [("S", [x]) for x in sts] + [("Q", I.Qn(0, a[big], 0))]),
                            f"R.b{tag}.i{i}"))
    return out


def test_ta_model_reads_an_osize_plant(threads):
    """G-2 (REVIEW 12b D-1): on _osize_items (256 train, 64 eval) the attribute-blind model fails R, train and eval,
    through osize alone (k3 scratch g2/osize_items.py: logit 0.867 / 1.000; with osize 0, 0.578 / 0.469 and no fail).
    REVIEW 12b's osize_fix.py found no such cue on the real draws (TA R 0.414 / 0.436, U 0.537 / 0.553, P 0.412 /
    0.449, train / K-EVAL-ID); gen.py check reads the live column on every set since G-2."""
    ta = G.ta_run(_osize_items(256, 1, 0), {"ev": _osize_items(64, 2, 1)}, lex(),
                  learners={"logit": {**G.LEARNERS["logit"], "batch": 32, "epochs": 20}})
    assert sorted(x.split(":")[0] for x in ta["fails"]) == ["cue model R (TA) ev", "cue model R (TA) train"], (
        ta["fails"], ta["ceiling"])

def test_pooled_vrank_on_a_handmade_f_item():
    """K3 round 13 (REVIEW 14b T-1): the pooled model's vrank, a statement's value rank among the asked attribute's
    statement values (0 the smallest, 1 the largest; 0 off the asked attribute), is the only reader of a value-order
    cue on F (the TA model skips F, the F model has no vrank, no oracle reads value ids on F). By hand (an F item):
    N0 = X, then N0 a0 v1; an ellipsis a1 v3 (N0's); N1 a1 v0 and the alias X a1 v6; N1 a2 v4; the question asks N0 a1
    (gold v6, the alias statement). Values v0 < v1 < ... < v6: vrank 0, 1/2, 0, 1, 0 and 0 on the none row (mutant
    m14_pooled_vrank_off: vrank all 0)."""
    p = pools()
    s1, s2, s3 = p.triple("S")
    nm = [(int(s1[i]), int(s2[i]), int(s3[i])) for i in range(2)]
    a0, a1, a2 = (int(x) for x in p["A"][:3])
    x = int(p["AL"][0])
    v = sorted(int(t) for t in p["V"][:7])
    St = I.St
    ex = [("S", [St("def", 0, alias=x), St("full", 0, a0, v[1])]), ("S", [St("ell", 0, a1, v[3])]),
          ("S", [St("full", 1, a1, v[0]), St("alias", 0, a1, v[6], x)]), ("S", [St("full", 1, a2, v[4])]),
          ("Q", I.Qn(0, a1, 0))]
    rec = I.render(I.Item("F", nm, ex), "F.b0.i0")
    st, qs, _ = O.parse(rec, lex())
    assert [s["kind"] for s in st[:qs[-1]["n"]]] == ["full", "ell", "full", "alias", "full"] and qs[-1]["gold"] == v[6]
    f = C.item_features(rec, lex())
    assert f["gold"] == 3 and len(f["X"]) == 6, (f["gold"], len(f["X"]))
    assert list(f["X"][:, f["names"].index("vrank")]) == [0.0, 0.5, 0.0, 1.0, 0.0, 0.0], list(
        f["X"][:, f["names"].index("vrank")])


def _attr_value_order_plant(rec, r):
    """REVIEW 14b's vrank_pooled_plant.py: on 9 of 10 items whose asked attribute has 2+ statements, the gold value id
    and the smallest value of the asked attribute's statements swap everywhere in the item (a relabeling: the world
    rule, purity and every rule blind to value ids are unchanged), so the gold is the smallest of them."""
    st, qs, _ = O.parse(rec, lex())
    q = qs[-1]
    vals = [s["val"] for s in st[:q["n"]] if s["attr"] == q["attr"]]
    if len(vals) < 2 or r.random() >= 0.9:
        return rec
    g, v = q["gold"], min(vals)
    if g not in vals or g == v:
        return rec
    out = copy.deepcopy(rec)
    for t in out["turns"]:
        t["ids"] = [v if x == g else g if x == v else x for x in t["ids"]]
    for qq in out.get("qs", []):
        qq["gold"] = v if qq["gold"] == g else g if qq["gold"] == v else qq["gold"]
    return out


def test_pooled_features_read_a_value_order_plant_on_f(threads):
    """K3 round 13 (REVIEW 14b T-1): the attribute value-order plant on F (1,024 train, 256 eval) fails the F-alone
    model (cuegate.run's family-alone fit on the pooled columns), train and eval, through vrank. REVIEW 14b on 8,192
    planted F among the stream's other families: pooled and F alone 0.960 / 0.953 with it, 0.629 / 0.668 without and no
    F fail; the F model 0.622 / 0.631 either way. Here 0.953 / 0.961 real, 0.617 / 0.641 with vrank 0 (k3 scratch
    r13/sizes.py)."""
    r = np.random.default_rng(1313)
    tr = [_attr_value_order_plant(x, r) for x in recs("F", n=512, seed=21) + recs("F", n=512, seed=22)]
    ev = [_attr_value_order_plant(x, r) for x in recs("F", "eval", n=256, seed=23)]
    F = [C.item_features(x, lex()) for x in tr]
    out = G.gate(F, {"ev": [C.item_features(x, lex()) for x in ev]},
                 {"logit": {**G.LEARNERS["logit"], "batch": 32, "epochs": 20}}, tag=" (F alone)")
    assert sorted(x.split(":")[0] for x in out["fails"]) == ["cue model F (F alone) ev", "cue model F (F alone) train"
                                                             ], (out["fails"], out["ceiling"])


DESIGN_CONSTANT = {     # K3 round 15 (REVIEW 16b T-2): the only columns SPEC 4's items never move, each by the design
    "pooled": {"ans_same_q_cur", "ans_attr_other_q", "k0", "nref1", "rep_max0", "rep_keys1", "qn_attrs2", "q_names3",
               "is_none"},
    "F": {"dist7", "next_ell", "pre_intro", "pre_def", "y_xg0", "y_xg11", "q_rank4", "k0", "k4", "q_nattr0",
          "q_nattr3", "nst4", "npq3", "a_N2", "a_O2"},
    "TA": {"share4", "k0", "k5", "e_rank5"},
    "TA-A": {"share4", "ell", "k5", "is_none", "e_rank5"}}


def test_every_cue_column_is_live():
    """K3 round 15 (REVIEW 16b T-2): a column that reads one value on every item makes the gate blind to its cue, as
    osize was from its introduction until G-2, through every review. Over three training blocks of 512 per family
    (seeds 0, 2 and 6: seed 6's F block holds y_xg10, an alias 10 exchanges after the asked definition, 5 of 13,824 F
    items in gen.py's sample; seed 2's holds cent_rank3) every column of the four models takes two or more values but
    DESIGN_CONSTANT's, which is also exactly the constant set of gen.py's own sample (k3/r15, 1,500 items per family).
    The pooled and TA-A models are read on their key rows only: their " none" row is 0 in every key column, so a
    column stuck at 1 would still vary with it in (mutant r15_pool_is_first_one); there is_none is constant by
    construction, and TA-A's e_rank5 (the asked name never introduced) occurs only on A items, which then hold no key.
    Why each is constant: pooled: no earlier question asks the final attribute but U's lure, which names the final
    entity and is followed by a restatement (ans_attr_other_q, ans_same_q_cur, qn_attrs2: SPEC 4), earlier questions
    name at most two entities (q_names3), every item names an entity (k0, rep_max0), reference statements are F's
    alone and F holds 2-4 (nref1), and a U key is stated as every other key of its item, so U and A's U contexts
    repeat 2 or more keys and the rest none (rep_keys1). F: d <= 12 (dist7); no ellipsis's antecedent states the asked
    attribute (next_ell; 0 of 14,722 in the sample); BD-21 puts every reference after both first mentions and
    definitions (pre_intro, pre_def); aliases sit in the definitions' 1-10 windows (y_xg0, y_xg11); F's asked name is
    introduced (q_rank4), F holds 2-4 entities (k0, k4), at most 14 statements (nst4), 1-3 questions (npq3), one or
    two other attributes of the asked name (q_nattr0, q_nattr3) and at most one named statement of the attribute per
    side (a_N2, a_O2). TA: 2-5 entities (k0, k5), at most 4 holders (share4), the asked name introduced (e_rank5);
    TA-A as TA, with no ellipsis outside F (ell). Mutants: one column per model read as 0 (r16_ta_e_span_dead,
    r16_ta_eq2_dead, r16_ta_after_qn_dead, r16_ta_x_last_stmt_dead, r16_ta_e_first_dead, r16_pool_is_pen_dead,
    r16_pool_own_sup_dead, r16_pool_ans_other_attr_dead, r16_f_y_kdef_ex_dead)."""
    its = {f: [x for sd in (0, 2, 6) for x in recs(f, n=512, seed=sd)] for f in "RUBFAP"}
    models = {"pooled": (lambda r: C.item_features(r, lex()), "RUBFAP"),
              "F": (lambda r: C.f_item_features(r, lex()), "F"),
              "TA": (lambda r: C.ta_item_features(r, lex()), "RUFP"),
              "TA-A": (lambda r: C.ta_item_features(r, lex(), none=True), "RUPA")}
    dead = {}
    for m, (fn, fams) in models.items():
        F = [f for f in (fn(r) for fam in fams for r in its[fam]) if f["X"] is not None and len(f["X"])]
        X = np.concatenate([f["X"] for f in F])
        assert all(f["names"] == F[0]["names"] for f in F) and X.shape[1] == len(F[0]["names"]), m
        if "is_none" in F[0]["names"]:
            X = X[X[:, F[0]["names"].index("is_none")] == 0]
        dead[m] = {c for j, c in enumerate(F[0]["names"]) if bool((X[:, j] == X[0, j]).all())}
    assert dead == DESIGN_CONSTANT, {m: {"dead": sorted(dead[m] - DESIGN_CONSTANT[m]),
                                         "live": sorted(DESIGN_CONSTANT[m] - dead[m])} for m in dead}


def test_b_groups_fail_on_any_learner_not_the_first(monkeypatch):
    """K3 round 16 (2026-11-22, REVIEW 17b; mutant r17_group_ceiling_first_learner survived): the B cube-group bar
    reads the largest group accuracy over the learners, as every family cell does. Planted with stand-in learners on
    an eval set of 16 cube groups: the first reads every question wrong, the second every one right; the group fail
    must come from the second (only the first read, the groups pass)."""
    E = [{"fam": "B", "grp": j % 4, "cheap_absent": True} for j in range(64)]
    T = [{"fam": "B", "grp": -1, "cheap_absent": True} for _ in range(64)]
    monkeypatch.setattr(G, "fit", lambda F, seed=0, **cfg: cfg["tag"])
    monkeypatch.setattr(G, "predict", lambda m, F: (np.full(len(F), 1.0 if m == "second" else 0.0), np.zeros(len(F))))
    r = G.gate(T, {"ev": E}, learners={"first": {"tag": "first"}, "second": {"tag": "second"}})
    assert r["n_groups"]["ev"] == 16 and r["groups"] == {"first": {"ev": 0.0}, "second": {"ev": 1.0}}
    assert [x for x in r["fails"] if "groups" in x] == ["cue model B groups ev: 1.000 > 0.125 (n 16)"], r["fails"]
