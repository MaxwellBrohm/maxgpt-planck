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


def test_gate_bars():
    assert G.BAR == {"R": .65, "U": .65, "B": .65, "F": .65, "A": .50, "P": .65} and G.GROUP_BAR == .125


@functools.lru_cache(maxsize=None)
def stream(fams, n, seed):
    """Training blocks at the stream's shares (skill.ROUND blocks per family), features."""
    g = S.Gen(pools(), "train")
    return tuple(C.item_features(I.render(it, f"{f}.b{b}.i{i}"), lex()) for f in fams for b in range(S.ROUND[f])
                 for i, it in enumerate(S.block(g, f, n, (97, ord(f), seed + b))))


def test_gate_passes_r_u_a_at_stream_shares(threads):
    """R, U and A at the stream's shares (5 : 5 : 2 blocks of 256), eval blocks of 256 scored by the training fit:
    the design passes and A stays at or under its absence half (0.50). A's both-items are cheap-indistinguishable
    from R and U items, so " none" is a prior there, and it loses only while values outnumber both-items: with one
    512-item block each of R, U and A (this test's first version, K3 round 1) the gate read A 0.963 (train) / 0.789
    (eval); 5 : 5 : 5 blocks of 256 read 0.791 / 0.520 (REVIEW 3b). With A's former both-items (one other entity
    stating the attribute) the gate read 0.98 on A, and with U's former counts 0.87 on U (code/notes.txt K3)."""
    ev = list(feats("RUA", "eval", n=256, seed=6))
    rep = G.gate(list(stream("RUA", 256, 50)), {"ev": ev})
    assert rep["fails"] == [], (rep["fails"], rep["ceiling"])
    assert max(rep["ceiling"]["A"].values()) <= 0.53 and max(rep["ceiling"]["U"].values()) < 0.65, rep["ceiling"]


F_ASKED = {"N", "O", "pre_intro", "pre_def", *[f"gap{i}" for i in range(6)], *[f"q_rank{i}" for i in range(5)],
           *[f"a_N{i}" for i in range(3)], *[f"a_O{i}" for i in range(3)],        # K3 round 3: the alias windows
           "y_xgap", *[f"y_xg{i}" for i in range(12)], "y_kwin", "y_owin", "y_owin_n", "y_near_is_k", "y_gaprank",
           "y_kdef", "y_kdef_ex"}


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
    assert n == 64 and F_ASKED <= set(a["names"]) and len(a["names"]) == 111


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
    """The F model's fails reach run()'s report (and gen.py check's): a planted F-model column marking the gold."""
    orig = C.f_item_features

    def planted(rec, lx):
        f = orig(rec, lx)
        col = np.zeros((len(f["X"]), 1), np.float32)
        col[f["gold"]] = 1
        return {**f, "X": np.hstack([f["X"], col]), "names": f["names"] + ["planted"]}
    monkeypatch.setattr(C, "f_item_features", planted)
    rep = G.run(list(recs("F", n=128, seed=6)), {"ev": list(recs("F", "eval", n=64, seed=7))}, lex(),
                learners={"logit": {**G.LEARNERS["logit"], "batch": 16, "epochs": 30}})
    assert rep["f_model"]["ceiling"]["F"]["train"] > 0.9 and rep["ceiling"]["F"]["train"] > 0.9, rep["ceiling"]
    assert any("cue model F (F model) train" in x for x in rep["fails"]), rep["fails"]


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
    the round-4 gate read REVIEW 5a's round-3 fresh sets at R 0.695-0.709, A 0.662-0.690, P 0.772-0.781.)"""
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
    assert lift > 0.05, (old["ceiling"], new["ceiling"])
