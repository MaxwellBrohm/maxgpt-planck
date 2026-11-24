"""The fitted cue-model gate (K3 round 1, 2026-09-27): learners over cues.item_features and the acceptance rule.

Each learner scores every candidate and takes a softmax over the question's candidates (cross-entropy on the gold):
  logit   a linear score (conditional logit)
  mlp     a 2-layer MLP per candidate (interactions of one candidate's columns with the item's)
  set     the MLP's candidate embedding joined with the max and the mean of every candidate's embedding in the item
          (DeepSets), so a candidate can be compared with the others (for example the most repeated key)
Accuracy is LIK's: the argmax with tied credit. The training number is a 2-fold cross-fit (even items fit the odd
ones and back); eval sets are scored by the fit on all training items. A family's ceiling is the maximum over the
learners; it passes at <= BAR + 3 x sqrt(BAR (1 - BAR) / n) (SPEC 5 C5). A's " none" decision is also read on the
items whose asked name and attribute both occur (the half O9|10 cannot answer), see gate().
K3 round 2: run() also fits the F model (cues.f_item_features: the asked attribute's statements as candidates, the
asked name read only through a closed column list, REVIEW 3a GATE-F-BLIND) on the F items; F's ceiling is the
maximum of both models. Its fails read "cue model F (F model) ...". B groups pass at C5's tolerance for the set's own
number of groups (REVIEW 3a GATE-CODE: it was 256 pairs on every set).
K3 round 3 (REVIEW 4a S-1): B's reversed twin pairs are replaced by cube groups (skill.group: an item and 3 copies
asking the asked name's two-token neighbours), each scored as the product of its 4 credits, bar GROUP_BAR.
K3 round 5 (REVIEW 6a G-1): run() also fits each of R, U, B, F and P alone on up to FAM_CAP of its own items (the
same columns and learners), and a family's ceiling is the largest of its pooled, alone (and F) models'.
K3 round 6 (REVIEW 7a S-1): every column above is blind to the question's name; the attribute-blind side (T's class:
the name read, the attribute never) gets its own model, ta_run over cues.ta_item_features, per family of TA_FAMS alone,
with T's bar, its fails tagged "(TA)", its ceiling "ta_model"["ceiling"] beside the name-blind "ceiling".
K3 round 7 (REVIEW 8a S-1): A's attribute-blind side, ta_a_run: the TA columns with " none" as a last candidate on R,
U, P and A at their stream shares; A's reading credits O9|10's half and passes at A's bar ("ta_model"["a_model"],
its ceiling "ta_model"["ceiling"]["A"]). REVIEW 8b T-2: every ceiling is the largest learner's (tests watch it)."""
from __future__ import annotations

import numpy as np
import torch

import cues as C
import kcommon as K
import oracles as O

BAR = {"R": .65, "U": .65, "B": .65, "F": .65, "A": .50, "P": .65}      # the cheap-rule bar per family (SPEC 5)
GROUP_BAR = O.GROUP_TH                                                  # B cube groups (SPEC 5, K3 round 3)
CAP = 81_920        # training items fitted: a seeded uniform draw from the whole stream (the K3 samples' size)
LEARNERS = {"logit": dict(hidden=0, epochs=12, lr=2e-2), "mlp": dict(hidden=64, epochs=12, lr=3e-3),
            "set": dict(hidden=64, pool=True, epochs=12, lr=3e-3)}


class Choice(torch.nn.Module):
    def __init__(self, d: int, hidden: int = 0, pool: bool = False):
        super().__init__()
        self.hidden, self.pool = hidden, pool
        if not hidden:
            self.head = torch.nn.Linear(d, 1)
            return
        self.enc = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden),
                                       torch.nn.ReLU())
        self.head = torch.nn.Sequential(torch.nn.Linear(hidden * (3 if pool else 1), hidden), torch.nn.ReLU(),
                                        torch.nn.Linear(hidden, 1))

    def forward(self, x, qid, nq):
        if not self.hidden:
            return self.head(x).squeeze(-1)
        h = self.enc(x)
        if self.pool:
            idx = qid[:, None].expand(-1, h.shape[1])
            mx = torch.zeros(nq, h.shape[1]).scatter_reduce(0, idx, h, "amax", include_self=False)
            cnt = torch.zeros(nq).index_add(0, qid, torch.ones_like(qid, dtype=h.dtype))
            mean = torch.zeros(nq, h.shape[1]).index_add(0, qid, h) / cnt[:, None]
            h = torch.cat([h, mx[qid], mean[qid]], -1)
        return self.head(h).squeeze(-1)


def _pack(F: list[dict]):
    X = torch.from_numpy(np.concatenate([f["X"] for f in F]))
    sizes = torch.tensor([len(f["X"]) for f in F])
    qid = torch.repeat_interleave(torch.arange(len(F)), sizes)
    return X, qid, torch.cumsum(sizes, 0) - sizes, torch.tensor([f["gold"] for f in F]), len(F)


def _seg_max(s, qid, nq):
    return torch.full((nq,), -1e30).scatter_reduce(0, qid, s, "amax")


def fit(F: list[dict], hidden=0, pool=False, epochs=12, lr=1e-2, batch=1024, seed=0) -> Choice:
    torch.manual_seed(seed)
    model = Choice(F[0]["X"].shape[1], hidden, pool)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for idx in torch.randperm(len(F), generator=g).split(batch):
            X, qid, start, gold, nq = _pack([F[int(i)] for i in idx])
            s = model(X, qid, nq)
            m = _seg_max(s.detach(), qid, nq)
            lse = m + torch.log(torch.zeros(nq).index_add(0, qid, torch.exp(s - m[qid])))
            loss = (lse - s[start + gold]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


@torch.no_grad()
def predict(model: Choice, F: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """-> (credit per question: the argmax with ties sharing it; 1 where " none", the last candidate, is a top pick)."""
    hit, pnone = [], []
    for lo in range(0, len(F), 4096):
        X, qid, start, gold, nq = _pack(F[lo:lo + 4096])
        s = model(X, qid, nq)
        m = _seg_max(s, qid, nq)
        top = (s >= m[qid] - 1e-6).float()
        hit.append((top[start + gold] / torch.zeros(nq).index_add(0, qid, top)).numpy())
        last = torch.cumsum(torch.tensor([len(f["X"]) for f in F[lo:lo + 4096]]), 0) - 1
        pnone.append((top[last] > 0).float().numpy())
    return np.concatenate(hit), np.concatenate(pnone)


def _both(f) -> bool:
    return not f["cheap_absent"]


def gate(train: list[dict], evals: dict | None = None, learners=None, seed=0, tag: str = "") -> dict:
    """-> {"table": {fam: {learner: {set: acc}}}, "ceiling": {fam: {set: acc}}, "groups", "a_both", "fails", "n"}.
    a_both (A's none decision where O9|10 cannot answer): per set, the learner's rate of " none" on A items whose
    asked name and attribute both occur, and on every other family's items with the same property; their
    difference is the lift a cheap cue gives A (0 when " none" there is a prior, not a detection); reported."""
    learners = learners or LEARNERS
    evals = evals or {}
    tab, groups, a_both, n_groups = {}, {}, {}, {}
    fams = [f for f in "RUBFAP" if any(x["fam"] == f for x in train)]
    for name, cfg in learners.items():
        a, b = train[0::2], train[1::2]
        (ha, na), (hb, nb) = predict(fit(b, seed=seed, **cfg), a), predict(fit(a, seed=seed, **cfg), b)
        per = {"train": (a + b, np.concatenate([ha, hb]), np.concatenate([na, nb]))}
        full = fit(train, seed=seed, **cfg) if evals else None
        for en, E in evals.items():
            per[en] = (E, *predict(full, E))
        for sn, (Fs, h, pn) in per.items():
            for fam in fams:
                sel = [i for i, f in enumerate(Fs) if f["fam"] == fam]
                if sel:
                    tab.setdefault(fam, {}).setdefault(name, {})[sn] = round(float(h[sel].mean()), 4)
            B = [i for i, f in enumerate(Fs) if f["fam"] == "B"]
            gr = [B[j:j + 4] for j in range(len(B) - 3) if [Fs[i]["grp"] for i in B[j:j + 4]] == [0, 1, 2, 3]]
            if gr:
                groups.setdefault(name, {})[sn] = round(float(np.mean([np.prod(h[g]) for g in gr])), 4)
                n_groups[sn] = len(gr)
            ab = [i for i, f in enumerate(Fs) if f["fam"] == "A" and _both(f)]
            ob = [i for i, f in enumerate(Fs) if f["fam"] != "A" and _both(f)]
            if ab and ob:
                a_both.setdefault(name, {})[sn] = {"A": round(float(pn[ab].mean()), 4),
                                                   "other": round(float(pn[ob].mean()), 4), "n": len(ab)}
    ceil, fails, n = {}, [], {}
    for fam, row in tab.items():
        for sn in row[next(iter(row))]:
            c = max(row[m][sn] for m in row)
            ceil.setdefault(fam, {})[sn] = c
            n[f"{fam}.{sn}"] = sum(f["fam"] == fam for f in (train if sn == "train" else evals[sn]))
            if not O.rate_ok(c, BAR[fam], n[f"{fam}.{sn}"]):
                fails.append(f"cue model {fam}{tag} {sn}: {c:.3f} > {BAR[fam]}")
    for sn in sorted({s for v in groups.values() for s in v}):
        c = max(v[sn] for v in groups.values() if sn in v)
        if not O.rate_ok(c, GROUP_BAR, n_groups[sn]):
            fails.append(f"cue model B groups{tag} {sn}: {c:.3f} > {GROUP_BAR} (n {n_groups[sn]})")
    return {"table": tab, "ceiling": ceil, "groups": groups, "n_groups": n_groups, "a_both": a_both, "fails": fails,
            "n": n}


def draw(n: int, cap: int, salt: int = 88) -> np.ndarray:
    """The training items the gate fits: all of them, or a seeded uniform draw of `cap` (never the first items)."""
    if n <= cap:
        return np.arange(n)
    return np.sort(np.random.default_rng([K.W, salt]).choice(n, cap, replace=False))


FAM_CAP = 40_960    # K3 round 5 (REVIEW 6a G-1): items per family-alone fit (the pooled draw held 10,240 R per half)
FAM_MODELS = "RUBFP"    # not A: its gold is always " none", so alone it learns that; A's bar is the mix's (SPEC 5)
TA_FAMS, TA_BAR = "RUFP", .65   # K3 round 6 (REVIEW 7a S-1): the attribute-blind model, T's class and bar (SPEC 5 C4);
                                # not B (its asked attribute is every entity's) nor A (its gold is " none")


def ta_run(train_recs: list[dict], eval_recs: dict, lex, fam_cap: int = FAM_CAP, learners=None, seed=0) -> dict:
    """K3 round 6 (REVIEW 7a S-1): the attribute-blind fitted model (cues.ta_item_features: candidates the asked
    entity's keys, nothing read of the final question's attribute) per family of TA_FAMS alone, on up to fam_cap of
    its training items (a seeded draw), 2-fold cross-fitted, each eval set scored by the full fit. Accuracy counts
    every item of the family, an item whose gold is none of the candidates as missed; it passes at T's bar TA_BAR
    with C5's tolerance. -> {"table": {fam: {learner: {set: acc}}}, "ceiling", "fails", "n", "n_train", "features"}."""
    learners = learners or LEARNERS
    out = {"table": {}, "ceiling": {}, "fails": [], "n": {}, "n_train": {}, "features": []}
    for fi, fam in enumerate(TA_FAMS if fam_cap > 0 else ""):
        own = [r for r in train_recs if r["fam"] == fam]
        if not own:
            continue
        tr = [C.ta_item_features(own[i], lex) for i in draw(len(own), fam_cap, 890 + fi)]
        ev = {n: [C.ta_item_features(r, lex) for r in v if r["fam"] == fam] for n, v in eval_recs.items()}
        ev = {n: v for n, v in ev.items() if v}
        ok = lambda F: [f for f in F if f["gold"] >= 0]                          # noqa: E731
        out["n_train"][fam] = len(tr)
        out["features"] = out["features"] or next((f["names"] for f in tr if f["names"]), [])
        for name, cfg in learners.items():
            a, b = ok(tr[0::2]), ok(tr[1::2])
            hits = sum(float(predict(fit(x, seed=seed, **cfg), y)[0].sum()) for x, y in ((b, a), (a, b)) if x and y)
            row = {"train": round(hits / len(tr), 4)}
            full = fit(ok(tr), seed=seed, **cfg) if ev and ok(tr) else None
            for en, E in ev.items():
                row[en] = round(float(predict(full, ok(E))[0].sum()) / len(E), 4) if full and ok(E) else 0.0
            out["table"].setdefault(fam, {})[name] = row
        for sn in out["table"][fam][next(iter(learners))]:
            c = max(v[sn] for v in out["table"][fam].values())
            n = len(tr) if sn == "train" else len(ev[sn])
            out["ceiling"].setdefault(fam, {})[sn] = c
            out["n"][f"{fam}.{sn}"] = n
            if not O.rate_ok(c, TA_BAR, n):
                out["fails"].append(f"cue model {fam} (TA) {sn}: {c:.3f} > {TA_BAR}")
    return out


TA_A_FAMS = "RUPA"      # K3 round 7 (REVIEW 8a S-1): A's attribute-blind side, pooled with its contexts' families


def ta_a_run(train_recs: list[dict], eval_recs: dict, lex, cap: int = CAP, learners=None, seed=0) -> dict:
    """K3 round 7 (REVIEW 8a S-1): A's attribute-blind side. The TA model skips A (its gold is never a key), so a
    reader of the asked name's side that says " none" where A's both-items gather (A-ONEKEY3: 0.56-0.57 on A with a
    mix gain) was never fitted. Here cues.ta_item_features(none=True) (the TA columns, " none" as a last candidate
    carrying the asked name's side, _ta_side) on R, U, P and A at their stream shares (a seeded draw of up to cap
    of their training items), 2-fold cross-fitted, eval sets by the full fit. An A item counts right when the pick is
    " none" or O9|10's cue fires (the bar's own cue, which every A1H-class rule reads), so A reads 0.50 plus half
    the both-items' " none" rate; it passes at A's bar with C5's tolerance, tagged "(TA)". R, U and P are reported
    (their bars are the TA model's). -> {"table": {learner: {set: {fam: acc}}}, "ceiling": {set: A's}, "none":
    {learner: {set: {"both", "other"}}} (the " none" rate on A's both-items and on R, U and P), "fails", "n",
    "n_train", "features"}; {} without A items."""
    learners = learners or LEARNERS
    own = [r for r in train_recs if r["fam"] in TA_A_FAMS]
    if not any(r["fam"] == "A" for r in own):
        return {}
    tr = [C.ta_item_features(own[i], lex, none=True) for i in draw(len(own), cap, 896)]
    assert all(f["gold"] >= 0 for f in tr), "an R, U, P or A item whose gold no candidate holds"
    ev = {n: [C.ta_item_features(r, lex, none=True) for r in v if r["fam"] in TA_A_FAMS] for n, v in eval_recs.items()}
    ev = {n: v for n, v in ev.items() if any(f["fam"] == "A" for f in v)}
    assert all(f["gold"] >= 0 for v in ev.values() for f in v), "an eval item whose gold no candidate holds"
    out = {"table": {}, "ceiling": {}, "none": {}, "fails": [], "n": {}, "n_train": len(tr), "features": tr[0]["names"]}
    for name, cfg in learners.items():
        a, b = tr[0::2], tr[1::2]
        fa, fb = fit(b, seed=seed, **cfg), fit(a, seed=seed, **cfg)        # fa scores a, fb scores b (cross-fit)
        (ha, na), (hb, nb) = predict(fa, a), predict(fb, b)
        per = {"train": (a + b, np.concatenate([ha, hb]), np.concatenate([na, nb]))}
        full = fit(tr, seed=seed, **cfg) if ev else None
        for en, E in ev.items():
            per[en] = (E, *predict(full, E))
        for sn, (Fs, h, pn) in per.items():
            right = np.where([f["fam"] == "A" and f["cheap_absent"] for f in Fs], 1.0, h)
            row = {}
            for fam in TA_A_FAMS:
                sel = [i for i, f in enumerate(Fs) if f["fam"] == fam]
                row[fam] = round(float(right[sel].mean()), 4) if sel else None
                out["n"][f"{fam}.{sn}"] = len(sel)
            out["table"].setdefault(name, {})[sn] = row
            ab = [i for i, f in enumerate(Fs) if f["fam"] == "A" and not f["cheap_absent"]]
            ob = [i for i, f in enumerate(Fs) if f["fam"] != "A"]
            out["none"].setdefault(name, {})[sn] = {"both": round(float(pn[ab].mean()), 4) if ab else None,
                                                    "other": round(float(pn[ob].mean()), 4), "n_both": len(ab)}
    for sn in out["table"][next(iter(learners))]:
        c = max(v[sn]["A"] for v in out["table"].values())
        out["ceiling"][sn] = c
        if not O.rate_ok(c, BAR["A"], out["n"][f"A.{sn}"]):
            out["fails"].append(f"cue model A (TA) {sn}: {c:.3f} > {BAR['A']}")
    return out


def run(train_recs: list[dict], eval_recs: dict, lex, cap: int = CAP, learners=None, fam_cap: int = FAM_CAP) -> dict:
    """The gate on record lists (gen.py check): train_recs, the kept training stream (draw()); eval_recs {name:
    records}. The pooled model on every family, then the F model on the F items; the column lists and sizes go into
    the report, "ceiling"["F"] is the larger of the two models' per set, "f_model" holds the F model's report.
    K3 round 5 (REVIEW 6a G-1): then a family-alone model per family of FAM_MODELS, fitted on up to fam_cap of that
    family's training items (a seeded draw per family; 0 skips them), each eval set's items of that family scored by
    it ("fam_models"); a family's ceiling is the largest of its models'. The pooled fit is a finite-sample ceiling
    that missed REVIEW 6a's R cue at 10,240 R items per half (0.643-0.645), which the same columns and learners show
    on 40,960 R items alone (0.663-0.667)."""
    all_train = train_recs
    train_recs = [train_recs[i] for i in draw(len(train_recs), cap)]
    F = [C.item_features(r, lex) for r in train_recs]
    E = {n: [C.item_features(r, lex) for r in v] for n, v in eval_recs.items() if v}
    rep = gate(F, E, learners)
    out = {**rep, "features": F[0]["names"], "n_train": len(F), "bar": BAR, "group_bar": GROUP_BAR}
    del F
    out["fam_models"] = {}
    for fi, fam in enumerate(FAM_MODELS if fam_cap > 0 else ""):
        own = [r for r in all_train if r["fam"] == fam]
        if not own:
            continue
        Ff = [C.item_features(own[i], lex) for i in draw(len(own), fam_cap, 880 + fi)]
        rf = gate(Ff, {n: [e for e in v if e["fam"] == fam] for n, v in E.items()
                       if any(e["fam"] == fam for e in v)}, learners, tag=f" ({fam} alone)")
        out["fam_models"][fam] = {k: rf[k] for k in ("table", "ceiling", "groups", "n_groups", "n")} | {
            "n_train": len(Ff)}
        out["fails"] = out["fails"] + rf["fails"]
        for sn, c in rf["ceiling"][fam].items():
            out["ceiling"].setdefault(fam, {})[sn] = max(c, out["ceiling"].get(fam, {}).get(sn, 0.0))
        for name, v in rf["groups"].items():
            for sn, c in v.items():
                out["groups"].setdefault(f"{name} ({fam} alone)", {})[sn] = c
        del Ff
    FF = [C.f_item_features(r, lex) for r in train_recs if r["fam"] == "F"]
    if FF:
        EF = {n: [C.f_item_features(r, lex) for r in v if r["fam"] == "F"] for n, v in eval_recs.items()}
        rf = gate(FF, {n: v for n, v in EF.items() if v}, learners, tag=" (F model)")
        out["f_model"] = {**rf, "features": FF[0]["names"], "n_train": len(FF)}
        out["fails"] = out["fails"] + rf["fails"]
        for sn, c in rf["ceiling"]["F"].items():
            out["ceiling"].setdefault("F", {})[sn] = max(c, out["ceiling"].get("F", {}).get(sn, 0.0))
    out["ta_model"] = ta_run(all_train, eval_recs, lex, fam_cap, learners)     # K3 round 6 (REVIEW 7a S-1)
    out["fails"] = out["fails"] + out["ta_model"]["fails"]
    ta_a = ta_a_run(all_train, eval_recs, lex, cap, learners) if fam_cap > 0 else {}     # K3 round 7 (REVIEW 8a
    if ta_a:                                                                            # S-1): A's TA side
        out["ta_model"]["a_model"] = ta_a
        out["ta_model"]["ceiling"]["A"] = ta_a["ceiling"]
        out["fails"] = out["fails"] + ta_a["fails"]
    return out
