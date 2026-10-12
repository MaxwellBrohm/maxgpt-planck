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
asking the asked name's two-token neighbours), each scored as the product of its 4 credits, bar GROUP_BAR."""
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
            fails.append(f"cue model B groups {sn}: {c:.3f} > {GROUP_BAR} (n {n_groups[sn]})")
    return {"table": tab, "ceiling": ceil, "groups": groups, "n_groups": n_groups, "a_both": a_both, "fails": fails,
            "n": n}


def draw(n: int, cap: int) -> np.ndarray:
    """The training items the gate fits: all of them, or a seeded uniform draw of `cap` (never the first items)."""
    if n <= cap:
        return np.arange(n)
    return np.sort(np.random.default_rng([K.W, 88]).choice(n, cap, replace=False))


def run(train_recs: list[dict], eval_recs: dict, lex, cap: int = CAP, learners=None) -> dict:
    """The gate on record lists (gen.py check): train_recs, the kept training stream (draw()); eval_recs {name:
    records}. The pooled model on every family, then the F model on the F items; the column lists and sizes go into
    the report, "ceiling"["F"] is the larger of the two models' per set, "f_model" holds the F model's report."""
    train_recs = [train_recs[i] for i in draw(len(train_recs), cap)]
    F = [C.item_features(r, lex) for r in train_recs]
    E = {n: [C.item_features(r, lex) for r in v] for n, v in eval_recs.items() if v}
    rep = gate(F, E, learners)
    out = {**rep, "features": F[0]["names"], "n_train": len(F), "bar": BAR, "group_bar": GROUP_BAR}
    FF = [C.f_item_features(r, lex) for r in train_recs if r["fam"] == "F"]
    if FF:
        EF = {n: [C.f_item_features(r, lex) for r in v if r["fam"] == "F"] for n, v in eval_recs.items()}
        rf = gate(FF, {n: v for n, v in EF.items() if v}, learners, tag=" (F model)")
        out["f_model"] = {**rf, "features": FF[0]["names"], "n_train": len(FF)}
        out["fails"] = out["fails"] + rf["fails"]
        for sn, c in rf["ceiling"]["F"].items():
            out["ceiling"].setdefault("F", {})[sn] = max(c, out["ceiling"].get("F", {}).get(sn, 0.0))
    return out
