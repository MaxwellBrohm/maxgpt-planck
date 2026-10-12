"""Graders for E002 (all likelihood read-outs; no generation).

An item is RIGHT when the gold candidate's summed log-prob is strictly greater than every other
candidate's, and every score is a finite number. Ties, missing golds and NaNs fail, so an "empty
answer" (a model that cannot tell the candidates apart) scores 0 and a plausible wrong answer
(another in-context value above the gold) scores 0.

Pass rule (REPORT 7.4, LEDGER P-001), fixed before any run, per seed, plain render:
  LW10  latest-wins at d10 on held-out wording = E001 same_k1 + same_k2 + same_k3 at d = 10,
        both item families (n = 192)
  TS10  two-slot at d10  (E001 twoslot, n = 64)
  NU10  no-update at d10 (E001 noupd,   n = 64)
  seed passes  <=>  LW10 >= 0.8 AND TS10 >= 0.8 AND NU10 >= 0.8
  model passes <=>  a majority of its seeds pass ("most seeds")
Lock-in rate = passing seeds / seeds. The in-training probe gives each seed's lock-in step.
"""
import math, random
from collections import defaultdict

THRESH = 0.8
LW_VARS = ("same_k1", "same_k2", "same_k3")


def right(scores):
    if not scores or "gold" not in scores or len(scores) < 2:
        return False
    vals = list(scores.values())
    if any(v is None or not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v) for v in vals):
        return False
    g = scores["gold"]
    return all(g > v for k, v in scores.items() if k != "gold")


def margin(scores):
    """gold minus the best other candidate (nats); None if not computable."""
    try:
        return scores["gold"] - max(v for k, v in scores.items() if k != "gold")
    except (KeyError, ValueError, TypeError):
        return None


def acc(flags):
    n = len(flags)
    if n == 0:
        return None
    p = sum(1 for f in flags if f) / n
    return {"acc": round(p, 4), "n": n, "se": round(math.sqrt(max(p * (1 - p), 1e-12) / n), 4)}


def cell(recs, render, vars_, d, fam=None):
    xs = [right(r["scores"]) for r in recs
          if r.get("render") == render and r.get("var") in vars_ and r.get("d") == d and (fam is None or r.get("fam") == fam)]
    return acc(xs)


def primary(new_recs, render="plain"):
    """-> dict(LW10, TS10, NU10, pass) from E001-item records of one run."""
    lw = cell(new_recs, render, LW_VARS, 10)
    ts = cell(new_recs, render, ("twoslot",), 10)
    nu = cell(new_recs, render, ("noupd",), 10)
    ok = all(c is not None and c["acc"] >= THRESH for c in (lw, ts, nu))
    return {"LW10": lw, "TS10": ts, "NU10": nu, "pass": ok}


def model_verdict(seed_passes):
    """seed_passes: list of bools. 'most seeds' = strict majority."""
    n = len(seed_passes)
    k = sum(1 for p in seed_passes if p)
    return {"seeds": n, "passing": k, "lock_in_rate": round(k / n, 3) if n else None, "pass": n > 0 and k > n / 2}


def lock_in_step(traj, keys=("LW", "TS", "NU")):
    """traj: list of dicts {step, LW, TS, NU} (accuracies) in step order. Returns the first step from
    which every key stays >= THRESH through the end of training, or None."""
    for i, t in enumerate(traj):
        if all(all(u.get(k) is not None and u[k] >= THRESH for k in keys) for u in traj[i:]):
            return t["step"]
    return None


def detail_new(new_recs):
    """Per render x variant x d accuracy (families pooled and split) and mean margins."""
    cells, marg, picks = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(int))
    for r in new_recs:
        ok = right(r["scores"])
        m = margin(r["scores"])
        for fam in (r["fam"], "all"):
            key = (r["render"], r["var"], fam, r["d"])
            cells[key].append(ok)
            if m is not None:
                marg[key].append(m)
            best = max(r["scores"].items(), key=lambda kv: kv[1])[0]
            picks[key][best] += 1
    out = {}
    for k, xs in cells.items():
        a = acc(xs)
        a["margin"] = round(sum(marg[k]) / len(marg[k]), 3) if marg[k] else None
        a["picks"] = dict(picks[k])
        out["|".join(map(str, k))] = a
    # paired all5 per scenario (E001 definition)
    by = defaultdict(dict)
    for r in new_recs:
        if r["var"] in ("same_k1", "noupd", "twoslot", "keyorig", "keycorr"):
            by[(r["render"], r["sid"])][r["var"]] = right(r["scores"])
    allc = defaultdict(list)
    for (render, sid), v in by.items():
        if len(v) == 5:
            fam, d, _ = sid.split("|")
            allc[(render, "PAIR_all5", "all", int(d))].append(all(v.values()))
    for k, xs in allc.items():
        out["|".join(map(str, k))] = acc(xs)
    return out


def old_summary(old_recs):
    """REPORT 3.3 columns from the old battery records (plain)."""
    def ok(r):
        return right(r["scores"])
    cells = defaultdict(list)
    for r in old_recs:
        if r["task"].startswith("U_k") and r["d"] in (4, 10):
            cells["latest_wins_d4-10"].append(ok(r))
        if r["task"].startswith("U_k") and r["d"] == 10:
            cells["latest_wins_d10"].append(ok(r))
        if r["task"] == "K_closedbook":
            cells["K_closed"].append(ok(r))
    rs = old_recs
    for a_, b_ in zip(rs, rs[1:]):
        if a_["task"] == "R1_owner" and a_["cond"] == "mine" and b_["task"] == "R1_owner" and b_["cond"] == "sister" and a_["d"] == 10:
            cells["owner_pair_d10"].append(ok(a_) and ok(b_))
        if a_["task"] == "P_myname" and b_["task"] == "P_yourname" and a_["d"] == 10:
            cells["persp_pair_d10"].append(ok(a_) and ok(b_))
    for i in range(len(rs) - 3):
        w = rs[i:i + 4]
        if [x["task"] for x in w] == ["R2_twohop", "R2_onehop", "R2_twohop", "R2_onehop"] and w[0]["d"] == 10:
            cells["twohop_pair_d10"].append(ok(w[0]) and ok(w[2]))
            cells["onehop_pair_d10"].append(ok(w[1]) and ok(w[3]))
    return {k: acc(v) for k, v in cells.items()}


def uprobe_summary(up_recs):
    cells = defaultdict(list)
    for r in up_recs:
        fam = r["task"].split("_")[0]
        cells[f"{fam}|d{r['d']}"].append(right(r["scores"]))
    return {k: acc(v) for k, v in cells.items()}


def paired(before, after, key="id", n_boot=10000, seed=0):
    """Paired change on the SAME items. before/after: lists of records with key, 'h' (prompt hash)
    and scores. Returns accuracy change with a bootstrap 95% CI and margin change with a t-based
    95% CI. Raises if the item sets or prompt hashes do not align."""
    b = {r[key]: r for r in before}
    a = {r[key]: r for r in after}
    if set(b) != set(a):
        raise ValueError(f"paired: item sets differ ({len(set(b) ^ set(a))} unmatched)")
    ids = sorted(b)
    for i in ids:
        if b[i].get("h") != a[i].get("h"):
            raise ValueError(f"paired: prompt hash mismatch at {i}")
    da = [int(right(a[i]["scores"])) - int(right(b[i]["scores"])) for i in ids]
    dm = []
    for i in ids:
        ma, mb = margin(a[i]["scores"]), margin(b[i]["scores"])
        if ma is not None and mb is not None:
            dm.append(ma - mb)
    n = len(ids)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = 0
        for _ in range(n):
            s += da[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    mean_dm = sum(dm) / len(dm)
    sd = math.sqrt(sum((x - mean_dm) ** 2 for x in dm) / (len(dm) - 1)) if len(dm) > 1 else float("nan")
    half = 1.96 * sd / math.sqrt(len(dm)) if len(dm) > 1 else float("nan")
    acc_b = sum(right(b[i]["scores"]) for i in ids) / n
    acc_a = sum(right(a[i]["scores"]) for i in ids) / n
    return {"n": n, "acc_before": round(acc_b, 4), "acc_after": round(acc_a, 4),
            "d_acc": round(sum(da) / n, 4), "d_acc_ci95": [round(lo, 4), round(hi, 4)],
            "gained": sum(1 for x in da if x > 0), "lost": sum(1 for x in da if x < 0),
            "d_margin": round(mean_dm, 4), "d_margin_ci95": [round(mean_dm - half, 4), round(mean_dm + half, 4)]}
