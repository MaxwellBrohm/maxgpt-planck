"""E006 audit: shared loaders and statistics, written independently of code/analyze_e006*.py, rules_e006.py and
passrule_e006.py (none of them is imported). Item generators (items_e004, pools_train) are imported as data only.
Paths are relative to this file: REPO = the maxgpt-planck checkout."""
import json, os, sys, hashlib
import numpy as np

AUDIT_DIR = os.path.dirname(os.path.abspath(__file__))
E6 = os.path.dirname(AUDIT_DIR)
REPO = os.path.dirname(os.path.dirname(E6))
E5 = os.path.join(REPO, "experiments", "E005_alias_eot")
E4 = os.path.join(REPO, "experiments", "E004_general_updating")
PFX = "HuggingFaceTB__SmolLM2-135M-Instruct__"
PASS9 = ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "C_noupd", "C_twoslot"]
FAM10 = PASS9 + ["ID"]
SEEDS = [1, 2, 3, 4, 5]
# DEVIATION F1: tag e005w1 reads e005w1_r2's record files. Set AUDIT_NOMAP=1 to read the killed attempt instead.
TAGMAP = {} if os.environ.get("AUDIT_NOMAP") else {"e005w1": "e005w1_r2"}


def code_path():
    p = os.path.join(E6, "code")
    if p not in sys.path:
        sys.path.insert(0, p)


def rec_file(tag, set_name, render, root=None):
    root = root or os.path.join(E6, "out")
    return os.path.join(root, f"{PFX}{TAGMAP.get(tag, tag) if root.startswith(E6) else tag}__{set_name}__{render}.jsonl")


def load(tag, set_name, render, root=None):
    p = rec_file(tag, set_name, render, root)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def tag_of(arm, s):
    return {"C": f"C{s}", "P": f"P{s}", "G": f"G{s}", "e005w": f"e005w{s}", "e004w": f"e004w{s}",
            "E005": f"s{s}", "E004": f"s{s}"}[arm]


def root_of(arm):
    return {"E005": os.path.join(E5, "out"), "E004": os.path.join(E4, "out")}.get(arm)


def lik_right(r):
    """My own LIK decision: the gold's summed log-prob strictly above every other candidate in the record."""
    g = r["scores"]["gold"]
    return all(g > v for k, v in r["scores"].items() if k != "gold")


def matrix(arm, set_name, render, kind, fams=None, seeds=SEEDS, stored=False):
    """{family: array (k seeds x n items)} of 0/1, rows in seed order, columns by record id within family.
    kind 'LIK' recomputes from scores; kind 'GEN' reads the stored strict flag. Returns None if any seed is
    missing or short (no imputation)."""
    out, ref_ids = {}, {}
    for s in seeds:
        recs = load(tag_of(arm, s), ("gen_" + set_name) if kind == "GEN" else set_name, render, root_of(arm))
        if recs is None:
            return None
        byf = {}
        for r in recs:
            if fams and r["family"] not in fams:
                continue
            v = (r["right"] if stored else lik_right(r)) if kind == "LIK" else r["strict"]
            byf.setdefault(r["family"], []).append((r["id"], v))
        for f, rows in byf.items():
            rows.sort()
            ids = [i for i, _ in rows]
            if f in ref_ids and ids != ref_ids[f]:
                return None
            ref_ids[f] = ids
            out.setdefault(f, []).append([int(v) for _, v in rows])
    res = {}
    for f, rows in out.items():
        if len(rows) != len(seeds):
            return None
        res[f] = np.array(rows, dtype=float)
    return res


def boot_pair(D, B=10000, seed=6006):
    """Two-way paired bootstrap of the mean of D (k x n): draw k seeds and n items with replacement, average D over
    the drawn seed x item cross product. My own implementation of the notes' text (not the analyzer's code)."""
    D = np.asarray(D, dtype=float)
    k, n = D.shape
    rng = np.random.default_rng(seed)
    si = rng.integers(0, k, size=(B, k))
    ii = rng.integers(0, n, size=(B, n))
    item_means = D[:, ii].mean(axis=2)            # k x B: per seed, mean over the drawn items
    vals = np.take_along_axis(item_means.T, si, axis=1).mean(axis=1)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(D.mean()), float(lo), float(hi)


def boot_conv(XA, XB, NT, B=10000, seed=6006, kind="rate"):
    """Chat bootstrap over seeds x conversations. XA, XB: k x c arrays of per-conversation counts (turns flagged,
    or checks passed); NT: k x c turns per conversation. rate: sum flagged / sum turns over the drawn pairs.
    checks: mean over drawn seeds of the sum over drawn conversations."""
    XA, XB, NT = (np.asarray(a, dtype=float) for a in (XA, XB, NT))
    k, c = XA.shape
    rng = np.random.default_rng(seed)
    si = rng.integers(0, k, size=(B, k))
    ci = rng.integers(0, c, size=(B, c))
    def stat(X):
        per_seed = X[:, ci].sum(axis=2)           # k x B
        tot = np.take_along_axis(per_seed.T, si, axis=1)
        if kind == "rate":
            nt = np.take_along_axis(NT[:, ci].sum(axis=2).T, si, axis=1)
            return tot.sum(axis=1) / nt.sum(axis=1)
        return tot.mean(axis=1)
    vals = stat(XA) - stat(XB)
    if kind == "rate":
        point = XA.sum() / NT.sum() - XB.sum() / NT.sum()
    else:
        point = XA.sum(axis=1).mean() - XB.sum(axis=1).mean()
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fmt(t):
    return f"{t[0]:+.3f}, CI {t[1]:+.3f} to {t[2]:+.3f}"
