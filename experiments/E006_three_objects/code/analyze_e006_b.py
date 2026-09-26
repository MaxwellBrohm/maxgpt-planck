"""E006 analysis, data layer (CPU only, no model). Reads scored records and turns them into the matrices the rules
use: right[seed][item] per (tag, set, render, part, family), with LIK right recomputed from the raw scores
(metrics_ft.right: gold strictly above every other candidate) and GEN right = the stored strict flag.
A record file counts only when complete (E004's rule: the eval draw 640, BIG 576, H5L 192, AL 64 records).
Item facts come from the rebuilt items (draw 4004 via evidence_e005, BIG and H5L via items_big_e006):
  intro        the asked object's introduction order (1-3)
  other_ind    another object's pronoun/ellipsis correction comes after the asked object's latest (the audit split)
  nl_right     the critique's rule NL ("latest statement of any object other than the last statement's object")
               picks the gold
  gold_last    the dialogue's last value statement is about the asked object
Per scored item: the LIK/GEN pick's role (evidence_e005.role_of), whether the pick is the dialogue's last
statement's value, and the recency index (rank of the last statement's value among the LIK candidates and its
log-prob margin over the gold, on items whose gold is not the last statement)."""
import os
from collections import defaultdict

import analyze_e004 as A
import evidence_e005 as V
import items_big_e006 as BG
import metrics_ft as MF

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
SLUG = MODEL.replace("/", "__")
SEEDS = (1, 2, 3, 4, 5)
N_SET = {"e004": 640, "big": 576, "h5l": 192, "al": 64, "kbig": 441}
_ITEMS = {}


def rec_path(out_dir, tag, name):
    if tag.startswith("mac_al:"):
        return os.path.join(out_dir, f"{tag.split(':', 1)[1]}__{name}.jsonl")
    return os.path.join(out_dir, f"{SLUG}__{tag}__{name}.jsonl")


def records(out_dir, tag, set_name, render="plain", part="LIK"):
    name = (f"gen_{set_name}" if part == "GEN" else set_name) + f"__{render}"
    rs = A.load(rec_path(out_dir, tag, name))
    if rs is None or (set_name in N_SET and len(rs) != N_SET[set_name]):
        return None
    return rs


def fam_of(r, set_name):
    return r.get("cell") if set_name == "al" else (r.get("family") or "all")


def matrix(out_dir, tag, set_name, render="plain", part="LIK"):
    """-> {family: {item key: 0/1}} or None."""
    rs = records(out_dir, tag, set_name, render, part)
    if rs is None:
        return None
    out = defaultdict(dict)
    for r in rs:
        key = r.get("idx", r["id"]) if set_name in N_SET else r["id"]
        out[fam_of(r, set_name)][key] = int(MF.right(r["scores"])) if part == "LIK" else int(bool(r["strict"]))
    return dict(out)


def arm(out_dir, tags, set_name, family, render="plain", part="LIK"):
    """tags: {seed: tag} -> {seed: {item: 0/1}} over the seeds with a complete file (missing seeds left out)."""
    out = {}
    for s, t in tags.items():
        m = matrix(out_dir, t, set_name, render, part)
        if m is not None and family in m:
            out[s] = m[family]
    return out


def complete(x, seeds=SEEDS):
    return x is not None and all(s in x for s in seeds)


def pooled(x):
    vals = [v for s in x for v in x[s].values()]
    return sum(vals) / len(vals) if vals else None


def items():
    """{(set, family, idx): item} for draw 4004, BIG and H5L."""
    if not _ITEMS:
        _ITEMS.update({("e004", f, i): it for (f, i), it in V.eval_items().items()})
        for f, its in BG.load().items():
            _ITEMS.update({("big" if f != "H5L" else "h5l", f, it["idx"]): it for it in its})
    return _ITEMS


def facts(it):
    st = it["stmts"]
    lastobj = st[-1]["obj"]
    nl = [s for s in st if s["obj"] != lastobj]
    f = V.facts(it)
    return {"intro": BG.intro_order(st, it["asked"]), "other_ind": f["other_ind_after"],
            "nl_right": bool(nl) and nl[-1]["value"] == it["gold"], "gold_last": lastobj == it["asked"],
            "last_value": st[-1]["value"]}


def rows(out_dir, tag, set_name, render="plain"):
    """-> [row] per item with LIK/GEN right, pick roles, last-statement picks, recency index and item facts."""
    lik = records(out_dir, tag, set_name, render, "LIK")
    gen = records(out_dir, tag, set_name, render, "GEN")
    if lik is None:
        return None
    its, out = items(), {}
    for part, rs in (("LIK", lik), ("GEN", gen or [])):
        for r in rs:
            it = its[(set_name, r["family"], r["idx"])]
            row = out.setdefault((r["family"], r["idx"]), dict(family=r["family"], idx=r["idx"], **facts(it)))
            if part == "LIK":
                ok, pick = MF.right(r["scores"]), V.lik_pick(r)
                lv = row["last_value"]
                lab = next((k for k, v in r["cand_vals"].items() if v == lv), None)
                if lab and lab != "gold":
                    sc = r["scores"]
                    row["rec_rank"] = 1 + sum(1 for v in sc.values() if v > sc[lab])
                    row["rec_margin"] = sc[lab] - sc["gold"]
            else:
                ok, pick = bool(r["strict"]), V.gen_pick(r["reply"], it)
            row[part], (row[part + "_role"], _) = int(ok), V.role_of(it, pick)
            row[part + "_last"] = (not ok) and pick is not V.TIE and pick == row["last_value"]
    return list(out.values())


def split(rows_by_seed, family, key, part="LIK"):
    """-> {level: (right, n)} pooled over seeds, for one family."""
    out = defaultdict(lambda: [0, 0])
    for rs in rows_by_seed.values():
        for r in rs or []:
            if r["family"] == family and part in r:
                out[r[key]][0] += r[part]
                out[r[key]][1] += 1
    return {k: tuple(v) for k, v in sorted(out.items(), key=lambda t: str(t[0]))}


def wrong(rows_by_seed, family, part="LIK"):
    """-> (n wrong, {role: count}, n whose pick is the dialogue's last statement's value)."""
    roles, n, last = defaultdict(int), 0, 0
    for rs in rows_by_seed.values():
        for r in rs or []:
            if r["family"] == family and part in r and not r[part]:
                n += 1
                roles[r[part + "_role"]] += 1
                last += bool(r.get(part + "_last"))
    return n, dict(roles), last


def recency(rows_by_seed, family):
    xs = [(r["rec_rank"], r["rec_margin"]) for rs in rows_by_seed.values() for r in rs or []
          if r["family"] == family and "rec_rank" in r]
    if not xs:
        return None
    return {"n": len(xs), "mean_rank": sum(a for a, _ in xs) / len(xs), "top1": sum(a == 1 for a, _ in xs) / len(xs),
            "mean_margin_vs_gold": sum(b for _, b in xs) / len(xs)}


def reweighted_intro(sp, weights):
    """sp: split by intro {1: (r, n), ...}; weights: {1: w, ...} -> the accuracy reweighted to that intro mix."""
    tot = sum(weights.values())
    return sum(weights[k] / tot * sp[k][0] / sp[k][1] for k in weights if k in sp and sp[k][1])
