"""E005 item evidence (notes.txt EXTRA REPORTS 1-4, and the FAIL sub-readings, which now report evidence instead of
E004's fixed strings). Joins every scored record of the E004 eval draw (4004) to its rebuilt item (items_e004,
unchanged) by (family, idx), checks the join, and reports what the model did item by item. CPU only; no model.
Per item (facts):
  form      reference form of the asked object's latest statement: full, head, pron, ell, alias, or orig (never
            corrected)
  place     for an alias latest: adjacent (the very next turn after the asked object's previous statement),
            filler (only fillers between) or B_sep (another object's statement between); else None
  other_after      another object has a statement after the asked object's latest
  other_ind_after  ... and one of those is a pronoun or ellipsis correction (the audit's H5 split)
Per pick (the LIK argmax value, or the first value of the whole pool the GEN reply names):
  gold | tie (LIK: gold tied for the top) | A_prev (the asked object's statement just before its latest) |
  A_earlier (an older statement of the asked object) | other_after / other_before (another object's value stated
  after / before the asked object's latest) | not_in_context (a pool value the dialogue never states) | none
  (the reply names no value); lure = the picked value's statement is a frame-echo lure.
Chat first-line strict (the audit's variant): the chat reply's first line graded as a stopped reply ("eos")."""
from collections import Counter, defaultdict

import gen_grade as G
import items_e004 as E
import metrics_ft as MF

PLACES = ("adjacent", "filler", "B_sep")
FORMS = ("alias", "ell", "pron", "full", "head", "orig")
ROLES = ("gold", "tie", "A_prev", "A_earlier", "other_after", "other_before", "not_in_context", "none")
TIE = object()
_ITEMS = {}


def eval_items():
    if not _ITEMS:
        _ITEMS.update({(f, it["idx"]): it for f, its in E.draw("eval").items() for it in its})
    return _ITEMS


def facts(it):
    A = [s for s in it["stmts"] if s["obj"] == it["asked"]]
    la = A[-1]
    after = [s for s in it["stmts"] if s["obj"] != it["asked"] and s["turn"] > la["turn"]]
    form = "orig" if la["role"] == "orig" else la["ref"]
    place = None
    if form == "alias":
        prev = A[-2]
        between = [s for s in it["stmts"] if prev["turn"] < s["turn"] < la["turn"]]
        place = "B_sep" if between else ("adjacent" if la["turn"] - prev["turn"] == 1 else "filler")
    return dict(form=form, place=place, other_after=bool(after),
                other_ind_after=any(s["role"] != "orig" and s["ref"] in ("pron", "ell") for s in after))


def role_of(it, v):
    """-> (role, lure) of a picked value (see the module docstring)."""
    if v is TIE:
        return "tie", False
    if v is None:
        return "none", False
    if v == it["gold"]:
        return "gold", False
    A = [s for s in it["stmts"] if s["obj"] == it["asked"]]
    lure = any(s["value"] == v and s.get("lure") for s in it["stmts"])
    if len(A) >= 2 and A[-2]["value"] == v:
        return "A_prev", lure
    if any(s["value"] == v for s in A[:-1]):
        return "A_earlier", lure
    oth = [s for s in it["stmts"] if s["obj"] != it["asked"] and s["value"] == v]
    if any(s["turn"] > A[-1]["turn"] for s in oth):
        return "other_after", lure
    if oth:
        return "other_before", lure
    return "not_in_context", False


def lik_pick(rec):
    sc = rec["scores"]
    if MF.right(sc):
        return rec["cand_vals"]["gold"]
    best = max(sc.values())
    tops = [k for k, v in sc.items() if v == best]
    return TIE if "gold" in tops else rec["cand_vals"][tops[0]]


def gen_pick(reply, it):
    text = G.norm(reply)
    hits = [(h[0], v) for v in it["values"] for h in [G.value_hits(text, v)] if h]
    return min(hits)[1] if hits else None


def first_line_strict(rec, it):
    gold, pool, obj = G.spec_e004(it)
    first = (rec.get("reply") or "").strip().split("\n")[0]
    return G.grade(first, "eos", gold, pool, obj)["strict"]


def join(lik_recs, gen_recs):
    """-> rows, one per item: family, idx, facts, LIK/GEN right, role, lure. Raises on a record that does not match
    its rebuilt item (gold, vtype, k, d, latest_ref)."""
    items, rows = eval_items(), {}
    for part, recs in (("LIK", lik_recs), ("GEN", gen_recs)):
        for r in recs or []:
            it = items[(r["family"], r["idx"])]
            if (r.get("vtype"), r.get("k"), r.get("d")) != (it["vtype"], it["k"], it["d"]) or \
                    r.get("latest_ref") != it.get("meta", {}).get("latest_ref") or \
                    (part == "LIK" and r["cand_vals"]["gold"] != it["gold"]):
                raise ValueError(f"record {r['family']}#{r['idx']} does not match the eval draw")
            row = rows.setdefault((r["family"], r["idx"]), dict(family=r["family"], idx=r["idx"], **facts(it)))
            if part == "LIK":
                ok, pick = MF.right(r["scores"]), lik_pick(r)
            else:
                ok, pick = bool(r["strict"]), gen_pick(r["reply"], it)
            row[part], (row[part + "_role"], row[part + "_lure"]) = ok, role_of(it, pick)
    return list(rows.values())


def acc(rows, part):
    xs = [r[part] for r in rows if part in r]
    return {"acc": round(sum(xs) / len(xs), 4) if xs else None, "n": len(xs)}


def cells(rows, key, fams=None):
    out = defaultdict(dict)
    for r in rows:
        if fams is None or r["family"] in fams:
            out[r["family"]].setdefault(r[key], []).append(r)
    return {f: {lv: {p: acc(rs, p) for p in ("LIK", "GEN")} for lv, rs in sorted(d.items(), key=lambda t: str(t[0]))}
            for f, d in out.items()}


def wrong_picks(rows):
    """-> {part: {role: count}, part_lure: count} over the wrong items of rows."""
    out = {}
    for p in ("LIK", "GEN"):
        bad = [r for r in rows if p in r and not r[p]]
        out[p] = dict(Counter(r[p + "_role"] for r in bad))
        out[p + "_lure"] = sum(r[p + "_lure"] for r in bad)
        out[p + "_n_wrong"] = len(bad)
    return out


def evidence(rows):
    h12 = [r for r in rows if r["family"] in ("H1", "H2")]
    al = [r for r in h12 if r["form"] == "alias"]
    ell = [r for r in rows if r["form"] == "ell"]
    h5 = [r for r in rows if r["family"] == "H5"]
    groups = {"H1/H2 alias": al, "H1/H2 not alias": [r for r in h12 if r["form"] != "alias"]}
    groups.update({f: [r for r in rows if r["family"] == f] for f in E.FAMILIES if f not in ("H1", "H2")})
    return {
        "form": cells(rows, "form"),
        "alias": {"all": {p: acc(al, p) for p in ("LIK", "GEN")},
                  **{f: {p: acc([r for r in al if r["family"] == f], p) for p in ("LIK", "GEN")} for f in ("H1", "H2")},
                  "place": {pl: {p: acc([r for r in al if r["place"] == pl], p) for p in ("LIK", "GEN")}
                            for pl in PLACES}},
        "ell_other_after": cells(ell, "other_after"),
        "h5": {"other_ind_after": {str(k): {p: acc([r for r in h5 if r["other_ind_after"] == k], p)
                                            for p in ("LIK", "GEN")} for k in (True, False)},
               "form": cells(h5, "form").get("H5", {})},
        "wrong": {g: wrong_picks(rs) for g, rs in groups.items() if rs},
    }


def chat_cells(gen_chat, gen_plain):
    """-> {family: {strict, first_line, capped, plain}} from the chat GEN records (and the plain ones, if any)."""
    items, out = eval_items(), defaultdict(lambda: defaultdict(list))
    for r in gen_chat or []:
        it = items[(r["family"], r["idx"])]
        out[r["family"]]["strict"].append(bool(r["strict"]))
        out[r["family"]]["first_line"].append(first_line_strict(r, it))
        out[r["family"]]["capped"].append(r.get("stop") == "cap")
    for r in gen_plain or []:
        out[r["family"]]["plain"].append(bool(r["strict"]))
    return {f: {k: round(sum(v) / len(v), 4) if v else None for k, v in d.items()} for f, d in out.items()}
