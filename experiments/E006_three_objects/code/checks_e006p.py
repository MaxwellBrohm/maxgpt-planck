"""E006 per-example checks for arm P (notes.txt ARMS P; used by test_train_e006p.py, validate_e006.py and
mutation_e006.py). An example WITHOUT "p_change" must pass E005's own table (checks_e005_b.all_problems) unchanged.
A changed example must pass E005's table except the ONE structure line the change is meant to break, and the
change's own declared structure:
  split   (ALIAS) E005's check_alias_struct may report only "adjacent/filler with A aliased: B not all before A's
          original or all after the alias"; and: case latest or earlier, placement adjacent or filler, B has >= 1
          correction, B's original is the dialogue's first statement and comes before A's original, every B
          correction comes after A's latest statement, B's statements keep their order.
  yfirst  (IND) E005's check_ind_struct is replaced (it requires X first and object 0 = X): the first statement
          is Y's original; then X's original, X's corrections (1-3), X's last an ellipsis or pronoun directly
          after X's previous statement; then Y's corrections (1-2), all after X's last; the first of them names Y
          (full or head); kind ind_x asks X, ind_y asks Y; no alias.
Every other E005 line (references, alias refs, values, answer, echo, heldout, render) applies to every example."""
import checks_e005_b as CB5

SPLIT_ALLOWED = "adjacent/filler with A aliased: B not all before A's original or all after the alias"


def check_split(ex):
    p = [x for x in CB5.check_alias_struct(ex) if x != SPLIT_ALLOWED]
    if ex.get("case") not in ("latest", "earlier") or ex.get("placement") not in ("adjacent", "filler"):
        p.append(f"split outside latest/earlier x adjacent/filler: {ex.get('case')} {ex.get('placement')}")
    S, A = ex["stmts"], ex["asked"]
    B = 1 - A
    ib = [i for i, s in enumerate(S) if s["obj"] == B]
    ia = [i for i, s in enumerate(S) if s["obj"] == A]
    if len(ib) < 2:
        return p + ["split with B uncorrected"]
    if ib[0] != 0 or not ib[0] < ia[0]:
        p.append("split: B's original is not first / not before A's original")
    if S[ib[0]]["role"] != "orig" or any(S[i]["role"] != "corr" for i in ib[1:]):
        p.append("split: B's statements out of order")
    if not all(i > ia[-1] for i in ib[1:]):
        p.append("split: a B correction is not after A's latest statement")
    return p


def check_yfirst(ex):
    p = CB5._generic(ex)
    kind_asked = {"ind_x": "X", "ind_y": "Y"}.get(ex["kind"])
    if kind_asked is None or ex.get("aliases"):
        return p + [f"undeclared IND kind/aliases {ex['kind']}"]
    S = ex["stmts"]
    objs = [s["obj"] for s in S]
    Y, X = objs[0], 1 - objs[0]
    if S[0]["role"] != "orig":
        p.append("yfirst: the first statement is not an original")
    asked_label = "Y" if ex["asked"] == Y else "X"
    if asked_label != kind_asked:
        p.append(f"yfirst: kind {ex['kind']} but the asked object is {asked_label}")
    ix = [i for i, o in enumerate(objs) if o == X]
    iy = [i for i, o in enumerate(objs) if o == Y]
    if not ix or ix != list(range(1, 1 + len(ix))):
        p.append("yfirst: X's chain is not contiguous right after Y's original")
    kx, ky = len(ix) - 1, len(iy) - 1
    if not (1 <= kx <= 3 and 1 <= ky <= 2):
        p.append(f"yfirst: k out of range: X {kx}, Y {ky}")
    if ix and (S[ix[0]]["role"] != "orig" or any(S[i]["role"] != "corr" for i in ix[1:])):
        p.append("yfirst: X's statements out of order")
    if ix and S[ix[-1]]["ref"] not in ("ell", "pron"):
        p.append(f"yfirst: X's last correction is {S[ix[-1]]['ref']}, not ellipsis/pronoun")
    if ix and iy[1:] and not all(i > ix[-1] for i in iy[1:]):
        p.append("yfirst: a Y correction comes before X's last")
    if iy[1:] and S[iy[1]]["ref"] not in ("full", "head"):
        p.append(f"yfirst: Y's first correction after X is {S[iy[1]]['ref']} (must name Y)")
    if any(s["ref"] == "alias" for s in S):
        p.append("alias correction in IND")
    return p


def all_problems_p(ex):
    """E005's per-example table with the P structure rule; -> {check name: [problems]} (empty = pass)."""
    ch = ex.get("p_change")
    if ch is None:
        return CB5.all_problems(ex)
    if ch not in ("split", "yfirst") or (ch == "split") != (ex.get("block") == "alias") or \
            (ch == "yfirst") != (ex.get("block") == "ind"):
        return {"structure": [f"undeclared p_change {ch!r} in block {ex.get('block')}"]}
    out = {}
    for name, fn in CB5.PER_EX:
        if name == "structure":
            out[name] = check_split(ex) if ch == "split" else check_yfirst(ex)
        else:
            out[name] = fn(ex)
    return {k: v for k, v in out.items() if v}
