"""Paired comparison of two models' likelihood rows (prereg draft s4, P-004; diagnostic, never in a composite or a
claim). Each model's rows are scored on the gold against the foils of the gold's token count under ITS OWN
tokenizer (lik_score.py), and two tokenizers can pick different foils, so a comparison of two models uses, per row,
only the foils BOTH tokenizers give the gold's count (the common foils; a row with none is left out and counted).
Both models are then judged on the same candidate set: right iff the gold's stored score is strictly above every
common foil's (lik_score.right, the E004 rule), margin = gold minus the best common foil.
  compare(rows_a, rows_b)  rows: two lik_run rows.jsonl of one render. Refused (ValueError): different row ids,
      renders, golds, candidates or lead-ins (not the same lik_rows build). Per key: n, n_both (scored under both,
      not over_ctx), n_common (rows with >= 1 common foil), n_equal_both (every candidate one count under both
      tokenizers), acc_a, acc_b, diff = acc_a - acc_b, chance (mean 1 / (1 + common foils)), n_nonfinite_a / _b,
      and for BIND pairs_a / pairs_b over twins with common foils in both.
CLI: python3 -B lik_compare.py A/template/rows.jsonl B/template/rows.jsonl   (JSON; exit 2 when refused)"""
import json
import sys
from collections import defaultdict

import lik_score as LS

SAME = ("key", "pair_id", "render", "lead", "gold", "candidates")


def join(rows_a, rows_b):
    A, B = {r["id"]: r for r in rows_a}, {r["id"]: r for r in rows_b}
    if len(A) != len(rows_a) or len(B) != len(rows_b):
        raise ValueError("a row id twice in one model's rows")
    if set(A) != set(B):
        raise ValueError(f"row ids differ: {len(set(A) ^ set(B))} rows in one model only")
    for i in sorted(A):
        bad = [k for k in SAME if A[i][k] != B[i][k]]
        if bad:
            raise ValueError(f"row {i}: {bad} differ between the models (not the same lik_rows build and render)")
    return [(A[i], B[i]) for i in sorted(A)]


def common_foils(a, b):
    both = set(a["foils"]) & set(b["foils"])
    return [c for c in a["candidates"] if c in both]


def judge(r, foils):
    sub = {c: r["scores"][c] for c in [r["gold"]] + foils}
    return LS.right(sub, r["gold"]), LS.margin(sub, r["gold"])


def compare(rows_a, rows_b):
    by = defaultdict(list)
    for a, b in join(rows_a, rows_b):
        by[a["key"]].append((a, b))
    out = {}
    for key in sorted(by, key=lambda k: (LS.ORDER.index(k) if k in LS.ORDER else len(LS.ORDER), k)):
        pairs = by[key]
        ok = [(a, b) for a, b in pairs if not a["over_ctx"] and not b["over_ctx"]]
        res = []
        for a, b in ok:
            foils = common_foils(a, b)
            if foils:
                (ra, ma), (rb, mb) = judge(a, foils), judge(b, foils)
                res.append(dict(id=a["id"], pair_id=a["pair_id"], ra=ra, rb=rb, ma=ma, mb=mb,
                                chance=1 / (1 + len(foils))))
        acc_a, acc_b = LS.rate([x["ra"] for x in res]), LS.rate([x["rb"] for x in res])
        s = dict(n=len(pairs), n_both=len(ok), n_common=len(res),
                 n_equal_both=sum(a["equal"] and b["equal"] for a, b in ok),
                 acc_a=acc_a, acc_b=acc_b, diff=None if acc_a is None else acc_a - acc_b,
                 chance=LS.rate([x["chance"] for x in res]), n_nonfinite_a=sum(x["ma"] is None for x in res),
                 n_nonfinite_b=sum(x["mb"] is None for x in res))
        if key == "BIND":
            tw = defaultdict(list)
            for x in res:
                tw[x["pair_id"]].append(x)
            full = [v for v in tw.values() if len(v) == 2]
            s["pairs"] = dict(n=len(full), acc_a=LS.rate([all(x["ra"] for x in v) for v in full]),
                              acc_b=LS.rate([all(x["rb"] for x in v) for v in full]))
        out[key] = s
    return out


def load(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__)
        return 2
    try:
        res = compare(load(argv[0]), load(argv[1]))
    except ValueError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(json.dumps(dict(a=argv[0], b=argv[1], families=res), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
