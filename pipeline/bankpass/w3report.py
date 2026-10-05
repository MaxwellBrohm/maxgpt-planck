"""W3 stage report and full refresh (PC CPU, outside the holds). Per bank: lines generated (parsed from finished
calls), items (templatized), gated (passed s2c/s2d and the pre-judge exact dedup), judged (both non-author judges in),
kept (gates + both judges keep, or gates alone for unjudged banks), pending, drop codes, author shares of generated
and kept, and the s1 target. Plus the call-level counts per teacher and kind (calls, parse problems, errors, output
tokens, seconds) and the derived summaries (topics selected, votes, relation labels, word labels).

    python -m bankpass.w3report --root STAGE [--refresh] [--workers 6] [--out report.json]"""
import argparse
import collections
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from bankpass import store, w3state as WS

SHORT = store.SHORT


def _refresh_one(args):
    root, t = args
    os.nice(10)
    WS.install_human()
    return t, WS.refresh(root, teachers=(t,))


def refresh_all(root, workers=3):
    with ProcessPoolExecutor(min(workers, 3)) as ex:
        return dict(ex.map(_refresh_one, [(root, t) for t in WS.ORDER]))


def shares(rows):
    n = len(rows) or 1
    c = collections.Counter(SHORT[r] for r in rows)
    return {k: round(v / n, 3) for k, v in sorted(c.items())}


def bank_table(its, recs, targets):
    gen = collections.defaultdict(list)
    for r in recs:
        if r.get("error") or r["kind"] in WS.ITEMLESS:
            continue
        for _ in r.get("lines") or []:
            gen[r["bank"]].append(r["author"]["model"])
    out = {}
    for bank, rows in sorted(WS.by_bank(its).items()):
        gated = [i for i in rows if i["status"] == "kept"]
        judged = [i for i in gated if WS.needs_judges(i) and len(i["judges"]) >= 2]
        kept = [i for i in rows if i.get("w3") == "kept"]
        drops = collections.Counter(i["drop"] for i in rows if i["status"] == "dropped")
        drops.update(i.get("drop_w3") for i in rows if i.get("w3") == "dropped" and i["status"] == "kept")
        out[bank] = {"target": targets.get(bank), "generated": len(gen.get(bank, [])), "items": len(rows),
                     "gated": len(gated), "judged": len(judged) if WS.needs_judges(rows[0]) else None,
                     "kept": len(kept), "pending": sum(i.get("w3") == "pending" for i in rows),
                     "authors_generated": shares(gen.get(bank, [])),
                     "authors_kept": shares([i["author"]["model"] for i in kept]),
                     "drops": dict(drops.most_common())}
    return out


def call_table(recs):
    out = collections.defaultdict(lambda: {"calls": 0, "problems": 0, "errors": 0, "out_tokens": 0, "seconds": 0.0})
    for r in recs:
        k = out[f"{SHORT[r['author']['model']]}:{r['kind']}"]
        k["calls"] += 1
        k["errors"] += bool(r.get("error"))
        k["problems"] += bool(r.get("problem"))
        k["out_tokens"] += r.get("n_out") or 0
        k["seconds"] += r.get("seconds") or 0.0
    return {k: dict(v, seconds=round(v["seconds"], 1)) for k, v in sorted(out.items())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    WS.install_human()
    if a.refresh:
        print(json.dumps({"refreshed": refresh_all(a.root, a.workers)}), flush=True)
    with open(os.path.join(a.root, "plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    recs = WS.records(a.root)
    its = WS.view(a.root, recs)
    rep = {"plan_sha256": plan["sha256"], "banks": bank_table(its, recs, plan["targets"]), "calls": call_table(recs)}
    out = a.out or os.path.join(a.root, "report.json")
    with open(out + ".tmp", "w") as f:
        json.dump(rep, f, indent=1, sort_keys=True)
    os.replace(out + ".tmp", out)
    tot = collections.Counter()
    for b in rep["banks"].values():
        tot.update({k: b[k] or 0 for k in ("generated", "items", "gated", "kept", "pending")})
    print(json.dumps({"out": out, "totals": dict(tot), "banks": len(rep["banks"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
