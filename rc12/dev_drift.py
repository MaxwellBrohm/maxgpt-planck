"""RC-12 STEP 10a: batch-composition drift read from the stored dev runs, no model and no GPU (the GPU-free stand-in
for verify_dev_runs R1 / R2, which replay on the engine).
Every seed's --own-cf twin replays the 48 OWN conversations with the same engine, render, seed and per-reply seeds,
but in lockstep batches of 48 instead of the dev run's 640 (608 from turn 2). Up to and including the first turn
where the twin's history departs from the dev run's (the cf rewrite, or an earlier reply that differs), the two
runs answer the SAME prompt (same user turns, same earlier assistant texts, same truncation). On those turns the
engine's own output (raw when stored, else reply; and the stop reason) should be equal unless the batch changes the
numerics. Margins (log-prob gaps at the first differing token) need a forward pass and are not computed here.
  python -B dev_drift.py --root <copy of rc12_dev> [--out runs/dev_panel/drift.json]"""
import argparse
import json
import os

import dev_panel as DP


def gen(t):
    return (t["raw"] if t.get("raw") is not None else t["reply"]), t["stop"]


def first_diff(a, b):
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), n)


def compare(dev, cf):
    """per OWN conversation: turns on identical prompts, how many give equal output, the first differing turn."""
    by_id = {r["id"]: r for r in dev}
    out = []
    for c in cf:
        d = by_id[c["id"]]
        same_prompt = equal = 0
        diff = None
        for td, tc in zip(d["turns"], c["turns"]):
            assert td["user"] == tc["user"] and td["i"] == tc["i"]
            if td["dropped"] != tc["dropped"]:
                break
            same_prompt += 1
            gd, gc = gen(td), gen(tc)
            if gd != gc:
                diff = dict(turn=td["i"], char=first_diff(gd[0], gc[0]), stops=(gd[1], gc[1]),
                            dev=gd[0][:90], twin=gc[0][:90])
                break
            equal += 1
            if td["reply"] != tc["reply"]:      # the cf rewrite took here: later prompts differ
                break
        out.append(dict(id=c["id"], same_prompt=same_prompt, equal=equal, diff=diff))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = []
    for render in DP.RENDERS:
        for model in DP.CORE + DP.EXTRAS:
            for seed in DP.SEEDS:
                dev, cf = DP.run_rows(args.root, model, render, seed)
                cmp = compare(dev, cf)
                diffs = [x["diff"] for x in cmp if x["diff"]]
                res.append(dict(model=model, render=render, seed=seed, convs=len(cmp),
                                turns_same_prompt=sum(x["same_prompt"] for x in cmp),
                                turns_equal=sum(x["equal"] for x in cmp), convs_with_diff=len(diffs),
                                first_turn1_diffs=sum(d["turn"] == 1 for d in diffs),
                                diffs=[dict(id=x["id"], **x["diff"]) for x in cmp if x["diff"]][:5]))
    for r in res:
        print(f"{r['model']:28s} {r['render']:8s} {r['seed']:6s} same-prompt turns {r['turns_same_prompt']:3d} "
              f"equal {r['turns_equal']:3d} convs with a difference {r['convs_with_diff']:2d} / {r['convs']}"
              f" (at turn 1: {r['first_turn1_diffs']})")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=0)


if __name__ == "__main__":
    main()
