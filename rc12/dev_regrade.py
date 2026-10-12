"""RC-12 regrade of stored dev runs with the working-tree graders (CPU only: no model, no GPU). Max, 2026-10-02: took all
recommendations in rc12/DECISIONS_FOR_MAX.md (item 3): after the recap frames joined the stale-value exemption
(graders.STALE_BEFORE), every stored dev-baseline run is regraded and rescored, then dev_panel.py and dev_s11.py
re-run on the regraded copy.
  python -B dev_regrade.py --src <copy of the PC's ~/planck/runs/rc12_dev> --dst <new dir> [--procs 6]
Per run dir with DONE (<model>/<render>/<seed>[_owncf]): each row -> runner.make_row(its record, the STORED turns,
render, seed, responder, train_seed, own_cf): the stored replies, stops and dropped counts, nothing replayed. Written
to the same path under --dst: transcripts.jsonl, scores.jsonl (= score.score_lines(rows)), meta.json (copied), DONE
last. The source is only read. Reported per run: rows whose probes, unit, flags, leaks or ack marks changed, every
changed probe (id, turn, old and new failing clauses), and EXPECTED: True when every change is of the kind the
change being regraded can make (--expect):
  item3   (default; 2026-10-02) every change is a probe that lost exactly the clause v3_other and nothing else moved
  t0gaps  (2026-10-04, notes STEP 11b: the T0 audit's two grader gaps, inflected values and the closing-quote
          sentence end) only probes, units and the role-leak scan moved, never the loop flags or the ack marks, and no
          probe gained or lost a degenerate, echo, unsure or format clause (those read neither values nor sentences)
Writes <dst>/regrade.json; exit 1 if any change is not of that kind."""
import argparse
import json
import os
import shutil
import sys
from multiprocessing import Pool

import runner as RN
import score as S

FIELDS = ("probes", "unit", "flags", "leaks", "ack_repeat", "ack_of_answer")
RECS = {}
UNMOVED = ("_degen", "_echo", "_unsure", "f3_short", "f4_rule", "f5_old")


def expected_item3(c):
    return set(c["fields"]) <= {"probes", "unit"} and c["probes"] and all(
        set(p["old"]) - set(p["new"]) == {"v3_other"} and set(p["new"]) <= set(p["old"]) for p in c["probes"])


def expected_t0gaps(c):
    """a probe whose OWN source parse (d0_source) flips has its other clauses graded or skipped as a block, so only
    the remaining probes are held to the clause rule."""
    moved = {x for p in c["probes"] if "d0_source" not in set(p["old"]) ^ set(p["new"])
             for x in set(p["old"]) ^ set(p["new"])}
    return set(c["fields"]) <= {"probes", "unit", "leaks"} and not any(
        x.startswith("loop_") or x.endswith(UNMOVED) for x in moved)


EXPECT = {"item3": expected_item3, "t0gaps": expected_t0gaps}


def run_dirs(src):
    out = []
    for model in sorted(os.listdir(src)):
        for render in ("template", "plain"):
            d = os.path.join(src, model, render)
            if not os.path.isdir(d):
                continue
            out += [os.path.join(model, render, x) for x in sorted(os.listdir(d))
                    if os.path.exists(os.path.join(d, x, "DONE"))]
    return out


def regrade(job):
    src, dst, rel, expect = job
    if not RECS:
        RECS.update({r["id"]: r for r in RN.load()})
    rows = [json.loads(line) for line in open(os.path.join(src, rel, "transcripts.jsonl"))]
    new_rows, changed = [], []
    for r in rows:
        n = RN.make_row(RECS[r["id"]], r["turns"], r["render"], r["seed"], r["responder"], r["train_seed"], r["own_cf"])
        assert n["turns"] == r["turns"] and n["cf_unswapped"] == r["cf_unswapped"], r["id"]
        moved = [f for f in FIELDS if n[f] != r[f]]
        if moved:
            probes = [dict(turn=a["turn"], old=a["fails"], new=b["fails"]) for a, b in zip(r["probes"], n["probes"])
                      if a != b]
            changed.append(dict(id=r["id"], family=r["family"], cell=r["cell"], fields=moved, unit=[r["unit"], n["unit"]],
                                probes=probes))
        new_rows.append(n)
    out = os.path.join(dst, rel)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "transcripts.jsonl"), "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in new_rows)
    with open(os.path.join(out, "scores.jsonl"), "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in S.score_lines(new_rows))
    shutil.copy2(os.path.join(src, rel, "meta.json"), os.path.join(out, "meta.json"))
    open(os.path.join(out, "DONE"), "w").write("regraded by dev_regrade.py\n")
    expected = all(EXPECT[expect](c) for c in changed)
    return dict(run=rel, rows=len(rows), changed=changed, expected=expected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--expect", default="item3", choices=sorted(EXPECT))
    args = ap.parse_args()
    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
    assert src != dst and not dst.startswith(src + os.sep), "the regraded copy must not sit inside the source"
    jobs = [(src, dst, rel, args.expect) for rel in run_dirs(src)]
    with Pool(args.procs) as pool:
        res = pool.map(regrade, jobs, chunksize=1)
    rows = sum(r["rows"] for r in res)
    ch = [c for r in res for c in r["changed"]]
    ups = sum(c["unit"][1] > c["unit"][0] for c in ch)
    print(f"runs {len(res)}, rows {rows}, rows changed {len(ch)} (unit up {ups}, down "
          f"{sum(c['unit'][1] < c['unit'][0] for c in ch)}, same {sum(c['unit'][1] == c['unit'][0] for c in ch)}), "
          f"probes changed {sum(len(c['probes']) for c in ch)}, all expected {all(r['expected'] for r in res)}")
    by = {}
    for r in res:
        for c in r["changed"]:
            k = (r["run"].split(os.sep)[0], r["run"].split(os.sep)[1], c["family"], c["cell"])
            by[k] = by.get(k, 0) + 1
    for k, v in sorted(by.items()):
        print("  ", " ".join(k), v)
    with open(os.path.join(dst, "regrade.json"), "w") as f:
        json.dump(dict(src=src, expect=args.expect, runs=len(res), rows=rows, rows_changed=len(ch), per_run=res), f,
                  indent=0)
    return 0 if all(r["expected"] for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())
