"""RC-12 regrade of stored dev runs with the working-tree graders (CPU only: no model, no GPU). Max, 2026-10-02: took all
recommendations in rc12/DECISIONS_FOR_MAX.md (item 3): after the recap frames joined the stale-value exemption
(graders.STALE_BEFORE), every stored dev-baseline run is regraded and rescored, then dev_panel.py and dev_s11.py
re-run on the regraded copy.
  python -B dev_regrade.py --src <copy of the PC's ~/planck/runs/rc12_dev> --dst <new dir> [--procs 6]
Per run dir with DONE (<model>/<render>/<seed>[_owncf]): each row -> runner.make_row(its record, the STORED turns,
render, seed, responder, train_seed, own_cf): the stored replies, stops and dropped counts, nothing replayed. Written
to the same path under --dst: transcripts.jsonl, scores.jsonl (= score.score_lines(rows)), meta.json (copied), DONE
last, all in <run>.partial, then dev_batch.publish_run (every file and the dir fsynced, the rename, the parent fsynced;
cuts audit row 143, 2026-10-11), so a power cut never leaves a DONE beside empty or cut files; a rerun replaces an
earlier copy (moved aside to <run>.old.partial in one rename, removed after the publish), and <dst>/regrade.json is
written the same way, after the earlier one is removed at the start of a rerun. Run dirs named *.partial are never
read. The source is only read. Reported per run: rows whose probes, unit, flags, leaks or ack marks changed, every
changed probe (id, turn, old and new failing clauses), and EXPECTED: True when every change is of the kind the
change being regraded can make (--expect):
  item3   (default; 2026-10-02) every change is a probe that lost exactly the clause v3_other and nothing else moved
  t0gaps  (2026-10-04, notes STEP 11b: the T0 audit's two grader gaps, inflected values and the closing-quote
          sentence end) only probes, units and the role-leak scan moved, never the loop flags or the ack marks, and no
          probe gained or lost a degenerate, echo, unsure or format clause (those read neither values nor sentences)
  a2      (2026-10-19, lock step A2, notes STEP 16: "You asked if X", the new abstain cues, a hedge inside a report of
          the user's words, a quoted question the user asked) only probes and units moved, never the loop flags, the
          ack marks or the role-leak scan, and a probe gained or lost only clauses that read the assertion rule or the
          abstain cue (A2_MOVED; v3_shotgun through the answer-first waiver below, which reads the assertion rule),
          never GAINING v2_gold, a2_cue or d4_assert (A2_ONLY_LOST, review r1 2026-11-19: A2 only widens matching); a
          probe whose OWN source parse flips is exempt from the clause rule, as in t0gaps; a change with no changed
          probe (the unit alone) is refused, as in first and cap (review r2, 2026-11-19)
  first   (2026-11-03, rc12/FRONTIER.md: the answer-first waiver) every changed probe only lost v3_other and/or
          v3_shotgun, nothing gained, only probes and units moved
  cap     (2026-11-03, the cap amendment in grade_loop) every changed probe only lost degenerate clauses, nothing
          gained, only probes and units moved
  a2b     (2026-11-19, lock step A2b, notes STEP 16A2B: the LOOKUP not-found cues and the a4 exemption) every changed
          probe is a LOOKUP record's X probe (read from the dev records, not from the graders) that only lost a2_cue
          and/or a4_voice and gained nothing, so none goes from pass to fail; only probes and units moved, no unit
          went down, and a change with no changed probe is refused
  a2c     (2026-11-19, lock step A2c, notes STEP 16A2C, LOCK_STEPS M7: a3_value reads a pool value in any case on
          LOOKUP's X probe) every changed probe is a LOOKUP record's X probe (read from the dev records) that only
          GAINED a3_value and lost nothing, so a probe can go from pass to fail and never back; only probes and units
          moved, no unit went up, and a change with no changed probe is refused (G-VAL and the P probe are unchanged)
  a2b_r2  (2026-11-22, the late review of lock step A2b, notes STEP 16A2B-R2: the LOOKUP not-found cues narrowed
          where they let wrong replies through and widened where they failed an answer word or an opening; run from
          the A2c copies) every changed probe is a LOOKUP record's X probe (read from the dev records) whose only
          moved clause is a2_cue, gained or lost (the a4 frame and every other clause are unchanged); only probes
          and units moved, and a change with no changed probe is refused. The late review's second round (notes
          STEP 16A2B-R3) moves the same clause only and is checked with the same rule
  a2d     (2026-11-23, lock step A2d, notes STEP 16A2D: the LOOKUP not-found cues read fail closed) every changed
          probe is a LOOKUP record's X probe (read from the dev records) that only GAINED a2_cue and lost nothing,
          so a probe can go from pass to fail and never back; only probes and units moved, no unit went up, and a
          change with no changed probe is refused
Writes <dst>/regrade.json; exit 1 if any change is not of that kind."""
import argparse
import json
import os
import shutil
import sys
from multiprocessing import Pool

import dev_batch as DB
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


# v3_shotgun since the merge onto the Nov 3 graders (2026-11-19): the answer-first waiver (graders.answer_first) asks
# what the first sentence asserts, so an A2 change there grants or withdraws the waiver of v3_other and v3_shotgun
A2_MOVED = {"v2_gold", "v3_shotgun", "v3_other", "a2_cue", "d4_assert"}
# review r1 (2026-11-19): A2 only widens what counts as a mention, an assertion or an abstain cue, so these three can
# only be lost; a gain is a regression (a gold no longer matched or asserted, a cue no longer seen)
A2_ONLY_LOST = {"d4_assert", "a2_cue", "v2_gold"}


def expected_a2(c):
    if not c["probes"]:                          # review r2 (2026-11-19): the unit alone moved
        return False
    gained = {x for p in c["probes"] if "d0_source" not in set(p["old"]) ^ set(p["new"])
              for x in set(p["new"]) - set(p["old"])}
    moved = {x for p in c["probes"] if "d0_source" not in set(p["old"]) ^ set(p["new"])
             for x in set(p["old"]) ^ set(p["new"])}
    return set(c["fields"]) <= {"probes", "unit"} and moved <= A2_MOVED and not gained & A2_ONLY_LOST


def expected_first(c):
    """the answer-first waiver (2026-11-03, rc12/FRONTIER.md) only ever removes v3_other or v3_shotgun: every changed
    probe lost some of those two and nothing else, gained nothing, and only probes and units moved."""
    return set(c["fields"]) <= {"probes", "unit"} and c["probes"] and all(
        set(p["old"]) - set(p["new"]) <= {"v3_other", "v3_shotgun"} and set(p["new"]) <= set(p["old"])
        for p in c["probes"])


CAP_CLAUSES = {"v1_degen", "a1_degen", "f1_degen", "d1_degen", "r1_degen", "loop_RUNAWAY"}


def expected_cap(c):
    """the cap amendment (2026-11-03, grade_loop: a reply cut at the cap is degenerate only if it loops) only ever
    removes a degenerate clause: every changed probe lost some of CAP_CLAUSES and nothing else, gained nothing, and
    only probes and units moved (the flags still report RUNAWAY)."""
    return set(c["fields"]) <= {"probes", "unit"} and c["probes"] and all(
        set(p["old"]) - set(p["new"]) <= CAP_CLAUSES and set(p["new"]) <= set(p["old"]) for p in c["probes"])


A2B_LOST = {"a2_cue", "a4_voice"}


def lookup_x(rid, turn):
    """the probe at that turn of record rid is a LOOKUP X probe (the dev records, loaded here when not yet)."""
    if not RECS:
        RECS.update({r["id"]: r for r in RN.load()})
    rec = RECS.get(rid)
    p = next((p for p in rec["probes"] if p["turn"] == turn), None) if rec else None
    return p is not None and rec["family"] == "LOOKUP" and p["kind"] == "X"


def expected_a2b(c):
    """A2b (2026-11-19, rc12/notes.txt STEP 16A2B): see the module docstring."""
    return set(c["fields"]) <= {"probes", "unit"} and bool(c["probes"]) and all(
        lookup_x(c["id"], p["turn"]) and set(p["old"]) - set(p["new"]) <= A2B_LOST and set(p["new"]) < set(p["old"])
        for p in c["probes"]) and c["unit"][1] >= c["unit"][0]


def expected_a2c(c):
    """A2c (2026-11-19, rc12/notes.txt STEP 16A2C): see the module docstring."""
    if not c["probes"] or not set(c["fields"]) <= {"probes", "unit"} or c["unit"][1] > c["unit"][0]:
        return False
    return all(lookup_x(c["id"], p["turn"]) and set(p["new"]) - set(p["old"]) == {"a3_value"}
               and set(p["old"]) < set(p["new"]) for p in c["probes"])


EXPECT = {"item3": expected_item3, "t0gaps": expected_t0gaps, "first": expected_first, "cap": expected_cap,
          "a2": expected_a2, "a2b": expected_a2b}
EXPECT["a2c"] = expected_a2c                     # lock step A2c (2026-11-19)


def expected_a2b_r2(c):
    """the late review of A2b (2026-11-22, rc12/notes.txt STEP 16A2B-R2): see the module docstring."""
    moved = [set(p["old"]) ^ set(p["new"]) for p in c["probes"]]
    return bool(moved) and set(c["fields"]).issubset({"probes", "unit"}) and all(
        m == {"a2_cue"} and lookup_x(c["id"], p["turn"]) for m, p in zip(moved, c["probes"]))


EXPECT["a2b_r2"] = expected_a2b_r2               # the late review of lock step A2b (2026-11-22)


def expected_a2d(c):
    """lock step A2d (2026-11-23, rc12/notes.txt STEP 16A2D): see the module docstring."""
    moved = [(set(p["new"]) - set(p["old"]), set(p["old"]) - set(p["new"])) for p in c["probes"]]
    ok_fields = set(c["fields"]) <= {"probes", "unit"}
    return len(moved) > 0 and ok_fields and c["unit"][1] <= c["unit"][0] and all(
        g == {"a2_cue"} and not lost and lookup_x(c["id"], p["turn"]) for (g, lost), p in zip(moved, c["probes"]))


EXPECT["a2d"] = expected_a2d                     # lock step A2d (2026-11-23)


def run_dirs(src):
    out = []
    for model in sorted(os.listdir(src)):
        for render in ("template", "plain"):
            d = os.path.join(src, model, render)
            if not os.path.isdir(d):
                continue
            out += [os.path.join(model, render, x) for x in sorted(os.listdir(d))
                    if not x.endswith(".partial") and os.path.exists(os.path.join(d, x, "DONE"))]
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
    tmp, aside = out + ".partial", out + ".old.partial"
    shutil.rmtree(tmp, ignore_errors=True)       # a cut regrade's leftovers
    shutil.rmtree(aside, ignore_errors=True)
    os.makedirs(tmp)
    with open(os.path.join(tmp, "transcripts.jsonl"), "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in new_rows)
    with open(os.path.join(tmp, "scores.jsonl"), "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in S.score_lines(new_rows))
    shutil.copy2(os.path.join(src, rel, "meta.json"), os.path.join(tmp, "meta.json"))
    with open(os.path.join(tmp, "DONE"), "w") as f:
        f.write("regraded by dev_regrade.py\n")
    if os.path.exists(out):                      # a rerun: the earlier copy goes aside in one rename, so a cut never
        os.replace(out, aside)                   # leaves a half-deleted run beside its DONE
    DB.publish_run(tmp, out)
    shutil.rmtree(aside, ignore_errors=True)
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
    stale = os.path.join(dst, "regrade.json")
    if os.path.exists(stale):                    # a rerun: the earlier summary goes first, so a cut never leaves it
        os.remove(stale)                         # describing a --dst that is half redone
        DB.fsync_path(dst, is_dir=True)
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
    path = os.path.join(dst, "regrade.json")
    with open(path + ".tmp", "w") as f:
        json.dump(dict(src=src, expect=args.expect, runs=len(res), rows=rows, rows_changed=len(ch), per_run=res), f,
                  indent=0)
    DB.fsync_path(path + ".tmp")
    os.replace(path + ".tmp", path)
    DB.fsync_path(dst, is_dir=True)
    return 0 if all(r["expected"] for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())
