"""dev_regrade.py writes durably (cuts audit row 143, 2026-10-11: its DONE used to be written in place, beside files no
fsync had reached). In-process, no model: two stored runs are made with dev_batch.py and fake:IDEAL_ALT (greedy and
its own-cf twin, 6 conversations), then regraded with the working-tree graders. os.fsync and os.replace are recorded in
call order (test_dev_batch_durable.recorded). Checks:
  files     transcripts.jsonl, scores.jsonl, meta.json and DONE of every regraded run are fsynced BEFORE the
            <run>.partial rename, the run dir before it and its parent after it (test_dev_batch_durable.check_run)
  content   the regraded rows and scores equal the stored ones (same graders: nothing moves), meta.json is the
            source's, DONE says "regraded"; main() exits 0 and writes regrade.json with 2 runs
  json      regrade.json is fsynced before its rename into place, --dst after it, no regrade.json.tmp left; on a
            rerun the earlier regrade.json is removed, and --dst fsynced, before the runs begin
  rerun     a second main() over the same --dst replaces every run (no error, the same bytes): the earlier copy
            goes aside to <run>.old.partial in one rename before the publish, and is gone after it
  leftover  a <run>.partial and <run>.old.partial a cut left in --dst (a stray file in each) are cleared and the
            run republished,
            never published
  midcut    a cut between the aside rename and the publish (no <run>, a whole <run>.old.partial and a <run>.partial
            with DONE): the next regrade makes the run whole and leaves no leftover
  skip      a source run dir named *.partial, even with DONE inside, is not a run (run_dirs)
Run: python3 -B test_dev_regrade.py   (exit 1 on any failure)"""
import os
import shutil
import sys
import tempfile

import dev_batch as DB
import dev_regrade as DR
from test_dev_batch_durable import check_run, key, recorded

RUNS = (os.path.join("IDEAL_ALT", "plain", "greedy"), os.path.join("IDEAL_ALT", "plain", "greedy_owncf"))
FILES = ("transcripts.jsonl", "scores.jsonl", "meta.json")


def make_src(root):
    old = sys.argv
    sys.argv = ["dev_batch.py", "--responder", "fake:IDEAL_ALT", "--render", "plain", "--root", root, "--limit", "6",
                "--seeds", "greedy"]
    try:
        DB.main()
    finally:
        sys.argv = old


def regrade_main(src, dst):
    old = sys.argv
    sys.argv = ["dev_regrade.py", "--src", src, "--dst", dst, "--procs", "1"]
    try:
        return DR.main()
    finally:
        sys.argv = old


def read(p):
    with open(p, "rb") as f:
        return f.read()


def checks(base):
    fails = []
    src, dst = os.path.join(base, "src"), os.path.join(base, "dst")
    make_src(src)
    if DR.run_dirs(src) != list(RUNS):
        return [f"setup: run_dirs {DR.run_dirs(src)}"]
    ev = recorded(lambda: [DR.regrade((src, dst, rel, "item3")) for rel in RUNS])
    for rel in RUNS:
        d = os.path.join(dst, rel)
        fails += check_run(ev, d)
        for n in FILES[:2]:
            if os.path.exists(os.path.join(d, n)) and read(os.path.join(d, n)) != read(os.path.join(src, rel, n)):
                fails.append(f"content: {rel}/{n} differs from the stored run (same graders)")
        if os.path.exists(os.path.join(d, "meta.json")) and read(os.path.join(d, "meta.json")) != read(
                os.path.join(src, rel, "meta.json")):
            fails.append(f"content: {rel}/meta.json is not the source's")
        if os.path.exists(os.path.join(d, "DONE")) and b"regraded" not in read(os.path.join(d, "DONE")):
            fails.append(f"content: {rel}/DONE does not say regraded")
        if os.path.exists(d + ".partial"):
            fails.append(f"files: {rel}.partial left behind")
    first = {rel: read(os.path.join(dst, rel, "transcripts.jsonl")) for rel in RUNS
             if os.path.exists(os.path.join(dst, rel, "transcripts.jsonl"))}
    rev = recorded(lambda: [DR.regrade((src, dst, rel, "item3")) for rel in RUNS])   # in-process: Pool hides renames
    for rel in RUNS:
        d = os.path.abspath(os.path.join(dst, rel))
        moves = [e[1] for e in rev if e[0] == "replace" and e[1] in (d + ".old.partial", d)]
        if moves != [d + ".old.partial", d]:
            fails.append(f"rerun: {rel} not moved aside in one rename before the publish ({len(moves)} renames)")
        if os.path.exists(d + ".old.partial"):
            fails.append(f"rerun: {rel}.old.partial left behind")
    open(os.path.join(dst, "regrade.json"), "w").write('{"stale": true}\n')     # an earlier pass's summary
    rc, seen, real = None, [], DR.Pool
    marker = os.path.join(base, "pool_marker")

    def pool(*a, **k):                            # the moment the runs begin: is the earlier summary gone, durably?
        seen.append(os.path.exists(os.path.join(dst, "regrade.json")))
        open(marker, "w").close()
        os.replace(marker, marker + ".at_pool")   # a recorded event that marks this moment in jev
        return real(*a, **k)
    DR.Pool = pool
    try:
        jev = recorded(lambda: ev.append(("rc", regrade_main(src, dst))))
        rc = [e[1] for e in ev if e[0] == "rc"][-1]
    except Exception as e:  # noqa: BLE001
        return fails + [f"rerun: main() over an existing --dst raised {type(e).__name__}: {str(e)[:120]}"]
    finally:
        DR.Pool = real
    if seen != [False]:
        fails.append(f"json: the earlier regrade.json was still there when the rerun's runs began ({seen})")
    at = [i for i, e in enumerate(jev) if e == ("replace", os.path.abspath(marker + ".at_pool"))]
    if not at or ("fsync", key(dst)) not in jev[:at[0]]:
        fails.append("json: the earlier regrade.json's removal was not fsynced before the rerun's runs began")
    if rc != 0:
        fails.append(f"content: main() exit {rc}, want 0 (no change under the same graders)")
    j = os.path.join(dst, "regrade.json")
    if not os.path.exists(j):
        fails.append("json: no regrade.json")
    else:
        if '"runs": 2' not in open(j).read():
            fails.append("json: regrade.json does not count 2 runs")
        idx = [i for i, e in enumerate(jev) if e == ("replace", os.path.abspath(j))]
        if not idx or ("fsync", key(j)) not in jev[:idx[-1]]:
            fails.append("json: regrade.json not fsynced before its rename into place")
        elif ("fsync", key(dst)) not in jev[idx[-1] + 1:]:
            fails.append("json: --dst not fsynced after regrade.json's rename")
    if os.path.exists(j + ".tmp"):
        fails.append("json: regrade.json.tmp left behind")
    for rel in RUNS:
        p = os.path.join(dst, rel, "transcripts.jsonl")
        if not os.path.exists(p) or read(p) != first.get(rel):
            fails.append(f"rerun: {rel} not replaced by the same bytes")
    rel = RUNS[0]
    os.makedirs(os.path.join(dst, rel + ".partial"), exist_ok=True)
    open(os.path.join(dst, rel + ".partial", "stray"), "w").write("cut\n")
    os.makedirs(os.path.join(dst, rel + ".old.partial"), exist_ok=True)
    open(os.path.join(dst, rel + ".old.partial", "stray"), "w").write("cut\n")
    try:
        DR.regrade((src, dst, rel, "item3"))
    except Exception as e:  # noqa: BLE001
        fails.append(f"leftover: a cut .partial or .old.partial made the regrade raise {type(e).__name__}")
    if os.path.exists(os.path.join(dst, rel, "stray")) or any(os.path.exists(os.path.join(dst, rel + x))
                                                             for x in (".partial", ".old.partial")):
        fails.append("leftover: a cut .partial or .old.partial was published or left")
    if not all(os.path.exists(os.path.join(dst, rel, n)) for n in FILES + ("DONE",)):
        fails.append("leftover: the run was not republished after clearing the leftovers")
    out = os.path.join(dst, rel)                 # the cut between the aside rename and the publish
    try:
        os.replace(out, out + ".old.partial")
        shutil.copytree(out + ".old.partial", out + ".partial")
    except Exception as e:  # noqa: BLE001
        return fails + [f"midcut: setup failed ({type(e).__name__}): an earlier case left the run dir unclean"]
    try:
        DR.regrade((src, dst, rel, "item3"))
    except Exception as e:  # noqa: BLE001
        fails.append(f"midcut: a cut between the aside rename and the publish made the rerun raise {type(e).__name__}")
    if not all(os.path.exists(os.path.join(out, n)) for n in FILES + ("DONE",)) or any(
            os.path.exists(out + x) for x in (".partial", ".old.partial")):
        fails.append("midcut: the run is not whole again, or a leftover remains")
    shutil.copytree(os.path.join(src, rel), os.path.join(src, rel + ".partial"))
    if DR.run_dirs(src) != list(RUNS):
        fails.append(f"skip: a *.partial source dir is read as a run ({DR.run_dirs(src)})")
    return fails


def main():
    with tempfile.TemporaryDirectory() as base:
        try:
            fails = checks(base)
        except Exception as e:  # noqa: BLE001  (a raising check is a failure line, not a crash)
            fails = [f"checks raised {type(e).__name__}: {str(e)[:200]}"]
    print("\n".join(f"FAIL {f}" for f in fails) or "ALL DEV_REGRADE CHECKS PASS (fake:IDEAL_ALT, no model)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
