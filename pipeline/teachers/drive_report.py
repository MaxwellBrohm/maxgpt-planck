"""Yield report for one drive.py run dir (DRY: FAKE skeletons, served teacher; nothing here is training data). CPU only.

    python3 -B pipeline/teachers/drive_report.py RUN_DIR --teacher NAME [--k 8] [--json OUT.dry.json] \
        [--export DIR] [--examples 10]

Reads the driver's own records (skeletons/, accepted/, rejects/, yield.jsonl) and reports:
  yield per attempt, per skeleton (up to --max-attempts tries) and on the first attempt;
  every checker code in checker.ORDER (59) plus the driver's own codes (TEACHER_ERROR, CHECKER_ERROR, DUP_*): primary
  rejects, attempts hit (accepted records' report-only codes included) and rejects failing on that code alone;
  finish reasons; the last driver session's wall time and teacher completion tokens per second, and the wall time
  and completion tokens summed over every session (a resumed run has more than one);
  diversity of the accepted chats: stats.text_stats (distinct-1/2/3, gzip ratio; all turns, user, assistant), the
  same on teacher-written turns only, and, with --k, on the first k accepted chats so teachers compare at equal n;
  yield per attempt kind (first, retry, repair: a --repair attempt is counted apart), the run flags seen in the
  records, and what the server applied (structured, dash ban, preset) over the teacher calls (2026-09-27).
--export writes accepted_texts.dry.jsonl (turn text + author, for count_planck.py on the PC, where the tokenizers
package lives) and --examples accepted plus --examples rejected records (seeded sample; rejects spread over the
primary codes) as dry jsonl."""
import argparse
import collections
import json
import os
import random
import sys

sys.dont_write_bytecode = True
PIPE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PIPE)
import checker  # noqa: E402
import driver  # noqa: E402
import shards  # noqa: E402
import stats  # noqa: E402

DRIVER_CODES = ["TEACHER_ERROR", "CHECKER_ERROR", "DUP_EXACT", "DUP_NEAR"]
DRY = {"status": "dry", "not_training_data": True}


def load(run):
    acc = list(shards.read(os.path.join(run, "accepted"), "accepted"))
    rej = list(shards.read(os.path.join(run, "rejects"), "rejects"))
    man = list(shards.read(os.path.join(run, "skeletons"), "skeletons"))
    snaps = []
    with open(os.path.join(run, "yield.jsonl"), encoding="utf-8") as f:
        snaps = [json.loads(ln) for ln in f if ln.strip()]
    return acc, rej, man, snaps


def sessions(snaps, tol=0.05):
    """[(start time, elapsed_s, completion_tokens)] per driver session. yield.jsonl restarts its session clock on a
    resume, so a run's wall time is the sum over sessions; a session is keyed by its start (time - elapsed_s). A
    session killed between snapshots loses up to --log-every seconds; drive.py snapshots when the server goes."""
    out = []
    for sn in sorted(snaps, key=lambda x: x["time"]):
        se = sn.get("session") or {}
        el = se.get("elapsed_s") or 0.0
        start = sn["time"] - el
        if out and abs(out[-1][0] - start) <= tol:
            out[-1] = (out[-1][0], max(out[-1][1], el), max(out[-1][2], se.get("completion_tokens") or 0))
        else:
            out.append((start, el, se.get("completion_tokens") or 0))
    return out


def teacher_turns(recs):
    return [t["text"] for r in recs for t in r["turns"] if str(t["author"]).startswith("teacher:")]


def diversity(recs):
    if not recs:
        return None
    out = stats.text_stats(recs)
    texts = teacher_turns(recs)
    toks = [stats.words(t) for t in texts]
    out["teacher_turns"] = {"turns": len(texts), "words": sum(map(len, toks)), "distinct_1": stats.distinct(toks, 1),
                            "distinct_2": stats.distinct(toks, 2), "distinct_3": stats.distinct(toks, 3),
                            "gzip_ratio": stats.gzip_ratio(texts)}
    out["chats"] = len(recs)
    return out


def code_table(acc, rej):
    prim, hit, only = collections.Counter(), collections.Counter(), collections.Counter()
    for r in rej:
        prim[r["primary"]] += 1
        for c in set(r["codes"]) | {r["primary"]}:
            hit[c] += 1
        failing = [c for c in r["codes"] if c not in checker.REPORT_ONLY]
        if len(failing) == 1:
            only[failing[0]] += 1
    for r in acc:
        for c in set(r["check"]["codes"]):
            hit[c] += 1
    names = checker.ORDER + DRIVER_CODES + sorted((set(prim) | set(hit)) - set(checker.ORDER) - set(DRIVER_CODES))
    return {c: {"primary": prim[c], "hit": hit[c], "only": only[c]} for c in names}


def by_kind(acc, rej):
    att, ok = collections.Counter(), collections.Counter()
    for r in acc:
        att[driver.attempt_kind(r)] += 1
        ok[driver.attempt_kind(r)] += 1
    for r in rej:
        att[driver.attempt_kind(r)] += 1
    return {k: {"attempted": att[k], "accepted": ok[k], "yield": round(ok[k] / att[k], 4) if att[k] else None}
            for k in driver.KINDS}


def decode_applied(acc, rej):
    """over the records of teacher calls: how many had structured output, the dash ban, and which preset."""
    tms = [r["provenance"]["teacher"] for r in acc] + [r["teacher"] for r in rej if r.get("teacher")]
    decs = [tm.get("decode") for tm in tms if tm.get("decode")]
    flags = sorted({json.dumps(r.get("run_flags"), sort_keys=True) for r in acc + rej})
    return {"run_flags": [json.loads(f) for f in flags], "calls_with_decode": len(decs),
            "structured": dict(collections.Counter(str(d.get("structured")) for d in decs)),
            "dash_ban": sum(1 for d in decs if d.get("ban")),
            "presets": dict(collections.Counter(d.get("preset") for d in decs))}


def pick_rejects(rej, n, rng):
    by = collections.defaultdict(list)
    for r in rej:
        by[r["primary"]].append(r)
    for v in by.values():
        rng.shuffle(v)
    order = sorted(by, key=lambda c: -len(by[c]))
    out = []
    while len(out) < n and any(by.values()):
        for c in order:
            if by[c] and len(out) < n:
                out.append(by[c].pop())
    return out


def report(run, teacher, k=None):
    acc, rej, man, snaps = load(run)
    skels = {m["skel"]["skel_id"] for m in man}
    ok_skel = {r["skel_id"] for r in acc}
    first_ok = sum(1 for r in acc if r["attempt"] == 0)
    attempts = len(acc) + len(rej)
    fin = collections.Counter([r.get("finish") for r in rej] + [r["check"].get("finish") for r in acc])
    final = snaps[-1] if snaps else {}
    sess = final.get("session", {})
    ses = sessions(snaps)
    out = {**DRY, "teacher": teacher, "run": os.path.basename(os.path.normpath(run)), "skeletons": len(skels),
           "attempts": attempts, "accepted": len(acc), "rejected": len(rej),
           "yield_attempt": round(len(acc) / attempts, 4) if attempts else None,
           "yield_skeleton": round(len(ok_skel) / len(skels), 4) if skels else None,
           "yield_first_attempt": round(first_ok / len(skels), 4) if skels else None,
           "accepted_on_retry": len(acc) - first_ok, "finish": dict(fin.most_common()),
           "driver_final": final.get("final"), "elapsed_s": sess.get("elapsed_s"),
           "sessions": len(ses), "elapsed_s_all_sessions": round(sum(x[1] for x in ses), 3),
           "completion_tokens_all_sessions": sum(x[2] for x in ses),
           "completion_tokens": sess.get("completion_tokens"),
           "completion_tokens_per_s": sess.get("completion_tokens_per_s"), "mean_latency_s": sess.get("mean_latency_s"),
           "codes": code_table(acc, rej), "diversity": diversity(acc), "by_attempt_kind": by_kind(acc, rej),
           "decode": decode_applied(acc, rej)}
    if k:
        first = sorted(acc, key=lambda r: r["conv_id"])[:k]
        out["diversity_at_k"] = {"k": k, "have": len(first), "stats": diversity(first) if len(first) == k else None}
    rep = stats.repetition(acc, top=3) if acc else None
    out["repetition_top"] = rep
    return out, acc, rej


def export(d, acc, rej, n_examples, seed=0):
    os.makedirs(d, exist_ok=True)
    rng = random.Random(seed)
    with open(os.path.join(d, "accepted_texts.dry.jsonl"), "w", encoding="utf-8") as f:
        for r in acc:
            f.write(json.dumps({**DRY, "conv_id": r["conv_id"], "turns": [
                {"role": t["role"], "author": t["author"], "text": t["text"]} for t in r["turns"]]}) + "\n")
    picks = rng.sample(acc, min(n_examples, len(acc)))
    with open(os.path.join(d, "accepted_examples.dry.jsonl"), "w", encoding="utf-8") as f:
        for r in picks:
            f.write(json.dumps({**DRY, **r}) + "\n")
    with open(os.path.join(d, "rejected_examples.dry.jsonl"), "w", encoding="utf-8") as f:
        for r in pick_rejects(rej, n_examples, rng):
            f.write(json.dumps({**DRY, **r}) + "\n")
    return len(picks)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--export", default=None)
    ap.add_argument("--examples", type=int, default=10)
    a = ap.parse_args(argv)
    out, acc, rej = report(a.run, a.teacher, a.k)
    if a.export:
        out["exported_accepted_examples"] = export(a.export, acc, rej, a.examples)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    brief = {k: out[k] for k in ("teacher", "skeletons", "attempts", "accepted", "yield_attempt", "yield_skeleton",
                                 "yield_first_attempt", "elapsed_s", "sessions", "elapsed_s_all_sessions",
                                 "completion_tokens_per_s", "by_attempt_kind")}
    brief["run_flags"] = out["decode"]["run_flags"]
    brief["top_primary"] = {c: v["primary"] for c, v in sorted(out["codes"].items(), key=lambda x: -x[1]["primary"])
                            if v["primary"]}
    print(json.dumps(brief, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
