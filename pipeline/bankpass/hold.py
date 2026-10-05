"""One teacher's work inside one gpu.lock hold (BANKPASS s2i, s6 W3). Against a running bankserve.py on loopback, it
loops: refresh the stage view (CPU), list everything this teacher still owes (its base plan calls, the calls derived
from finished outputs, the judgments of other teachers' items), and run them concurrently, base calls first, until
nothing is owed or the deadline. Every finished call is one fsynced JSONL record in calls/<teacher>.jsonl; a restart
skips done ids (a call that errored twice is given up and reported). A kind whose first 8 calls in a hold give 6 or
more parse problems is stopped for the hold (BAD_KIND, written to the log and bad_kinds.json) so a broken prompt
does not eat the hold. Exit 0: nothing owed; 3: deadline with work left; 4: server gone; 5: a bad kind.

    python -m bankpass.hold --root STAGE --teacher qwen3.5-9b --port 18851 --deadline EPOCH --workers 32 \
        --wordlist WORDLIST.tsv --max-ok '{"by": "Max", "date": "2026-10-04", ...}'"""
import argparse
import collections
import json
import os
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from bankpass import (client as CL, gen, store, w3amend, w3deps as WD, w3judge as WJ, w3plan, w3render as WR,
                      w3state as WS)

SMOKE_N, SMOKE_BAD = 8, 6
CODE_FILES = ("prompts", "prompts3", "prompts4", "judge3", "w3render", "w3items", "w3plan", "w3fill", "w3deps",
              "w3amend", "w3judge", "w3state", "itemize", "templatize", "gates", "judge", "bankserve", "client", "hold")


def code_hash():
    here = os.path.dirname(os.path.abspath(__file__))
    return store.sha256_text("".join(store.sha256_file(os.path.join(here, f + ".py")) for f in CODE_FILES))[:16]


def record(c, prompt, spec, row, max_ok, code, t0):
    lines, likes, problem = WR.parse(c, row.get("text"), WR.render(c)[2], spec)
    pv = WR.PROMPT_VERSIONS
    return {"schema": store.CALL_SCHEMA, "call_id": c["call_id"], "kind": c["kind"], "bank": c["bank"],
            "class": c["class"], "author": store.teacher_author(c["teacher"]), "plan": c, "prompt": prompt,
            "prompt_id": f"{pv['prompts']}+{pv['prompts3']}+{pv['prompts4']}:{c['kind']}{'.v2' if c.get('v') == 2 else ''}"
                         f":{c['bank']}",
            "prompt_sha256": store.sha256_text(prompt), "decode_sha256": spec["sha256"],
            "regex_sha256": WR.P3.D.sha(spec["regex"]), "seed": c["seed"], "preset": WR.preset(c),
            "server_decode": row.get("decode"), "raw": row.get("text"), "raw_sha256": store.sha256_text(row.get("text") or ""),
            "finish": row.get("finish"), "n_out": row.get("n_out"), "n_prompt": row.get("n_prompt"), "lines": lines,
            "likelihoods": likes, "problem": problem, "time": round(t0, 3), "seconds": round(time.time() - t0, 3),
            "mode": "real", "max_ok": max_ok, "code": code}


def one(cl, c, max_ok, code):
    t0 = time.time()
    prompt, spec, _ = WR.render(c)
    try:
        row = cl.generate(prompt, spec, c["seed"], WR.preset(c), call=c)
    except CL.CallError as e:
        return {"schema": store.CALL_SCHEMA, "call_id": c["call_id"], "kind": c["kind"], "bank": c["bank"],
                "author": store.teacher_author(c["teacher"]), "error": str(e)[:400], "time": round(t0, 3),
                "mode": "real", "max_ok": max_ok, "code": code}
    return record(c, prompt, spec, row, max_ok, code, t0)


def run_calls(cl, calls, path, deadline, workers, max_ok, code, log, bad, seen):
    """run calls not yet done; -> (n finished, server gone)."""
    done = WS.done_ids(store.read_jsonl(path))
    todo = iter([c for c in calls if c["call_id"] not in done])
    lock, futs, n, gone = threading.Lock(), {}, 0, False
    with ThreadPoolExecutor(workers) as ex:
        while True:
            while len(futs) < workers and not gone and time.time() < deadline:
                c = next(todo, None)
                if c is None:
                    break
                if c["kind"] in bad:
                    continue
                futs[ex.submit(one, cl, c, max_ok, code)] = c
            if not futs:
                break
            fin, _ = wait(futs, timeout=5, return_when=FIRST_COMPLETED)
            for f in fin:
                c = futs.pop(f)
                try:
                    rec = f.result()
                except CL.ServerGone as e:
                    gone = True
                    log(f"server gone: {e}")
                    continue
                with lock:
                    store.append_jsonl(path, rec)
                n += 1
                k = seen.setdefault(c["kind"], [0, 0])
                if k[0] < SMOKE_N and not c.get("retry_of"):     # retries are the hard cases: never a brake
                    k[0] += 1
                    k[1] += bool(rec.get("problem") or rec.get("error"))
                    if k[0] == SMOKE_N and k[1] >= SMOKE_BAD and c["kind"] not in bad:
                        bad.add(c["kind"])
                        log(f"BAD_KIND {c['kind']}: {k[1]} of the first {SMOKE_N} calls failed; e.g. "
                            f"{(rec.get('problem') or rec.get('error'))!r} raw {str(rec.get('raw'))[:200]!r}")
    return n, gone


def owed(root, teacher, plan, wl):
    """everything this teacher still owes, in run order: base calls, derived calls, judgments."""
    WS.refresh(root, kinds={"topic", "pool", "label", "notewas", "listname", "para", "attr", "pred", "topen"})
    recs = WS.records(root)
    done = WS.done_ids(recs)
    its = WS.view(root, recs)
    bcalls = w3amend.base_calls(plan)
    base = [c for c in w3plan.ordered(bcalls, teacher) + w3amend.retries(recs, teacher) if c["call_id"] not in done]
    chosen = WD.select_topics(its, recs, wl, bcalls)
    sel = {it["id"] for it, _ in chosen}
    derived = (WD.feature_calls(teacher, its, recs) + WD.vote_calls(teacher, its, recs, bcalls)
               + WD.topic_calls(teacher, chosen) + WJ.group_calls(teacher, chosen, done)
               + WJ.check_calls(teacher, its, done))
    derived = [c for c in derived if c["call_id"] not in done]
    judges = [c for c in WJ.judge_calls(teacher, its, sel) if c["call_id"] not in done]
    return base, derived, judges, len(chosen)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--teacher", required=True, choices=sorted(store.TEACHERS))
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--deadline", type=float, required=True, help="epoch seconds: no call is started after it")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--wordlist", required=True)
    ap.add_argument("--max-ok", required=True)
    ap.add_argument("--status-only", action="store_true", help="print what is owed and exit (CPU only)")
    a = ap.parse_args(argv)
    max_ok = json.loads(a.max_ok)
    WS.install_human()
    gen.check_out_path(os.path.join(a.root, "calls", "x.jsonl"), "real", max_ok)
    from bankpass import wordload
    wl = wordload.read(a.wordlist)
    with open(os.path.join(a.root, "plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    logp = os.path.join(a.root, "logs", f"hold_{store.SHORT[a.teacher]}.log")
    os.makedirs(os.path.dirname(logp), exist_ok=True)

    def log(msg):
        line = f"{time.strftime('%F %T')} [{a.teacher}] {msg}"
        print(line, flush=True)
        with open(logp, "a") as f:
            f.write(line + "\n")
    if a.status_only:
        b, d, j, nsel = owed(a.root, a.teacher, plan, wl)
        print(json.dumps({"base": len(b), "derived": len(d), "judge": len(j), "topics_selected": nsel,
                          "derived_kinds": dict(collections.Counter(c["kind"] for c in d))}))
        return 0 if not (b or d or j) else 3
    cl = CL.Client(f"http://127.0.0.1:{a.port}", a.teacher)
    cl.check()
    code, path, bad, seen = code_hash(), WS.calls_path(a.root, a.teacher), set(), {}
    log(f"start code {code} deadline in {round(a.deadline - time.time())} s workers {a.workers}")
    for rnd in range(12):
        b, d, j, nsel = owed(a.root, a.teacher, plan, wl)
        todo = [c for c in b + d + j if c["kind"] not in bad]
        log(f"round {rnd}: owed base {len(b)} derived {len(d)} judge {len(j)} (topics selected {nsel}); bad {sorted(bad)}")
        if not todo:
            break
        if time.time() >= a.deadline:
            log("deadline reached with work owed")
            return 3
        t0 = time.time()
        n, gone = run_calls(cl, todo, path, a.deadline, a.workers, max_ok, code, log, bad, seen)
        log(f"round {rnd}: {n} calls finished in {round(time.time() - t0)} s")
        if gone:
            return 4
    b, d, j, _ = owed(a.root, a.teacher, plan, wl)
    if bad:
        with open(os.path.join(a.root, "logs", f"bad_kinds_{store.SHORT[a.teacher]}.json"), "w") as f:
            json.dump(sorted(bad), f)
        return 5
    return 0 if not (b or d or j) else 3


if __name__ == "__main__":
    sys.exit(main())
