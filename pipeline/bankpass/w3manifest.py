"""W3 stage P manifest (PC CPU, after the last hold): per bank counts, author shares and drop codes, the GPU holds with
their times, and the hashes of every record file. No item text: this file goes into the repo
(pipeline/bankpass/manifests/), the items stay on the PC (Max's decision 2 (a)).

Counting rules. generated: lines parsed from finished author calls of the current version (calls an amendment
superseded are counted apart, under superseded_lines). gated: items that passed the s2c/s2d checks and the pre-judge
exact dedup. judged: gated items carrying both non-author judgments. kept: gated and both judges keep (unjudged banks:
gated). Topic-dependent banks (word sets, intents, topic openings) count only the selected topics.

    python -m bankpass.w3manifest --root STAGE --logs ~/planck/logs [--out manifest_w3.json]"""
import argparse
import collections
import json
import os
import re
import sys

import heldout
from bankpass import store, w3amend, w3deps as WD, w3report as WR, w3state as WS
from teachers import decode as D

SHORT = store.SHORT
HOLD_RE = re.compile(r"bp3hold \[(\S+) (\w+)\]: lock taken (\S+ \S+)")
REL_RE = re.compile(r"bp3hold \[(\S+) (\w+)\]: unloaded, releasing (\S+ \S+) .* held (\d+) s")
WAIT_RE = re.compile(r"(?:stage \d+ (\S+)|(\S+) probe): waiting for gpu.lock(?: \(try \d+\))? (\S+ \S+)")


def holds(log_paths):
    """[{teacher, mode, waited_from, taken, released, held_s}] from the master and probe logs, in time order."""
    out, waits = [], {}
    for p in log_paths:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = WAIT_RE.search(line)
                if m:
                    waits[m.group(1) or m.group(2)] = m.group(3)
                m = HOLD_RE.search(line)
                if m:
                    out.append({"teacher": m.group(1), "mode": m.group(2), "taken": m.group(3),
                                "waited_from": waits.pop(m.group(1), None)})
                m = REL_RE.search(line)
                if m:
                    for h in reversed(out):
                        if h["teacher"] == m.group(1) and "released" not in h:
                            h.update(released=m.group(3), held_s=int(m.group(4)))
                            break
    return sorted(out, key=lambda h: h["taken"])


def current(r):
    return not w3amend.superseded(r["call_id"]) and not (r.get("kind") == "line" and w3amend.amend4_bank(r["bank"])
                                                          and not w3amend.current6(r["call_id"]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--wordlist", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    WS.install_human()
    from bankpass import wordload
    wl = wordload.read(a.wordlist)
    with open(os.path.join(a.root, "plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    recs = WS.records(a.root)
    its = WS.view(a.root, recs)
    chosen = WD.select_topics(its, recs, wl, w3amend.base_calls(plan))
    sel = {it["id"] for it, _ in chosen}
    for it in its.values():          # topic-dependent banks: only the selected topics count
        tid = (it.get("features") or {}).get("topic_id")
        if tid and tid not in sel and it.get("w3") != "dropped":
            it.update(w3="dropped", drop_w3="TOPIC_NOT_SELECTED")
        if it["bank"] == "topic" and it["status"] == "kept" and it["id"] not in sel:
            it.update(w3="dropped", drop_w3="NOT_SELECTED")
    rows = WR.bank_table(its, [r for r in recs if current(r)], plan["targets"])
    sup = collections.Counter(r["bank"] for r in recs if not current(r) and not r.get("error")
                              for _ in r.get("lines") or [])
    for bank, row in rows.items():
        row["superseded_lines"] = sup.get(bank, 0)
    files = {}
    for t in WS.ORDER:
        for p in (WS.calls_path(a.root, t), WS.cache_path(a.root, t)):
            if os.path.exists(p):
                files[os.path.relpath(p, a.root)] = {"sha256": store.sha256_file(p), "bytes": os.path.getsize(p)}
    codes = collections.Counter(r.get("code") for r in recs)
    man = {"schema": "bankpass-w3-manifest-v1", "stage": "stageP_v1", "status": "W3 generated and judged; not frozen "
           "(W4: templatize review, dedup, frame caps, author thirds, p_exact, freeze)", "plan_sha256": plan["sha256"],
           "plan_version": plan["version"], "amendments": w3amend.__doc__.strip(), "gate_hash": heldout.gate_hash(),
           "rc12": heldout.RC12_STATUS, "phrase_ban": D.PHRASE_RULE, "dash_ban": D.DASH_RULE,
           "teachers": {t: dict(store.TEACHERS[t]) for t in WS.ORDER},
           "inputs_on_pc": plan["inputs"], "code_hashes": dict(codes), "files": files,
           "topics_selected": dict(collections.Counter(SHORT[a_] for _, a_ in chosen)),
           "banks": rows, "calls": WR.call_table(recs),
           "holds": holds([os.path.join(a.logs, "bp3_all.log"), os.path.join(a.logs, "bp3_probe.log")]),
           "licences": {"teacher items": "apache-2.0 (the three pinned teachers)", "fill values and voice seeds "
                        "(prompt side, templatized away)": "GeoNames cc-by-4.0, Nemotron-Personas-USA cc-by-4.0, "
                        "FAKE pools", "mined example sentences (prompt side)": "YouTube cc-by-4.0, OASST2 apache-2.0"}}
    out = a.out or os.path.join(a.root, "manifest_w3.json")
    with open(out + ".tmp", "w") as f:
        json.dump(man, f, indent=1, sort_keys=True)
    os.replace(out + ".tmp", out)
    print(json.dumps({"out": out, "banks": len(rows), "holds": len(man["holds"]), "topics": man["topics_selected"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
