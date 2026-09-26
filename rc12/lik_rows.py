"""RC-12 likelihood rows (prereg draft s4, P-004): the row builder. Diagnostic only: never in a composite or a claim.

Every dev probe with a CLOSED candidate set gets one row on GOLDEN history: the user turns and the IDEAL replies of
every turn before the probe (never a model's reply), then the probe turn, then a forced answer prefix (lead-in)
from pools_lik_prefix (its own pool and seed). lik_score.py scores the rows; this file loads no model and no
tokenizer, and runner.py is not involved.
Closed candidate set (decided 2026-09-26, rc12/notes.txt STEP 11):
  VAL probes whose gold is one of >= 2 candidates: RECALL P, CORR, BIND, TWOHOP (COMPOSE included, keyed COMPOSE
    and diagnostic as in score.py), TOPIC, ROLE P, LOOKUP P;
  OWN pick (DYN, gold_fn type "pick"): on golden history the IDEAL Q reply is the model's own reply, so the gold is
    the probe's ideal_gold (the option that reply picked; test_lik.py re-parses it with G-DYN's own parser) and
    the candidates are the probe's candidates (the offered options and the lure picks).
  Excluded, counted by reason: "knowledge" (K; knowledge-bearing is never pooled, L4), "no_gold" (ABS abstain
    probes, OWN list, LOOP, PERSIST FMT), "open_question" (ROLE X, graded by G-ROLEX: "Tell me a little about
    yourself." has no answer slot to force a value into), "one_candidate" (T0 and the RECALL abstain X probe: no
    other candidate to compare the gold with).
Row: id "<rid>:<turn>", rid, family, cell, key, pair_id (BIND twins), turn, messages (golden history ending with
  the probe's user turn), lead, gold, candidates (record order). The lead-in is drawn per record, except that BIND
  twins draw it from their pair_id (lead_key), so the two twins differ only in the order of the value statements.
CLI: python3 -B lik_rows.py [--data FILE] prints the row and exclusion counts per key."""
import argparse
import json
import os
from collections import Counter

import pools_lik_prefix as LP
import pools_vals as V
import score as S

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.join(HERE, "dev", "rc12_dev.jsonl")


def load(path=DEV):
    return [json.loads(line) for line in open(path)]


def key_of(rec):
    return "COMPOSE" if (rec["family"], rec["cell"]) in S.DIAG else rec["family"]


def closed(rec, probe):
    """-> (gold, None) when the probe has a closed candidate set, else (None, reason)."""
    if rec["knowledge"]:
        return None, "knowledge"
    if probe["grader"] == "ROLEX":
        return None, "open_question"
    if probe["grader"] == "DYN" and (probe.get("gold_fn") or {}).get("type") == "pick":
        gold = probe.get("ideal_gold")
    elif probe["grader"] == "VAL":
        gold = probe.get("gold")
    else:
        return None, "no_gold"
    if gold is None:
        return None, "no_gold"
    cands = probe.get("candidates") or []
    if len(cands) < 2:
        return None, "one_candidate"
    if gold not in cands:
        raise ValueError(f"{rec['id']} turn {probe['turn']}: gold {gold!r} is not a candidate")
    return gold, None


def lead_key(rec):
    """the lead-in stream key: the pair_id for twins (BIND: a minimal pair shares its lead-in), else the record id."""
    return rec["meta"].get("pair_id") or rec["id"]


def golden(rec, turn):
    """messages: user text and IDEAL reply of every turn before `turn`, then the user text of `turn`."""
    msgs, last = [], None
    for t in sorted(rec["turns"], key=lambda x: x["i"]):
        if t["i"] < turn:
            msgs.append({"role": "user", "content": t["text"]})
            msgs.append({"role": "assistant", "content": t["ideal"]})
        elif t["i"] == turn:
            last = t
    if last is None or len(msgs) != 2 * (turn - 1):
        raise ValueError(f"{rec['id']}: turns 1..{turn} are not all present")
    msgs.append({"role": "user", "content": last["text"]})
    return msgs


def build(recs):
    """-> (rows, excluded Counter {(key, reason): n})."""
    rows, excluded = [], Counter()
    for rec in recs:
        for p in rec["probes"]:
            gold, why = closed(rec, p)
            if why:
                excluded[(key_of(rec), why)] += 1
                continue
            msgs = golden(rec, p["turn"])
            if msgs[-1]["content"] != p["question"]:
                raise ValueError(f"{rec['id']} turn {p['turn']}: the probe turn is not the question")
            lead = LP.lead_for(lead_key(rec), p["turn"])
            hit = [c for c in p["candidates"] if V.mentions(lead, c, False)]
            if hit:
                raise ValueError(f"{rec['id']}: lead-in {lead!r} mentions {hit}")
            rows.append(dict(id=f"{rec['id']}:{p['turn']}", rid=rec["id"], family=rec["family"], cell=rec["cell"],
                             key=key_of(rec), pair_id=rec["meta"].get("pair_id"), turn=p["turn"], messages=msgs,
                             lead=lead, gold=gold, candidates=list(p["candidates"])))
    return rows, excluded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEV)
    args = ap.parse_args()
    rows, excluded = build(load(args.data))
    print(json.dumps(dict(rows=dict(sorted(Counter(r["key"] for r in rows).items())),
                          excluded={f"{k}:{w}": n for (k, w), n in sorted(excluded.items())}), indent=1))
    print(f"{len(rows)} rows")


if __name__ == "__main__":
    main()
