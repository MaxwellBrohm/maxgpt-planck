"""Yield tallies for the driver (SPEC 11 yield.jsonl): accepted / attempted overall, by primary reject code, by every
code, and by cell (kind x variant x d bin, one count per event of the attempt), plus accepted tokens per hour.

Counts cover the whole run (rebuilt from the shards on resume); the rate covers this session only, since a resumed
run has a gap nobody should average over. Tokens are words x 1.3 until the Planck tokenizer exists (flagged)."""
import collections
import json
import time

import records


class Tally:
    def __init__(self):
        self.attempts = 0
        self.accepted = 0
        self.tokens = 0
        self.by_primary = collections.Counter()
        self.by_code = collections.Counter()
        self.cell_att = collections.Counter()
        self.cell_acc = collections.Counter()
        self.session = {"attempts": 0, "accepted": 0, "tokens": 0, "calls": 0, "requests": 0,
                        "completion_tokens": 0, "latency_s": 0.0}
        self.t0 = time.time()

    def add(self, skel, ok, primary=None, codes=(), tokens=0, call=None, session=True):
        self.attempts += 1
        cells = records.cells(skel)
        for c in cells:
            self.cell_att[c] += 1
        if ok:
            self.accepted += 1
            self.tokens += tokens
            for c in cells:
                self.cell_acc[c] += 1
        else:
            self.by_primary[primary] += 1
            for c in set(codes) | {primary}:
                self.by_code[c] += 1
        if session:
            s = self.session
            s["attempts"] += 1
            s["accepted"] += int(ok)
            s["tokens"] += tokens if ok else 0
            if call:
                s["calls"] += 1
                s["requests"] += call.get("requests") or 0
                s["completion_tokens"] += (call.get("usage") or {}).get("completion_tokens") or 0
                s["latency_s"] += call.get("latency_s") or 0.0

    def snapshot(self, extra=None):
        el = max(1e-9, time.time() - self.t0)
        s = self.session
        cells = {}
        for c, n in sorted(self.cell_att.items()):
            cells["|".join(c)] = {"attempted": n, "accepted": self.cell_acc[c],
                                  "yield": round(self.cell_acc[c] / n, 4)}
        return {
            "time": round(time.time(), 3), "attempts": self.attempts, "accepted": self.accepted,
            "rejected": self.attempts - self.accepted,
            "yield": round(self.accepted / self.attempts, 4) if self.attempts else None,
            "accepted_tokens_est": self.tokens, "token_flag": records.TOKEN_FLAG,
            "by_primary": dict(self.by_primary.most_common()), "by_code": dict(self.by_code.most_common()),
            "cells": cells,
            "session": {**s, "elapsed_s": round(el, 3),
                        "accepted_tokens_per_hour": round(s["tokens"] * 3600 / el),
                        "completion_tokens_per_s": round(s["completion_tokens"] / el, 2),
                        "mean_latency_s": round(s["latency_s"] / s["calls"], 3) if s["calls"] else None},
            **(extra or {}),
        }

    def line(self):
        snap = self.snapshot()
        top = ", ".join(f"{k} {v}" for k, v in list(snap["by_primary"].items())[:4])
        y = f"{100 * snap['yield']:.1f}%" if snap["yield"] is not None else "-"
        return (f"accepted {snap['accepted']}/{snap['attempts']} ({y}) | "
                f"{snap['session']['accepted_tokens_per_hour']:,} est tok/h this session | top rejects: {top or '-'}")


def append(path, snap):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(snap, sort_keys=True) + "\n")
