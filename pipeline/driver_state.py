"""Run directory state for driver.py: config pinning, the skeleton manifest, and resume.

<out>/run.json                 pinned config (shard seed, register, generator version, max attempts, source, gate
                               hash); a resume with a different pinned value is refused
<out>/skeletons/skeletons-*.jsonl   every skeleton in dispatch order {"j", "skel"}, written BEFORE its first request
<out>/accepted/accepted-*.jsonl     accepted records        <out>/rejects/rejects-*.jsonl   one per failed attempt
<out>/yield.jsonl              yield snapshots

Resume is derived from the shards alone: an attempt is done when its accepted or reject record exists; a skeleton is
finished when it has an accepted record, or its last reject is terminal (SKEL_INFEASIBLE), or it used max_attempts.
The dedup index and the yield counts are rebuilt from the records; torn last lines are cut first (shards.repair)."""
import json
import os

import dedup
import shards

PINNED = ("shard_seed", "register", "gen_version", "max_attempts", "source", "gate_hash")
TERMINAL = {"SKEL_INFEASIBLE"}


def pin(out, cfg):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "run.json")
    want = {k: cfg[k] for k in PINNED}
    if os.path.exists(path):
        with open(path) as f:
            have = json.load(f)
        diff = {k: (have.get(k), want[k]) for k in PINNED if have.get(k) != want[k]}
        if diff:
            raise SystemExit(f"refusing to resume {out}: pinned config differs {diff}")
        have["sessions"] = have.get("sessions", 0) + 1
    else:
        have = {**want, "created": cfg.get("created"), "sessions": 1}
    have["n"] = max(have.get("n", 0), cfg["n"])
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(have, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return have


class State:
    """per-skeleton progress. skels holds only unfinished skeletons (finished ones are dropped to save memory)."""

    def __init__(self, out, max_attempts, near_t):
        self.out, self.max_attempts = out, max_attempts
        self.done = {}          # skel_id -> {attempt: primary or None (accepted)}
        self.skels = {}         # skel_id -> skeleton, unfinished only
        self.order = []         # unfinished skel ids in manifest order
        self.n_manifest = 0
        self.last_j = 0
        self.seen = set()
        self.index = dedup.Index(near_t)
        self.dup_pairs = []     # (skel_id, attempt) recorded twice: a driver bug, reported
        self.torn = {}

    def finished(self, sid):
        d = self.done.get(sid, {})
        if any(v is None for v in d.values()):
            return True
        if not d:
            return False
        last = d[max(d)]
        return last in TERMINAL or len(d) >= self.max_attempts

    def next_attempt(self, sid):
        d = self.done.get(sid, {})
        return max(d) + 1 if d else 0

    def mark(self, sid, attempt, primary):
        d = self.done.setdefault(sid, {})
        if attempt in d:
            self.dup_pairs.append((sid, attempt))
        d[attempt] = primary
        if self.finished(sid):
            self.skels.pop(sid, None)

    def load(self, tally, triple_fn):
        """rebuild progress, the dedup index and the tally from disk. -> (accepted records, reject records)."""
        for sub, prefix in (("accepted", "accepted"), ("rejects", "rejects"), ("skeletons", "skeletons")):
            d = os.path.join(self.out, sub)
            for p in shards.paths(d, prefix) if os.path.isdir(d) else []:
                dropped = shards.repair(p)
                if dropped:
                    self.torn[p] = dropped
        acc = list(shards.read(os.path.join(self.out, "accepted"), "accepted"))
        rej = list(shards.read(os.path.join(self.out, "rejects"), "rejects"))
        for r in acc:
            self.mark(r["skel_id"], r["attempt"], None)
            self.index.add(r["conv_id"], self.index.keys(r["turns"], r["slots"]))
        for r in rej:
            self.mark(r["skel_id"], r["attempt"], r["primary"])
        by_sid = {}
        for r in acc:
            by_sid.setdefault(r["skel_id"], []).append((True, r))
        for r in rej:
            by_sid.setdefault(r["skel_id"], []).append((False, r))
        for m in shards.read(os.path.join(self.out, "skeletons"), "skeletons"):
            sk, sid = m["skel"], m["skel"]["skel_id"]
            self.n_manifest += 1
            self.last_j = max(self.last_j, m["j"])
            self.seen.add(triple_fn(sk))
            for ok, r in by_sid.pop(sid, []):
                tally.add(sk, ok, r.get("primary"), r.get("codes", ()), r.get("est_tokens", 0), session=False)
            if not self.finished(sid):
                self.skels[sid] = sk
                self.order.append(sid)
        if by_sid:
            raise SystemExit(f"records for skeletons missing from the manifest: {sorted(by_sid)[:5]}")
        return len(acc), len(rej)
