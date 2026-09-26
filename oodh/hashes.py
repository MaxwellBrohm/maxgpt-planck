"""oodh/HASHES.txt, the public record of OOD-H Part 1 (hashes and counts only, never item text). build.py writes the
file; gate.py replaces the GATE section at its end, tied to the sha256 of the file it checked. The tree-id hash is the
prereg's: sha256 of the sorted tree ids joined by newlines, no trailing newline (as oodh/select.py hashes its list)."""
import hashlib
import os
from collections import Counter

KINDS = ("H-FACT", "H-ASK", "H-CORR", "H-ABS")
GATE_HEAD = "GATE (oodh/gate.py)"


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def ids_sha(tree_ids):
    return hashlib.sha256("\n".join(sorted(tree_ids)).encode()).hexdigest()


def counts_lines(recs):
    pk = Counter(p["probe_kind"] for r in recs for p in r["probes"])
    per = Counter(len(r["probes"]) for r in recs)
    cells = Counter(r["cell"] for r in recs)
    users = Counter(r["meta"]["n_thread_user_turns"] for r in recs)
    return [f"threads {len(recs)}; probes {sum(pk.values())}: " + ", ".join(f"{k} {pk[k]}" for k in KINDS),
            "probes per thread: " + ", ".join(f"{k}: {per[k]}" for k in sorted(per)),
            "cells (probe-kind mix): " + ", ".join(f"{c} {cells[c]}" for c in sorted(cells)),
            "thread user turns: " + ", ".join(f"{k}: {users[k]}" for k in sorted(users))]


def write_hashes(recs, short, n, data_path, hashes_path, inputs):
    status = (f"SHORT: {len(recs)} usable threads, target {n}; NOT the lock set" if short
              else f"COMPLETE: {n} threads chosen by the rule in oodh/build.py")
    lines = ["OOD-H Part 1 hashes (public: hashes and counts only; the items are sealed). Written by oodh/build.py.",
             f"status {status}",
             f"sealed/oodh/{os.path.basename(data_path)} sha256 {sha(data_path)}",
             f"sorted tree ids (newline-joined, no trailing newline) sha256 {ids_sha(r['tree_id'] for r in recs)}"]
    lines += counts_lines(recs)
    lines.append("inputs (sha256, first 16 hex): " + ", ".join(f"{os.path.basename(f)} {sha(f)[:16]}" for f in inputs))
    with open(hashes_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return lines


def put_gate(hashes_path, gate_lines):
    """replace the GATE section (everything from its head line on) with gate_lines."""
    keep = open(hashes_path).read().split("\n" + GATE_HEAD)[0].rstrip("\n") if os.path.exists(hashes_path) else ""
    with open(hashes_path, "w") as fh:
        fh.write((keep + "\n" if keep else "") + "\n".join(gate_lines) + "\n")
