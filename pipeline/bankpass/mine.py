"""Mined human examples (BANKPASS s0): when a bank class needs an example in its prompt, it is a sentence a person
wrote, mined from core v0 (non-reserve OASST2 user turns, YouTube transcripts) and cited by doc id; never a
Claude-written line. This file finds candidates per class with rubric patterns (Claude wrote the patterns; the
sentences are human) and keeps those that pass the same gates as a bank item.

    nice -n 10 python mine.py CORE_DIR RAW_OASST2_TREES_GZ OUT_DIR [--workers 6] [--per-class 300]

OOD-H: every OASST2 doc re-asserts corpus/oodh.in_oodh_reserve on its tree id (core v0 already left the reserve
out); oodh.leaked_reserve_trees runs on the raw trees file and is recorded; and a sentence whose normalized text
(oodh.prompt_key) occurs, word-aligned, inside any reserved tree's user turn is skipped when picked (OODH_TEXT).
Only counts are printed, never reserve text. NAME: a capitalized word after the first (a person, place or brand: "My
name is <a YouTuber>") drops the sentence, so no real name reaches a prompt or, copied, a bank line."""
import argparse
import collections
import hashlib
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]   # run as a script: keep bankpass/ off the path
_PIPE = os.path.dirname(_HERE)
for _p in (_PIPE, os.path.join(os.path.dirname(_PIPE), "corpus")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SENT = re.compile(r"(?<=[.!?])\s+")
OK_CHARS = re.compile(r"^[A-Za-z][A-Za-z ,.'!?]*$")
PRONOUN_I = re.compile(r"I(?:'[a-z]+)?[,.!?]*")
PAT = {   # class -> pattern on the lowercased sentence (rubric, never inserted into a chat)
    "K_plant": r"^(?:my (?:name|dog|cat|sister|brother|mom|dad|mother|father|wife|husband|son|daughter|friend|job|"
               r"favou?rite \w+|birthday|son's|daughter's) (?:is|was)\b|i (?:live|work) (?:in|as|at)\b|i'm (?:a|an) \w+\.?$)",
    "M_fix": r"^(?:sorry|actually|i mean|i meant|wait|no wait|my mistake|scratch that|let me rephrase)\b",
    "O_greet": r"^(?:hi|hello|hey|good (?:morning|afternoon|evening)|greetings|hey there|hi there)\b",
    "C_close": r"(?:\bthanks?\b|thank you).*(?:\bbye\b|that's all|that is all|for now|have a (?:good|nice))",
    "S_social": r"^(?:how are you|how's it going|how is your day|what's your name|who are you)\b",
    "R_rule": r"(?:\bfrom now on\b|\bplease (?:keep|make) (?:your|the) (?:answers?|replies|responses)\b|"
              r"\bstop (?:using|saying) the word\b|\bcall me \w+|\banswer in (?:one|\w+) (?:sentence|words?)\b)",
    "L_list": r"\bmy (?:shopping|grocery|packing|guest|reading) list\b",
}
PAT = {k: re.compile(v) for k, v in PAT.items()}
YT_CLASSES = {"K_plant", "O_greet", "C_close", "M_fix"}
MIN_W, MAX_W = 2, 25


def gate_reason(s):
    """the first gate a candidate fails, or None (the bank item checks that apply to a bare sentence)."""
    import gate as G
    import heldout
    import lexicons as L
    import hygiene
    if not OK_CHARS.match(s):
        return "CHARS"
    n = len(s.split())
    if not MIN_W <= n <= MAX_W:
        return "LEN"
    for code, rx in (("DASH", G.DASH_RE), ("MARKDOWN", L.MARKDOWN_RE), ("SAFETY", L.SAFETY_RE),
                     ("AI_ISM", L.AI_ISM_RE), ("ROLE_LABEL", L.ROLE_LABEL_RE)):
        if rx.search(s):
            return code
    if hygiene.aiism(s):
        return "AI_ISM"
    if heldout.vocab_hits(s):
        return "HELDOUT_VOCAB"
    if heldout.echo_hits(s):
        return "HELDOUT_ECHO"
    if any(w[:1].isupper() and not PRONOUN_I.fullmatch(w) for w in s.split()[1:]):
        return "NAME"
    return None


def candidates(text, classes):
    for j, s in enumerate(SENT.split(text.replace("’", "'").strip())):
        s = s.strip()
        low = s.lower()
        for cls in classes:
            if PAT[cls].search(low):
                yield cls, j, s


def mine_shard(job):
    core, rel, source = job
    from bankpass.wordlist import _open   # same reader (zst on the PC, gz or plain in tests)
    from oodh import in_oodh_reserve
    out, notes = [], collections.Counter()
    with _open(os.path.join(core, rel)) as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            meta = r.get("meta") or {}
            if source == "oasst2":
                if in_oodh_reserve(meta["tree_id"]):
                    notes["oodh_reserve_hit"] += 1
                    continue
                texts = [t["text"] for t in r.get("turns", []) if t["role"] == "user"]
                classes = tuple(PAT)
            else:
                texts, classes = [r["text"]], tuple(YT_CLASSES)
            for k, t in enumerate(texts):
                for cls, j, s in candidates(t, classes):
                    notes[f"cand_{cls}"] += 1
                    why = gate_reason(s)
                    if why:
                        notes[f"drop_{cls}_{why}"] += 1
                        continue
                    out.append({"class": cls, "text": s, "source": source, "doc_id": r["id"], "turn": k, "sent": j,
                                "doc_sha1": meta.get("sha1"), "url": meta.get("url"), "license": meta.get("license"),
                                "shard": rel, "line": i})
    return out, dict(notes)


class Reserve:
    """reserved OASST2 user turns (oodh.prompt_key form) with a word-pair index, so "does this sentence occur inside
    any reserved turn" checks a few turns, not all of them; plus the leaked_reserve_trees count (BANKPASS s2d:
    imported, never re-implemented)."""
    def __init__(self, raw_trees_gz):
        import gzip
        import oodh
        trees, self.turns, self.index = [], [], collections.defaultdict(set)
        with gzip.open(raw_trees_gz, "rt", encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)
                trees.append(t)
                if oodh.in_oodh_reserve(t["message_tree_id"]):
                    for u in oodh._user_turns(t["prompt"]):
                        k = oodh.prompt_key(u)
                        w = k.split()
                        for i in range(max(1, len(w) - 1)):
                            self.index[tuple(w[i:i + 2])].add(len(self.turns))
                        self.turns.append(k)
        self.n_reserved = sum(oodh.in_oodh_reserve(t["message_tree_id"]) for t in trees)
        self.n_leaked = len(oodh.leaked_reserve_trees(trees))

    def has(self, text):
        import oodh
        k = oodh.prompt_key(text)
        w = k.split()
        cands = self.index.get(tuple(w[:2]), set())
        return any(k in self.turns[i] for i in cands)


def pick(rows, per_class, ok=None, notes=None, seed="bankpass-mine-v0"):
    """per (class, source): exact duplicates (class, lowercased text) dropped, then the first per_class rows that
    pass ok() in a seeded hash order (a failure is counted in notes). Input order does not matter."""
    def key(r):
        return hashlib.sha256(f"{seed}|{r['doc_id']}|{r['turn']}|{r['sent']}".encode()).hexdigest()
    by, seen = collections.defaultdict(list), set()
    for r in sorted(rows, key=key):
        k = (r["class"], r["text"].lower())
        if k not in seen:
            seen.add(k)
            by[(r["class"], r["source"])].append(r)
    out = []
    for _, rs in sorted(by.items()):
        n = 0
        for r in rs:
            if n >= per_class:
                break
            if ok and not ok(r):
                if notes is not None:
                    notes[f"drop_{r['class']}_OODH_TEXT"] += 1
                continue
            out.append(r)
            n += 1
    return out


def main(argv=None):
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("core_dir")
    ap.add_argument("raw_trees")
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--per-class", type=int, default=300)
    a = ap.parse_args(argv)
    with open(os.path.join(a.core_dir, "MANIFEST.json"), "rb") as f:
        raw = f.read()
    jobs = [(a.core_dir, s["shard"], s["source"]) for s in json.loads(raw)["shards"]
            if s["source"] in ("oasst2", "youtube")]
    rows, notes = [], collections.Counter()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for out, nt in pool.imap_unordered(mine_shard, jobs):
            rows += out
            notes.update(nt)
    res = Reserve(a.raw_trees)
    chosen = pick(rows, a.per_class, ok=lambda r: not res.has(r["text"]), notes=notes)
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, "mined_examples_v0.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in chosen:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    per = collections.Counter((r["class"], r["source"]) for r in chosen)
    stats = {"input_manifest_sha256": hashlib.sha256(raw).hexdigest(), "shards": len(jobs), "notes": dict(notes),
             "passed_gates": len(rows), "chosen": {f"{c}/{s}": n for (c, s), n in sorted(per.items())},
             "reserved_trees_in_raw": res.n_reserved, "leaked_reserve_trees": res.n_leaked,
             "file_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
             "code_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}
    with open(os.path.join(a.out_dir, "mined_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in stats.items() if k != "notes"}, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
