"""W5 re-measure (BANKPASS s5; notes "OFFTOPIC CHOICE" method) on the bank probe's renders, with the frozen bank set
installed (real topics, topic word sets, word list families). PC CPU; counts only, no line text leaves the PC.

For every guided user topic turn and assistant filler reply of every parsed attempt: the share of the teacher's own
lines OFFTOPIC flags, and the share of lines it catches when the line is swapped for one of the same kind from a
chat of the same teacher whose topics are disjoint (seeded pick). Variants:
  stems        the checker as it runs (check_behav.chk_offtopic: Porter stems, lexicons.GENERIC, the installed sets)
  fam          s5's family table: a word counts as its wordlist_v1 headword; wanted = families of the topic text and
               its word set union (+ the previous user line for an assistant reply); generic families COMPUTED (W4's
               5% of topic sets, plus the top 200 families of the probe's assistant replies spread over half of the
               probe's topics) replace Claude's GENERIC; prev_assist: "cur" (2 shared or half, as the checker),
               "none" (dropped), "fam2" (2 shared families)
REQ_WORD: the share of placed required words missing, by the checker's forms (check_lines.word_forms_re) and by the
family's listed forms. VOCAB_OOL: the share of a turn's non-slot words whose family is outside RM, per role.

    python -m bankpass.w5measure --banks BANKS --probe PROBE --wordlist V1.tsv [--out measure.json]"""
import argparse
import collections
import json
import os
import re
import sys

import check_behav as CB
import check_lines as CLn
import lexicons as L
import parse
import topic_words as TW
from check_base import Ctx, words as cwords
from bankpass import load, store, w4banks as WB

KINDS = ("user", "assistant")
TOP_GENERIC, GENERIC_TOPIC_SHARE = 200, 0.5


def kind_of(t):
    it = t.get("intent") or ""
    if t["mode"] != "guided":
        return None
    if t["role"] == "user" and it.startswith("topic:"):
        return "user"
    if t["role"] == "assistant" and it.split(";")[0] in CB.FILLER_ASSIST:
        return "assistant"
    return None


def attempts(run_dir, skels):
    """(skeleton, {turn index: text}) for every attempt the parser could read."""
    for part in ("accepted", "rejects"):
        d = os.path.join(run_dir, part)
        for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            for r in store.read_jsonl(os.path.join(d, name)):
                sk = skels.get(r.get("skel_id"))
                if sk is None:
                    continue
                if part == "accepted":
                    yield sk, {i: t["text"] for i, t in enumerate(r["turns"])}
                elif r.get("raw"):
                    yield sk, parse.parse(r["raw"], parse.plan(sk))["turns"]


class Fam:
    def __init__(self, wl, generic):
        self.wl, self.generic = wl, set(generic)

    def content(self, s):
        return {self.wl.head(w.strip("'")) for w in cwords(s) if w not in L.STOPWORDS and w not in L.FUNCTION_WORDS
                and len(w) > 2} - self.generic

    def topic(self, text):
        return self.content(text) | self.content(" ".join(TW.related(text)))


def flags_fam(sk, texts, fam, prev_rule):
    topics, out = sk["topic_text"], set()
    all_topic = set().union(*(fam.topic(x) for x in topics.values()))
    prev_user = prev_assist = None
    for t in sk["turns"]:
        s, k = texts.get(t["i"]), kind_of(t)
        if s is not None and k:
            if k == "user":
                want = fam.topic(topics[t["intent"].split(":")[1]])
                own, shared = fam.content(s), fam.content(s) & fam.content(prev_assist or "")
                if (prev_rule == "cur" and (len(shared) >= 2 or len(shared) >= 0.5 * len(own) > 0)) or \
                        (prev_rule == "fam2" and len(shared) >= 2):
                    want = want | fam.content(prev_assist or "")
            else:
                want = all_topic | fam.content(prev_user or "")
            if not fam.content(s) & want:
                out.add(t["i"])
        if s is not None and t["role"] == "user":
            prev_user = s
        elif s is not None and t["role"] == "assistant" and not t.get("lookup_call"):
            prev_assist = s
    return out


def flags_stems(sk, texts):
    return {i for c, i, _ in CB.chk_offtopic(Ctx(sk, texts)) if c == "OFFTOPIC"}


def probe_generic(rows, fam0, top=TOP_GENERIC, share=GENERIC_TOPIC_SHARE):
    """families of the assistant replies present in chats covering share or more of the probe's topics."""
    where, topics = collections.defaultdict(set), set()
    for sk, texts in rows:
        tp = set(sk["topic_text"].values())
        topics |= tp
        for t in sk["turns"]:
            if t["role"] == "assistant" and texts.get(t["i"]):
                for h in fam0.content(texts[t["i"]]):
                    where[h] |= tp
    n = max(1, len(topics))
    ranked = sorted(((len(v) / n, h) for h, v in where.items() if len(v) / n >= share), reverse=True)
    return [h for _, h in ranked[:top]]


def swap_rates(rows, flag_fn, seed="w5-swap"):
    """{kind: [n, own flagged, swapped caught]} over one teacher's attempts."""
    lines = collections.defaultdict(list)
    for a, (sk, texts) in enumerate(rows):
        for t in sk["turns"]:
            if kind_of(t) and texts.get(t["i"]):
                lines[kind_of(t)].append((a, t["i"], texts[t["i"]]))
    out = {k: [0, 0, 0] for k in KINDS}
    cache = {}
    for k, ls in lines.items():
        for a, i, s in ls:
            sk, texts = rows[a]
            if a not in cache:
                cache[a] = flag_fn(sk, texts)
            tp = set(sk["topic_text"].values())
            donors = [x for x in ls if x[0] != a and not tp & set(rows[x[0]][0]["topic_text"].values())]
            out[k][0] += 1
            out[k][1] += i in cache[a]
            if donors:
                d = donors[int(store.sha256_text(f"{seed}:{a}:{i}")[:8], 16) % len(donors)]
                out[k][2] += i in flag_fn(sk, {**texts, i: d[2]})
    return out


def req_rates(rows, wl):
    n = miss_cur = miss_fam = 0
    for sk, texts in rows:
        alltext = " ".join(texts.values())
        for part, w in CLn.placed_words(sk):
            n += 1
            miss_cur += not CLn.word_forms_re(w, part).search(alltext)
            d = wl.families.get(wl.head(w)) or {"forms": {}}
            forms = {w} | {f for fs in d["forms"].values() for f in fs}
            miss_fam += not re.search(r"(?<![a-z])(?:" + "|".join(map(re.escape, sorted(forms, key=len, reverse=True)))
                                      + r")(?![a-z])", alltext, re.I)
    return {"placed": n, "missing_checker_forms": miss_cur, "missing_family_forms": miss_fam}


def vocab_ool(rows, wl):
    out = {k: [0, 0] for k in ("user", "assistant")}
    for sk, texts in rows:
        vals = {w for s in sk["slots"].values() for w in cwords(str(s.get("value") or ""))}
        for t in sk["turns"]:
            if t["role"] in out and t["mode"] == "guided" and texts.get(t["i"]):
                ws = [w for w in cwords(texts[t["i"]]) if w.isalpha() and w not in vals]
                out[t["role"]][0] += len(ws)
                out[t["role"]][1] += sum(not wl.in_list(w, "RM") for w in ws)
    return {k: round(b / a, 4) if a else None for k, (a, b) in out.items()}


def measure(bank_dir, probe_dir, wl):
    bs = load.from_dir(bank_dir)
    undo = load.install(bs)
    try:
        skels = {}
        for r in store.read_jsonl(os.path.join(probe_dir, "skels60.jsonl")):
            skels[r["skel_id"]] = r
        man = bs.manifest
        rows = {t: list(attempts(os.path.join(probe_dir, "dry", t, "run"), skels))
                for t in sorted(os.listdir(os.path.join(probe_dir, "dry")))}
        topic_generic = _w4_generic(bank_dir, wl)
        fam0 = Fam(wl, topic_generic)
        pg = probe_generic([x for v in rows.values() for x in v], fam0)
        fam = Fam(wl, set(topic_generic) | set(pg))
        rep = {"bank_manifest_sha256": store.sha256_file(os.path.join(bank_dir, "manifest.json")),
               "generic": {"topic_sets": len(topic_generic), "probe": len(pg), "lexicons_GENERIC": len(L.GENERIC)},
               "teachers": {}}
        for t, rs in rows.items():
            r = {"attempts": len(rs), "stems": swap_rates(rs, flags_stems)}
            for rule in ("cur", "none", "fam2"):
                r[f"fam_{rule}"] = swap_rates(rs, lambda sk, tx, rule=rule: flags_fam(sk, tx, fam, rule))
            r["req_word"], r["vocab_ool_rm"] = req_rates(rs, wl), vocab_ool(rs, wl)
            rep["teachers"][t] = r
        rep["topicwords_ref"] = man["banks"].get("topicwords", {}).get("ref")
        return rep
    finally:
        undo()


def _w4_generic(bank_dir, wl):
    """W4's topic-set generic families, recomputed from the frozen word sets (the manifest keeps only their hash)."""
    sets = collections.defaultdict(dict)
    for r in store.read_jsonl(os.path.join(bank_dir, "topicwords.jsonl"), tolerate_torn_tail=False):
        if r["status"] == "kept":
            sets[r["features"]["topic_id"]][r["author"]["model"]] = r["features"]
    return WB.generic_families(sets, wl)


def main(argv=None):
    from bankpass import wordload
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--wordlist", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rep = measure(a.banks, a.probe, wordload.read(a.wordlist))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1, sort_keys=True)
    print(json.dumps(rep, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
