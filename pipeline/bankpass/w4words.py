"""W4 word labels (BANKPASS s4; DECISIONS_BANKPASS_FOR_MAX "one engineering change"): the stage P teacher labels
merged into the W1 word list, written as wordlist_v1 (same columns plus pos and pos_votes; wordload.read takes pos
when the column is there). PC CPU, no model.

  POS      votes = the three teachers' pos2 labels (mapped below) plus the corpus proxy; a POS with 3 or more of the 4
           votes is "confirmed" (three agreeing teachers decide; the proxy only breaks a 2 to 1 split), anything else is
           "amb" (counts for VOCAB_OOL, never a required word). Words no teacher labelled stay "pending".
  -ly      an adverb joins its adjective's family when 2 of 3 teachers say "same" and the base is a confirmed adj.
  irregular past and participle forms join a confirmed verb's family when 2 of 3 teachers give the same form, the
           form is an eligible word, differs from the regular -ed form, and is either no ranked family or a ranked
           family confirmed as a verb ("went" joins go; "left", an adjective or amb, stays apart from leave).
A joined word that was its own family keeps its row, unranked, flagged joined:<head> (only the headword moves; its
own forms match as themselves). required_ok = ranked, unflagged, confirmed noun, verb or adj.

    python -m bankpass.w4words --root STAGE --wordlist V0.tsv --eligible eligible_words.full.txt --out V1.tsv"""
import argparse
import collections
import json
import sys

from bankpass import store, w3amend, w3state as WS

POS_MAP = {"noun": "noun", "verb": "verb", "adjective": "adj", "adverb": "adv", "pronoun": "func",
           "preposition": "func", "conjunction": "func", "determiner": "func", "interjection": "other",
           "number": "other", "other": "other"}
CONTENT = ("noun", "verb", "adj")
NEED_VOTES, NEED_AGREE = 3, 2


def current(r, kind):
    """a finished, parsed call of this kind from the current stage version (pos2, verbs on pos2, ly: w3amend marks
    the version 1 pos.* and verbs.pos.* calls superseded)."""
    if r.get("kind") != kind or r.get("problem") or r.get("error"):
        return False
    return not w3amend.superseded(r["call_id"])


def labels(recs):
    """{"pos": {word: {teacher: label}}, "verbs": {verb: {teacher: (past, part)}}, "ly": {(adj, adv): {t: same}}}"""
    out = {"pos": collections.defaultdict(dict), "verbs": collections.defaultdict(dict),
           "ly": collections.defaultdict(dict)}
    for r in recs:
        t = r["author"]["model"]
        if current(r, "pos"):
            for ln in r["lines"]:
                w, _, lab = ln.rpartition(": ")
                if lab in POS_MAP:
                    out["pos"][w.strip()].setdefault(t, POS_MAP[lab])
        elif current(r, "verbs"):
            for ln in r["lines"]:
                w, _, forms = ln.partition(": ")
                past, _, part = forms.partition(", ")
                if past and part:
                    out["verbs"][w.strip()].setdefault(t, (past.strip(), part.strip()))
        elif current(r, "ly"):
            for ln in r["lines"]:
                pair, _, v = ln.rpartition(": ")
                a, _, b = pair.partition(", ")
                if v in ("same", "different"):
                    out["ly"][(a.strip(), b.strip())].setdefault(t, v)
    return out


def decide(votes, proxy):
    """(pos, status) from {teacher: pos} and the proxy POS ("unk" casts no vote)."""
    c = collections.Counter(votes.values())
    if proxy in CONTENT or proxy == "func":
        c[proxy] += 1
    if not votes:
        return proxy, "pending"
    pos, n = c.most_common(1)[0] if c else (None, 0)
    return (pos, "confirmed") if n >= NEED_VOTES else ("amb", "amb")


def regular_ed(v):
    if v.endswith("e"):
        return v + "d"
    if v.endswith("y") and len(v) > 2 and v[-2] not in "aeiou":
        return v[:-1] + "ied"
    return v + "ed"


def joins(fams, lab, eligible):
    """[(form, head, rule)] to join, and {code: n} for the ones refused."""
    out, refused = [], collections.Counter()
    for (adj, adv), vs in sorted(lab["ly"].items()):
        if sum(v == "same" for v in vs.values()) < NEED_AGREE:
            refused["LY_DIFFERENT"] += 1
        elif adj not in fams or fams[adj]["pos"] != "adj" or fams[adj]["pos_status"] != "confirmed":
            refused["LY_BASE_NOT_ADJ"] += 1
        else:
            out.append((adv, adj, "ly"))
    for verb, vs in sorted(lab["verbs"].items()):
        if verb not in fams or fams[verb]["pos"] != "verb" or fams[verb]["pos_status"] != "confirmed":
            refused["IRR_NOT_VERB"] += 1
            continue
        for k in (0, 1):
            c = collections.Counter(f[k] for f in vs.values())
            form, n = c.most_common(1)[0]
            if n < NEED_AGREE or form == regular_ed(verb) or form == verb:
                continue
            other = fams.get(form)
            if form not in eligible:
                refused["IRR_NOT_ELIGIBLE"] += 1
            elif other and other["rank"] and (other["pos_status"], other["pos"]) != ("confirmed", "verb"):
                refused["IRR_OTHER_POS"] += 1
            elif (form, verb, "irr") not in out:
                out.append((form, verb, "irr"))
    return out, refused


def read_rows(path):
    head, cols, rows = [], None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                head.append(line.rstrip("\n"))
                continue
            p = line.rstrip("\n").split("\t")
            if cols is None:
                cols = p
            else:
                rows.append(dict(zip(cols, p)))
    return head, cols, rows


def _forms(cell):
    return {r: fs.split("|") for r, fs in (p.split(":", 1) for p in filter(None, cell.split(";")))}


def _cell(forms):
    return ";".join(f"{r}:{'|'.join(sorted(set(fs)))}" for r, fs in sorted(forms.items()))


def build(v0_path, recs, eligible, out_path, note=""):
    head, cols, rows = read_rows(v0_path)
    lab = labels(recs)
    by = {r["headword"]: r for r in rows}
    fams, stats = {}, collections.Counter()
    for r in rows:
        pos, st = decide(lab["pos"].get(r["headword"], {}), r["pos_proxy"])
        r["pos"], r["pos_status"] = pos, st
        r["pos_votes"] = ",".join(f"{store.SHORT[t]}:{p}" for t, p in sorted(lab["pos"].get(r["headword"], {}).items()))
        fams[r["headword"]] = {"pos": pos, "pos_status": st, "rank": r["rank"]}
        stats[f"pos_{st}"] += 1
    taken = {f: r["headword"] for r in rows for fs in _forms(r["forms"]).values() for f in fs}
    js, refused = joins(fams, lab, eligible)
    for form, h, rule in js:
        if taken.get(form) not in (None, h):
            refused["FORM_OF_OTHER"] += 1
            continue
        f = _forms(by[h]["forms"])
        f.setdefault(rule, []).append(form)
        by[h]["forms"] = _cell(f)
        taken[form] = h
        if form in by and form != h:
            by[form].update(rank="", lists="", flags=",".join(filter(None, [by[form]["flags"], "joined:" + h])))
        stats[f"join_{rule}"] += 1
    for r in rows:
        r["required_ok"] = str(int(bool(r["rank"]) and not r["flags"] and r["pos"] in CONTENT
                                   and r["pos_status"] == "confirmed"))
    out_cols = cols + [c for c in ("pos", "pos_votes") if c not in cols]
    code = store.sha256_file(__file__)
    with open(out_path + ".tmp", "w", encoding="utf-8") as f:
        f.write("\n".join(head[:3]) + "\n")
        f.write(f"# W4 v1: teacher labels merged (w4words.py sha256 {code}); {note}\n")
        f.write("# pos = 3 of 4 votes (three teachers + the proxy), else amb; pos_proxy kept as counted\n")
        f.write("\t".join(out_cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in out_cols) + "\n")
    import os
    os.replace(out_path + ".tmp", out_path)
    stats.update({f"refused_{k}": v for k, v in refused.items()})
    return dict(stats, sha256=store.sha256_file(out_path))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--wordlist", required=True)
    ap.add_argument("--eligible", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    recs = [r for r in WS.records(a.root) if r.get("kind") in ("pos", "verbs", "ly")]
    with open(a.eligible, encoding="utf-8") as f:
        elig = {ln.strip() for ln in f if ln.strip()}
    calls = {t: store.sha256_file(WS.calls_path(a.root, t)) for t in WS.ORDER}
    note = "stage calls sha256 " + ", ".join(f"{store.SHORT[t]} {h[:16]}" for t, h in calls.items())
    print(json.dumps(build(a.wordlist, recs, elig, a.out, note), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
