"""Controlled word list, step 3: the CLI over wordfam (BANKPASS s4). Writes per sample the word list TSV, the eligible
word set (the nonce filter's English list), wordlist_stats.json (counts, drops, PROGRAM-list check, and for two
samples the top-2,000 / 5,000 / 20,000 Jaccard that decides whether the full read runs: either of the first two under
0.95), and vocab_candidates.txt (the union of the rows read plus the PROGRAM words: the full read's --vocab).

    python wordstats.py OUT_DIR counts_A.tsv.gz [counts_B.tsv.gz]"""
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]   # run as a script: keep bankpass/ off the path
_PIPE = os.path.dirname(_HERE)
if _PIPE not in sys.path:
    sys.path.insert(0, _PIPE)

from bankpass import wordfam as WF  # noqa: E402


PROGRAM_WORDS = ("monday tuesday wednesday thursday friday saturday sunday january february march april may june july "
                 "august september october november december red blue green yellow purple orange pink brown grey "
                 "black white one two three four five six seven eight nine ten eleven twelve first second third "
                 "fourth fifth noon midnight").split()


def stats(res):
    fl = res["flags"]
    drops = {}
    for h in res["fam"]:
        for f in fl[h]:
            drops[f] = drops.get(f, 0) + 1
    prog = {w: {"eligible": w in res["elig"], "rank": res["rank"].get(res["head_of"].get(w, w)),
                "flags": fl.get(w, [])} for w in PROGRAM_WORDS}
    n_forms = sum(len(v) for f in res["fam"].values() for v in f["forms"].values())
    pos = {}
    for h in res["ranked"][:5000]:
        pos[res["pos"][h][0]] = pos.get(res["pos"][h][0], 0) + 1
    return {"rows_read": len(res["rows"]), "eligible_words": len(res["elig"]), "families": len(res["fam"]),
            "forms_joined": n_forms, "ranked_families": len(res["ranked"]), "flagged_families": drops,
            "required_ok_RM": sum(WF.required_ok(res, h) for h in res["ranked"][:5000]),
            "lists": {n: min(cap, len(res["ranked"])) for n, cap in WF.LISTS}, "pos_proxy_RM": pos,
            "program_words": prog, "docs": res["head"]["docs"]}


def jaccard_top(a, b, n):
    x, y = set(a["ranked"][:n]), set(b["ranked"][:n])
    return round(len(x & y) / len(x | y), 4) if x | y else 1.0


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    out_dir, paths = argv[0], argv[1:]
    res, out = {}, {"files": {}}
    cand = set(PROGRAM_WORDS)
    for p in paths:
        lab = os.path.basename(p).split("_", 1)[1].split(".")[0]
        res[lab] = WF.build(p)
        cand |= set(res[lab]["rows"])
        tsv = WF.write(res[lab], out_dir, lab)
        el = os.path.join(out_dir, f"eligible_words.{lab}.txt")
        with open(el, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(res[lab]["elig"])) + "\n")
        out[lab] = stats(res[lab])
        for fp in (p, tsv, el):
            out["files"][os.path.basename(fp)] = hashlib.sha256(open(fp, "rb").read()).hexdigest()
    if len(res) == 2:
        a, b = (res[k] for k in sorted(res))
        out["stability"] = {f"jaccard_top{n}": jaccard_top(a, b, n) for n in (2000, 5000, 20000)}
        out["stability"]["full_read_needed"] = min(out["stability"]["jaccard_top2000"],
                                                   out["stability"]["jaccard_top5000"]) < 0.95
    vp = os.path.join(out_dir, "vocab_candidates.txt")
    with open(vp, "w", encoding="utf-8") as f:
        f.write("# union of the count-table rows read here plus the PROGRAM words (the full read's --vocab)\n")
        f.write("\n".join(sorted(cand)) + "\n")
    out["files"]["vocab_candidates.txt"] = hashlib.sha256(open(vp, "rb").read()).hexdigest()
    out["wordfam_sha256"] = hashlib.sha256(open(WF.__file__, "rb").read()).hexdigest()
    out["wordstats_sha256"] = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    with open(os.path.join(out_dir, "wordlist_stats.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in out.items() if k != "files"}, sort_keys=True)[:4000])


if __name__ == "__main__":
    sys.exit(main())
