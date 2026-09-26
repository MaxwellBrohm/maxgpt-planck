"""Calibrate the LoC OCR filter (ocr.py) on Project Gutenberg paragraphs. CPU only, no model.

    python corpus/ocr_calibrate.py PG.jsonl.gz [PG2 ...] --loc LOC.jsonl.gz|- [--pg-docs 300]
        [--loc-docs 600] [--out stats/core_v0/ocr_calibration.json] [--samples 20]

Project Gutenberg text is transcribed and proofread by people, so its paragraphs are the clean
reference. PG also holds a fixed share of bytes that every grid value of a rule drops (tables
drawn with '|' and '+', ASCII art, index pages, Greek), so the budget is marginal: each rule's
threshold is the strictest grid value whose PG byte loss exceeds the loss at the loosest grid
value by at most PG_BUDGET (0.25% of PG paragraph bytes). The report prints PG paragraphs dropped
at the loosest value (what PG always loses) and those lost at the chosen value only. The report then gives what each rule drops
on LoC (after ocr.reflow), random examples of LoC paragraphs each rule drops and of the kept LoC
paragraphs nearest the thresholds (so a reader can check the cut by eye), and the doc-level
dictionary-rate percentiles of PG books (clean_book's floor is PG's p1). LoC may be read from
stdin ('-'): a range-limited stream, nothing written to disk. A truncated gzip tail is ignored.
"""
import argparse
import gzip
import json
import random
import sys

import numpy as np

import hygiene as H
import ocr

PG_BUDGET = 0.0025
GRID = dict(min_dict_rate=[0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
            max_junk_rate=[0.005, 0.01, 0.02, 0.03, 0.05, 0.08],
            max_frag_rate=[0.10, 0.15, 0.20, 0.25, 0.30, 0.40])


def docs(path, n, want_lang):
    fh = gzip.open(sys.stdin.buffer if path == "-" else path, "rt", encoding="utf-8")
    k = 0
    try:
        for line in fh:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                break
            if (d.get("metadata") or {}).get("language") not in want_lang:
                continue
            yield d
            k += 1
            if k >= n:
                break
    except (EOFError, OSError):
        return


def para_rows(text, words, tag):
    for p in text.split("\n\n"):
        if p.strip():
            n, dr, junk, frag, lw = ocr.features(p, words)
            yield (n, dr, junk, frag, lw, len(p.encode("utf-8")), tag, p)


def drop_mask(a, rule, t, short):
    n = a["n"]
    if rule == "max_junk_rate":
        return a["junk"] > t
    if rule == "max_frag_rate":
        return a["frag"] > t
    return (n >= short) & (a["dict"] < t)


def arrays(rows):
    return {k: np.array([r[i] for r in rows], dtype=float)
            for i, k in enumerate(("n", "dict", "junk", "frag", "long", "bytes"))}


def share(x, rule, t, short):
    return float(x["bytes"][drop_mask(x, rule, t, short)].sum() / x["bytes"].sum())


def loosest(rule):
    return min(GRID[rule]) if rule == "min_dict_rate" else max(GRID[rule])


def pick(pg, rule, short):
    """Strictest grid value whose PG byte loss is within PG_BUDGET of the loosest value's."""
    base, ok = share(pg, rule, loosest(rule), short), None
    vals = sorted(GRID[rule], reverse=rule != "min_dict_rate")
    for t in vals:
        if share(pg, rule, t, short) - base <= PG_BUDGET:
            ok = t
        else:
            break
    return ok


def show(rows, idx, title, n, rnd):
    print(f"\n### {title} ({len(idx)}):")
    for i in rnd.sample(list(idx), min(n, len(idx))):
        r = rows[i]
        print(f"  [n={r[0]} dict={r[1]:.2f} junk={r[2]:.3f} frag={r[3]:.2f}] {r[7][:150]!r}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pg", nargs="+")
    ap.add_argument("--loc", required=True)
    ap.add_argument("--pg-docs", type=int, default=300)
    ap.add_argument("--loc-docs", type=int, default=600)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    words, short = ocr.dictionary(), ocr.SHORT_TOKENS
    pg_rows, pg_doc_rates, per = [], [], max(1, a.pg_docs // len(a.pg))
    for path in a.pg:
        for d in docs(path, per, {"en"}):
            t = H.normalize(d.get("text") or "")
            pg_rows += list(para_rows(t, words, "pg"))
            pg_doc_rates.append(ocr.doc_dict_rate(t, words))
    loc_rows, loc_books = [], []
    for d in docs(a.loc, a.loc_docs, {"english"}):
        raw = d.get("text") or ""
        loc_rows += list(para_rows(ocr.reflow(raw, words), words, d.get("id")))
        loc_books.append((d.get("id"), raw))
    pg, lc = arrays(pg_rows), arrays(loc_rows)
    chosen = {r: pick(pg, r, short) for r in GRID}
    rep = {"pg_budget": PG_BUDGET, "short_tokens": short, "chosen": chosen,
           "pg": {"docs": len(pg_doc_rates), "paragraphs": len(pg_rows), "bytes": int(pg["bytes"].sum())},
           "loc": {"docs": len(loc_books), "paragraphs": len(loc_rows), "bytes": int(lc["bytes"].sum())},
           "rules": {}}
    for rule, vals in GRID.items():
        rep["rules"][rule] = {str(t): {"pg_bytes_share": round(share(pg, rule, t, short), 5),
                                       "loc_bytes_share": round(share(lc, rule, t, short), 5)}
                              for t in vals}
    th = dict(ocr.THRESHOLDS, **{k: v for k, v in chosen.items() if v is not None})
    shortm = lambda x: (x["n"] < short) & (x["long"] == 0)
    rep["short_no_word"] = {"pg_bytes_share": round(float(pg["bytes"][shortm(pg)].sum() / pg["bytes"].sum()), 5),
                            "loc_bytes_share": round(float(lc["bytes"][shortm(lc)].sum() / lc["bytes"].sum()), 5)}
    q = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    rep["pg_doc_dict_rate_quantiles"] = {str(x): round(float(np.quantile(pg_doc_rates, x)), 4) for x in q}
    th["min_doc_dict_rate"] = float(np.floor(np.quantile(pg_doc_rates, 0.01) * 100) / 100)
    kept_share, loc_doc_rates, book_drops = [], [], {}
    for bid, raw in loc_books:
        text, c, why = ocr.clean_book(raw, words, th)
        kept_share.append(c["letters_kept"] / max(1, c["letters_in"]))
        loc_doc_rates.append(ocr.doc_dict_rate(text, words))
        if why:
            book_drops[why] = book_drops.get(why, 0) + 1
    rep["loc_after_filter"] = {
        "kept_letter_share_quantiles": {str(x): round(float(np.quantile(kept_share, x)), 4) for x in q},
        "doc_dict_rate_quantiles": {str(x): round(float(np.quantile(loc_doc_rates, x)), 4) for x in q},
        "book_drops_at_thresholds": book_drops, "thresholds": th}
    rnd = random.Random(0)
    for rule in GRID:
        t, lo = th[rule], loosest(rule)
        m_lo, m_t = drop_mask(pg, rule, lo, short), drop_mask(pg, rule, t, short)
        show(pg_rows, np.flatnonzero(m_lo), f"PG dropped by {rule} at its loosest value {lo}",
             a.samples, rnd)
        show(pg_rows, np.flatnonzero(m_t & ~m_lo), f"PG dropped by {rule} at {t} only", a.samples,
             rnd)
        show(loc_rows, np.flatnonzero(drop_mask(lc, rule, t, short)),
             f"LoC dropped by {rule} at {t}", a.samples, rnd)
    keep_pg, keep = np.ones(len(pg_rows), bool), np.ones(len(loc_rows), bool)
    for rule in GRID:
        keep_pg &= ~drop_mask(pg, rule, th[rule], short)
        keep &= ~drop_mask(lc, rule, th[rule], short)
    keep_pg &= ~shortm(pg)
    keep &= ~shortm(lc)
    rep["combined_paragraph_filter"] = {
        "pg_bytes_share_dropped": round(1 - float(pg["bytes"][keep_pg].sum() / pg["bytes"].sum()), 5),
        "loc_bytes_share_dropped": round(1 - float(lc["bytes"][keep].sum() / lc["bytes"].sum()), 5)}
    near = sorted(np.flatnonzero(keep & (lc["n"] >= short)), key=lambda i: lc["dict"][i])[:a.samples]
    print("\n### kept LoC paragraphs with the lowest dictionary rate:")
    for i in near:
        r = loc_rows[i]
        print(f"  [n={r[0]} dict={r[1]:.2f} junk={r[2]:.3f} frag={r[3]:.2f}] {r[7][:150]!r}")
    print(json.dumps(rep, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
