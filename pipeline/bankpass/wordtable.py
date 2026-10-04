"""Count table between wordlist.py (counting) and wordfam.py (families and lists): sources pooled into the BANKPASS
s4 groups, the per-group DF floor applied, one gzip TSV row per word that clears the floor in 2+ voting groups (a
superset of the eligible words, which need 3), with a JSON header line holding the docs per group and the run meta.

Groups: speech (youtube, irc), chat (oasst2, dolly, stackexchange), prose (news, wikimedia, pressbooks, oercommons,
pdr), books (gutenberg), web (cccc), howto (foodista); loc (OCR newspapers and books) is counted and reported but
does not vote and adds nothing to the context features."""
import gzip
import json

GROUPS = {"speech": ("youtube", "irc"), "chat": ("oasst2", "dolly", "stackexchange"),
          "prose": ("news", "wikimedia", "pressbooks", "oercommons", "pdr"), "books": ("gutenberg",),
          "web": ("cccc",), "howto": ("foodista",), "loc": ("loc",)}
VOTING = ("speech", "chat", "prose", "books", "web", "howto")
GROUP_OF = {s: g for g, ss in GROUPS.items() for s in ss}
CTX_FEATS = ("noun", "verb", "adj", "a", "an", "pl", "mass", "mid", "capmid")
MIN_DOCS, MIN_FRAC, PRE_GROUPS = 5, 5e-4, 2


def floor(docs):
    return max(MIN_DOCS, MIN_FRAC * docs)


def pool_groups(by_group):
    """{group: {"docs", "feats"}} -> (docs per group, df per group {g: Counter}, tf and ctx summed over voting
    groups). Each group's tf and ctx counters are released once summed, so two full copies never coexist."""
    import collections
    docs = {g: by_group.get(g, {}).get("docs", 0) for g in GROUPS}
    empty = collections.Counter()
    df = {g: by_group[g]["feats"]["df"] if g in by_group else empty for g in GROUPS}
    other = {f: collections.Counter() for f in ("tf",) + CTX_FEATS}
    for g in VOTING:
        if g in by_group:
            for f in other:
                other[f].update(by_group[g]["feats"][f])
                by_group[g]["feats"][f] = None if f != "df" else by_group[g]["feats"][f]
    return docs, df, other


def write(path, by_group, meta):
    docs, df, other = pool_groups(by_group)
    floors = {g: floor(docs.get(g, 0)) for g in GROUPS}
    words = set()
    for g in VOTING:
        words |= {w for w, n in df[g].items() if n >= floors[g]}
    rows = []
    for w in sorted(words):
        votes = sum(df[g][w] >= floors[g] for g in VOTING)
        if votes >= PRE_GROUPS:
            rows.append([w] + [df[g][w] for g in GROUPS] + [other[f][w] for f in ("tf",) + CTX_FEATS])
    head = dict(meta, docs=docs, floors=floors, columns=["word"] + [f"df_{g}" for g in GROUPS] + ["tf"]
                + list(CTX_FEATS), rule=f"rows: DF >= max({MIN_DOCS}, {MIN_FRAC} x docs) in >= {PRE_GROUPS} of "
                + ", ".join(VOTING))
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("#" + json.dumps(head, sort_keys=True) + "\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    return len(rows)


def read(path):
    """-> (header dict, {word: {column: int}})."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        head = json.loads(f.readline()[1:])
        cols = head["columns"]
        rows = {}
        for line in f:
            p = line.rstrip("\n").split("\t")
            rows[p[0]] = dict(zip(cols[1:], map(int, p[1:])))
    return head, rows
