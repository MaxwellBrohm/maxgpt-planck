"""Dedup priority of the strict-open core v0, one order for every pass: tier first, then the
source's place in PRIORITY within the tier (chat anchors first), then manifest order. Pass A
(stream_core.py, exact dedup) commits files in this order and core_finalize.py's near-dedup keep
rule ranks shards by (tier, PRIORITY) too, so an exact and a near copy shared by two sources go
to the same source.

A file that failed and is retried in a later run commits after files ranked below it (a late
file). Its documents still win exact dedup: best_owner() gives, for every key already in the
store, the best rank among the files that own it, and stream_work.commit_file keeps a document
whose owners all rank below its own file. The owners' copies are then recorded as displaced in the
late file's file_done event, the store gains an override row (KeyStore.commit_file), and
core_finalize.py drops the displaced rows before near-dedup (core_displaced.py). The stage1
output of such a run holds both copies until finalize; the final output is the one a run without
the failure gives.
"""
PRIORITY = ["oasst2", "dolly", "stackexchange", "irc", "wikimedia", "news", "pressbooks",
            "oercommons", "foodista", "pdr", "gutenberg", "youtube", "loc", "cccc"]


def prio(src) -> int:
    return PRIORITY.index(src) if src in PRIORITY else len(PRIORITY)


def fid_of(entry) -> str:
    return f"{entry['dataset']}/{entry['path']}"


def manifest_ranks(manifest) -> dict:
    """-> {fid: rank} for every manifest file; rank 0 wins exact dedup over rank 1, and so on."""
    files = manifest["files"]
    order = sorted(range(len(files)), key=lambda i: (files[i].get("tier") or 0,
                                                     prio(files[i]["source"]), i))
    return {fid_of(files[i]): r for r, i in enumerate(order)}


def best_owner(store, table, values, rank) -> dict:
    """-> {value: (rank, fid)} of the best-ranked committed owner of every value already in the
    store table ("keys" or "ids"). A file missing from `rank` ranks first (-1): its documents
    are never displaced."""
    out = {}
    for k, f in store.owners(table, values):
        fid = store.fid(f)
        r = rank.get(fid, -1)
        if k not in out or r < out[k][0]:
            out[k] = (r, fid)
    return out
