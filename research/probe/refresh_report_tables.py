"""Replace each generated table in lanes/probe.md with a fresh copy from results.json
(identifies a table by its header row; run analyze.py first)."""
import io, contextlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report_tables as T

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lanes", "probe.md")


def cap(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*a)
    lines = [l for l in buf.getvalue().split("\n") if l.startswith("|")]
    return lines


def replace_table(text, new_lines, occurrence=0):
    header = new_lines[0]
    lines = text.split("\n")
    hits = [i for i, l in enumerate(lines) if l == header]
    if len(hits) <= occurrence:
        raise SystemExit(f"table header not found: {header[:80]}")
    i = hits[occurrence]
    j = i
    while j < len(lines) and lines[j].startswith("|"):
        j += 1
    return "\n".join(lines[:i] + new_lines + lines[j:])


if __name__ == "__main__":
    s = open(P).read()
    jobs = [((T.ability, "greedy"), 0), ((T.recall_distance, ["greedy", "sampled"]), 0), ((T.e2_curves,), 0),
            ((T.taxonomy, "failure_taxonomy_greedy_multiturn"), 0), ((T.corrections, "greedy"), 0),
            ((T.misc, "greedy"), 0), ((T.ctxcap, "greedy"), 0)]
    if "--sampled" in sys.argv:
        jobs += [((T.misc, "sampled"), 1), ((T.ability, "sampled"), 1)]
    for (args, occ) in jobs:
        s = replace_table(s, cap(*args), occ)
    open(P, "w").write(s)
    print("refreshed", len(jobs), "tables")
