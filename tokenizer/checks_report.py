"""Markdown rendering of checks.py results (works on the in-memory dict or on the JSON read back)."""
from __future__ import annotations

from checks_bpt import POOLED
from spec import size_label


def _k(d: dict) -> list:
    """Size keys in numeric order (ints in memory, strings after a JSON round trip)."""
    return sorted(d, key=int)


def _get(d: dict, v):
    return d[v] if v in d else d[str(v)]


def _fmt(x, nd=3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def table_md(res: dict) -> list[str]:
    bpt = res["bytes_per_token"]
    sizes = _k(bpt)
    srcs = [s for s in res["sources"]] + [POOLED]
    first = _get(bpt, sizes[0])
    out = ["## Bytes per token on held-out, per source (higher is better)", "",
           "| source | docs | MB | " + " | ".join(size_label(int(v)) for v in sizes) + " |",
           "|---|---:|---:|" + "---:|" * len(sizes)]
    for s in srcs:
        row = [s, str(first[s]["docs"]), f"{first[s]['bytes'] / 1e6:.2f}"]
        row += [_fmt(_get(bpt, v)[s]["bpt"]) for v in sizes]
        out.append("| " + " | ".join(row) + " |")
    return out + [""]


def p098_md(r: dict) -> list[str]:
    out = ["## P-098: nested-truncated vs separately trained, same sample", ""]
    if r.get("status") == "not_run":
        return out + [f"Not run: {r['reason']}.", ""]
    idn = r["identity"]
    out += [f"Size {size_label(int(r['size']))}. Vocab identical: {idn['vocab_identical']}. "
            f"Merges identical: {idn['merges_identical']} ({idn['n_merges'][0]} vs {idn['n_merges'][1]}).", "",
            "| source | truncated bpt | separate bpt | truncated vs separate (%) |", "|---|---:|---:|---:|"]
    for s, x in r["per_source"].items():
        out.append(f"| {s} | {_fmt(x['bpt_truncated'])} | {_fmt(x['bpt_separate'])} | {_fmt(x['rel_diff_pct'], 4)} |")
    out += ["", f"Ledger default ('truncated 8k compresses about as well'): "
            f"{'holds' if r['default_holds'] else 'fails'} ({r['verdict']}).", ""]
    return out


def p101_md(r: dict) -> list[str]:
    out = ["## P-101: chat-phrase superword tokens at a fixed vocabulary size", ""]
    if r.get("status") == "not_run":
        return out + [f"Not run: {r['reason']}.", ""]
    srcs = r["chat_sources"]
    out += [f"Base {size_label(int(r['base_vocab']))}; phrases mined from the training chat "
            f"({', '.join(f'{s} {n} docs' for s, n in r['mined_from'].items())}); "
            f"{r['distinct_phrases_mined']} distinct 2-4-word phrases. Each arm truncates the base by N "
            "merge-made tokens and adds the top N phrases, so the total stays the same. Gain = rise in "
            "bytes/token on held-out chat.", "",
            "| N | ranking | pooled gain (%) | " + " | ".join(f"{s} gain (%)" for s in srcs)
            + " | real added-token gain, pooled (%) | cost of dropping N merges (%) | phrase tokens used |",
            "|---:|---|---:|" + "---:|" * len(srcs) + "---:|---:|---:|"]
    for a in r["arms"]:
        g = a["gain_pct"]
        out.append(f"| {a['n']} | {a['ranking']} | {_fmt(g[POOLED], 2)} | "
                   + " | ".join(_fmt(g[s], 2) for s in srcs)
                   + f" | {_fmt((a.get('real_gain_pct') or {}).get(POOLED), 2)}"
                   + f" | {_fmt(a['drop_merges_gain_pct'][POOLED], 2)} | {a['phrase_tokens_used']} |")
    freq = [a for a in r["arms"] if a["ranking"] == "freq"]
    if freq:
        top = freq[-1]["examples"][:10]
        out += ["", "Most frequent phrases: " + ", ".join(f"`{e['phrase']!r}` ({e['count']})" for e in top)]
    out += ["", f"Bar (>= {r['bar_pct']}% on the specified frequency ranking): "
            f"{'passes' if r['passes_bar'] else 'fails'}. Best frequency-ranked gain {_fmt(r['best_gain_pct_freq'], 2)}%; "
            f"best of any ranking {_fmt(r['best_gain_pct_any'], 2)}%. With real added tokens (which also "
            f"fire inside words) the best frequency-ranked gain is {_fmt(r.get('best_real_gain_pct_freq'), 2)}%: "
            f"{'passes' if r.get('passes_bar_real') else 'fails'}.", ""]
    return out


def p097_md(r: dict) -> list[str]:
    out = ["## P-097: name tokenization across surface forms (200 common US names)", "",
           "| size | 1 token ' Name' | 1 token 'Name' | 1 token ' name' | 1 token 'NAME' | mean tokens ' Name' "
           "| count differs | first id differs | first piece differs | ' Name' and 'Name' split alike |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for v in _k(r):
        a = _get(r, v)["all"]
        one = a["one_token_frac"]
        out.append(f"| {size_label(int(v))} | {_fmt(one['space_cap'])} | {_fmt(one['cap'])} | "
                   f"{_fmt(one['space_lower'])} | {_fmt(one['upper'])} | {_fmt(a['mean_tokens']['space_cap'], 2)} | "
                   f"{_fmt(a['count_differs_frac'])} | {_fmt(a['first_id_differs_frac'])} | "
                   f"{_fmt(a['first_piece_differs_frac'])} | {_fmt(a['same_split_space_frac'])} |")
    return out + ["", "Fractions of names. Definitions in tokenizer/checks_names.py.", ""]


def to_markdown(res: dict) -> str:
    lines = ["# Tokenizer v0 CPU checks", "",
             f"Sizes: {', '.join(size_label(int(v)) for v in res['sizes'])}. "
             f"Held-out sources: {', '.join(res['sources'])}.", ""]
    lines += table_md(res) + p098_md(res["P-098"]) + p101_md(res["P-101"]) + p097_md(res["P-097"])
    lines += ["## P-100", "", f"Out of scope for v0: {res['P-100']['reason']}.", ""]
    return "\n".join(lines)
