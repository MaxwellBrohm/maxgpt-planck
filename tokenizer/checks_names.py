"""P-097 CPU count: how a name's tokenization changes with its surface form, per vocab size.

NAMES is a measurement list written here (100 common US first names, 100 common US surnames, no
duplicates). It is never training text. Each name is encoded (add_special_tokens=False) in 4 forms:
  " Name"  mid-sentence, capitalized (how a name is usually planted)
  "Name"   sentence- or line-initial
  " name"  lower-cased by the user
  "NAME"   upper-cased
Per vocab size and per list (first, last, all):
  one_token_frac[form]     fraction of names whose form is a single token (" Name" is the headline)
  mean_tokens[form]        mean tokens per name in that form
  count_differs_frac       fraction of names whose 4 forms do not all take the same number of tokens
  first_id_differs_frac    fraction of names whose 4 forms do not all start with the same token id
                           (expected near 1: " N" and "N" are different tokens by construction)
  first_piece_differs_frac same, after taking the first token's text, dropping one leading space
                           and lower-casing it: do the forms at least start with the same subword?
  same_split_space_frac    fraction of names where " Name" and "Name" split at the same places
                           (token texts equal once the leading space is dropped), the planted vs
                           sentence-initial recall case
"""
from __future__ import annotations

FIRST_NAMES = """James Mary Robert Patricia John Jennifer Michael Linda David Elizabeth William Barbara
Richard Susan Joseph Jessica Thomas Sarah Christopher Karen Charles Lisa Daniel Nancy Matthew Betty
Anthony Sandra Mark Margaret Donald Ashley Steven Kimberly Andrew Emily Paul Donna Joshua Michelle
Kenneth Carol Kevin Amanda Brian Melissa George Deborah Timothy Stephanie Ronald Dorothy Jason Rebecca
Edward Sharon Jeffrey Laura Ryan Cynthia Jacob Amy Gary Kathleen Nicholas Angela Eric Shirley Jonathan
Brenda Stephen Emma Larry Anna Justin Pamela Scott Nicole Brandon Samantha Benjamin Katherine Samuel
Christine Gregory Helen Alexander Debra Patrick Rachel Frank Carolyn Raymond Janet Jack Maria Dennis
Olivia Jerry Heather""".split()

SURNAMES = """Smith Johnson Williams Brown Jones Garcia Miller Davis Rodriguez Martinez Hernandez Lopez
Gonzalez Wilson Anderson Powell Taylor Moore Jackson Martin Lee Perez Thompson White Harris Sanchez
Clark Ramirez Lewis Robinson Walker Young Allen King Wright Perry Torres Nguyen Hill Flores Green Adams
Nelson Baker Hall Rivera Campbell Mitchell Carter Roberts Gomez Phillips Evans Turner Diaz Parker Cruz
Edwards Collins Reyes Stewart Morris Morales Murphy Cook Rogers Gutierrez Ortiz Morgan Cooper Peterson
Bailey Reed Kelly Howard Ramos Kim Cox Ward Richardson Watson Brooks Chavez Wood Jenkins Bennett Gray
Mendoza Ruiz Hughes Price Alvarez Castillo Sanders Patel Myers Long Ross Foster Jimenez""".split()

FORMS = {"space_cap": lambda n: " " + n, "cap": lambda n: n, "space_lower": lambda n: " " + n.lower(),
         "upper": lambda n: n.upper()}


def _norm_first(text: str) -> str:
    return (text[1:] if text.startswith(" ") else text).lower()


def encode_forms(tok, name: str) -> dict[str, list[int]]:
    return {f: tok.encode(fn(name), add_special_tokens=False).ids for f, fn in FORMS.items()}


def token_texts(tok, ids: list[int]) -> list[str]:
    return [tok.decode([i], skip_special_tokens=False) for i in ids]


def name_row(tok, name: str) -> dict:
    ids = encode_forms(tok, name)
    firsts = {f: token_texts(tok, v[:1])[0] for f, v in ids.items()}
    a, b = token_texts(tok, ids["space_cap"]), token_texts(tok, ids["cap"])
    a[0] = a[0][1:] if a[0].startswith(" ") else a[0]
    return {"name": name, "n_tokens": {f: len(v) for f, v in ids.items()},
            "count_differs": len({len(v) for v in ids.values()}) > 1,
            "first_id_differs": len({v[0] for v in ids.values()}) > 1,
            "first_piece_differs": len({_norm_first(t) for t in firsts.values()}) > 1,
            "same_split_space": a == b}


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}

    def frac(key):
        return round(sum(r[key] for r in rows) / n, 4)
    return {"n": n,
            "one_token_frac": {f: round(sum(r["n_tokens"][f] == 1 for r in rows) / n, 4) for f in FORMS},
            "mean_tokens": {f: round(sum(r["n_tokens"][f] for r in rows) / n, 4) for f in FORMS},
            "count_differs_frac": frac("count_differs"), "first_id_differs_frac": frac("first_id_differs"),
            "first_piece_differs_frac": frac("first_piece_differs"),
            "same_split_space_frac": frac("same_split_space")}


def p097(toks: dict[int, object], first=None, last=None, n_examples: int = 8) -> dict:
    """{vocab size: {"first": summary, "last": summary, "all": summary, "examples": [...]}}."""
    first = FIRST_NAMES if first is None else first
    last = SURNAMES if last is None else last
    out = {}
    for v in sorted(toks):
        rf = [name_row(toks[v], x) for x in first]
        rl = [name_row(toks[v], x) for x in last]
        worst = sorted(rf + rl, key=lambda r: (-sum(r["n_tokens"].values()), r["name"]))[:n_examples]
        out[v] = {"first": summarize(rf), "last": summarize(rl), "all": summarize(rf + rl),
                  "examples": [{"name": r["name"], "n_tokens": r["n_tokens"],
                                "forms": {f: token_texts(toks[v], toks[v].encode(fn(r["name"]),
                                                                                  add_special_tokens=False).ids)
                                          for f, fn in FORMS.items()}} for r in worst]}
    return out
