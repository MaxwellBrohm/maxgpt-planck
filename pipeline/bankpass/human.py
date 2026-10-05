"""Human pools (BANKPASS s3 P; W2): first names from SSA baby names (by era and sex), surnames from the 2010 Census,
cities from GeoNames cities15000 with their countries. Every candidate the selection looks at becomes an item record
(store.make_item, author kind "human" with the s2h source fields), kept or dropped with a code; candidates never
reached are counted in the stats, not stored. Deterministic: rank orders and a seeded hash order, no clock.

Drop codes, in the order they are tried (cities try FEATURE and DUP_NAME first, surnames CASE_AMBIG first):
  pool checks (gates.pool_checks: HELDOUT_POOL, DASH, MARKDOWN, EMOJI, SAFETY, NON_ENGLISH, LEN_ITEM)
  FORM               anything but letters, spaces and inner apostrophes ("St. Louis", "Saint-Denis")
  PROGRAM_COLLISION  equal to a closed-list value (May, June, Orange, Brown): the checker matches values literally
  WORD_COLLISION     a one-word value that is an English word in the W1 list and not flagged proper ("Will", "Grace",
                     "Nice", "Reading"): sentence-initial "Will you" would match the name Will, and lowercase-style
                     user turns match values case-insensitively (check_base.has)
  SAFETY_LIST        holds an LDNOOBW term (the real rubric list; SAFETY above is the FAKE checker list)
  CASE_AMBIG         surnames only: Census drops punctuation and case, so MCDONALD, ODONNELL, DELEON cannot be
                     cased from the file (any MC name; O, MAC, DE, DI, DA, DEL, DELA, LA, LE, VAN, VON, ST, DU plus a
                     remainder of 3+ letters that is itself a listed surname)
  DUP_NAME           cities only: a smaller city with the same ASCII name (the largest keeps it)
  FEATURE            cities only: not a city feature code (sections, historical, abandoned places)"""
import collections
import csv
import hashlib
import io
import os
import re
import zipfile

import fake_data as F
import heldout
from bankpass import gates, sources, store

ERAS = (("older", 1930, 1959), ("middle", 1960, 1989), ("younger", 1990, 9999))
NAME_TARGET, SURNAME_TARGET, CITY_TARGET = 5000, 2000, 3000
COMMON_DEPTH, RARE_MIN = 1000, 100           # names: common band = a cell's top 1,000; rare = deeper, 100+ births
SEX_PURE = 0.9                               # share of births that makes a name F or M; otherwise U (either)
SURNAME_SCAN, SURNAME_RARE_TO = 3000, 25000      # common: first passing in rank order
CITY_CODES = {"PPL", "PPLA", "PPLA2", "PPLA3", "PPLA4", "PPLA5", "PPLC", "PPLG"}
PREFIXES = ("MAC", "DELA", "DEL", "VAN", "VON", "DE", "DI", "DA", "LA", "LE", "ST", "DU", "O")
FORM_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?(?: [A-Za-z]+(?:'[A-Za-z]+)?)*")
PROGRAM = {v.lower() for vals in (F.WEEKDAYS, F.MONTHS, F.COLOURS, F.NUMBER_WORDS, F.ORDINALS, F.TIMES) for v in vals}


def unsafe_re(terms):
    terms = sorted({t.strip().lower() for t in terms if t.strip()}, key=len, reverse=True)
    if not terms:
        return None
    return re.compile(r"(?<![a-z])(?:" + "|".join(re.escape(t) for t in terms) + r")(?![a-z])", re.I)


class Ctx:
    """what the value checks need: the W1 word list (WordList or None), the LDNOOBW regex, the gate hash."""
    def __init__(self, wordlist=None, unsafe=None):
        self.wl, self.unsafe = wordlist, unsafe
        self.gate = {"gate_hash": heldout.gate_hash(), "rc12": heldout.RC12_STATUS,
                     "oodh": "not an OASST2 source"}

    def common_word(self, value):
        if not self.wl or " " in value:
            return False
        fam = self.wl.families.get(self.wl.head(value))
        return bool(fam) and "proper" not in fam["flags"]


def value_drop(value, vtype, ctx):
    """the first drop code for a pool value, or None."""
    hits = gates.pool_checks(value, vtype)
    if hits:
        return hits[0][0]
    if not FORM_RE.fullmatch(value):
        return "FORM"
    if value.lower() in PROGRAM:
        return "PROGRAM_COLLISION"
    if ctx.common_word(value):
        return "WORD_COLLISION"
    if ctx.unsafe and ctx.unsafe.search(value):
        return "SAFETY_LIST"
    return None


def _item(bank, n, value, rec, rel, row, features, ctx, drop):
    author = {"kind": "human", "source": sources.source_block(rec, rel, row)}
    gates_ = dict(ctx.gate, hits=[drop] if drop else [])
    return store.make_item("P", bank, n, value, author, status="dropped" if drop else "kept", drop=drop,
                           features=features, gates=gates_)


def hash_order(seed, keys):
    return sorted(keys, key=lambda k: hashlib.sha256(f"{seed}|{k}".encode()).hexdigest())


def _split(total, parts):
    return [total // parts + (i < total % parts) for i in range(parts)]


# ---- SSA first names ------------------------------------------------------------------------------------------
def ssa_counts(zip_path):
    """{name: {"F": {era: n}, "M": {era: n}}}, {era: total births}, last year: births 1930 on, per ERAS."""
    names = collections.defaultdict(lambda: {"F": collections.Counter(), "M": collections.Counter()})
    totals, last = collections.Counter(), 0
    with zipfile.ZipFile(zip_path) as z:
        for fn in sorted(z.namelist()):
            m = re.fullmatch(r"yob(\d{4})\.txt", fn)
            if not m:
                continue
            year = int(m.group(1))
            era = next((e for e, lo, hi in ERAS if lo <= year <= hi), None)
            last = max(last, year)
            if not era:
                continue
            for line in z.read(fn).decode("ascii").splitlines():
                name, sex, n = line.strip().split(",")
                names[name][sex][era] += int(n)
                totals[era] += int(n)
    return names, totals, last


def name_profiles(names, totals):
    """per name: sex label, share_f, era (highest births per era total), eras in use, births, overall rank."""
    out = {}
    for name, d in names.items():
        f, m = sum(d["F"].values()), sum(d["M"].values())
        rate = {e: (d["F"][e] + d["M"][e]) / totals[e] for e, _, _ in ERAS if totals[e]}
        top = max(rate.values())
        share_f = f / (f + m)
        out[name] = {"sex": "F" if share_f >= SEX_PURE else "M" if share_f <= 1 - SEX_PURE else "U",
                     "share_f": round(share_f, 3), "cell_sex": "F" if share_f >= 0.5 else "M",
                     "era": max(rate, key=lambda e: (rate[e], e)), "eras": [e for e in rate if rate[e] >= top / 2],
                     "births": f + m, "era_births": {e: d["F"][e] + d["M"][e] for e in rate}}
    for i, name in enumerate(sorted(out, key=lambda k: (-out[k]["births"], k)), 1):
        out[name]["ssa_rank"] = i
    return out


def build_names(rec, ctx, zip_rel="names.zip", target=NAME_TARGET, seed="bankpass-names-v0"):
    """-> (items, stats). Six cells (era x sex), each half common band (a cell's top), half rare band (seeded)."""
    names, totals, last = ssa_counts(os.path.join(rec["_dir"], zip_rel))
    prof = name_profiles(names, totals)
    cells = [(e, s) for e, _, _ in ERAS for s in ("F", "M")]
    items, stats = [], {"universe": len(prof), "last_year": last, "cells": {}}
    row_of = lambda nm: f"{nm}: births {prof[nm]['births']} from 1930 to {last}, yob files"
    for (era, sex), quota in zip(cells, _split(target, len(cells))):
        ranked = sorted((n for n, p in prof.items() if p["era"] == era and p["cell_sex"] == sex),
                        key=lambda n: (-prof[n]["era_births"][era], n))
        pos = {n: i for i, n in enumerate(ranked, 1)}
        common = ranked[:COMMON_DEPTH]
        rare = hash_order(seed + era + sex, [n for n in ranked[COMMON_DEPTH:] if prof[n]["era_births"][era] >= RARE_MIN])
        got = collections.Counter()
        for band, cands, want in (("common", common, quota - quota // 2), ("rare", rare, quota // 2)):
            for nm in cands:
                if got[band] >= want:
                    break
                drop = value_drop(nm, "name", ctx)
                p = prof[nm]
                feats = {k: p[k] for k in ("sex", "share_f", "era", "eras", "births", "ssa_rank")}
                feats.update(band=band, cell=f"{era}.{sex}", cell_rank=pos[nm])
                items.append(_item("pool.name", len(items), nm, rec, zip_rel, row_of(nm), feats, ctx, drop))
                got[band] += not drop
        stats["cells"][f"{era}.{sex}"] = {"quota": quota, "kept": dict(got), "ranked": len(ranked),
                                          "rare_candidates": len(rare)}
    return items, stats


# ---- Census 2010 surnames ----------------------------------------------------------------------------------------
def census_rows(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        fn = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        text = z.read(fn).decode("utf-8-sig")
    rows = [r for r in csv.DictReader(io.StringIO(text)) if r["name"] and r["name"] != "ALL OTHER NAMES"]
    return fn, rows


def case_ambiguous(name, listed):
    if name.startswith("MC"):
        return True
    return any(name.startswith(p) and len(name) - len(p) >= 3 and name[len(p):] in listed for p in PREFIXES)


def build_surnames(rec, ctx, zip_rel="names.zip", target=SURNAME_TARGET, seed="bankpass-surnames-v0"):
    fn, rows = census_rows(os.path.join(rec["_dir"], zip_rel))
    listed = {r["name"] for r in rows}
    rows.sort(key=lambda r: int(r["rank"]))
    common = rows[:SURNAME_SCAN]
    rare = [r for _, r in sorted((hashlib.sha256(f"{seed}|{r['name']}".encode()).hexdigest(), r)
                                 for r in rows[SURNAME_SCAN:SURNAME_RARE_TO])]
    items, got = [], collections.Counter()
    for band, cands, want in (("common", common, target - target // 2), ("rare", rare, target // 2)):
        for r in cands:
            if got[band] >= want:
                break
            v = r["name"].capitalize()
            drop = "CASE_AMBIG" if case_ambiguous(r["name"], listed) else value_drop(v, "surname", ctx)
            feats = {"rank": int(r["rank"]), "count": int(r["count"]), "band": band}
            items.append(_item("pool.surname", len(items), v, rec, zip_rel, f"{fn} rank {r['rank']}", feats, ctx,
                               drop))
            got[band] += not drop
    return items, {"universe": len(rows), "kept": dict(got)}


# ---- GeoNames cities ----------------------------------------------------------------------------------------------
def countries(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.rstrip("\n").split("\t")
            out[p[0]] = p[4]
    return out


def build_cities(rec, ctx, target=CITY_TARGET):
    cc = countries(os.path.join(rec["_dir"], "countryInfo.txt"))
    with zipfile.ZipFile(os.path.join(rec["_dir"], "cities15000.zip")) as z:
        lines = z.read("cities15000.txt").decode("utf-8").splitlines()
    rows = [line.split("\t") for line in lines if line.strip()]
    same = collections.Counter(r[2].lower() for r in rows)
    rows.sort(key=lambda r: (-int(r[14] or 0), int(r[0])))
    items, seen, kept = [], set(), 0
    for r in rows:
        if kept >= target:
            break
        v = store.straight(r[2])
        drop = "FEATURE" if r[7] not in CITY_CODES else "DUP_NAME" if v.lower() in seen else value_drop(v, "city", ctx)
        if r[7] in CITY_CODES:
            seen.add(v.lower())
        feats = {"country_code": r[8], "country": cc.get(r[8]), "population": int(r[14] or 0), "feature_code": r[7],
                 "geonameid": int(r[0]), "same_name": same[r[2].lower()]}
        items.append(_item("pool.city", len(items), v, rec, "cities15000.zip", f"geonameid {r[0]}", feats, ctx, drop))
        kept += not drop
    return items, {"universe": len(rows), "kept": kept}


def tally(items, *keys):
    """counts of kept items by feature value(s), for the manifest (counts only, no item text)."""
    out = collections.Counter()
    for it in items:
        if it["status"] == "kept":
            out["|".join(str(it["features"].get(k)) for k in keys)] += 1
    return dict(sorted(out.items()))
