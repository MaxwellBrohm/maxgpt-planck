"""Fixture sources for the W2 tests: tiny stand-ins for the five downloads, written into a temp root through
sources.fetch with a fake getter (no network), plus a fixture word list. Every name, city and line here is FAKE
fixture text written for the tests; nothing reaches a bank outside a temp directory."""
import csv
import io
import zipfile
from unittest import mock

from _corpus import corpus  # noqa: F401  (puts the pipeline on sys.path)
from bankpass import sources, wordload

ERA_YEARS = (1935, 1975, 2005)
# (name, sex, births in 1935, 1975, 2005)
SSA = [("Agatha", "F", 900, 50, 5), ("Bertram", "M", 800, 40, 5), ("Corinne", "F", 30, 700, 20),
       ("Desmond", "M", 20, 650, 30), ("Elowen", "F", 5, 30, 600), ("Fintan", "M", 5, 20, 640),
       ("Jordan", "F", 0, 300, 300), ("Jordan", "M", 0, 320, 280), ("May", "F", 700, 10, 5),
       ("Will", "M", 600, 30, 20), ("Novak", "M", 500, 20, 10), ("Bimbo", "F", 400, 10, 5),
       ("Gwendolyn", "F", 650, 40, 5), ("Horace", "M", 550, 30, 5), ("Imogen", "F", 10, 20, 500),
       ("Jasper", "M", 10, 25, 520), ("Kerensa", "F", 20, 450, 10), ("Lachlan", "M", 15, 430, 10)]
CENSUS = [("SMITH", 1, 2442977), ("BROWN", 2, 1437026), ("LAMBERT", 3, 500000), ("MCDONALD", 4, 400000),
          ("DONNELL", 5, 300000), ("ODONNELL", 6, 290000), ("YOUNG", 7, 280000), ("NOVAK", 8, 270000),
          ("ALDERTON", 9, 260000), ("BRISCOE", 10, 250000), ("CARWELL", 11, 240000), ("DUNMORE", 12, 230000)]
# geonameid, asciiname, feature code, country, population
CITIES = [(1, "Vantorra", "PPLC", "AA", 900000), (2, "Kelmouth", "PPLX", "AA", 800000),
          (3, "Nice", "PPLA", "BB", 700000), (4, "Vantorra", "PPL", "BB", 600000), (5, "Saint-Ober", "PPL", "BB", 500000),
          (6, "Orange", "PPL", "AA", 400000), (7, "Kelmouth", "PPLA2", "AA", 300000), (8, "Brimsby", "PPL", "BB", 200000)]
UNSAFE = ["bimbo", "brimsby", "bad phrase", "bimbo"]


def _zip(files):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for name, text in files.items():
            z.writestr(name, text)
    return b.getvalue()


def ssa_zip():
    files = {}
    for k, y in enumerate(ERA_YEARS):
        files[f"yob{y}.txt"] = "\n".join(f"{n},{s},{c[k]}" for n, s, *c in SSA if c[k]) + "\n"
    files["NationalReadMe.pdf"] = "readme"
    return _zip(files)


def census_zip():
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["name", "rank", "count", "prop100k", "cum_prop100k", "pctwhite", "pctblack", "pctapi", "pctaian",
                "pct2prace", "pcthispanic"])
    for n, r, c in CENSUS:
        w.writerow([n, r, c, "1", "1", "(S)", "1", "1", "1", "1", "1"])
    w.writerow(["ALL OTHER NAMES", "0", "29312001", "1", "1", "1", "1", "1", "1", "1", "1"])
    return _zip({"Names_2010Census.csv": out.getvalue()})


def cities_zip():
    rows = []
    for gid, name, code, cc, pop in CITIES:
        r = [str(gid), name, name, "", "0", "0", "P", code, cc, "", "01", "", "", "", str(pop), "", "0", "UTC", "2026"]
        rows.append("\t".join(r))
    return _zip({"cities15000.txt": "\n".join(rows) + "\n"})


COUNTRY_INFO = "#ISO\tISO3\tISO-Numeric\tfips\tCountry\n" + "AA\tAAA\t1\tAA\tAaland Test\nBB\tBBB\t2\tBB\tBeeland Test\n"
EVIDENCE = {"ssa_names": "license https://creativecommons.org/publicdomain/zero/1.0/",
            "census_surnames": 'href="//www2.census.gov/topics/genealogy/2010surnames/names.zip"',
            "geonames_cities": "This work is licensed under a Creative Commons Attribution 4.0 License,",
            "nemotron_personas_usa": "---\nlicense: cc-by-4.0\n---\n", "ldnoobw": "Attribution 4.0 International"}
DATA = {"ssa_names": {"names.zip": ssa_zip}, "census_surnames": {"names.zip": census_zip, "surnames.pdf": lambda: b"%PDF"},
        "geonames_cities": {"cities15000.zip": cities_zip, "countryInfo.txt": lambda: COUNTRY_INFO.encode()},
        "ldnoobw": {"en": lambda: ("\n".join(UNSAFE) + "\n\n").encode()}}


PARQUET = "data/train-00000-of-00011.parquet"


def make_sources(root, evidence=None, skip=()):
    """fetch every fixture source into root with a fake getter. Nemotron is fetched as ONE unpinned stand-in shard
    (the registry is patched for the fetch only); the build test reads its rows through a patched reader."""
    nemo = sources.SOURCES["nemotron_personas_usa"]
    with mock.patch.dict(nemo, files={PARQUET: (nemo["files"][PARQUET][0], None)}):
        return _fetch_all(root, dict(EVIDENCE, **(evidence or {})), skip)


def _fetch_all(root, ev, skip):
    by_url = {}
    for name, s in sources.SOURCES.items():
        for rel, (url, *_) in s["evidence"].items():
            by_url[url] = ev[name].encode()
        for rel, (url, _pin) in s["files"].items():
            if name in DATA and rel in DATA[name]:
                by_url[url] = DATA[name][rel]
            elif rel == PARQUET:
                by_url[url] = b"PAR1 fixture stand-in"

    def get(url, path):
        if url not in by_url or any(s in url for s in skip):
            raise OSError("fixture: not served")
        body = by_url[url]
        body = body() if callable(body) else body
        with open(path, "wb") as f:
            f.write(body if isinstance(body, bytes) else body.encode())
    return {n: sources.fetch(root, n, get=get) for n in sources.SOURCES}


def wordlist():
    def fam(flags=()):
        return {"pos": "noun", "pos_status": "pending", "rank": 1, "lists": {"RS", "RM", "RL"}, "forms": {},
                "flags": set(flags), "required_ok": False, "a": 0, "an": 0, "mass": 0, "tf": 1}
    fams = {"will": fam(), "nice": fam(), "young": fam(), "jasper": fam(["proper"]), "smith": fam(["proper"])}
    return wordload.WordList(fams, "computed", "wordlist@fixture:computed", {})


def persona_rows():
    def row(uuid, name, persona, age=40, sex="Female"):
        other = {f: f"{name} does other things" for f in ("professional_persona", "sports_persona", "arts_persona",
                                                          "travel_persona", "culinary_persona")}
        return dict(other, uuid=uuid, persona=f"{name} {persona}", age=age, sex=sex, occupation="clerk", state="ZZ")
    rows = [row("u1", "Ann de la Vega", "keeps a tidy garden and a calm head."),
            row("u2", "Bo Linn", "plans every weekend, and Bo writes it all down."),
            row("u3", "Cy Moor", "runs three miles at 5 each morning."),
            row("u4", "Di Hale", "is a young reader of maps.", age=15),
            row("u5", "Ed Rowe", "keeps a tidy garden and a calm head."),
            row("u6", "Fay Lund", "collects old postcards and fixes radios.", sex="Male")]
    rows.append(dict(rows[5], uuid="u7", persona="At forty, Gil Park sorts mail."))
    rows.append(dict(row("u8", "Hal Ives", ""), persona="Hal Ives, a calm cook who hums while he works."))
    rows.append(dict(row("u9", "Ivy Moss", ""), persona="Ivy Moss's garden is her pride."))
    return [(r, PARQUET, i) for i, r in enumerate(rows)]
