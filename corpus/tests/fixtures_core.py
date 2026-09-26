"""Hand-made raw files for the core v0 readers (readers_core.py), one per new schema, written into
a pytest tmp dir at test time. Each planted case carries a MARKER that must reach no output, and
each kept case a KEEP tag that must. Field shapes copy the real files (see readers_core.py)."""
import os

from fixtures import SENTS, prose, write_jsonl

BY = "Creative Commons - Attribution - https://creativecommons.org/licenses/by/4.0/"
BY3 = "Creative Commons - Attribution - https://creativecommons.org/licenses/by/3.0/"
BY_SA = "Creative Commons - Attribution Share-Alike - https://creativecommons.org/licenses/by-sa/4.0/"
CC0 = "Creative Commons Zero - Public Domain - https://creativecommons.org/publicdomain/zero/1.0/"
BY_NC = ("Creative Commons - Attribution NonCommercial - "
         "https://creativecommons.org/licenses/by-nc/4.0/")
GERMAN = ("Wir sind gestern mit dem Zug nach Hamburg gefahren und haben dort Freunde besucht. "
          "Das Wetter war schlecht, aber das Essen im kleinen Restaurant war sehr gut. ") * 3


def youtube():
    def yt(i, t, text, **kw):
        return dict(dict(channel_id=f"UC{i % 3}", title=f"Video {i}", description="d", tags=None,
                         published_time=t, cataloged_time="2024-02-05 08:57:07", duration=60 + i,
                         id=f"vid{i}", text=" " + text,
                         metadata={"url": f"https://www.youtube.com/watch?v=vid{i}",
                                   "license": BY}), **kw)
    return [yt(0, "2016-10-09T23:39:12Z", prose("KEEP_YT0", 5)),
            yt(1, "2022-11-30T23:59:59Z", prose("KEEP_YT1", 5)),
            yt(2, "2023-03-24T09:35:58Z", "YT_POST_MARKER " + prose("y2", 5)),
            yt(3, "2019-01-01T00:00:00Z", "YT_GERMAN_MARKER " + GERMAN),
            yt(4, "2018-01-01T00:00:00Z", "YT_NC_MARKER " + prose("y4", 5),
               metadata={"url": "u", "license": BY_NC})]


def news():
    def nw(i, outlet, created, url, text):
        return {"id": i, "text": text, "source": f"news-{outlet}", "added": "2024-05-27T14:45:34",
                "created": created, "metadata": {"license": BY, "url": url, "author": "A. Writer"}}
    gv = "https://globalvoices.org"
    return [nw(1, "globalvoices", "", f"{gv}/2004/10/26/a/", prose("KEEP_NEWS_URL", 5)),
            nw(2, "globalvoices", "", f"{gv}/2022/12/02/b/", "NEWS_URLPOST_MARKER " + prose("n2", 5)),
            nw(3, "globalvoices", "", f"{gv}/about/", prose("KEEP_NEWS_UNDATED", 5)),
            nw(4, "globalvoices", "", f"{gv}/about/x/",
               "NEWS_LATE_MARKER " + prose("n4", 5) + " We asked ChatGPT about it."),
            nw(5, "360info", "Published on December 13, 2021", "https://360info.org/x/",
               prose("KEEP_NEWS_TEXTDATE", 5)),
            nw(6, "altnews", "1st December 2022", "https://www.altnews.in/englishclone/y/",
               "NEWS_EDGE_MARKER " + prose("n6", 5)),
            nw(7, "milwaukeenns", "sometime last spring",
               "http://247wallst.com/special-report/2017/08/18/z/",
               "NEWS_FOREIGN_URL_MARKER " + prose("n7", 5)),
            nw(8, "oxpeckers", "08 Jul", "https://oxpeckers.org/2013/07/rhino/",
               prose("KEEP_NEWS_NOYEAR", 5))]


def pressbooks():
    def pb(i, created, text, lic=BY):
        return {"id": f"https://x.pressbooks.pub/b/chapter/{i}/", "text": text,
                "source": "pressbooks", "added": "2025-03-22T05:08:50", "created": created,
                "metadata": {"license": lic, "url": f"https://x.pressbooks.pub/b/chapter/{i}/",
                             "book_url": "https://x.pressbooks.pub/b/", "title": "Book",
                             "author": "A. Author", "institution": "", "subject": "Education"}}
    table = "|\n" + "\n|\n".join(SENTS[:4]) + "\n||\n| ✓ |\n" + prose("KEEP_PB_TABLE", 4)
    return [pb(1, "04-9-2021", prose("KEEP_PB1", 5)), pb(2, "09-30-2024", "PB_POST_MARKER " + prose("p", 5)),
            pb(3, "04-9-2021", "PB_NC_MARKER " + prose("p3", 5), BY_NC),
            pb(4, "10-23-2020", table, CC0), pb(5, "02-30-2020", "PB_BADDATE_MARKER " + prose("p5", 5))]


def oercommons():
    def oer(i, created, text, lic=BY):
        u = f"https://oercommons.org/courseware/lesson/{i}/overview"
        return {"id": u, "text": text, "source": "oercommons", "added": "2025-03-18T00:34:10",
                "created": created, "metadata": {"license": lic, "url": u, "title": f"L{i}",
                                                 "author": None}}
    return [oer(1, "08/27/2020", prose("KEEP_OER1", 5)),
            oer(2, "Activity/Lab", prose("KEEP_OER_SHIFTED", 5), "Public Domain"),
            oer(3, "13/45/2020", "OER_BADDATE_MARKER " + prose("o3", 5)),
            oer(4, "01/05/2023", "OER_POST_MARKER " + prose("o4", 5)),
            oer(5, None, prose("KEEP_OER_NONE", 5))]


def foodista():
    def fd(i, created, text):
        return {"id": i, "text": text, "source": "foodista", "added": "2024-05-23T00:42:18",
                "created": created, "metadata": {"license": BY3, "url": f"https://f.com/{i}",
                                                 "authors": ["B. Cook"]}}
    return [fd(0, "2007-12-09T00:00:00", prose("KEEP_FOOD0", 5)), fd(1, None, prose("KEEP_FOOD_UNDATED", 5)),
            fd(2, "2023-01-02T00:00:00", "FOOD_POST_MARKER " + prose("f2", 5))]


def pdr(kind):
    def pd(slug, date, text):
        return {"id": slug, "text": text, "source": "public-domain-review", "date": date,
                "author": "", "type": kind, "added": "2024-05-01T21:38:18",
                "metadata": {"license": BY_SA, "url": f"https://publicdomainreview.org/{slug}/"}}
    return [pd("same-slug", "Aug 2, 2011", prose(f"KEEP_PDR_{kind}", 5)),
            pd("late", "Mar 26, 2024", "PDR_POST_MARKER " + prose("r", 5))]


def ocr_page(k, lines_per_page=12, width=48):
    """One scanned page: running head + page number, then hard-wrapped prose, split words."""
    words = " ".join(SENTS[(k + j) % len(SENTS)] for j in range(6)).split()
    out, cur = [], ""
    for w in words:
        if len(cur) + 1 + len(w) > width:
            if len(w) >= 6 and len(cur) + 4 < width:          # split the word at the margin
                out.append(cur + " " + w[:3] + "-")
                cur = w[3:]
            else:
                out.append(cur)
                cur = w
        else:
            cur = (cur + " " + w).strip()
    out.append(cur + " ")
    head = [f"RHEAD_MARKER TALES OF THE HOUSE ", "", "", f"{10 + k} ", "", ""]
    return "\n".join(head + [x + " " for x in out[:lines_per_page]])


def loc():
    pages = "\n\n\n".join(ocr_page(k) for k in range(6))
    junk = "\n\n".join(["OCRJUNK_MARKER %^^W* ^^^^\"V ^'^iS^.* -^^", "♦", "• •", "o« •", ", ^■*. vv"])
    book = junk + "\n\n\n[library of congress.] LOCSTAMP_MARKER\n\n\n" + pages + "\n\n\n0 014 012 345 6\n"
    allj = "\n\n".join(["LOCJUNK_BOOK_MARKER %^^W* ^^^^\"V ^'^iS^.* -^^ ~~ ^^"] * 40 + [prose("j", 3)])

    def lc(i, text, year=1890, lang="english"):
        return {"id": f"loc{i}", "source": "loc_books", "added": "2024-05-14T16:22:35",
                "text": text, "metadata": {"license": "Public Domain", "title": f"T{i}",
                                           "author": "Doe, Jane", "year": year, "language": lang,
                                           "item_url": f"https://www.loc.gov/item/loc{i}"}}
    return [lc(1, book), lc(2, allj), lc(3, "LOC_POST_MARKER " + book, year=2023),
            lc(4, "LOC_FRENCH_MARKER " + book, lang="french")]


def gutenberg():
    body = "\n\n".join(prose(f"g{k}", 4) for k in range(40))
    mid = "\n\n".join(prose(f"m{k}", 4) for k in range(40))
    book = ("Produced by PGCREDIT_MARKER Jane Doe and the Online Distributed\nProofreading Team\n\n"
            "Transcriber's Note: PGNOTE_MARKER spelling kept as printed.\n\n"
            "THE HOUSE BY THE LAKE\n\n" + body +
            "\n\nKEEP_PG_MID The Project Gutenberg Etext of a quoted charter follows.\n\n" + mid +
            "\n\nEnd of Project Gutenberg's The House by the Lake, PGTAIL_MARKER by Jane Doe")
    return [{"id": "77", "text": book, "source": "project gutenberg", "added": "2024-05-14",
             "metadata": {"license": "Public Domain", "language": "en", "title": "The House",
                          "url": "https://www.gutenberg.org/ebooks/77.txt.utf-8"}}]


def write_core_raw(root):
    """-> {source: [(path, rel)]} of the fixture files, laid out as fetch_manifest leaves them."""
    files = {"youtube": [("common-pile__youtube/00000_youtube_commons.jsonl.gz", youtube())],
             "news": [("common-pile__news/v0/documents/00000_globalvoices.jsonl.gz", news())],
             "pressbooks": [("common-pile__pressbooks/00000_pressbooks.json.gz", pressbooks())],
             "oercommons": [("common-pile__oercommons/00000_oercommons.json.gz", oercommons())],
             "foodista": [("common-pile__foodista/v0/documents/00000_foodista.jsonl.gz", foodista())],
             "pdr": [(f"common-pile__public_domain_review/v0/00000_{k}s.jsonl.gz", pdr(k))
                     for k in ("collection", "essay")],
             "loc": [("common-pile__library_of_congress/data/00000_loc_books.jsonl.gz", loc())],
             "gutenberg": [("common-pile__project_gutenberg/v0/documents/00000_pg.jsonl.gz",
                            gutenberg())]}
    out = {}
    for src, items in files.items():
        for rel, rows in items:
            write_jsonl(os.path.join(root, rel), rows)
            out.setdefault(src, []).append((os.path.join(root, rel), rel))
    return out
