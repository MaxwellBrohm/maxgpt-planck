"""Tiny hand-made raw starter files mimicking each real schema (see readers_cp.py for the facts).
They are written into a pytest tmp dir at test time (about 40 KB gzipped, 170 KB of text), never
into the repo.
Every planted case carries a MARKER string so a test can prove it reached no output."""
import gzip
import json
import os


SENTS = [
    "The kettle was still warm when we came back from the market.",
    "She told me that the bus would be late again this morning.",
    "If you want the garden to grow, you have to water it every day.",
    "We walked along the river and talked about what to cook for dinner.",
    "He asked whether the library was open on Sundays, and I said it was.",
    "It is easier to fix a small leak now than a big one later.",
    "My neighbor keeps a list of the birds that visit her feeder.",
    "They painted the fence blue because the old color had faded.",
    "When the power went out, we read by candlelight until it came back.",
    "I think the best part of the trip was the quiet road by the lake.",
    "You can store the bread in a cloth bag so it does not dry out.",
    "The class voted to plant a tree in the corner of the yard.",
    "Our dog waits by the door at five because that is when we walk.",
    "He wrote the recipe on a card and pinned it above the stove.",
    "There is a small shop on the corner that sells fresh bread.",
    "After the rain stopped, the children ran out to jump in the puddles.",
]
GERMAN = ("Wir sind gestern mit dem Zug nach Hamburg gefahren und haben dort Freunde besucht. "
          "Das Wetter war schlecht, aber das Essen im kleinen Restaurant war sehr gut. ") * 3
RUSSIAN = ("Мы вчера ездили на поезде в город и навестили старых друзей. "
           "Погода была плохой, но обед в маленьком кафе был очень вкусным. ") * 3
SWEDISH = ["jag gick hem från jobbet tidigt idag", "vi ska äta middag klockan sju",
           "det regnar hela dagen här hos oss", "har du sett min nya cykel ute på gården",
           "nej men den ser fin ut tycker jag", "vi ses på fredag efter jobbet"]
BY_SA = "Creative Commons - Attribution Share-Alike - https://creativecommons.org/licenses/by-sa/"
BY_NC_SA = ("Creative Commons - Attribution NonCommercial Share-Alike - "
            "https://creativecommons.org/licenses/by-nc-sa/4.0/")


BOILER = ("BOILERPLATE_MARKER This page uses cookies to improve your experience, and by going on "
          "you agree to our use of cookies and to the terms of the policy that you can read here.")
NEAR_BOILER = "NEAR_BOILER line that nine pages share, which is just under the threshold of ten."
DUP_GROUP_LINE = "DUP_GROUP line in nine distinct pages, one of them crawled twice."
SE_REPEAT = "SE_REPEAT Thanks, that answers my question."
RECRAWL = "RECRAWL_LINE the page content that each crawl of the same address repeats."
WIKI_BANNER = ("WIKI_BANNER_MARKER The developer team is making some changes to how accounts work "
               "and you can read about it on the project page.")
IRC_NICK = "ZorblaxQuint"          # a planted nickname: must never reach the output
SHARED_CONTEXT = "The shared context paragraph that three rows quote. " + " ".join(SENTS[:3])


def prose(tag, n=5, sep=" "):
    k = sum(map(ord, str(tag)))
    return sep.join(SENTS[(k + 5 * j) % len(SENTS)] for j in range(n)) + f" (Note {tag}.)"


from fixtures_chat import msg, oasst_trees, tree, tree_ids  # noqa: E402,F401


def cp(i, text, created, **meta):
    return {"id": str(i), "text": text, "source": "fixture", "added": "2024-06-03T21:29:47Z",
            "created": created, "metadata": meta}


def cccc_docs():
    docs = [cp(f"c{i}", prose(f"c{i}", 6), f"2019-03-{1 + i % 28:02d}T05:33:58.000Z",
               url=f"example.org/{i}") for i in range(25)]
    docs[0]["text"] += "\n" + DUP_GROUP_LINE
    dup = docs[0]["text"].replace("\n", "  \r\n") + "\r\n"
    docs += [cp("c-post", "DATEGATE_POST_MARKER " + prose("cp", 6), "2023-02-01T00:00:00.000Z"),
             cp("c-edge-ok", prose("edge", 6), "2022-11-30T23:59:59.000Z"),
             cp("c-edge-drop", "BOUNDARY_DROP_MARKER " + prose("ed", 6), "2022-12-01T00:00:00Z"),
             cp("c-de", GERMAN, "2019-01-01T00:00:00Z"), cp("c-ru", RUSSIAN, "2019-01-01"),
             cp("c-short", "Too short to keep.", "2019-01-01"), cp("c-empty", "", "2019-01-01"),
             cp("c-dup", dup, "2019-01-02"),
             cp("c-baddate", "BADDATE_MARKER " + prose("bd", 6), "sometime in 2019"),
             cp("c-bponly", BOILER + "\nA short note about this page and the rest of it.", "2019-01-03")]
    docs += [cp(f"c-recrawl{k}", prose(f"rc{k}", 6) + "\n" + RECRAWL, f"20{13 + k % 9}-05-01",
                url=("https://www." if k % 2 else "http://") + "example.org/recrawl/")
             for k in range(10)]                     # one page in ten snapshots: kept
    for d in docs[1:11]:
        d["text"] += "\n" + BOILER                  # 10 docs + c-bponly: boilerplate
    for d in docs[11:20]:
        d["text"] += "\n" + NEAR_BOILER             # 9 docs: under the threshold, kept
    for d in docs[20:25] + docs[11:14]:
        d["text"] += "\n" + DUP_GROUP_LINE          # 8 docs + c0 + c-dup (its copy): 9 groups
    return docs


def se_docs():
    lic = BY_SA + "3.0/"
    docs = [cp(i, prose(f"s{i}", 7), "2012-05-01T10:00:00", license=lic, all_licenses=[lic],
               site="cooking.stackexchange.com") for i in range(15)]
    docs += [cp(100, "LICENSE_NC_MARKER " + prose("nc", 6), "2012-01-01", license=BY_NC_SA,
                all_licenses=[BY_NC_SA], site="cooking.stackexchange.com"),
             cp(101, "ALL_LICENSES_MARKER " + prose("al", 6), "2012-01-01", license=lic,
                all_licenses=[lic, "https://creativecommons.org/licenses/by-nd/4.0/"],
                site="cooking.stackexchange.com"),
             cp(102, "SE_POST_MARKER " + prose("sp", 6), "2024-01-01T00:00:00", license=lic,
                all_licenses=[lic], site="cooking.stackexchange.com"),
             cp(103, "NO_LICENSE_MARKER " + prose("nl", 6), "2012-01-01",
                site="cooking.stackexchange.com"),
             cp(104, "COMPONENT_NONE_MARKER " + prose("cn", 6), "2012-01-01", license=lic,
                all_licenses=[lic, "None"], site="cooking.stackexchange.com"),
             cp(105, prose("sl", 6) + "\nSE_LATE_MARKER ChatGPT generated answers are not allowed.",
                "2020-06-20T20:12:12", license=lic, all_licenses=[lic],
                site="cooking.stackexchange.com"),
             cp(106, prose("sy", 6) + "\nSE_YEAR_MARKER Funny how obvious it is now in 2024.",
                "2014-06-12", license=lic, all_licenses=[lic], site="cooking.stackexchange.com")]
    for d in docs[:11]:
        d["text"] += "\n" + SE_REPEAT
    return docs


def irc_log(tag, n=8):
    lines = [f"[13:{10 + j:02d}] <nick{j % 3}> {SENTS[(len(tag) + j) % len(SENTS)]}"
             for j in range(n)]
    lines.insert(2, "=== nick0 is now known as nick9")
    lines.insert(4, f"[13:3{len(tag) % 10}]  * nick1 waves ({tag})")
    return "\n".join(lines) + "\n"


def irc_docs():
    pd = dict(license="Public Domain", authors=[])
    docs = [cp(f"2015-04-17-#chan{i}", irc_log(f"i{i}"), "2015-04-17", channel=f"#chan{i}", **pd)
            for i in range(12)]
    for d in docs[:4]:
        d["text"] += (f"[14:01] <{IRC_NICK}> {SENTS[3]}\n[14:02] <nick2> zorblaxquint: {SENTS[4]}\n"
                      f"[14:03]  * {IRC_NICK} nods at nick2\n[14:04] <garden> {SENTS[2]}\n")
    swedish = "\n".join(f"[10:{j:02d}] <sv{j % 2}> {s}" for j, s in enumerate(SWEDISH * 2))
    docs += [cp("2015-04-17-#short", "[10:00] <a> hi\n[10:01] <b> hey\n", "2015-04-17", **pd),
             cp("2015-04-17-#ubuntu-se", swedish, "2015-04-17", channel="#ubuntu-se", **pd),
             cp("2023-06-01-#chan", "IRC_POST_MARKER\n" + irc_log("late"), "2023-06-01", **pd),
             cp("2015-04-17-#empty", "", "2015-04-17", **pd)]
    return docs


def wiki_docs():
    lic = BY_SA + "4.0/"
    kw = dict(license=lic, wiki="wikivoyage.com")
    docs = [cp(f"0-{i}", prose(f"w{i}", 6), "2019-06-01T00:00:00", namespace="0",
               title=f"Place {i}", **kw) for i in range(12)]
    docs += [cp("10-1", "TEMPLATE_MARKER " + prose("tp", 6), "2019-01-01", namespace="10", **kw),
             cp("0-900", "WIKI_POST_MARKER " + prose("wp", 6), "2024-06-11T03:07:42",
                namespace="0", **kw),
             cp("0-901", "WIKI_NC_MARKER " + prose("wn", 6), "2019-01-01", namespace="0",
                license=BY_NC_SA, wiki="wikivoyage.com"),
             cp("0-0", "DUP_ID_MARKER " + prose("di", 6), "2019-01-01", namespace="0", **kw),
             cp("1-5", "WIKI_TALK_MARKER " + prose("wt", 6), "2019-01-01", namespace="1", **kw),
             cp("3-5", "WIKI_USERTALK_MARKER " + prose("wu", 6), "2019-01-01", namespace="3",
                **kw),
             cp("2-5", "WIKI_USER_MARKER " + prose("wr", 6), "2019-01-01", namespace="2", **kw),
             cp("102-5", "WIKI_NS102_MARKER " + prose("wx", 6), "2019-01-01", namespace="102",
                **kw),
             cp("102-6", prose("cookbook", 6), "2019-01-01", namespace="102", license=lic,
                wiki="wikibooks.com"),
             cp("0-902", prose("wy", 6) + "\nWIKI_YEAR_MARKER Copyright 2003-2024 Wikimedia.",
                "2019-07-11", namespace="0", **kw)]
    for d in docs[:10]:
        d["text"] += "\n" + WIKI_BANNER
    return docs


def book(tag):
    body = "\n\n".join(prose(f"{tag}-{k}", 4) for k in range(30))
    return f"PG_HEADER_MARKER {tag}\n\n{body}\n\nPG_LICENSE_MARKER end of {tag}"


def gutenberg_docs(cccc_first_text):
    pd = dict(license="Public Domain")
    docs = [cp(i, book(f"b{i}"), None, language="en", title=f"Book {i}", **pd) for i in range(4)]
    for d in docs:
        del d["created"]
    docs += [cp(50, "FRENCH_MARKER " + book("fr"), None, language="fr", **pd),
             cp(51, cccc_first_text, None, language="en", **pd)]
    return docs


def write_jsonl(path, rows, gz=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    with (gzip.open(path, "wb") if gz else open(path, "wb")) as f:
        f.write(data)


def write_raw(root):
    """Write the fixture starter tree; returns facts the tests assert on."""
    trees, ok, res = oasst_trees()
    write_jsonl(f"{root}/OpenAssistant__oasst2/2023-11-05_oasst2_all.trees.jsonl.gz", trees)
    dolly = [{"instruction": prose(f"di{i}", 2), "context": prose(f"dc{i}", 2) if i % 2 else "",
              "response": prose(f"dr{i}", 3), "category": "open_qa"} for i in range(20)]
    dolly.append({"instruction": "Tope or Rope?", "context": "", "response": "Tope",
                  "category": "classification"})
    dolly.append({"instruction": prose("dai", 2), "context": "", "category": "open_qa",
                  "response": "AIISM_DOLLY_MARKER Voting is a personal choice, and as an AI, I "
                              "can't tell you which to prefer. " + prose("dair", 2)})
    dolly += [{"instruction": prose(f"sh{k}", 2), "context": SHARED_CONTEXT, "category": "qa",
               "response": prose(f"shr{k}", 3)} for k in range(3)]
    write_jsonl(f"{root}/databricks__databricks-dolly-15k/databricks-dolly-15k.jsonl", dolly, gz=False)
    cccc = cccc_docs()
    write_jsonl(f"{root}/common-pile__cccc/CC-MAIN-2019-13/cccc-CC-MAIN-2019-13-0000.json.gz", cccc)
    write_jsonl(f"{root}/common-pile__stackexchange/cooking.stackexchange.com/documents/"
                "00000_se.jsonl.gz", se_docs())
    write_jsonl(f"{root}/common-pile__ubuntu_irc/raw/documents/00000_ubuntu.jsonl.gz", irc_docs())
    write_jsonl(f"{root}/common-pile__wikimedia/00000_wikivoyage.com.jsonl.gz", wiki_docs())
    write_jsonl(f"{root}/common-pile__project_gutenberg/v0/documents/00000_pg.jsonl.gz",
                gutenberg_docs(cccc[0]["text"]))
    with open(f"{root}/FETCHED.json", "w") as f:
        f.write("{}")
    return {"ok_trees": ok, "reserved_trees": res}
