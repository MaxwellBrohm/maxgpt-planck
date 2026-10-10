"""OOD-H Part 1 review page for Max (rc12/DECISIONS_FOR_MAX.md item 13: fixed "only after Max's review"). No model.
Reads the file oodh/HASHES.txt names; REFUSES (exit 3, writes nothing) unless its sha256 and its sorted tree-id list
sha256 both match. Writes ONE self-contained page, sealed/oodh/REVIEW.html (item text: sealed, gitignored): no network
(no external script, font, image or link; CSP default-src 'none'), opens by double-click, light and dark. Per thread:
the human chat, a check of the conversation, one check per probe (kind in plain words, expected answer, IDEAL reply),
each Looks right / Problem / Not sure plus a note; marks in localStorage (try/catch), "Export my marks" downloads JSON.
Sample: the SAMPLE_N lowest sha256(SAMPLE_SALT + tree id). Time: words read (chat, questions, IDEAL) at WPM plus
SEC_PER_CHECK per check. Prints counts and hash prefixes only. Run: python3 -B oodh/review_view.py"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
import select  # noqa: E402,F401  the stdlib module, cached first: oodh/ holds a select.py of its own
sys.path[:0] = [HERE]
import argparse  # noqa: E402
import hashlib  # noqa: E402
import html  # noqa: E402
import re  # noqa: E402

import build as B  # noqa: E402  (read_jsonl, SEALED)
import hashes as H  # noqa: E402  (sha, ids_sha)

SAMPLE_SALT, SAMPLE_N, WPM, SEC_PER_CHECK = "planck-oodh-maxreview-v1:", 30, 200, 30
KIND_WORDS = {"H-FACT": "asks for a fact the person gave", "H-ASK": "asks what the person asked for",
              "H-CORR": "a correction: asks for the value the person changed it to",
              "H-ABS": "should say it does not know (nobody in the chat said it)"}


BUTTONS = [("looks_right", "Looks right"), ("problem", "Problem"), ("not_sure", "Not sure")]


class Refused(Exception):
    pass


def sample_key(tree_id):
    return hashlib.sha256((SAMPLE_SALT + tree_id).encode()).hexdigest()


def pick_sample(recs, n=SAMPLE_N):
    return set(sorted((r["tree_id"] for r in recs), key=sample_key)[:n])


def expected_hashes(hashes_path):
    """(file name, file sha256, tree-id list sha256) from HASHES.txt; Refused if either line is missing."""
    data = ids = None
    for line in open(hashes_path, encoding="utf-8"):
        m = re.fullmatch(r"sealed/oodh/(\S+) sha256 ([0-9a-f]{64})", line.strip())
        data = data or (m and m.groups())
        m = re.fullmatch(r"sorted tree ids .* sha256 ([0-9a-f]{64})", line.strip())
        ids = m.group(1) if m else ids
    if not data or not ids:
        raise Refused(f"{hashes_path} names no data file or no tree-id hash")
    return data[0], data[1], ids


def load_checked(sealed, hashes_path):
    name, want, want_ids = expected_hashes(hashes_path)
    path = os.path.join(sealed, name)
    got = H.sha(path)
    if got != want:
        raise Refused(f"sha256 mismatch on {name}: HASHES.txt {want[:16]}, file {got[:16]}")
    recs = B.read_jsonl(path)
    got_ids = H.ids_sha(r["tree_id"] for r in recs)
    if got_ids != want_ids:
        raise Refused(f"tree-id list sha256 mismatch: HASHES.txt {want_ids[:16]}, file {got_ids[:16]}")
    return name, got, recs


def E(s):
    return html.escape("" if s is None else str(s), quote=True)


def thread_turns(r):
    return [t for t in r["turns"] if "history_reply" in t]


def read_words(r):
    texts = [x for t in thread_turns(r) for x in (t["text"], t["history_reply"])]
    texts += [x for p in r["probes"] for x in (p["question"], p.get("ideal"))]
    return sum(len(str(x or "").split()) for x in texts)


def check_html(r, pid, kind, title, body):
    return (f'<div class="check" data-t="{E(r["id"])}" data-tree="{E(r["tree_id"])}" data-p="{E(pid)}" '
            f'data-k="{E(kind)}" data-mark=""><div class="kind">{title}</div>{body}<div class="btns">'
            + "".join(f'<button type="button" data-m="{m}">{w}</button>' for m, w in BUTTONS) + '</div><textarea '
            'aria-label="Notes" placeholder="Notes: what is wrong, or why you are not sure"></textarea></div>')


def probe_html(r, p, j, m):
    out = [f'<p class="q">{E(p["question"])}</p>']
    if p["grader"] == "ABS":
        out.append('<p class="exp">Expected: the assistant says it was never told this. Check that the answer is '
                   'nowhere above, neither from the person nor from the assistant.</p>')
    else:
        out.append(f'<p class="exp">Expected answer: <span class="gold">{E(p["gold"])}</span></p>')
        alts = [a for a in p.get("accepted") or [] if a != p["gold"]]
        out += [f'<p class="exp">Also counts as right: {", ".join(E(a) for a in alts)}</p>'] if alts else []
        out += ['<p class="exp">The earlier value, wrong unless the reply says it changed: '
                f'{", ".join(E(s) for s in p["stale"])}</p>'] if p.get("stale") else []
        out += [f'<p class="exp">The person said it in message {" and ".join(E(i) for i in p["src"])}.</p>'
                ] if p.get("src") else []
    out.append(f'<p class="exp">A reply that passes: <span class="ideal">{E(p.get("ideal"))}</span></p>')
    title = f'Question {j} of {m}: {E(KIND_WORDS.get(p["probe_kind"], p["probe_kind"]))}'
    return check_html(r, f'q{p["turn"]}', p["probe_kind"], title, "".join(out))


def thread_html(r, label, in_sample):
    chat = "".join(f'<div class="msg user"><div class="who">Person, message {E(t["i"])}</div>'
                   f'<p class="say">{E(t["text"])}</p></div><div class="msg bot"><div class="who">Assistant '
                   f'(a volunteer wrote this)</div><p class="say">{E(t["history_reply"])}</p></div>'
                   for t in thread_turns(r))
    checks = [check_html(r, "conversation", "CONVERSATION", "The conversation itself",
                         '<p class="q">Does it read like a real person? Is anything offensive or private?</p>')]
    checks += [probe_html(r, p, j, len(r["probes"])) for j, p in enumerate(r["probes"], 1)]
    return (f'<section class="thread" id="{E(r["id"])}" data-sample="{int(in_sample)}"><h2>{label}</h2>'
            f'<div class="chat">{chat}</div><h3>Checks</h3>{"".join(checks)}</section>')


def minutes(words, checks):
    return round(words / WPM + checks * SEC_PER_CHECK / 60)


CSS = """:root{--bg:#f6f5f1;--fg:#1c1c1a;--mut:#64635d;--card:#fff;--line:#dcdad2;--user:#dfe9fb;--bot:#eeede7;
--acc:#2350b8;--ok:#1d7439;--bad:#b3261e;--uns:#8a5d00}@media (prefers-color-scheme:dark){:root{--bg:#151514;
--fg:#ebeae5;--mut:#a3a29b;--card:#1e1e1c;--line:#3a3934;--user:#1d2b44;--bot:#292926;--acc:#93b2ff;--ok:#62c483;
--bad:#ff8f86;--uns:#e3b34f}}*{box-sizing:border-box}html{color-scheme:light dark}body{margin:0;background:var(--bg);
color:var(--fg);overflow-wrap:anywhere;font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,
sans-serif}main{max-width:780px;margin:0 auto;padding:0 16px 96px}h1{font-size:26px;margin:28px 0 8px}
h2{font-size:18px;margin:0 0 8px}h3{font-size:15px;color:var(--mut);margin:18px 0 0;text-transform:uppercase}
.intro p,.intro li{max-width:68ch}.fine{font-size:14px;color:var(--mut)}.bar{position:sticky;top:0;z-index:2;
background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;flex-wrap:wrap;gap:8px 14px;
align-items:center;font-size:15px}#prog{font-weight:600;margin-right:auto}.bar button,.btns button{font:inherit;
font-size:15px;padding:6px 12px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);
cursor:pointer}#warn{background:var(--bad);color:var(--bg);padding:8px 16px}.thread{background:var(--card);
border:1px solid var(--line);border-radius:12px;padding:16px;margin:22px 0}.msg{border-radius:12px;padding:8px 14px;
margin:8px 0;max-width:92%}.user{background:var(--user);margin-left:auto}.bot{background:var(--bot)}
.who{font-size:13px;font-weight:600;color:var(--mut)}.say{margin:2px 0 0;white-space:pre-wrap}
.check{border-top:1px solid var(--line);padding:14px 0 6px 12px;border-left:4px solid transparent;margin-top:10px}
.check[data-mark=looks_right]{border-left-color:var(--ok)}.check[data-mark=problem]{border-left-color:var(--bad)}
.check[data-mark=not_sure]{border-left-color:var(--uns)}.kind{font-size:13px;font-weight:600;color:var(--acc)}
.q{font-size:17px;font-weight:600;margin:4px 0}.exp{margin:4px 0}.gold{font-weight:700}
.btns{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 8px}.btns button[aria-pressed=true]{color:var(--bg);
font-weight:600}button[data-m=looks_right][aria-pressed=true]{background:var(--ok);border-color:var(--ok)}
button[data-m=problem][aria-pressed=true]{background:var(--bad);border-color:var(--bad)}
button[data-m=not_sure][aria-pressed=true]{background:var(--uns);border-color:var(--uns)}textarea{width:100%;
min-height:46px;font:inherit;font-size:15px;padding:8px;border-radius:8px;border:1px solid var(--line);background:var(
--bg);color:var(--fg)}[hidden]{display:none!important}"""

JS = """(function(){var B=document.body,sha=B.getAttribute("data-sha"),KEY="oodh-review-"+sha.slice(0,16),S={};
function warn(){document.getElementById("warn").hidden=false}try{S=JSON.parse(localStorage.getItem(KEY)||"{}")||{}}
catch(e){S={};warn()}function save(){try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){warn()}}
var only=document.getElementById("only"),checks=[].slice.call(document.querySelectorAll(".check"));function key(
c){return c.getAttribute("data-t")+"|"+c.getAttribute("data-p")}function get(c){return S[key(c)]||{mark:"",note:""}}
function put(c){return S[key(c)]||(S[key(c)]={mark:"",note:""})}function inS(c){return c.closest(
".thread").getAttribute("data-sample")==="1"}function shown(){return checks.filter(function(
c){return !only.checked||inS(c)})}function paint(c){var m=get(c).mark||"";c.setAttribute("data-mark",m);
[].forEach.call(c.querySelectorAll("button"),function(b){b.setAttribute("aria-pressed",String(b.getAttribute(
"data-m")===m))})}function progress(){var v=shown(),n=0,p=0,u=0;v.forEach(function(c){var m=get(c).mark;if(m)n++;if(
m==="problem")p++;if(m==="not_sure")u++});document.getElementById(
"prog").textContent="Marked "+n+" of "+v.length+" checks ("+p+" problem, "+u+" not sure)"}function filter(
){[].forEach.call(document.querySelectorAll(".thread"),function(t){t.hidden=only.checked&&t.getAttribute(
"data-sample")!=="1"});try{localStorage.setItem(KEY+"-all",only.checked?"0":"1")}catch(e){}progress()}checks.forEach(
function(c){var ta=c.querySelector("textarea");ta.value=get(c).note||"";paint(c);c.addEventListener("click",function(
e){var b=e.target.closest("button");if(!b)return;var r=put(c),m=b.getAttribute("data-m");r.mark=r.mark===m?"":m;save(
);paint(c);progress()});ta.addEventListener("input",function(){put(c).note=ta.value;save()})});
try{only.checked=localStorage.getItem(KEY+"-all")!=="1"}catch(e){}only.addEventListener("change",filter);filter();
document.getElementById("next").addEventListener("click",function(){var c=shown().filter(function(c){return !get(
c).mark})[0];if(c)c.scrollIntoView({behavior:"smooth",block:"center"})});document.getElementById(
"export").addEventListener("click",function(){var marks=checks.filter(function(c){var r=get(c);return r.mark||r.note}
).map(function(c){var r=get(c);return {thread:c.getAttribute("data-t"),tree_id:c.getAttribute("data-tree"),
probe:c.getAttribute("data-p"),kind:c.getAttribute("data-k"),in_sample:inS(c),mark:r.mark,note:r.note}});
var a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify({file:B.getAttribute(
"data-file"),file_sha256:sha,exported:new Date().toISOString(),marks:marks},null,1)],{type:"application/json"}));
a.download="oodh_review_marks.json";B.appendChild(a);a.click();setTimeout(function(){URL.revokeObjectURL(a.href);
a.remove()},2000)})})();"""


def render(recs, name, sha):
    """the page and its counts. Threads in file order; the sample is numbered in that order."""
    sample = pick_sample(recs)
    parts, k, words, checks = [], 0, 0, 0
    for r in recs:
        if r["tree_id"] in sample:
            k, words, checks = k + 1, words + read_words(r), checks + 1 + len(r["probes"])
        label = f"Sample {k} of {len(sample)}" if r["tree_id"] in sample else "Not in the sample"
        parts.append(thread_html(r, f"{label} ({E(r['id'])})", r["tree_id"] in sample))
    all_words, all_checks = sum(read_words(r) for r in recs), sum(1 + len(r["probes"]) for r in recs)
    head = (f'<header class="intro"><h1>OOD-H review: {len(sample)} conversations</h1>'
            '<p>Each conversation below is a real chat between a person and a volunteer who played the assistant '
            '(OpenAssistant data, held out of all of Planck\'s training). After it come 2 or 3 test questions. At test '
            'time the model reads the chat, is asked these questions, and must answer from what the <strong>person'
            '</strong> said.</p><p>For each question, check:</p><ol><li>Is the expected answer right, and is it the '
            'only fair answer?</li><li>Is the question fair, given the conversation? Could a careful reader answer '
            'it?</li></ol><p>For each conversation, check:</p><ol start="3"><li>Does it read like a real person?</li>'
            '<li>Is anything offensive or private (a real full name, address, phone number)?</li></ol>'
            '<p>Click <strong>Looks right</strong>, <strong>Problem</strong> or <strong>Not sure</strong>. For a '
            'problem, write one line in the note saying what is wrong. Marks save in this browser as you go. When you '
            'are done, click <strong>Export my marks</strong> and send the file it downloads.</p>'
            f'<p>How long: the sample has {words:,} words to read and {checks} checks, about '
            f'{minutes(words, checks)} minutes (at {WPM} words a minute plus {SEC_PER_CHECK} seconds a check). All '
            f'{len(recs)}: {all_words:,} words, {all_checks} checks, about {minutes(all_words, all_checks)} minutes.'
            f'</p><p class="fine">Which {len(sample)}: for each conversation, sha256 of "{E(SAMPLE_SALT)}" followed by '
            f'its tree id; the {len(sample)} lowest are the sample. The rule is fixed in oodh/review_view.py, so '
            f'nobody chose them. Untick "Sample only" to see all {len(recs)}. File {E(name)}, sha256 {E(sha)}, '
            'matches oodh/HASHES.txt.</p></header>')
    bar = ('<div class="bar"><span id="prog"></span><label><input type="checkbox" id="only" checked> Sample only'
           '</label><button type="button" id="next">Next unmarked</button><button type="button" id="export">Export '
           'my marks</button></div><div id="warn" hidden>This browser is not saving your marks (private window or '
           'blocked storage). Export before you close the page.</div>')
    page = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; '
            'script-src \'unsafe-inline\'"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>OOD-H review</title><style>{CSS}</style></head><body data-sha="{E(sha)}" data-file="{E(name)}">'
            f'{bar}<main>{head}{"".join(parts)}</main><script>{JS}</script></body></html>\n')
    stats = dict(threads=len(recs), probes=sum(len(r["probes"]) for r in recs), sample=len(sample),
                 sample_probes=checks - len(sample), words=words, minutes=minutes(words, checks))
    return page, stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sealed", default=B.SEALED)
    ap.add_argument("--hashes", default=os.path.join(HERE, "HASHES.txt"))
    ap.add_argument("--out", default=None, help="default <sealed>/REVIEW.html; must be under a sealed/ directory")
    a = ap.parse_args(argv)
    out = a.out or os.path.join(a.sealed, "REVIEW.html")
    if "sealed" not in os.path.abspath(out).split(os.sep):
        print(f"refusing --out outside a sealed/ directory: {out}", file=sys.stderr)
        return 2
    try:
        name, sha, recs = load_checked(a.sealed, a.hashes)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    page, st = render(recs, name, sha)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"sha256 check PASS: {name} {sha[:16]} and its tree-id list match oodh/HASHES.txt\n"
          f"threads {st['threads']}; probes {st['probes']}; sample {st['sample']} threads, {st['sample_probes']} "
          f"probes, {st['words']:,} words, about {st['minutes']} minutes\npage {out}: {len(page.encode()):,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
