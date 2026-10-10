"""OOD-H Part 1 review page for Max (rc12/DECISIONS_FOR_MAX.md item 13: fixed "only after Max's review"). No model.
Reads the file oodh/HASHES.txt names; REFUSES (exit 3, writes nothing) unless its sha256 and its sorted tree-id list
sha256 both match. Writes ONE self-contained page, sealed/oodh/REVIEW.html (item text: sealed, gitignored): no network
(no external script, font, image or link; CSP default-src 'none'), opens by double-click, light and dark. Per thread:
the human chat, a check of the conversation, one check per probe (kind in plain words, expected answer, IDEAL reply),
each Looks right / Problem / Not sure plus a note; marks in localStorage (try/catch), "Export my marks" downloads JSON.
Sample: the SAMPLE_N lowest sha256(SAMPLE_SALT + tree id). Time: words read (chat, questions, IDEAL) at WPM plus
SEC_PER_CHECK per check. Prints counts and hash prefixes only. Run: python3 -B oodh/review_view.py
Re-check (DESIGN s15): add --only IDS --changes sealed/oodh/review_c3/c3_decisions.txt (MODES, parse_changes)."""
import os, sys  # noqa: E401
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
import select  # noqa: E402,F401  the stdlib module, cached first: oodh/ holds a select.py of its own
sys.path[:0] = [HERE]
import argparse, hashlib, html, re  # noqa: E401,E402
import build as B, hashes as H  # noqa: E401,E402  (read_jsonl, SEALED; sha, ids_sha)

SAMPLE_SALT, SAMPLE_N, WPM, SEC_PER_CHECK = "planck-oodh-maxreview-v1:", 30, 200, 30
KIND_WORDS = {"H-FACT": "asks for a fact the person gave", "H-ASK": "asks what the person asked for",
              "H-CORR": "a correction: asks for the value the person changed it to",
              "H-ABS": "should say it does not know (nobody in the chat said it)"}
BUTTONS = [("looks_right", "Looks right"), ("problem", "Problem"), ("not_sure", "Not sure")]
MODES = {False: ("REVIEW.html", "oodh-review-", "oodh_review_marks.json"),    # page, storage key prefix, export name
         True: ("REVIEW_RECHECK.html", "oodh-recheck-", "oodh_recheck_marks.json")}   # the re-check page (--only)
REF, GONE = re.compile(r"\b(\d{3}) t(\d+)\b"), re.compile(r"\b(drop|dropped|removed?)\b", re.I)  # see parse_changes
PART = re.compile(r"(?:\((?:[^()]|\([^()]*\))*\)|[^,;()])+")     # a run up to the next ',' or ';' outside brackets
E = lambda s: html.escape("" if s is None else str(s), quote=True)  # noqa: E731
thread_turns = lambda r: [t for t in r["turns"] if "history_reply" in t]  # noqa: E731
read_words = lambda r: sum(len(str(x or "").split()) for x in [x for t in thread_turns(r) for x in (t["text"], t[
    "history_reply"])] + [x for p in r["probes"] for x in (p["question"], p.get("ideal"))])  # noqa: E731
Refused = type("Refused", (Exception,), {})


def pick_sample(recs, n=SAMPLE_N):
    key = lambda t: hashlib.sha256((SAMPLE_SALT + t).encode()).hexdigest()  # noqa: E731
    return set(sorted((r["tree_id"] for r in recs), key=key)[:n])


def load_checked(sealed, hashes_path):
    """(name, sha256, records) of the file HASHES.txt names; Refused unless its sha256 and tree-id list sha256 match"""
    data, ids = (re.search(rf"^[ \t]*{k} sha256 ([0-9a-f]{{64}})[ \t]*$", open(hashes_path, encoding="utf-8").read(),
                           re.M) for k in (r"sealed/oodh/(\S+)", "sorted tree ids .*"))
    if not data or not ids:
        raise Refused(f"{hashes_path} names no data file or no tree-id hash")
    (name, want), want_ids, path = data.groups(), ids.group(1), os.path.join(sealed, data.group(1))
    if (got := H.sha(path)) != want:
        raise Refused(f"sha256 mismatch on {name}: HASHES.txt {want[:16]}, file {got[:16]}")
    recs = B.read_jsonl(path)
    if (got_ids := H.ids_sha(r["tree_id"] for r in recs)) != want_ids:
        raise Refused(f"tree-id list sha256 mismatch: HASHES.txt {want_ids[:16]}, file {got_ids[:16]}")
    return name, got, recs


def parse_changes(text):
    """{(thread id, turn): [(category, note)]} from the third review's decisions file: the block indented under a
    top-level line ending in ':', a category per line indented 1 or 2 ("<category>: 017 t3 (note), 019 t4 note; ..."),
    deeper lines continuing it. A note runs to the next ',' or ';' outside brackets; "021 t6 and 071 t5 note": both."""
    out = {}
    for cat in [c.strip() for b in re.findall(r"^\S.*:\n((?:[ \t]+\S.*(?:\n|$))+)", text, re.M)
                for c in re.sub(r"\n {3,}", " ", b).split("\n") if c.strip()]:
        head = cat[:REF.search(cat).start()].strip().rstrip(":").strip() if REF.search(cat) else ""
        for part in PART.findall(cat):
            refs = list(REF.finditer(part))
            note = re.sub(r"^\(([^()]*)\)$", r"\1", part[refs[-1].end():].strip().rstrip(".")).strip() if refs else ""
            for m in refs:
                out.setdefault((f"oodh-{m.group(1)}", int(m.group(2))), []).append((head, note))
    return out


def check_html(r, pid, kind, title, body):
    return (f'<div class="check" data-t="{E(r["id"])}" data-tree="{E(r["tree_id"])}" data-p="{E(pid)}" '
            f'data-k="{E(kind)}" data-mark=""><div class="kind">{title}</div>{body}<div class="btns">'
            + "".join(f'<button type="button" data-m="{m}">{w}</button>' for m, w in BUTTONS) + '</div><textarea '
            'aria-label="Notes" placeholder="Notes: what is wrong, or why you are not sure"></textarea></div>')


def probe_html(r, p, j, m, chg=""):
    out = [chg, f'<p class="q">{E(p["question"])}</p>']
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


def thread_html(r, label, in_sample, changes=None):
    def line(notes, gone=False):    # each question: changed in the third review or not; a removed one: at the top
        said = E("; ".join(c + (f": {n}" if n else "") for c, n in notes or []))
        return "" if notes is None else '<p class="exp fine">Not changed in the third review.</p>' if not notes else (
            f'<p class="chg{" gone" * gone}">Changed in the third review: {said}.{gone * " Not on this page."}</p>')
    mine = [(t, n, bool(GONE.search(n[0])) or t not in {p["turn"] for p in r["probes"]})
            for (tid, t), ns in sorted((changes or {}).items()) if tid == r["id"] for n in ns]
    chat = "".join(f'<div class="msg user"><div class="who">Person, message {E(t["i"])}</div>'
                   f'<p class="say">{E(t["text"])}</p></div><div class="msg bot"><div class="who">Assistant '
                   f'(a volunteer wrote this)</div><p class="say">{E(t["history_reply"])}</p></div>'
                   for t in thread_turns(r))
    checks = [check_html(r, "conversation", "CONVERSATION", "The conversation itself",
                         '<p class="q">Does it read like a real person? Is anything offensive or private?</p>')]
    checks += [probe_html(r, p, j, len(r["probes"]), line(None if changes is None else [
        n for t, n, g in mine if t == p["turn"] and not g])) for j, p in enumerate(r["probes"], 1)]
    return (f'<section class="thread" id="{E(r["id"])}" data-sample="{int(in_sample)}"><h2>{label}</h2>'
            f'{"".join(line([n], True) for t, n, g in mine if g)}<div class="chat">{chat}</div><h3>Checks</h3>'
            f'{"".join(checks)}</section>')


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
--bg);color:var(--fg)}[hidden]{display:none!important}.chg{margin:4px 0;font-weight:600;color:var(--uns)}"""
JS = """(function(){var B=document.body,sha=B.getAttribute("data-sha"),KEY="@KEY@",S={};
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
try{only.checked=!only.disabled&&localStorage.getItem(KEY+"-all")!=="1"}catch(e){}only.addEventListener("change",filter
);filter();document.getElementById("next").addEventListener("click",function(){var c=shown().filter(function(c){
return !get(c).mark})[0];if(c)c.scrollIntoView({behavior:"smooth",block:"center"})});document.getElementById(
"export").addEventListener("click",function(){var marks=checks.filter(function(c){var r=get(c);return r.mark||r.note}
).map(function(c){var r=get(c);return {thread:c.getAttribute("data-t"),tree_id:c.getAttribute("data-tree"),
probe:c.getAttribute("data-p"),kind:c.getAttribute("data-k"),in_sample:inS(c),mark:r.mark,note:r.note}});
var a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify({file:B.getAttribute(
"data-file"),file_sha256:sha,exported:new Date().toISOString(),marks:marks},null,1)],{type:"application/json"}));
a.download="@EXPORT@";B.appendChild(a);a.click();setTimeout(function(){URL.revokeObjectURL(a.href);a.remove()},2000)
})})();"""


def render(recs, name, sha, only=None, changes=None):
    sample, rc = pick_sample(recs), only is not None
    num = {t: i for i, t in enumerate([r["tree_id"] for r in recs if r["tree_id"] in sample], 1)}
    shown = [r for r in recs if not rc or r["id"] in only]
    parts = [thread_html(r, (f"Sample {num[r['tree_id']]} of {len(sample)}" if r["tree_id"] in num else "Not in the "
                             f"sample") + f" ({E(r['id'])})", r["tree_id"] in sample, changes) for r in shown]
    count = lambda rs: (sum(map(read_words, rs)), sum(1 + len(r["probes"]) for r in rs))  # noqa: E731
    mins = lambda w, c: round(w / WPM + c * SEC_PER_CHECK / 60)  # noqa: E731
    insample, (_, key, export) = [r for r in shown if r["tree_id"] in sample], MODES[rc]
    (words, checks), (aw, ac) = count(shown if rc else insample), count(recs)
    intro = (f'<h1>OOD-H re-check: {len(shown)} conversations</h1><p>These are the {len(shown)} conversations from '
             'your sample that changed after your review: the third review (DESIGN s15) fixed some of their '
             'questions. Please check them again, the same way as before. The expected answers, the answers that '
             'also count as right and the replies that pass are the <strong>new</strong> ones, and each question '
             'says whether the third review changed it. Marks here are kept apart from your first review.</p>' if rc
             else f'<h1>OOD-H review: {len(sample)} conversations</h1><p>Each conversation below is a real chat '
             'between a person and a volunteer who played the assistant (OpenAssistant data, held out of all of '
             'Planck\'s training). After it come 2 or 3 test questions. At test time the model reads the chat, is '
             'asked these questions, and must answer from what the <strong>person</strong> said.</p>')
    head = (f'<header class="intro">{intro}<p>For each question, check:</p><ol><li>Is the expected answer right, and '
            'is it the only fair answer?</li><li>Is the question fair, given the conversation? Could a careful reader '
            'answer it?</li></ol><p>For each conversation, check:</p><ol start="3"><li>Does it read like a real '
            'person?</li><li>Is anything offensive or private (a real full name, address, phone number)?</li></ol>'
            '<p>Click <strong>Looks right</strong>, <strong>Problem</strong> or <strong>Not sure</strong>. For a '
            'problem, write one line in the note saying what is wrong. Marks save in this browser as you go. When you '
            f'are done, click <strong>Export my marks</strong> and send the file it downloads, {export}.</p>'
            f'<p>How long: {f"these {len(shown)} conversations have" if rc else "the sample has"} {words:,} words to '
            f'read and {checks} checks, about {mins(words, checks)} minutes (at {WPM} words a minute plus '
            f'{SEC_PER_CHECK} seconds a check).' + ('' if rc else f' All {len(recs)}: {aw:,} words, {ac} checks, '
            f'about {mins(aw, ac)} minutes.') + '</p><p class="fine">' + ('' if rc else f'Which {len(sample)}: for '
            f'each conversation, sha256 of "{E(SAMPLE_SALT)}" followed by its tree id; the {len(sample)} lowest are '
            'the sample. The rule is fixed in oodh/review_view.py, so nobody chose them. Untick "Sample only" to see '
            f'all {len(recs)}. ') + f'File {E(name)}, sha256 {E(sha)}, matches oodh/HASHES.txt.</p></header>')
    bar = (f'<div class="bar"><span id="prog"></span><label{" hidden" * rc}><input type="checkbox" id="only" '
           f'{"disabled" if rc else "checked"}> Sample only</label><button type="button" id="next">Next unmarked'
           '</button><button type="button" id="export">Export my marks</button></div><div id="warn" hidden>This '
           'browser is not saving your marks (private window or blocked storage). Export before you close the page.'
           '</div>')
    page = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; '
            'script-src \'unsafe-inline\'"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>OOD-H {"re-check" if rc else "review"}</title><style>{CSS}</style></head><body data-sha="'
            f'{E(sha)}" data-file="{E(name)}">{bar}<main>{head}{"".join(parts)}</main><script>'
            f'{JS.replace("@KEY@", key + sha[:16]).replace("@EXPORT@", export)}</script></body></html>\n')
    return page, dict(threads=len(shown), probes=sum(len(r["probes"]) for r in shown), sample=len(insample),
                      sample_probes=sum(len(r["probes"]) for r in insample), words=words, minutes=mins(words, checks),
                      changed=page.count('<p class="chg">'), gone=page.count('<p class="chg gone">'))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sealed", default=B.SEALED)
    ap.add_argument("--hashes", default=os.path.join(HERE, "HASHES.txt"))
    ap.add_argument("--out", help="default <sealed>/REVIEW.html, with --only REVIEW_RECHECK.html; under a sealed/ dir")
    ap.add_argument("--only", help="comma list of thread ids (oodh-017,oodh-019): the re-check page")
    ap.add_argument("--changes", help="the third review's decisions file: which questions it changed")
    a = ap.parse_args(argv)
    only = None if a.only is None else set(filter(None, map(str.strip, a.only.split(","))))
    out = a.out or os.path.join(a.sealed, MODES[only is not None][0])
    if "sealed" not in os.path.abspath(out).split(os.sep):
        return print(f"refusing --out outside a sealed/ directory: {out}", file=sys.stderr) or 2
    try:
        name, sha, recs = load_checked(a.sealed, a.hashes)
    except Refused as e:
        return print(f"REFUSED: {e}", file=sys.stderr) or 3
    if only is not None and (not only or only - {r["id"] for r in recs}):
        return print(f"refusing --only: {len(only - {r['id'] for r in recs})} ids not in {name}", file=sys.stderr) or 2
    changes = None if a.changes is None else parse_changes(open(a.changes, encoding="utf-8").read())
    page, st = render(recs, name, sha, only, changes)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"sha256 check PASS: {name} {sha[:16]} and its tree-id list match oodh/HASHES.txt\n"
          f"threads {st['threads']}; probes {st['probes']}; sample {st['sample']} threads, {st['sample_probes']} "
          f"probes, {st['words']:,} words, about {st['minutes']} minutes\npage {out}: {len(page.encode()):,} bytes"
          + ("" if changes is None else f"\nchange lines on {st['changed']} probes, {st['gone']} removed questions"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
