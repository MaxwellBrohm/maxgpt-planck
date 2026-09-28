"""Decoding controls for the served teachers (D3, 2026-09-27): pure Python, no vLLM import, so the Mac tests, the
driver side and serve_http share one definition. DRY prep: nothing here is training data.

  label_regex(lines, sep)   structured output (vLLM StructuredOutputsParams(regex=...), xgrammar backend): the planned
                            label lines in order, one line each, then END (SPEC 5f). lines = [[label, literal or None]];
                            a literal (an exact or tool line, --structured labels_exact) is written as fixed text.
                            sep joins the lines: "\\n", or "\\n+" for Ministral, which puts a blank line between lines
                            (probe 2026-09-27, CPU: the "\\n" form accepts 0 of 69 of its own well-formed outputs, "\\n+"
                            61 of 69; Qwen and Gemma pass 61/62 and 58/58 either way; on the GPU Ministral under "\\n"
                            ran on with " END. END." in 5 of 30).
  dash_ban_ids(vocab)       the dash-token ban (applied as logit_bias -100): every token whose bytes hold U+2012 to
                            U+2015, U+2212, "--" or " - ", plus every token ending in " -" (it makes " - " before a
                            token that starts with a space). The probe's set: 140 / 185 / 149 ids for Ministral / Qwen
                            / Gemma, under the V2 runner's 1024-id logit_bias cap; it took Ministral from 26 to 28 of
                            30 outputs with a dash to 0 of 30 at 1.00 to 1.03x throughput. Byte-fallback fragments of a
                            dash are NOT banned (their first bytes are shared with curly quotes); the checker's DASH
                            still rejects whatever gets through.
Presets (D4) live in serve.TEACHERS next to the teacher they belong to."""
import hashlib
import re

STRUCTURED = ("off", "labels", "labels_exact")
SEPS = ("\\n", "\\n+")                       # regex text: one newline, or one or more
LABEL_OK = re.compile(r"^[UAT][1-9][0-9]*$")
LINE_ANY = "[^\\n]+"
META = set("\\.^$|?*+()[]{}")               # regex metacharacters; nothing else is escaped (xgrammar's parser)

DASH_RULE = "dash-v1"
DASH_CHARS = "\u2012\u2013\u2014\u2015\u2212"
DASH_BYTES = tuple(c.encode("utf-8") for c in DASH_CHARS) + (b"--", b" - ")
LOGIT_BIAS_CAP = 1024                         # vLLM 0.30 V2 model runner: logit_bias ids per request (probe)
BAN_BIAS = -100.0


def esc(text):
    """a literal line as regex text: metacharacters backslashed, every other character as itself."""
    return "".join("\\" + c if c in META else c for c in text)


def label_regex(lines, sep="\\n"):
    """[[label, literal or None], ...] -> the regex of the whole output: 'U1: <line>' + sep + ... + sep + 'END'."""
    if sep not in SEPS:
        raise ValueError(f"line separator {sep!r} is not one of {SEPS}")
    if not lines:
        raise ValueError("no planned lines")
    parts, seen = [], set()
    for item in lines:
        lab, lit = item
        if not isinstance(lab, str) or not LABEL_OK.match(lab) or lab in seen:
            raise ValueError(f"bad or repeated label {lab!r}")
        seen.add(lab)
        if lit is None:
            parts.append(f"{lab}: {LINE_ANY}")
        elif isinstance(lit, str) and lit.strip() and "\n" not in lit and "\r" not in lit:
            parts.append(f"{lab}: {esc(lit)}")
        else:
            raise ValueError(f"bad literal for {lab}: {lit!r}")
    return sep.join(parts + ["END"])


def constraint_ok(c):
    """a request's constraint: {"mode": "labels" | "labels_exact", "lines": [[label, literal or None], ...]}."""
    if not isinstance(c, dict) or c.get("mode") not in STRUCTURED[1:] or not isinstance(c.get("lines"), list):
        return False
    lines = c["lines"]
    if not all(isinstance(x, list) and len(x) == 2 for x in lines):
        return False
    if c["mode"] == "labels" and any(x[1] is not None for x in lines):
        return False
    return True


def dash_ban_ids(vocab):
    """vocab = [bytes] per token id -> (sorted banned ids, {set: count})."""
    contains, end_sh = [], []
    for i, b in enumerate(vocab):
        if not b:
            continue
        if any(d in b for d in DASH_BYTES):
            contains.append(i)
        elif b.endswith(b" -"):
            end_sh.append(i)
    return sorted(contains + end_sh), {"contains": len(contains), "end_space_hyphen": len(end_sh)}


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ids_sha(ids):
    return sha(",".join(str(int(i)) for i in ids))


def logit_bias(ids):
    if len(ids) > LOGIT_BIAS_CAP:
        raise ValueError(f"{len(ids)} banned ids exceed vLLM's logit_bias cap of {LOGIT_BIAS_CAP} per request")
    return {int(i): BAN_BIAS for i in ids}
