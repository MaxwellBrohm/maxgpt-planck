"""Stack Exchange post HTML -> plain text, and comment markup -> plain text (readers_se_dump.py).

Post bodies in the dumps are rendered HTML. The output matches the layout of the Common Pile
StackExchange text that the starter used: block elements (p, div, headings, pre, list items, table
rows, hr) each start a new line, joined by ONE newline; readers_se_dump.py puts a blank line between
posts. Rules:
- inline text: runs of whitespace become one space, entities are decoded (html.parser does it);
- <pre>: whitespace kept as written (code and verse);
- <br>: a line break;
- lists: '- item' (ul) or '1. item' (ol), one per line, nested lists included;
- <blockquote> (quotes and spoilers): its lines prefixed '> ';
- tables: one row per line, cells joined by ' | ';
- links keep their text and lose the URL; images are dropped (alt text too); script/style dropped.
Comments are stored as Markdown-lite: '[text](url)' becomes 'text', **bold**, __bold__, *italic* and
`code` lose their marks (a spaced '2 * 3' stays), whitespace collapses.
"""
import re
from html.parser import HTMLParser

_WS = re.compile(r"\s+")
_BLANKS = re.compile(r"\n{3,}")
_MD_LINK = re.compile(r"\[([^\[\]\n]{1,500})\]\((?:https?://|/)[^)\s]{0,1000}\)")
_MD_MARKS = [(re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*"), r"\1"),         # **bold**
             (re.compile(r"__(?=\S)(.+?)(?<=\S)__"), r"\1"),                 # __bold__
             (re.compile(r"(?<![\w*])\*(?=[^\s*])([^*\n]+?)(?<=\S)\*(?![\w*])"), r"\1"),  # *it*
             (re.compile(r"`([^`\n]+)`"), r"\1")]                            # `code`
BLOCK = frozenset("p div h1 h2 h3 h4 h5 h6 hr dl dt dd table thead tbody tr figure "
                  "figcaption section article details summary".split())
SKIP = frozenset({"script", "style"})


class _Buf:
    def __init__(self):
        self.parts, self.bol = [], True       # bol: at the beginning of a line


class _Html2Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.bufs, self.lists, self.pre, self.skip = [_Buf()], [], 0, 0
        self.prefix, self.cells = "", 0

    def _put(self, s):
        b = self.bufs[-1]
        if b.bol:
            s = s.lstrip(" ")
            if not s:
                return
            s, self.prefix = self.prefix + s, ""
        elif s.startswith(" ") and b.parts and b.parts[-1].endswith(" "):
            s = s[1:]
            if not s:
                return
        b.parts.append(s)
        b.bol = s.endswith("\n")

    def _brk(self):
        b = self.bufs[-1]
        if not b.bol:
            b.parts.append("\n")
            b.bol = True

    def handle_starttag(self, tag, attrs):
        if tag in SKIP:
            self.skip += 1
        elif tag == "br":
            b = self.bufs[-1]
            b.parts.append("\n")
            b.bol = True
        elif tag == "blockquote":
            self._brk()
            self.bufs.append(_Buf())
        elif tag in ("ul", "ol"):
            self._brk()
            self.lists.append([tag, 0])
        elif tag == "li":
            self._brk()
            if self.lists:
                self.lists[-1][1] += 1
                kind, n = self.lists[-1]
                indent = "  " * (len(self.lists) - 1)
                self.prefix = indent + (f"{n}. " if kind == "ol" else "- ")
            else:
                self.prefix = "- "
        elif tag == "pre":
            self._brk()
            self.pre += 1
        elif tag == "tr":
            self._brk()
            self.cells = 0
        elif tag in ("td", "th"):
            if self.cells:
                self._put(" | ")
            self.cells += 1
        elif tag in BLOCK:
            self._brk()

    def handle_endtag(self, tag):
        if tag in SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag == "blockquote" and len(self.bufs) > 1:
            inner = _finish(self.bufs.pop())
            self._brk()
            if inner:
                self.bufs[-1].parts.append("\n".join("> " + x if x else ">"
                                                     for x in inner.split("\n")) + "\n")
                self.bufs[-1].bol = True
        elif tag in ("ul", "ol"):
            if self.lists:
                self.lists.pop()
            self._brk()
        elif tag == "pre":
            self.pre = max(0, self.pre - 1)
            self._brk()
        elif tag in BLOCK or tag == "li":
            self._brk()

    def handle_data(self, data):
        if self.skip:
            return
        if self.pre:
            b = self.bufs[-1]
            if b.bol and self.prefix:
                data, self.prefix = self.prefix + data, ""
            b.parts.append(data)
            b.bol = data.endswith("\n")
        else:
            self._put(_WS.sub(" ", data))


def _finish(buf) -> str:
    text = "".join(buf.parts)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return _BLANKS.sub("\n\n", text).strip("\n")


def html_to_text(html: str) -> str:
    """A post body (HTML) -> plain text."""
    if not html:
        return ""
    p = _Html2Text()
    p.feed(html)
    p.close()
    while len(p.bufs) > 1:                    # an unclosed blockquote: keep its text
        p.handle_endtag("blockquote")
    return _finish(p.bufs[0])


def comment_to_text(text: str) -> str:
    """A comment's stored text (Markdown-lite) -> plain text: links keep their text; bold, italic
    and code marks are removed; whitespace collapses."""
    t = _MD_LINK.sub(r"\1", text or "")
    for rx, rep in _MD_MARKS:
        t = rx.sub(rep, t)
    return _WS.sub(" ", t).strip()
