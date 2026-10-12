"""Date parsing for the core v0 readers (CORPUS 3.1 date gate, ruling Q6: keep documents created
before 2022-12-01 wherever the source records a date).

hygiene.date_status() reads only ISO dates. The core sources write dates in other ways (read from
the real files on the PC, 2026-09-26, range-limited streams of the first file of each source):

| source      | field                  | examples                                                     |
|-------------|------------------------|--------------------------------------------------------------|
| youtube     | published_time (top)   | 2016-10-09T23:39:12Z                                         |
| pressbooks  | created                | 09-30-2024, 04-9-2023 (month first)                          |
| oercommons  | created                | 08/27/2020 (month first); also 'Activity/Lab', 'Neil Greenwood' (a shifted column, no date) |
| pdr         | date (top)             | Mar 26, 2024                                                 |
| news        | created                | 'Published on December 13, 2021', '1st December 2022', 'Oct. 20, 2023', '26 July 2015 at 18:13', 'Sunday, 26 May 2024', '08 Jul' (no year), '4. juna, 2014.' (Bosnian), '' (globalvoices) |
| news        | metadata.url           | https://globalvoices.org/2004/10/26/slug/ (year/month/day in the path) |
| loc         | metadata.year (int)    | 1862                                                         |

A parsed date is a partial ISO string: 'YYYY-MM-DD', 'YYYY-MM' or 'YYYY'. gate() keeps a partial
date only when its LATEST possible day is before the cutoff: '2022-11' is kept, '2022' is not (it
may be December 2022). Day and month are range-checked against the calendar.
"""
import calendar
import re

DATE_CUTOFF = "2022-12-01"
MIN_YEAR, MAX_YEAR = 1000, 2100

MONTHS = {"jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3, "apr": 4,
          "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8,
          "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10, "nov": 11,
          "november": 11, "dec": 12, "december": 12}
_MON = "(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + ")"
_ISO = re.compile(r"^\s*(\d{4})-(\d{1,2})-(\d{1,2})(?![\d])")
_ISO_YM = re.compile(r"^\s*(\d{4})-(\d{1,2})\s*$")
_YEAR = re.compile(r"^\s*(\d{4})\s*$")
_NUM = re.compile(r"^\s*(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\s*$")
_MDY = re.compile(r"\b" + _MON + r"\.?\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s+(\d{4})\b", re.I)
_DMY = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\.?\s+(?:of\s+)?" + _MON + r"\.?\s*,?\s+(\d{4})\b",
                  re.I)
_URL = re.compile(r"/((?:19|20)\d\d)/(0?[1-9]|1[0-2])(?:/(0?[1-9]|[12]\d|3[01]))?/")
_HOST = re.compile(r"^[a-z][a-z0-9+.-]*://([^/?#]+)", re.I)


def _ymd(y, m, d=None):
    """-> 'YYYY-MM-DD' ('YYYY-MM' when d is None), or None when out of range."""
    y, m = int(y), int(m)
    if not (MIN_YEAR <= y <= MAX_YEAR and 1 <= m <= 12):
        return None
    if d is None:
        return f"{y:04d}-{m:02d}"
    d = int(d)
    if not 1 <= d <= calendar.monthrange(y, m)[1]:
        return None
    return f"{y:04d}-{m:02d}-{d:02d}"


def parse_date(value, numeric_order=None):
    """A recorded date value -> partial ISO date, or None when it does not parse.

    numeric_order='mdy' reads 'MM/DD/YYYY' and 'MM-D-YYYY' (pressbooks, oercommons); without it an
    all-numeric date other than ISO does not parse (day-first and month-first are ambiguous).
    An int (or an all-digit string) is a year (LoC metadata.year)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return f"{value:04d}" if MIN_YEAR <= value <= MAX_YEAR else None
    s = str(value).strip()
    if not s:
        return None
    m = _ISO.match(s)
    if m:
        return _ymd(*m.groups())
    m = _ISO_YM.match(s)
    if m:
        return _ymd(*m.groups())
    m = _YEAR.match(s)
    if m:
        return parse_date(int(m.group(1)))
    m = _NUM.match(s)
    if m:
        a, b, y = m.groups()
        return _ymd(y, a, b) if numeric_order == "mdy" else None
    hits = []
    m = _MDY.search(s)
    if m:
        hits.append((m.start(), _ymd(m.group(3), MONTHS[m.group(1).lower()], m.group(2))))
    m = _DMY.search(s)
    if m:
        hits.append((m.start(), _ymd(m.group(3), MONTHS[m.group(2).lower()], m.group(1))))
    return min(hits)[1] if hits else None


def url_host(url) -> str:
    m = _HOST.match(str(url or "").strip())
    h = m.group(1).lower().split("@")[-1].split(":")[0] if m else ""
    return h[4:] if h.startswith("www.") else h


def url_date(url, host_must_contain=None):
    """The /YYYY/MM/ or /YYYY/MM/DD/ date in a URL path -> partial ISO date, or None.
    host_must_contain: only trust URLs on the outlet's own host (milwaukeenns records links to
    other sites, whose path dates are the linked page's, not this document's)."""
    if not url:
        return None
    if host_must_contain and host_must_contain.lower() not in url_host(url):
        return None
    m = _URL.search(str(url))
    if not m:
        return None
    y, mo, d = m.groups()
    return _ymd(y, mo, d) if d else _ymd(y, mo)


def gate(partial, cutoff: str = DATE_CUTOFF) -> str:
    """'ok' when the latest day the partial date can mean is before cutoff, else 'post_cutoff'."""
    if len(partial) == 4:
        latest = partial + "-12-31"
    elif len(partial) == 7:
        latest = partial + "-31"
    else:
        latest = partial
    return "ok" if latest < cutoff else "post_cutoff"


def has_digit(value) -> bool:
    return any(c.isdigit() for c in str(value))


def classify(value, cutoff=DATE_CUTOFF, numeric_order=None, url=None, url_host_hint=None,
             nondate_is_undated=False):
    """-> (status, date, basis). status: 'ok', 'post_cutoff', 'undated' or 'unparseable'.

    Order: the recorded value; if it is missing or does not parse, the URL path date (only when
    url is given, and only on the outlet's host when url_host_hint is given); then 'undated' for
    a missing value, 'unparseable' for a value that is present but does not parse. With
    nondate_is_undated, a value with no digit at all ('Activity/Lab': a shifted column, not a
    date) counts as missing."""
    missing = value is None or (isinstance(value, str) and not value.strip())
    if not missing and nondate_is_undated and not isinstance(value, int) and not has_digit(value):
        missing = True
    p = None if missing else parse_date(value, numeric_order)
    if p:
        return gate(p, cutoff), p, "field"
    if url is not None:
        u = url_date(url, url_host_hint)
        if u:
            return gate(u, cutoff), u, "url"
    return ("undated" if missing else "unparseable"), None, None
