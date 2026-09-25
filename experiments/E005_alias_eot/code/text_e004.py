"""E004 shared text helpers (no model, no tokenizer): word splitting, value matching, the frame-echo test of
notes.txt DEFINITIONS, and the fixed word lists (determiners, stopwords, object-free verbs, grader cue lists).
Imported by the training generator, its self-test, and later by the eval builders, oracles and grader, so the
definitions stay in one place. Fixed before any E004 item or training example was drawn."""
import re

DETERMINERS = {"the", "a", "an", "this", "that", "these", "those", "my", "your", "our", "his", "her", "their",
               "its", "some"}
# content words = lowercased words minus STOPWORDS, values, markers and OBJECT_FREE_VERBS (notes DEFINITIONS)
STOPWORDS = DETERMINERS | {
    "i", "i'm", "i've", "i'd", "i'll", "me", "you", "you're", "you'll", "we", "we're", "we've", "they", "they're",
    "it", "it's", "it'll", "he", "she", "us", "them", "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "have", "has", "had", "will", "would", "should", "can", "could", "to", "of", "in", "on",
    "at", "for", "with", "from", "by", "up", "out", "off", "over", "into", "until", "and", "or", "but", "so",
    "as", "if", "then", "than", "what", "which", "when", "where", "who", "how", "why", "what's", "when's",
    "where's", "that's", "there", "here", "just", "now", "again", "after", "all", "also", "please", "let's",
    "one", "get", "got", "going", "go", "like", "know", "tell", "remind", "day", "month", "color", "city"}
OBJECT_FREE_VERBS = {
    "move", "moved", "push", "pushed", "change", "changed", "switch", "switched", "shift", "shifted", "bump",
    "bumped", "make", "made", "put", "set", "book", "booked", "rebooked", "reschedule", "rescheduled",
    "postpone", "postponed", "relocate", "relocated", "return", "returned", "swap", "swapped", "pick", "picked",
    "choose", "chose", "happen", "happens", "happening", "hold", "holding", "held", "take", "takes", "took"}
# grader clause lists (notes (c) clauses 5 and 6); the grader imports these
NEGATION_CUES = ["not", "n't", "no longer", "never", "instead", "rather than", "no,"]
HEDGES = ["i think", "maybe", "probably", "might", "perhaps", "not sure", "i don't know", "i believe"]

_WORD = re.compile(r"<o>|<v>|[a-z0-9]+(?:'[a-z]+)?")


def words(text):
    """lowercased words; apostrophe contractions kept whole; <o>/<v> placeholders kept."""
    return _WORD.findall(text.lower())


def value_re(v):
    """whole-word matcher: capitalized values (Monday, May, Paris) match case-sensitively so that the modal
    'may' is not the month; lowercase values (colours, sports) match in any case."""
    flags = 0 if v[:1].isupper() else re.I
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])", flags)


def values_in(text, pool):
    """values of pool that occur in text, in pool order."""
    return [v for v in pool if value_re(v).search(text)]


def count_value(text, v):
    return len(value_re(v).findall(text))


def normalize(text, obj_terms=(), values=()):
    """word list with every object term (full phrase or head noun; longest first) -> <o> and every value -> <v>.
    Template placeholders {o} and {v} map to <o> and <v> too."""
    t = text.replace("{o}", " <o> ").replace("{v}", " <v> ")
    for term in sorted(obj_terms, key=len, reverse=True):
        t = re.sub(r"(?<![A-Za-z])" + re.escape(term) + r"(?![A-Za-z])", " <o> ", t, flags=re.I)
    for v in values:
        t = value_re(v).sub(" <v> ", t)
    return words(t)


def grams(ws, n=3):
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def qualifying(g):
    """a shared run counts as a frame only if it holds >= 2 words that are not <o>/<v>/determiners."""
    return sum(1 for w in g if w not in ("<o>", "<v>") and w not in DETERMINERS) >= 2


def echo_runs(a_words, b_words):
    """3-word runs shared by two normalized word lists that qualify as a frame echo (notes DEFINITIONS)."""
    return sorted(g for g in grams(a_words) & grams(b_words) if qualifying(g))


def content_words(text, extra_exclude=()):
    ex = STOPWORDS | OBJECT_FREE_VERBS | set(w.lower() for w in extra_exclude)
    return [w for w in words(text) if w not in ex and w not in ("<o>", "<v>")]


def has_cue(text, cues):
    """True if any cue occurs in text (lowercased); word cues match whole words, 'n't' and 'no,' as substrings."""
    t = " " + text.lower() + " "
    for c in cues:
        if c in ("n't", "no,"):
            if c in t:
                return True
        elif re.search(r"(?<![a-z'])" + re.escape(c) + r"(?![a-z])", t):
            return True
    return False
