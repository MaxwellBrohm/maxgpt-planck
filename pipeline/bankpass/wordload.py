"""Word list loader (BANKPASS s4, s5; the W0 loaders): the controlled word list TSV (wordfam.write) -> families, the
RS / RM / RL lists and the forms table, with provenance. FAKE (fake_data WORD_NOUNS / VERBS / ADJS) is the fallback:
with no path and PLANCK_WORDLIST unset, load() returns it, marked FAKE.

A TSV is used only when its header names the input manifest and both code hashes, its columns are the ones wordfam
writes, and, when the caller passes one, its sha256 equals the expected hash (the MANIFEST.sha256 line written with
it on the PC; manifest_sha reads it). Anything else raises Refused.

Required words need teacher POS labels (s4: a proxy POS without 2 of 3 teachers agreeing is "amb", never a required
word), so a list whose rows say pos_status pending offers none: required() is empty and the req_* pools stay FAKE.
What it offers now: head() (s5's family matcher: a word is a headword only through the forms table; an unlisted word
matches as itself), in_list(), features() (article, number and mass from core v0 counts, the s3 P replacement of
pools.features' vowel rule; an unknown first word falls back to the vowel rule, marked), seed_nouns() (prompt-side
seeds for pool and topic calls, s3 P / T: corpus words, so a proxy noun is enough for a nudge) and english() (every
eligible word; install_english adds them to pools._ENGLISHISH, the nonce filter's list, with an undo)."""
import os
import re

import fake_data as F
from bankpass import store

ENV = "PLANCK_WORDLIST"
NEED = ("headword", "rank", "lists", "pos_proxy", "pos_status", "forms", "tf", "article", "a", "an", "mass_ctx",
        "flags", "required_ok")
USABLE_POS = ("confirmed", "fake")          # pos_status values whose POS may pick a required word
_PROV = re.compile(r"input manifest sha256 ([0-9a-f]{64}); wordlist\.py sha256 ([0-9a-f]{64}); "
                   r"wordfam\.py sha256 ([0-9a-f]{64})")


class Refused(RuntimeError):
    pass


class WordList:
    def __init__(self, families, provenance, ref, meta):
        self.families, self.provenance, self.ref, self.meta = families, provenance, ref, meta
        self.form_of = {f: h for h, d in families.items() for fs in d["forms"].values() for f in fs}
        self.rule_of = {f: r for d in families.values() for r, fs in d["forms"].items() for f in fs}

    def head(self, word):
        w = word.lower()
        return self.form_of.get(w, w)

    def in_list(self, word, name="RM"):
        d = self.families.get(self.head(word))
        return bool(d) and name in d["lists"]

    def ranked(self, name="RM", pos=None):
        out = [(d["rank"], h) for h, d in self.families.items()
               if name in d["lists"] and (pos is None or d["pos"] == pos)]
        return [h for _, h in sorted(out)]

    def required(self, pos, name="RM"):
        return [h for h in self.ranked(name, pos) if self.families[h]["required_ok"]
                and self.families[h]["pos_status"] in USABLE_POS]

    def seed_nouns(self, name="RM"):
        return [h for h in self.ranked(name, "noun") if self.families[h]["required_ok"]]

    def english(self):
        return set(self.families) | set(self.form_of)

    def features(self, value):
        ws = value.lower().split()
        d = self.families.get(ws[0])
        if d and d["a"] + d["an"] >= 3:
            art, basis = ("an" if d["an"] > d["a"] else "a"), "counted"
        else:
            art, basis = ("an" if ws[0][:1] in "aeiou" else "a"), "vowel_rule"
        plural = self.rule_of.get(ws[-1]) == "s"
        last = self.families.get(ws[-1])
        mass = bool(last) and last["mass"] >= 3 and last["mass"] > last["a"] + last["an"]
        return {"article": "" if plural or mass else art, "number": "pl" if plural else "sg", "mass": mass,
                "basis": basis}


def _forms(cell):
    out = {}
    for part in filter(None, cell.split(";")):
        rule, fs = part.split(":", 1)
        out[rule] = fs.split("|")
    return out


def read(path, expect_sha256=None):
    sha = store.sha256_file(path)
    if expect_sha256 and sha != expect_sha256:
        raise Refused(f"{path}: sha256 {sha[:12]} is not the expected {expect_sha256[:12]}")
    meta, cols, fams = None, None, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                m = _PROV.search(line)
                if m:
                    meta = {"input_manifest_sha256": m.group(1), "wordlist_sha256": m.group(2),
                            "wordfam_sha256": m.group(3)}
                continue
            p = line.rstrip("\n").split("\t")
            if cols is None:
                cols = p
                if meta is None:
                    raise Refused(f"{path}: no provenance header (input manifest and code hashes)")
                if any(c not in cols for c in NEED):
                    raise Refused(f"{path}: columns {sorted(set(NEED) - set(cols))} missing")
                continue
            r = dict(zip(cols, p))
            fams[r["headword"]] = {"pos": r.get("pos") or r["pos_proxy"], "pos_status": r["pos_status"],
                                   "rank": int(r["rank"]) if r["rank"] else None,
                                   "lists": set(filter(None, r["lists"].split(","))), "forms": _forms(r["forms"]),
                                   "flags": set(filter(None, r["flags"].split(","))),
                                   "required_ok": r["required_ok"] == "1", "a": int(r["a"]), "an": int(r["an"]),
                                   "mass": int(r["mass_ctx"]), "tf": int(r["tf"])}
    if cols is None:
        raise Refused(f"{path}: no rows")
    return WordList(fams, "computed", f"wordlist@{sha[:12]}:computed", dict(meta, file_sha256=sha, path=path))


def fake():
    fams = {}
    for pos, words in (("noun", F.WORD_NOUNS), ("verb", F.WORD_VERBS), ("adj", F.WORD_ADJS)):
        for i, w in enumerate(words, 1):
            fams.setdefault(w, {"pos": pos, "pos_status": "fake", "rank": i, "lists": {"RS", "RM", "RL"},
                                "forms": {}, "flags": set(), "required_ok": True, "a": 0, "an": 0, "mass": 0,
                                "tf": 0})
    return WordList(fams, "FAKE", "wordlist@fake:FAKE", {})


def load(path=None, expect_sha256=None):
    p = path or os.environ.get(ENV)
    return read(p, expect_sha256) if p else fake()


def manifest_sha(manifest_path, rel):
    """the sha256 a `sha256sum` style manifest lists for rel (a path relative to the manifest's directory)."""
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            h, _, name = line.strip().partition("  ")
            if name == rel:
                return h
    raise Refused(f"{manifest_path}: no line for {rel}")


def install_english(wl):
    """add the list's words to pools._ENGLISHISH in place (nonce values never collide with an English word);
    returns undo()."""
    import pools as P
    added = wl.english() - P._ENGLISHISH
    P._ENGLISHISH |= added
    return lambda: P._ENGLISHISH.difference_update(added)
