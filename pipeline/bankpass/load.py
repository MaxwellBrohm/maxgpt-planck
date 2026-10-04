"""Bank loaders (BANKPASS s6 W0): a frozen bank directory -> the exact shapes the skeleton code reads (banks.BANKS,
banks.MARKERS / ERR_FIX_MARKERS, banks_keys.KEYS, pools.POOLS, topic_words), with per-bank refs
"<bank>@<sha>:<author mix>". FAKE stays the fallback: with no directory (or PLANCK_BANKS unset) every bank is the
FAKE one, and a bank the directory does not hold falls back to FAKE by itself, keeping its ":FAKE" ref, so a partly
real set still fails closed at admit (FAKE_PROVENANCE). A directory is used only after admit.check_dir passes; any
problem raises LoadRefused listing them.

install(bankset) swaps the loaded content into the running modules IN PLACE (dicts and lists keep their identity,
because events_*.py hold `from banks_keys import KEYS`) and returns an undo function. assemble.finish still marks
every skeleton fake=True; switching that to the refs is the owner's change, not made here."""
import copy
import os

import banks as B
import banks_keys as BK
import pools as P
import topic_words as TW
from bankpass import admit, specs, store

ENV = "PLANCK_BANKS"


class LoadRefused(RuntimeError):
    def __init__(self, probs):
        self.probs = probs
        flat = [f"{b}: {c} {d}" for b, ps in probs.items() for c, d in ps]
        super().__init__(f"bank directory refused ({len(flat)} problems): " + "; ".join(flat[:8]))


class BankSet:
    def __init__(self):
        self.banks = copy.deepcopy(B.BANKS)
        self.keys = copy.deepcopy(BK.KEYS)
        self.markers, self.err_markers = list(B.MARKERS), list(B.ERR_FIX_MARKERS)
        self.pools = {vt: (list(p.values), p.provenance, p.source, p.license) for vt, p in P.POOLS.items()}
        self.topic_words = copy.deepcopy(TW.TOPIC_WORDS)
        self.refs = {b: f"{b}@fake:FAKE" for b in specs.line_specs()}
        self.p_exact, self.version, self.manifest = {}, "FAKE", None

    @property
    def fake_banks(self):
        return sorted(b for b, r in self.refs.items() if r.endswith(":FAKE"))

    @property
    def provenance(self):
        return "FAKE" if self.fake_banks else "BANKSET"


def fake():
    return BankSet()


def _kept(bank_dir, meta):
    return [r for r in store.read_jsonl(os.path.join(bank_dir, meta["file"]), tolerate_torn_tail=False)
            if r["status"] == "kept"]


def from_dir(bank_dir, allow_fixture=False, allow_dry=False):
    man, probs = admit.check_dir(bank_dir, allow_fixture, allow_dry)
    if not admit.ok(probs):
        raise LoadRefused(probs)
    bs = BankSet()
    bs.version, bs.manifest = man["version"], man
    for bank, meta in sorted(man["banks"].items()):
        recs = sorted(_kept(bank_dir, meta), key=lambda r: r["id"])
        texts = [r["text"] for r in recs]
        if bank in B.BANKS:
            bs.banks[bank] = texts
        elif bank == "marker.fix":
            bs.markers = [t.rstrip() + " " for t in texts]
        elif bank == "marker.err":
            bs.err_markers = [t.rstrip() + " " for t in texts]
        elif bank.startswith("key."):
            _, key, role = bank.split(".")
            if role in ("query", "corr"):
                bs.keys[key][role] = [(r["features"]["form"], r["text"]) for r in recs]
            else:
                bs.keys[key][role] = texts
        elif bank.startswith("label."):
            bs.keys[bank.split(".", 1)[1]]["label"] = texts[0]
        elif bank.startswith("pool.") or bank == "topic":
            vt = bank.split(".", 1)[1] if bank != "topic" else "topic"
            src = (recs[0]["author"].get("source") or {}).get("name") if recs else None
            lic = sorted({r["author"].get("license") or (r["author"].get("source") or {}).get("license")
                          for r in recs})
            bs.pools[vt] = (texts, meta["ref"].rsplit(":", 1)[1], src or meta.get("class"), ",".join(map(str, lic)))
        elif bank == "topicwords":
            for r in recs:
                f = r["features"]
                bs.topic_words[f["topic"]] = tuple(" ".join(f[k]) for k in ("nouns", "verbs", "adjs"))
        else:
            raise LoadRefused({bank: [("UNLOADABLE", bank)]})     # admit refuses it first; never skipped
        bs.refs[bank] = meta["ref"]
        if "p_exact" in meta:
            bs.p_exact[bank] = meta["p_exact"]
    return bs


def load(bank_dir=None, **kw):
    """the bank set to use: bank_dir, else $PLANCK_BANKS, else FAKE."""
    d = bank_dir or os.environ.get(ENV)
    return from_dir(d, **kw) if d else fake()


def _swap_dict(d, new):
    old = dict(d)
    d.clear()
    d.update(new)
    return lambda: (d.clear(), d.update(old))


def _swap_list(lst, new):
    old = list(lst)
    lst[:] = new
    return lambda: lst.__setitem__(slice(None), old)


def install(bs):
    """put a bank set into the running modules; returns undo(). Pool values pass heldout.pool_ok again here."""
    undo = [_swap_dict(B.BANKS, bs.banks), _swap_dict(BK.KEYS, bs.keys), _swap_list(B.MARKERS, bs.markers),
            _swap_list(B.ERR_FIX_MARKERS, bs.err_markers), _swap_dict(TW.TOPIC_WORDS, bs.topic_words)]
    pools = {}
    for vt, (vals, prov, src, lic) in bs.pools.items():
        keep = [v for v in vals if P.heldout.pool_ok(v)]
        pools[vt] = P.Pool(vt, keep, prov, src, lic, [v for v in vals if v not in keep])
    undo.append(_swap_dict(P.POOLS, pools))
    old_prov, old_tw = B.PROVENANCE, TW.PROVENANCE
    B.PROVENANCE = bs.provenance
    TW.PROVENANCE = "FAKE" if bs.refs.get("topicwords", ":FAKE").endswith(":FAKE") else bs.refs["topicwords"]
    TW.words.cache_clear()

    def restore():
        for u in reversed(undo):
            u()
        B.PROVENANCE, TW.PROVENANCE = old_prov, old_tw
        TW.words.cache_clear()
    return restore
