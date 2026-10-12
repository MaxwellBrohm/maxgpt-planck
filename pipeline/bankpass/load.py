"""Bank loaders (BANKPASS s6 W0): a frozen bank directory -> the exact shapes the skeleton code reads (banks.BANKS,
banks.MARKERS / ERR_FIX_MARKERS, banks_keys.KEYS, pools.POOLS, topic_words), with per-bank refs
"<bank>@<sha>:<author mix>". FAKE stays the fallback: with no directory (or PLANCK_BANKS unset) every bank is the
FAKE one, and a bank the directory does not hold falls back to FAKE by itself, keeping its ":FAKE" ref, so a partly
real set still fails closed at admit (FAKE_PROVENANCE). A directory is used only after admit.check_dir passes; any
problem raises LoadRefused listing them.

install(bankset) swaps the loaded content into the running modules IN PLACE (dicts and lists keep their identity,
because events_*.py hold `from banks_keys import KEYS`) and returns an undo function. W4 adds list names, the lookup
kinds with their attributes and predicates, relation sex, topic required words, counted articles and per-bank p_exact;
assemble.finish marks a skeleton fake when any ref it records is FAKE."""
import copy
import os

import banks as B
import banks_keys as BK
import fake_data as F
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
        self.refs.update({f"label.{k}": f"label.{k}@fake:FAKE" for k in BK.KEYS})    # W4: a FAKE key label counts
        self.p_exact, self.version, self.manifest = {}, "FAKE", None
        self.features, self.aux = {}, {}     # pool value features (sex, era, country ...); aux bank paths (unloaded)
        self.list_names, self.req_words = dict(B.LIST_NAMES), {}     # W4: list names, topic required words
        self.list_refs = dict(B.LIST_REFS)
        self.lookup = None         # W4: (ENTITY_KINDS, ATTR_TYPES, LOOKUP_PRED_ATTR) from pool.entity_kind features
        self.sex = None            # W4: (FEMALE, MALE) from pool.relation features

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
        elif bank.startswith("listname."):
            bs.list_names[bank.split(".", 1)[1]] = texts[0]
            bs.list_refs[bank.split(".", 1)[1]] = meta["ref"]
        elif bank.startswith("pool.") or bank == "topic":
            vt = bank.split(".", 1)[1] if bank != "topic" else "topic"
            src = (recs[0]["author"].get("source") or {}).get("name") if recs else None
            lic = sorted({r["author"].get("license") or (r["author"].get("source") or {}).get("license")
                          for r in recs})
            bs.pools[vt] = (texts, meta["ref"].split(":", 1)[1], src or meta.get("class"), ",".join(map(str, lic)))
            bs.features[vt] = {r["text"]: r.get("features") or {} for r in recs}
            if vt == "relation":
                bs.sex = ({r["text"] for r in recs if r["features"].get("sex") == "F"},
                          {r["text"] for r in recs if r["features"].get("sex") == "M"})
            if vt == "entity_kind":
                at = [a for r in recs for a in r["features"]["attrs"]]
                bs.lookup = ({r["text"]: [a[0] for a in r["features"]["attrs"]] for r in recs},
                             {a[0]: a[1] for a in at}, {a[0]: a[2] for a in at})
        elif bank == "topicwords":
            words, req = merge_topicwords(recs)
            bs.topic_words.update(words)
            bs.req_words.update(req)
        else:
            raise LoadRefused({bank: [("UNLOADABLE", bank)]})     # admit refuses it first; never skipped
        bs.refs[bank] = meta["ref"]
        if "p_exact" in meta:
            bs.p_exact[bank] = meta["p_exact"]
    bs.aux = {b: os.path.join(bank_dir, m["file"]) for b, m in sorted((man.get("aux") or {}).items())}
    return bs


def merge_topicwords(recs):
    """W4 topicwords records (one per topic and author) -> ({topic: (nouns, verbs, adjs) strings, the union in author
    order}, {topic: {part: [required families]}})."""
    merged, req = {}, {}
    for r in recs:
        f = r["features"]
        m = merged.setdefault(f["topic"], ([], [], []))
        for k, part in enumerate(("nouns", "verbs", "adjs")):
            m[k].extend(w for w in f[part] if w not in m[k])
        q = req.setdefault(f["topic"], {"noun": [], "verb": [], "adj": []})
        for part, ws in (f.get("req") or {}).items():
            q[part].extend(w for w in ws if w not in q[part])
    return {t: tuple(" ".join(x) for x in m) for t, m in merged.items()}, req


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


def _swap_set(st, new):
    old = set(st)
    st.clear()
    st.update(new)
    return lambda: (st.clear(), st.update(old))


def install(bs):
    """put a bank set into the running modules; returns undo(). Pool values pass heldout.pool_ok again here."""
    undo = [_swap_dict(B.BANKS, bs.banks), _swap_dict(BK.KEYS, bs.keys), _swap_list(B.MARKERS, bs.markers),
            _swap_list(B.ERR_FIX_MARKERS, bs.err_markers), _swap_dict(TW.TOPIC_WORDS, bs.topic_words)]
    pools = {}
    for vt, (vals, prov, src, lic) in bs.pools.items():
        keep = [v for v in vals if P.heldout.pool_ok(v)]
        pools[vt] = P.Pool(vt, keep, prov, src, lic, [v for v in vals if v not in keep])
    undo.append(_swap_dict(P.POOLS, pools))
    undo += [_swap_dict(B.LIST_NAMES, bs.list_names), _swap_dict(B.LIST_REFS, bs.list_refs),
             _swap_dict(B.P_EXACT_BANK, bs.p_exact), _swap_dict(TW.REQ_WORDS, bs.req_words),
             _swap_dict(P.VALUE_FEATURES, bs.features)]
    if bs.lookup:
        undo += [_swap_dict(F.ENTITY_KINDS, bs.lookup[0]), _swap_dict(F.ATTR_TYPES, bs.lookup[1]),
                 _swap_dict(B.LOOKUP_PRED_ATTR, bs.lookup[2])]
    if bs.sex:
        undo += [_swap_set(BK.FEMALE, bs.sex[0]), _swap_set(BK.MALE, bs.sex[1])]
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
