"""Per-bank admit checks (BANKPASS s0, s2h; SPEC 0 "fail closed"). A frozen bank directory is used only when every
bank it lists passes every check below; load.py raises on any problem. Codes:

  MANIFEST      manifest missing, wrong schema, or status not "frozen" (a dry or fixture set needs an explicit flag)
  GATE_STALE    the manifest's or an item's gate hash differs from heldout.gate_hash() (RC-12 or OOD-H landed: re-gate)
  BANK_FILE / BANK_HASH / BANK_COUNT   file missing, sha256 differs from the manifest, kept/dropped counts differ
  ITEM_SCHEMA   bad schema, wrong bank, duplicate id, bad status, or a dropped item without a code
  FAKE_PROVENANCE  a fake-authored item in a non-fixture set
  AUTHOR        an author block missing a s2h field, an unpinned teacher, or a license off the D8 audit table
  AUTHOR_KIND   human pools must be human, computed banks computed, every other bank teacher-written
  AUTHOR_SHARE  a teacher bank of 30+ kept items without all three teachers within 1/3 +- 0.05
  JUDGE         a teacher item kept without two non-author judges both voting keep (judge.vote)
  GATE_HITS     a kept item that carries held-out or item-check hits
  HOLES         a line template that breaks its bank's hole set (specs.hole_problems)
  RECHECK       a kept line that fails the item checks when re-run now
  DUP           two kept items with the same dedup key
  EMPTY         no kept item
  UNLOADABLE    a bank load.py cannot install (specs.loadable): refused, never silently skipped
A manifest's "aux" entries (prompt-side seeds, rubric lists: seed.*, rubric.*, and pools with no skeleton consumer
yet) get every check above except UNLOADABLE; load.py never installs them. A kept pool value (pool.*) and a kept
persona seed (seed.persona) are re-run through their gates too (RECHECK)."""
import json
import os

import heldout
from bankpass import gates, judge, specs, store

SHARE_MIN_N, SHARE_TOL = 30, 0.05
JUDGED_CLASSES = {"K", "M", "O", "C", "S", "R", "L", "U", "Y"}


def expected_kind(bank):
    if bank.startswith(("seed.", "rubric.")):
        return "human"
    if bank.startswith("pool."):
        vt = bank.split(".", 1)[1]
        return "human" if vt in specs.HUMAN_POOLS else "computed" if vt in specs.COMPUTED_POOLS else "teacher"
    return "computed" if bank == "wordlist" else "teacher"


def check_bank(bank_dir, bank, meta, fixture=False):
    path = os.path.join(bank_dir, meta.get("file") or bank + ".jsonl")
    if not os.path.exists(path):
        return [("BANK_FILE", path)]
    out = []
    if store.sha256_file(path) != meta.get("sha256"):
        out.append(("BANK_HASH", bank))
    try:
        recs = store.read_jsonl(path, tolerate_torn_tail=False)
    except ValueError as e:
        return out + [("ITEM_SCHEMA", str(e))]
    kept = [r for r in recs if r.get("status") == "kept"]
    if len(kept) != meta.get("kept") or len(recs) - len(kept) != meta.get("dropped"):
        out.append(("BANK_COUNT", f"{len(kept)} kept, {len(recs) - len(kept)} dropped"))
    ids, keys, line_spec = set(), {}, specs.line_specs().get(bank)
    gh = heldout.gate_hash()
    for r in recs:
        if r.get("schema") != store.SCHEMA or r.get("bank") != bank or r.get("id") in ids \
                or r.get("status") not in store.STATUSES or (r.get("status") == "dropped" and not r.get("drop")):
            out.append(("ITEM_SCHEMA", str(r.get("id"))))
        ids.add(r.get("id"))
    for r in kept:
        a, rid = r.get("author") or {}, r["id"]
        if a.get("kind") == "fake" and not fixture:
            out.append(("FAKE_PROVENANCE", rid))
        if a.get("kind") != "fake":
            out += [("AUTHOR", f"{rid}: {p}") for p in store.author_problems(a)]
        if a.get("kind") not in (expected_kind(bank), "fake"):
            out.append(("AUTHOR_KIND", f"{rid}: {a.get('kind')}"))
        if a.get("kind") == "teacher" and r.get("class") in JUDGED_CLASSES and judge.vote(r):
            out.append(("JUDGE", f"{rid}: {judge.vote(r)}"))
        g = r.get("gates") or {}
        if g.get("gate_hash") != gh:
            out.append(("GATE_STALE", rid))
        if g.get("hits"):
            out.append(("GATE_HITS", rid))
        if line_spec:
            out += [("HOLES", f"{rid}: {p}") for p in specs.hole_problems(bank, r["text"], line_spec)]
            hits = [h for h in gates.item_checks(r["text"], bank, line_spec) if h[0] != "BANK_PROMPT_ECHO"]
            if hits:
                out.append(("RECHECK", f"{rid}: {hits[0][0]}"))
        elif bank.startswith("pool.") or bank == "seed.persona":
            hits = gates.pool_checks(r["text"], bank[5:]) if bank.startswith("pool.") else gates.seed_checks(r["text"])
            out += [("RECHECK", f"{rid}: {hits[0][0]}")] if hits else []
        k = store.norm_key(r["text"])
        if k in keys:
            out.append(("DUP", f"{rid} = {keys[k]}"))
        keys[k] = rid
    if not kept:
        out.append(("EMPTY", bank))
    teach = [r for r in kept if (r.get("author") or {}).get("kind") == "teacher"]
    if len(teach) >= SHARE_MIN_N:
        shares = store.author_shares(teach)
        if set(shares) != set(store.TEACHERS) or any(abs(s - 1 / 3) > SHARE_TOL for s in shares.values()):
            out.append(("AUTHOR_SHARE", json.dumps(shares, sort_keys=True)))
    return out


def check_dir(bank_dir, allow_fixture=False, allow_dry=False):
    """-> (manifest or None, {bank or "*": [(code, detail)]}); empty problem lists mean admitted."""
    mpath = os.path.join(bank_dir, "manifest.json")
    if not os.path.exists(mpath):
        return None, {"*": [("MANIFEST", "no manifest.json")]}
    with open(mpath, encoding="utf-8") as f:
        man = json.load(f)
    top = []
    if man.get("schema") != "bankpass-manifest-v1":
        top.append(("MANIFEST", "schema"))
    if man.get("fixture") and not allow_fixture:
        top.append(("MANIFEST", "a fixture set"))
    if man.get("status") != "frozen" and not (allow_dry and man.get("status") == "dry"):
        top.append(("MANIFEST", f"status {man.get('status')!r}"))
    if man.get("gate_hash") != heldout.gate_hash():
        top.append(("GATE_STALE", "manifest"))
    probs = {"*": top}
    for bank, meta in sorted((man.get("banks") or {}).items()):
        probs[bank] = check_bank(bank_dir, bank, meta, fixture=bool(man.get("fixture")))
        if not specs.loadable(bank):
            probs[bank].append(("UNLOADABLE", bank))
    for bank, meta in sorted((man.get("aux") or {}).items()):
        probs["aux:" + bank] = check_bank(bank_dir, bank, meta, fixture=bool(man.get("fixture")))
    return man, probs


def ok(probs):
    return not any(probs.values())
