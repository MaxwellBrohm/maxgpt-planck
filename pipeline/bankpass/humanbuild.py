"""W2 build: human pools, persona seeds and the safety rubric from the fetched sources into one bank directory, its
manifest, and the admit result (BANKPASS s3 P, Q, Z; s6 W2). PC CPU, no model.

    python -m bankpass.humanbuild SOURCES_ROOT OUT_DIR VERSION WORDLIST_TSV WORDLIST_MANIFEST [--only bank,bank]

Banks the skeleton installs (manifest "banks"): pool.name, pool.city. Kept in the same directory and checked by admit
but never installed (manifest "aux"): pool.surname (no skeleton consumer yet), seed.persona (prompt side),
rubric.safety (checker rubric). A source that sources.require refuses is skipped and listed under "missing" with the
reason, so a partial set is visible in its own manifest; the banks it would have made stay FAKE in load.py.
Output stays on the PC (item files are private, Max's decision 2 (a)); the manifest holds hashes, counts, author
shares, licences and drop counts only, and is what goes into the repo."""
import json
import os
import sys

import heldout
from bankpass import admit, human, humanseed, sources, store, wordload

CODE = ("sources.py", "human.py", "humanseed.py", "humanbuild.py", "gates.py", "store.py", "admit.py", "wordload.py")
LOADED = {"pool.name", "pool.city"}
PLAN = (  # bank, source, builder (None: humanseed, by bank), target
    ("rubric.safety", "ldnoobw", None, None),
    ("pool.name", "ssa_names", human.build_names, human.NAME_TARGET),
    ("pool.surname", "census_surnames", human.build_surnames, human.SURNAME_TARGET),
    ("pool.city", "geonames_cities", human.build_cities, human.CITY_TARGET),
    ("seed.persona", "nemotron_personas_usa", None, humanseed.PERSONA_TARGET),
)
TALLY = {"pool.name": (("sex",), ("era",), ("band",), ("cell", "band")), "pool.surname": (("band",),),
         "pool.city": (("country",),), "seed.persona": (("age_band",), ("sex",))}


def code_hashes():
    here = os.path.dirname(os.path.abspath(__file__))
    return {f: store.sha256_file(os.path.join(here, f)) for f in CODE}


def source_summary(rec):
    keep = ("title", "license", "basis", "revision", "status", "retrieved", "notes")
    out = {k: rec.get(k) for k in keep if rec.get(k) is not None}
    out["files"] = {r: {k: v[k] for k in ("url", "sha256", "bytes", "retrieved")} for r, v in rec["files"].items()}
    out["evidence"] = {r: {k: v.get(k) for k in ("url", "sha256", "licence_match")} for r, v in rec["evidence"].items()}
    return out


def build(src_root, out_dir, version, wl, only=None):
    """-> (manifest, admit problems). Writes <bank>.jsonl, manifest.json and admit.json into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    ctx, recs, missing, stats, made = human.Ctx(wl), {}, {}, {}, {}
    for bank, src, fn, target in PLAN:
        if only and bank not in only:
            continue
        try:
            rec = recs[src] = sources.require(src_root, src)
        except sources.SourceRefused as e:
            missing[bank] = str(e)
            continue
        rec["_dir"] = os.path.join(src_root, src)
        if bank == "rubric.safety":
            items, st = humanseed.build_safety(rec, ctx.gate)
            ctx.unsafe = human.unsafe_re(humanseed.safety_terms(items))
        elif bank == "seed.persona":
            items, st = humanseed.build_personas(rec, ctx.gate, target)
        else:
            if ctx.unsafe is None:
                missing[bank] = "no LDNOOBW list: SAFETY_LIST cannot run"
                continue
            items, st = fn(rec, ctx, target=target)
        for keys in TALLY.get(bank, ()):
            st["by_" + "_".join(keys)] = human.tally(items, *keys)
        stats[bank], made[bank] = st, items
        store.write_jsonl(os.path.join(out_dir, bank + ".jsonl"), items)
    banks = {b: {"class": made[b][0]["class"] if made[b] else "P", "kind": "human", "target": t}
             for b, _, _, t in PLAN if b in made}
    for b in banks:
        banks[b]["stats"] = stats[b]
    code = code_hashes()
    extra = {"sources": {s: source_summary(r) for s, r in recs.items()}, "missing": missing, "code": code,
             "wordlist": wl.ref if wl else None, "note": "NOT TRAINING DATA until RC-12 clears (DECON_PENDING); "
             "pool.name and pool.city are the only banks load.py installs from this set"}
    man = store.build_manifest(out_dir, version, heldout.gate_hash(), heldout.RC12_STATUS,
                               {b: m for b, m in banks.items() if b in LOADED},
                               code_sha256=store.sha256_text(json.dumps(code, sort_keys=True)),
                               aux={b: m for b, m in banks.items() if b not in LOADED}, extra=extra)
    _, probs = admit.check_dir(out_dir)
    with open(os.path.join(out_dir, "admit.json"), "w", encoding="utf-8") as f:
        json.dump({"gate_hash": heldout.gate_hash(), "ok": admit.ok(probs),
                   "problems": {b: [list(p) for p in ps] for b, ps in probs.items()}}, f, indent=1, sort_keys=True)
    return man, probs


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    src_root, out_dir, version, tsv, wl_manifest = a[:5]
    only = set(a[6].split(",")) if len(a) > 6 and a[5] == "--only" else None
    rel = os.path.relpath(tsv, os.path.dirname(wl_manifest))
    wl = wordload.read(tsv, wordload.manifest_sha(wl_manifest, rel))
    man, probs = build(src_root, out_dir, version, wl, only)
    for part in ("banks", "aux"):
        for b, m in sorted((man.get(part) or {}).items()):
            print(f"{part}:{b} kept {m['kept']} dropped {m['dropped']} drops {m['drops']} sha {m['sha256'][:12]}")
    print("missing:", json.dumps(man["missing"]))
    bad = {b: ps for b, ps in probs.items() if ps}
    print("admit:", "OK" if not bad else json.dumps({b: ps[:5] for b, ps in bad.items()}))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
