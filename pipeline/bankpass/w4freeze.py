"""W4 freeze (BANKPASS s2f-s2h, s6 W4; PC CPU, no model): the stage P judged view -> a frozen bank directory whose
manifest admit.py passes, and the manifest copy for the repo (counts, shares, drop codes, hashes; no item text).

Banks (installed by load.py): every line bank, the teacher pools, pool.avoid_word (computed), pool.city (human_v0,
copied byte for byte), topic, topicwords, label.<key>, listname.<type>. Aux (admit-checked, never installed):
intent, open.topic_spec, instr.para (kept only when its checklist check passed), label.notewas and the
human_v0 aux files (pool.surname, seed.persona, rubric.safety). Entity attributes and their predicates travel as
features of pool.entity_kind (same author: w3deps asks a kind's own author for both). A bank admit refuses is left out of the manifest
(it stays FAKE when loaded) and listed under "excluded" with its problem codes: fail closed per bank, never silently.

    python -m bankpass.w4freeze --root STAGE --human HUMAN_V0 --wordlist-v0 V0.tsv --wordlist V1.tsv \
        --rc12-dev rc12_dev.jsonl --trees OASST2.trees.jsonl.gz --mined mined_examples_v0.jsonl --out BANKS_V1"""
import argparse
import json
import os
import shutil
import sys

import heldout
from bankpass import admit, specs, store, trim, w3amend, w3deps as WD, w3state as WS, w4banks as WB, w4feats as WF, w4gates

VERSION = "bankpass-v1"
TEACHER_POOLS = ("job", "hobby", "food", "grocery", "chore", "plan", "object", "pet_kind", "pet_name",
                 "assistant_name", "relation", "entity_kind")
LISTNAME_SEED = "bankpass-w4-listname"


def _topic_scoped(items, selected):
    for i in items:
        tid = (i.get("features") or {}).get("topic_id")
        if tid and tid not in selected and i.get("w3") == "kept":
            i.update(w3="dropped", drop_w3="TOPIC_NOT_SELECTED")
    return items


def collect(root, wl0, wl1, g, votes=None):
    """{bank: (records, meta)} for banks and aux, plus stage facts. votes: the W4 re-vote records (w4vote, amendment
    7) that replace the stage's label votes; None keeps the stage's."""
    recs = WS.records(root)
    its = WS.view(root, recs)
    with open(os.path.join(root, "plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    chosen = WD.select_topics(its, recs, wl0, w3amend.base_calls(plan))
    selected = {it["id"] for it, _ in chosen}
    wordset_ids = {r["plan"]["topic_id"] for r in recs if r["kind"] == "wordset"}
    bb, tg = WB.by_bank(its), plan["targets"]
    banks, aux = {}, {}
    for bank in specs.line_specs():
        if bb.get(bank):
            banks[bank] = WB.line_bank(bb[bank], tg.get(bank), g, wl1)
    relf, (attrs, attr_drops) = WF.relation_features(recs), WF.entity_attrs(its, g)
    for vt in TEACHER_POOLS:
        if bb.get("pool." + vt):
            banks["pool." + vt] = WB.pool_bank(bb["pool." + vt], vt, g, wl1, relf, attrs)
    text = {i["id"]: i["text"] for i in bb.get("topic", []) if i["id"] in selected}
    mismatch = {b: WF.topic_text_check(bb.get(b, []), text) for b in ("topicwords.raw", "intent", "open.topic_spec")}
    banks["topic"] = WB.topic_bank(bb.get("topic", []), selected, WF.topic_groups(recs, text), g)
    WF.drop_without_sets(banks["topic"][0], WF.wordset_topics(bb.get("topicwords.raw", [])))
    trim.author_thirds(banks["topic"][0], LISTNAME_SEED)
    alive = {r["id"] for r in banks["topic"][0] if r["status"] == "kept"}       # the W4 gates may drop a topic
    banks["topicwords"] = WB.topicwords_bank(_topic_scoped(bb.get("topicwords.raw", []), alive), alive, wl1, g)
    for r in banks["topicwords"][0]:
        r["bank"], r["id"] = "topicwords", r["id"].replace("topicwords.raw", "topicwords", 1)
    win = WF.vote_winners(votes, "vote2") if votes is not None else WF.vote_winners(recs)
    for bank in sorted(b for b in bb if b.startswith("label.")):
        target = aux if bank == "label.notewas" else banks
        target[bank] = WF.chosen_bank(bb[bank], win.get(bank), "LABEL_NOT_CHOSEN", g)
    for bank in sorted(b for b in bb if b.startswith("listname.")):
        banks[bank] = WF.chosen_bank(bb[bank], WF.first_kept(bb[bank], LISTNAME_SEED), "LISTNAME_NOT_CHOSEN", g)
    checks = WF.para_checks(recs)
    para = [WB.as_record(i) for i in bb.get("instr.para", [])]
    for r in para:
        if r["status"] == "kept" and checks.get(r["id"]) is not True:
            r.update(status="dropped", drop="PARA_CHECK_FAIL" if r["id"] in checks else "PARA_CHECK_MISSING")
    aux["instr.para"] = WB.gate_kept(para, g, value=True), {}
    for bank in ("intent", "open.topic_spec"):
        rs = WB.exact_dedup(WB.gate_kept([WB.as_record(i) for i in _topic_scoped(bb.get(bank, []), alive)], g,
                                         value=True))
        trim.author_thirds(rs, LISTNAME_SEED)
        aux[bank] = rs, {}
    trim.author_thirds(aux["instr.para"][0], LISTNAME_SEED)
    facts = {"plan_sha256": plan["sha256"], "topics_selected": len(selected),
             "selected_without_wordset_calls": len(selected - wordset_ids), "votes": "w4 re-vote (amendment 7)"
             if votes is not None else "stage P",
             "topics_kept": len(alive), "topic_text_mismatch": mismatch, "label_votes": len(win),
             "entity_attr_drops": attr_drops}
    return banks, aux, facts


def write_all(out, banks, aux, human_dir, computed):
    """write every bank file; the human bank set's files are copied byte for byte, as its manifest lists them (banks
    stay banks, aux stays aux; a human_v1 with SSA names brings pool.name in by itself)."""
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(human_dir, "manifest.json"), encoding="utf-8") as f:
        hman = json.load(f)
    meta = {}
    for part in ("banks", "aux"):
        for b, m in sorted((hman.get(part) or {}).items()):
            shutil.copyfile(os.path.join(human_dir, m["file"]), os.path.join(out, b + ".jsonl"))
            meta[(part, b)] = {k: v for k, v in m.items() if k in ("class", "kind", "target")}
    banks = dict(banks, **computed)
    for part, d in (("banks", banks), ("aux", aux)):
        for bank, (rs, m) in d.items():
            store.write_jsonl(os.path.join(out, bank + ".jsonl"), rs)
            cls = rs[0]["class"] if rs else None
            meta[(part, bank)] = dict({k: v for k, v in m.items() if k != "generic_families"}, **{
                "class": cls, "kind": admit.expected_kind(bank)})
            if "generic_families" in m:
                meta[(part, bank)].update(generic_n=len(m["generic_families"]),
                                          generic_sha256=store.sha256_text("\n".join(m["generic_families"])))
    return meta


def freeze(out, meta, extra):
    """manifest, admit, drop refused banks from the manifest, again until admitted."""
    excluded = {}
    while True:
        banks = {b: m for (p, b), m in meta.items() if p == "banks" and b not in excluded}
        aux = {b: m for (p, b), m in meta.items() if p == "aux" and b not in excluded}
        store.build_manifest(out, VERSION, heldout.gate_hash(), heldout.RC12_STATUS, banks, aux=aux,
                             extra=dict(extra, excluded=excluded), code_sha256=code_hashes())
        _, probs = admit.check_dir(out)
        bad = {b.replace("aux:", ""): sorted({c for c, _ in ps}) for b, ps in probs.items() if ps and b != "*"}
        if probs.get("*"):
            raise SystemExit(f"manifest refused: {probs['*']}")
        if not bad:
            return excluded
        for b, codes in bad.items():
            excluded[b] = {"codes": codes, "n": sum(1 for _ in probs.get(b, probs.get("aux:" + b, [])))}


def code_hashes():
    here = os.path.dirname(os.path.abspath(__file__))
    names = ("w4freeze.py", "w4banks.py", "w4gates.py", "w4words.py", "trim.py", "admit.py", "store.py", "load.py")
    return store.sha256_text(json.dumps({n: store.sha256_file(os.path.join(here, n)) for n in names}, sort_keys=True))


def main(argv=None):
    ap = argparse.ArgumentParser()
    for a in ("--root", "--human", "--wordlist-v0", "--wordlist", "--rc12-dev", "--trees", "--mined", "--out"):
        ap.add_argument(a, required=True)
    ap.add_argument("--votes", default=None, help="the w4vote output dir (amendment 7); default: the stage's votes")
    a = ap.parse_args(argv)
    from bankpass import wordload
    WS.install_human(a.human)
    wl0, wl1 = wordload.read(a.wordlist_v0), wordload.read(a.wordlist)
    g = w4gates.Gates.from_paths(a.rc12_dev, a.trees, a.mined, os.path.join(a.human, "rubric.safety.jsonl"))
    from bankpass import w4vote
    banks, aux, facts = collect(a.root, wl0, wl1, g, w4vote.records(a.votes) if a.votes else None)
    author = {"kind": "computed", "code_sha256": store.sha256_file(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "w4banks.py")), "input_sha256": wl1.meta["file_sha256"]}
    computed = {"pool.avoid_word": WB.avoid_bank(wl1, author, g)}
    computed.update({f"pool.req_{p}": WB.req_bank(wl1, p, author, g) for p in ("noun", "verb", "adj")})
    meta = write_all(a.out, banks, aux, a.human, computed)
    extra = {"stage": facts, "w4_gates": g.describe(), "wordlist": wl1.ref, "wordlist_v0": wl0.ref,
             "human_bankset": os.path.basename(os.path.normpath(a.human))}
    excluded = freeze(a.out, meta, extra)
    with open(os.path.join(a.out, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    print(json.dumps({"banks": len(man["banks"]), "aux": len(man.get("aux", {})), "excluded": excluded,
                      "manifest_sha256": store.sha256_file(os.path.join(a.out, "manifest.json"))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
