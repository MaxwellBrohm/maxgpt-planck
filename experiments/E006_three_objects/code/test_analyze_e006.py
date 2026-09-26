"""E006 analyzer tests (notes.txt CODE TO WRITE, test_analyze_e006). No model; reads E005's and E004's records
read-only and writes only into a temporary directory.
  E005 through the new code: the pass rule reads PARTIAL-X (H5), leave-one-seed-out keeps it only without s1 or s4,
    the fewest flips are 2 items on s5 C_twoslot, alias DATA GAP (5 HIGH), stopping learned on 5 of 5 (audit Q1, Q2,
    Q4); the chat measures give the audit's first-sentence template shares .58/.57/.50/.59/.54 (untouched .01),
    checks 30/33/33/32/29 (untouched 21) and capped turns 87/25/62/18/59 (untouched 68)
  Q-device plumbing: C = E005's own records gives C - E005 = 0 with a 0..0 interval on every family and set
  Q-H5 plumbing: synthetic BIG/H5L records with known right patterns give RESTORED (P 1.0, C .75, L4 .9, e005w = C,
    H5L equal), then NOT THE CAUSE (P = C), then the NO DROP flag (L4 = C)
  Q-chat plumbing: C = E005's transcripts, G = the untouched model's: PARTLY PROTECTS, failing (c) and (d); knowledge
    C = E005's kbig, G = E004's: NO EFFECT
  readings: a second reading with the same file name is refused (FileExistsError), index.txt gets one line per reading
usage: python -B test_analyze_e006.py   (exit 0 = all pass)"""
import json
import os
import shutil
import sys
import tempfile
sys.dont_write_bytecode = True

import analyze_e006 as AN
import analyze_e006_c as QC
import analyze_e006_d as QD
import chatmeasures_e006 as CM
import items_big_e006 as BG
import passrule_e006 as PR

HERE = os.path.dirname(os.path.abspath(__file__))
EXPS = os.path.dirname(os.path.dirname(HERE))
E5, E4 = os.path.join(EXPS, "E005_alias_eot"), os.path.join(EXPS, "E004_general_updating")
SLUG = "HuggingFaceTB__SmolLM2-135M-Instruct"
FAIL = []


def ok(c, m):
    print(("PASS  " if c else "FAIL  ") + m, flush=True)
    if not c:
        FAIL.append(m)


def link(src_dir, src_tag, dst_dir, dst_tag, names):
    for n in names:
        s = os.path.join(src_dir, f"{SLUG}__{src_tag}__{n}")
        if os.path.exists(s):
            os.symlink(s, os.path.join(dst_dir, f"{SLUG}__{dst_tag}__{n}"))


def synth(out, tag, rule):
    """BIG + H5L LIK/GEN plain records; rule(family, idx) -> right (bool)."""
    D = BG.load()
    for set_name, fams in (("big", BG.BIG_FAMS), ("h5l", ("H5L",))):
        L, G = [], []
        for f in fams:
            for it in D[f]:
                r = rule(f, it["idx"])
                other = next(v for v in it["candidates"] if v != it["gold"])
                L.append({"set": set_name, "render": "plain", "id": len(L), "family": f, "idx": it["idx"],
                          "cand_vals": {"gold": it["gold"], "c1": other},
                          "scores": {"gold": 0.0 if r else -5.0, "c1": -1.0}})
                G.append({"set": set_name, "render": "plain", "id": len(G), "family": f, "idx": it["idx"],
                          "strict": r, "reply": (it["gold"] if r else other) + "."})
        for name, recs in ((set_name, L), ("gen_" + set_name, G)):
            with open(os.path.join(out, f"{SLUG}__{tag}__{name}__plain.jsonl"), "x") as fh:
                fh.write("".join(json.dumps(x) + "\n" for x in recs))


def main():
    rd = PR.arm_readings(os.path.join(E5, "out"), [f"s{s}" for s in range(1, 6)])
    ok(rd["label"] == "PARTIAL-X" and "H5" in rd["detail"], f"E005 pass rule {rd['label']} ({rd['detail']})")
    ok([k for k, v in rd["loo"].items() if v == "PARTIAL-X"] == ["without s1", "without s4"], f"LOO {rd['loo']}")
    ok(rd["flips"] == (2, [(5, "LIK", "C_twoslot")]), f"fewest flips {rd['flips']}")
    ok(rd["alias"]["label"] == "DATA GAP" and rd["alias"]["n_high"] == 5, f"alias {rd['alias']['label']}")
    ok(rd["stopping"]["seeds_stop"] == [True] * 5 and rd["any_seed"] == "PARTIAL-X", "stopping 5/5, any-seed same")
    tr = lambda t: CM.summary(CM.load(os.path.join(E5, "transcripts", f"{SLUG}__{t}__greedy.jsonl")))
    S = [tr(f"s{s}") for s in range(1, 6)]
    b = tr("base")
    ok([round(x["TF"], 2) for x in S] == [0.58, 0.57, 0.50, 0.59, 0.54] and round(b["TF"], 2) == 0.01, "TF shares")
    ok([x["CHECKS"] for x in S] == [30, 33, 33, 32, 29] and b["CHECKS"] == 21, "checks passed")
    ok([x["CAP_n"] for x in S] == [87, 25, 62, 18, 59] and b["CAP_n"] == 68, "capped turns")
    tmp = tempfile.mkdtemp(prefix="e006_an_")
    try:
        out, trd = os.path.join(tmp, "out"), os.path.join(tmp, "transcripts")
        os.makedirs(out)
        os.makedirs(trd)
        names = [f"{a}{x}__{r}.jsonl" for a in ("", "gen_") for x in ("e004",) for r in ("plain", "chat")] + \
            [f"{x}__plain.jsonl" for x in ("kbig", "khard", "new", "extra", "old", "uprobe", "cross")]
        for s in range(1, 6):
            link(os.path.join(E5, "out"), f"s{s}", out, f"C{s}", names)
            link(os.path.join(E4, "out"), f"s{s}", out, f"G{s}", ["kbig__plain.jsonl"])
            for r in ("plain", "chat"):
                os.symlink(os.path.join(E5, "al", "out", f"e005_s{s}__al__{r}.jsonl"),
                           os.path.join(out, f"{SLUG}__C{s}__al__{r}.jsonl"))
            os.symlink(os.path.join(E5, "transcripts", f"{SLUG}__s{s}__greedy.jsonl"),
                       os.path.join(trd, f"{SLUG}__C{s}__greedy.jsonl"))
            os.symlink(os.path.join(E5, "transcripts", f"{SLUG}__base__greedy.jsonl"),
                       os.path.join(trd, f"{SLUG}__G{s}__greedy.jsonl"))
        dev, L = QD.q_device(out, os.path.join(EXPS, "E006_three_objects"), trd)
        zero = [x for x in L if ": +0.000, CI +0.000 to +0.000" in x]
        ok(dev["n_ci"] == 59 and dev["n_clear"] == 0 and len(zero) == 59, f"Q-device C = E005: {dev}, zero lines {len(zero)}")
        rq, L = QD.q_chat(out, trd, os.path.join(EXPS, "E006_three_objects"), {"stopping": {"learned": False}})
        ok(rq["label"]["label"] == "PARTLY PROTECTS" and rq["label"]["failed"] == ["(c) probe checks", "(d) stopping"],
           f"Q-chat {rq['label']}")
        ok(rq["knowledge"]["label"] == "NO EFFECT", f"knowledge {rq['knowledge']['label']} {rq['knowledge']['K']}")
        for s in range(1, 6):
            synth(out, f"C{s}", lambda f, i: i % 4 != 0)
            synth(out, f"P{s}", lambda f, i: True)
            synth(out, f"e004w{s}", lambda f, i: i % 10 != 0)
            synth(out, f"e005w{s}", lambda f, i: i % 4 != 0)
        r, L = QC.q_h5(out)
        ok(r["label"]["label"] == "RESTORED" and r["label"]["suffix"] == "no loss where the gold is last" and
           abs(r["d"]["LIK"][0] - 0.25) < 1e-9 and not r["label"]["flags"], f"Q-H5 synthetic 1: {r['label']} d {r['d']['LIK'][:3]}")
        for s in range(1, 6):
            for n in ("big", "gen_big", "h5l", "gen_h5l"):
                os.remove(os.path.join(out, f"{SLUG}__P{s}__{n}__plain.jsonl"))
                os.remove(os.path.join(out, f"{SLUG}__e004w{s}__{n}__plain.jsonl"))
            synth(out, f"P{s}", lambda f, i: i % 4 != 0)
            synth(out, f"e004w{s}", lambda f, i: i % 4 != 0)
        r, L = QC.q_h5(out)
        ok(r["label"]["label"] == "NOT THE CAUSE" and r["label"]["flags"] == ["NO DROP TO RESTORE ON CUDA"],
           f"Q-H5 synthetic 2: {r['label']}")
        rdir = os.path.join(tmp, "readings")
        orig = AN.time.strftime
        AN.time.strftime = lambda fmt, *a: "20990101-000000" if fmt == "%Y%m%d-%H%M%S" else orig(fmt, *a)
        try:
            AN.main(["--trigger", "t", "--out-dir", out, "--tr-dir", trd, "--readings-dir", rdir])
            try:
                AN.main(["--trigger", "t", "--out-dir", out, "--tr-dir", trd, "--readings-dir", rdir])
                ok(False, "a second reading with the same name was written")
            except FileExistsError:
                ok(True, "a second reading with the same name is refused")
        finally:
            AN.time.strftime = orig
        ok(len(open(os.path.join(rdir, "index.txt")).read().splitlines()) == 1, "index.txt: one line per reading")
    finally:
        shutil.rmtree(tmp)
    print("RESULT: " + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
