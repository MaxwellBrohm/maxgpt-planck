"""SCREENS.txt STAGE 2 VERDICTS (2026-10-11): the text tables, from the files stage2_report.py wrote. No rule is applied
here; every number is printed from results.json, c6_readings.json or base_replay.json.
  python stage2_tables.py [--dest EXPERIMENTS_DIR]
Writes experiments/S004_canon, S005_forget_gate, S006_mtp_aux, S007_smeared_key/tables.txt and
experiments/screens/c6_tables.txt (the C6 readings of all 9 contrasts, SINK, GATE, the per-module readings) and
experiments/screens/base_tables.txt (BASE-REPLAY's readings and labels, the BASE level over seeds 101 to 105).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import screens_lib as L
from report_lib import fmt, p3

STAGE2 = ("S004", "S005", "S006", "S007")
SENS = {"at_registered_sd": "at the registered SD", "at_pooled_sd": "at the observed pooled SD"}
KEYS = ("CHAT", "PROSE", "cccc", "gutenberg", "wikimedia")


def screen_tables(r: dict) -> str:
    (key, c), = r["contrasts"].items()
    arm = key.split(".")[1]
    o = [f"{r['screen']} STAGE 2 VERDICT TABLES (analyze.py verdicts on the copied seed runs, 2026-10-11; {r['entry']})",
         "A 5M loss reading only (C8): bpb at 5M, 250.0M slots, starter mix, tok_v0_8k. No RC-12 or Tier 0 score.",
         f"Class {r['class']}, k {r['k']}, seeds {' '.join(map(str, r['seeds']))}, pick {r['pick']}.",
         "d = arm - BASE (bpb; negative = arm better).", "", "C7 READINGS",
         "arm          metric     d s101    d s102    dbar      thr      delta    SD_ref   df  p (two-sided)  label"]
    for k in KEYS:
        x = c["readings"][k]
        o.append(f"{arm:12s} {k:10s} {fmt(x['d'][0])}  {fmt(x['d'][1])}  {fmt(x['dbar'])}  {fmt(x['thr'], sign=False)}  "
                 f"{fmt(x['delta'], sign=False)}  {fmt(x['sd_ref'], sign=False)}  {x['df']}   {p3(x['p_two_sided'])}  "
                 f"{x['label']}")
    fam = len(r["holm"]["family"])
    o += ["", f"VERDICT {key}: {c['verdict']}; labels {', '.join(c['labels']) or 'none'}; Holm (family of {fam} so far): "
          f"{c['holm']} (p {p3(c['readings']['CHAT']['p_two_sided'])})"]
    for m, v in r["pooled_sia_check"].items():
        o.append(f"C4 pooled SIA check {m}: SD {100 * v['sd_rel']:.4f}% of BASE vs limit {100 * v['limit_rel']:.4f}% "
                 f"(df {v['df']}, {v['arms']} arms): {'ok' if v['ok'] else 'FAILED'}")
    o += [f"At E3 arm A's mean (SD_ref, thr and delta scaled): {r['verdict_at_e3_arm_a_mean']}",
          f"BASE replay label: {r['base_replay']['label']}; onset: {r['base_replay']['onset_label']}", "",
          "C5 REPORTED, NO RULE (d s101, d s102; F = final checkpoint, M = mean of the 3 scored checkpoints)"]
    for k, x in r["c5_reported"].items():
        o.append(f"{arm:12s} {k:19s} {fmt(x['d'][0])}  {fmt(x['d'][1])}  mean {fmt(x['dbar'])}  BASE "
                 f"{fmt(x['base'][0], sign=False)} {fmt(x['base'][1], sign=False)}")
    o += ["", "CLIP RATE AND SPEED (log.jsonl gnorm over the 1.0 clip; median tok/s from the third record)"]
    for n, x in r["clip"].items():
        o.append(f"{n:24s} clipped {x['clipped']:3d} of {x['logged']}  rate {100 * x['clip_rate']:.2f}%  "
                 f"gnorm max {x['gnorm_max']:.4f}  tok/s {r['tok_s'][n]:,.0f}")
    o += ["", "IND ONSET (AMENDMENT BASE-DIAG ONSET RULE; steps)"]
    on = r["onset"]
    if isinstance(on, str):
        o.append(on)
        if "onset_descriptive" in r:
            x = r["onset_descriptive"]
            o.append(f"descriptive (C6): arm {x['arm_onset_half']}  own BASE {x['base_onset_half']}  d {x['d']}  "
                     f"earlier {x['earlier']}, later {x['later']}, same {x['same']}")
    else:
        s = r["onset_sigma"]
        o.append(f"sigma_seed {s['sigma_seed']:.3f} (BASE onset_half 101..105: {s['base_onset_half']}), df {s['df']}")
        o.append(f"arm {on['arm_onset_half']}  BASE {on['base_onset_half']}  d {on['d']}  dbar {on['dbar']:+.1f}  "
                 f"thr {on['thr']:.1f}  p {p3(on['p_two_sided'])}  {on['label']}  Holm {on['holm']}")
    o += ["", "C6 READINGS (AMENDMENT C6-DIAG READING RULE; E3 PART 4d SDs, absolute units; c6_tables.txt has all 9)"]
    for m, x in r["c6"]["readings"].items():
        ar, ba = (list(x["arm"].values()), list(x["base"].values()))
        tail = (f"thr {x['thr']:.5f}  p {p3(x['p_two_sided'])}  {x['label']}  class {x['class']}  Holm {x['holm']}"
                + (f"  [{', '.join(x['labels'])}]" if x.get("labels") else "") if x.get("reading") == "rule" else
                f"descriptive: higher {x.get('higher')}, lower {x.get('lower')}")
        o.append(f"{m:9s} arm {fmt(ar[0], sign=False)} {fmt(ar[1], sign=False)}  BASE {fmt(ba[0], sign=False)} "
                 f"{fmt(ba[1], sign=False)}  d {fmt(x['d'][0])} {fmt(x['d'][1])}  dbar {fmt(x['dbar'])}  {tail}")
        if "class_registered" in x:
            o.append(f"{'':9s} re-read SD {x['sd_ref']:.5f} replaces the {x['class_registered']} SD "
                     f"{x['sd_ref_registered']:.5f} ({'larger' if x['sd_ref'] > x['sd_ref_registered'] else 'SMALLER'})")
            for f, s in ((f, x["sensitivity"][f]) for f in SENS if f in x.get("sensitivity", {})):
                o.append(f"{'':9s} {SENS[f]}: SD {s['sd_ref']:.5f}, df {s['df']}, thr {s['thr']:.5f}  "
                         f"p {p3(s['p_two_sided'])}  {s['label']}  Holm {s['holm']}  (reported, no rule)")
    ni = r["c6"].get("near_init")
    if ni:
        o.append(f"NEAR ITS INIT ({r['screen']}): {'fires' if ni['fires'] else 'does not fire'}; per seed "
                 f"{json.dumps(ni['per_seed'], sort_keys=True)}; threshold {ni['threshold']}")
    if "s005_flag" in r:
        f = r["s005_flag"]
        o.append(f"S005 DIAGNOSTICS flag (BETTER on CHAT with LOWER HD_asst): {'FLAGGED' if f['flagged'] else 'not flagged'}"
                 f" (CHAT {f['chat_label']}, HD_asst {f['hd_asst_label']})")
    if "mtp_parts" in r:
        o += ["", "S006 OBJECTIVE PARTS (log.jsonl, last 2% of steps; w_t = mtp_weight x the LR factor)"]
        for n, x in r["mtp_parts"].items():
            o.append(f"{n:20s} next-token loss {x['ntp_loss_last2pct']:.5f}  t+2 loss {x['mtp_loss_last2pct']:.5f}  "
                     f"w_t first {x['mtp_w_first']} final {x['mtp_w_final']}")
    return "\n".join(o) + "\n"


def c6_tables(c6: dict) -> str:
    o = ["C6 READINGS, ALL CONTRASTS (SCREENS.txt AMENDMENT C6-DIAG READING RULE, read 2026-10-11 at STAGE 2 VERDICTS)",
         "d = arm - BASE at seeds 101, 102; IND in nats, HD in bpb; SD_ref from E3 PART 4d (results_part4d.json)", ""]
    for m, v in c6["metrics"].items():
        o.append(f"{m}: noise checks {json.dumps(v.get('noise_checks'), sort_keys=True)}")
        for k, x in sorted(v["contrasts"].items()):
            if "label" in x:
                o.append(f"  {k:18s} d {fmt(x['d'][0])} {fmt(x['d'][1])}  dbar {fmt(x['dbar'])}  thr {x['thr']:.5f}  "
                         f"p {p3(x['p_two_sided'])}  {x['label']:10s} {x['class']:9s} Holm {x['holm']}"
                         + (f"  [{', '.join(x['labels'])}]" if x.get("labels") else ""))
            else:
                o.append(f"  {k:18s} d {fmt(x['d'][0])} {fmt(x['d'][1])}  dbar {fmt(x['dbar'])}  descriptive "
                         f"(higher {x['higher']}, lower {x['lower']})")
        s = v.get("sensitivity")
        if s:
            o.append(f"  SENSITIVITY (reported, no rule; added after the audit): re-read {', '.join(s['reread'])}; "
                     f"at a SMALLER SD than registered: {', '.join(s['smaller_than_registered']) or 'none'}")
            for f in ("at_registered_sd", "at_pooled_sd"):
                if f not in s:
                    continue
                o.append(f"  {SENS[f]}:")
                for k, x in s[f]["readings"].items():
                    o.append(f"    {k:18s} SD {x['sd_ref']:.5f} df {x['df']}  thr {x['thr']:.5f}  p {p3(x['p_two_sided'])}  "
                             f"{x['label']:10s} Holm {x['holm']}")
                o.append(f"    Holm over the {len(s[f]['holm'])} holds for: "
                         f"{', '.join(k for k, h in s[f]['holm'].items() if h == 'holds') or 'none'}")
    o += ["", f"S005 flag: {json.dumps(c6['s005_flag'], sort_keys=True)}",
          f"Scorer replay |second - first| (e3_5m_A_s1): {json.dumps(c6['scorer_replay'], sort_keys=True)}", "",
          "CHECKS (1) hd.json oasst2 bpb vs scored F(CHAT), relative; (2) checkpoint and code sha256; (3) |ind - hook|"]
    for n, x in sorted(c6["checks"].items()):
        o.append(f"  {n:26s} {'ok' if x['ok'] else 'FAILED ' + '; '.join(x['why'])}  (1) {x['check1_rel']:.1e}  "
                 f"(3) {x['check3_ind_minus_hook_abs']:.1e}")
    rep = c6["reported"]
    o += ["", "SINK per layer (head mean, query positions >= 64, 200 oasst2 windows) and GATE mean per layer (gate-on runs)"]
    for n, x in sorted(rep["runs"].items()):
        o.append(f"  {n:26s} SINK " + " ".join(f"{v:.4f}" for v in x["sink_layer_mean"]))
        if x.get("gate_mean"):
            o.append(f"  {'':26s} GATE " + " ".join(f"{sum(r) / len(r):.3f}" for r in x["gate_mean"]))
    sp = rep["s001_sink_paired"]
    o.append("S001 SINK gate on minus off per layer, s101 / s102 (seeds with a negative difference):")
    for i, neg in enumerate(sp["seeds_negative_per_layer"]):
        a, b = (sp["gate_on_minus_off_per_seed"][s][i] for s in ("101", "102"))
        o.append(f"  layer {i}: {a:+.5f} / {b:+.5f}  negative at {neg or 'none'}")
    o.append("NEAR ITS INIT: " + json.dumps({k: v.get("fires") for k, v in rep["near_init"].items()}, sort_keys=True))
    o.append("SHARED BASE AT SEEDS 101-105 BESIDE E3 ARM A (reported, no rule):")
    for m, x in c6["base_by_seed"].items():
        e = x["e3_arm_a"]
        o.append(f"  {m:8s} BASE " + " ".join(f"{v:.5f}" for v in x["base"].values())
                 + f"  E3 A {min(e):.5f} to {max(e):.5f} (mean {sum(e) / len(e):.5f})")
    return "\n".join(o) + "\n"


def base_tables(b: dict) -> str:
    o = ["BASE REPLAY AND BASE LEVEL (SCREENS.txt AMENDMENT BASE-REPLAY, AMENDMENT BASE-DIAG; read 2026-10-11)",
         "D = mean(rep1, rep2) - base_s101; H = hookoff - mean(rep1, rep2); t = x / (SD_rep sqrt(3/2)), df 3, bar 3.182"]
    for k, x in b["readings"].items():
        o.append(f"  {k:10s} SD_rep {x['sd_rep']:.5f}  rep1-rep2 {x['rep1_minus_rep2']:+.5f}  D {x['D']['value']:+.5f} "
                 f"(t {x['D']['t']:+.2f}, {x['D']['reads']})  H {x['H']['value']:+.5f} (t {x['H']['t']:+.2f}, "
                 f"{x['H']['reads']})")
    o.append(f"onset_half {json.dumps(b['onset_half'], sort_keys=True)}; D_onset {b['D_onset']}")
    o += [f"  {k}: {v}; onset {b['onset_labels'].get(k, 'not applicable')}" for k, v in sorted(b["labels"].items())]
    lv = b["base_level_101_105"]
    o.append("BASE LEVEL vs E3 Part 1 arm A (8 seeds), t = gap / (E3 sigma_seed sqrt(1/n + 1/8)), p on t(7):")
    for k in ("CHAT", "PROSE"):
        for n in ("n5", "n3"):
            x = lv[k][n]
            o.append(f"  {k:5s} {n} seeds {x['seeds']}: gap {x['gap']:+.5f} ({100 * x['gap_rel']:+.2f}%)  t {x['t']:.2f}  "
                     f"p {p3(x['p_two_sided'])}  above all 8 {x['above_all_8']}")
    o.append(f"  reading: {lv['seed_draw']}")
    return "\n".join(o) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.dirname(L.HERE))
    a = ap.parse_args(argv)
    for sid in STAGE2:
        d = os.path.join(a.dest, L.SCREENS[sid]["dir"])
        open(os.path.join(d, "tables.txt"), "w").write(screen_tables(json.load(open(os.path.join(d, "results.json")))))
    sx = os.path.join(a.dest, "screens")
    open(os.path.join(sx, "c6_tables.txt"), "w").write(c6_tables(json.load(open(os.path.join(sx, "c6_readings.json")))))
    open(os.path.join(sx, "base_tables.txt"), "w").write(base_tables(json.load(open(os.path.join(sx, "base_replay.json")))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
