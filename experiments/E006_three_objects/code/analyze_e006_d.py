"""E006 analysis: Q-chat, knowledge and Q-device (notes.txt Q-CHAT, Q-DEVICE). CPU only, no model.
q_chat   chat probe per assistant turn (chatmeasures_e006), G vs C, two-way bootstraps (rules_e006) -> PROTECTS /
         PARTLY PROTECTS / SMALL EFFECT / WORSE / NO EFFECT; knowledge K on the UNEXPOSED kbig items (replay_ref's
         strict exposure list) with the items x seeds bootstrap -> its own label; a blinded sample for reading
q_device C minus E005 (Mac records), every family / AL cell / continuity set / knowledge set / chat measure, as
         sentences with CIs, no label; BIG: C minus e005w; the scoring-device and training-device parts; the
         untouched model on CUDA vs E004's records and transcript; the TF32 twins"""
import hashlib
import json
import os
import random

import analyze_e004 as A
import analyze_e005_b as B5
import analyze_e006_b as B
import chatmeasures_e006 as CM
import passrule_e006 as PR
import rules_e006 as RL
import rules_e004 as RU

BASE_TRANSCRIPT_SHA = "ffa5ee69"


def transcript(tr_dir, tag, slug=B.SLUG):
    p = os.path.join(tr_dir, f"{slug}__{tag}__greedy.jsonl")
    return CM.load(p) if os.path.exists(p) else None


def by_conv(convs):
    out = {}
    for r in CM.rows(convs):
        out.setdefault(r["conv"], []).append(r)
    return out


def chat_side(tr_dir, prefix):
    tr = {s: transcript(tr_dir, f"{prefix}{s}") for s in B.SEEDS}
    tr = {s: c for s, c in tr.items() if c and len(c) == 53}
    rows = {s: by_conv(c) for s, c in tr.items()}
    chk = {s: {k: v[0] for k, v in CM.checks(c).items()} for s, c in tr.items()}
    return tr, rows, chk


def kbig(out_dir, prefix, exclude=()):
    x = B.arm(out_dir, {s: f"{prefix}{s}" for s in B.SEEDS}, "kbig", "all")
    return {s: {i: v for i, v in d.items() if i not in exclude} for s, d in x.items()}


def q_chat(out_dir, tr_dir, exp_dir, g_reading):
    L, res = [], {}
    _, gr, gc = chat_side(tr_dir, "G")
    trc, cr, cc = chat_side(tr_dir, "C")
    full = len(gr) == 5 and len(cr) == 5
    m = {k: RL.chat_boot(gr, cr, k) if full else None for k in ("TF", "TF2", "LOOP")}
    chk = RL.checks_boot(gc, cc) if full else None
    share = lambda rows, k: sum(r[k] for d in rows.values() for rs in d.values() for r in rs) / \
        max(1, sum(len(rs) for d in rows.values() for rs in d.values()))
    stops = bool(g_reading.get("stopping", {}).get("learned"))
    lab = RL.q_chat(m["TF"], m["TF2"], share(cr, "TF"), share(gr, "TF"), share(cr, "TF2"), share(gr, "TF2"), m["LOOP"],
                    chk, stops)
    res.update(label=lab, measures=m, checks=chk)
    L.append(f"Q-chat (G vs C, chat probe): {lab['label']}" + (f" (failed: {', '.join(lab['failed'])})" if lab["failed"] else "")
             + (" [near untouched: TF_G <= 0.10]" if full and share(gr, "TF") <= 0.10 else ""))
    for k in ("TF", "TF2", "LOOP"):
        L.append("   " + RL.sentence(f"G - C {k}", m[k]))
    L.append("   " + RL.sentence("G - C CHECKS (per model, of 73)", chk))
    for tag in ["base"] + [f"{a}{s}" for a in ("C", "G", "e005w") for s in B.SEEDS]:
        c = transcript(tr_dir, tag)
        if c:
            s = CM.summary(c)
            L.append(f"   chat {tag}: TF {s['TF']:.2f} TF2 {s['TF2']:.2f} WHOLE {s['WHOLE']:.2f} LOOP {s['LOOP']:.3f} "
                     f"(old rule {s['LOOP_OLD']:.3f}) CAP {s['CAP_n']} EOS {s['EOS_n']} CHECKS {s['CHECKS']}/{s['checks_total']} "
                     f"same-first {s['same_prev']}/{s['n_prev']} latest-plan {s['latest_plan']:.2f} "
                     f"stopped-turn tokens {s['mean_new_tok_stopped']} sig {s['prompt_sig']}")
    exp = json.load(open(os.path.join(exp_dir, "replay_ref", "exposed_kbig.json")))
    K = {}
    for name, excl in (("unexposed (the reading)", set(exp["strict"])), ("all 441", set()),
                       ("unexposed, loose exposure", set(exp["loose"]))):
        g, c = kbig(out_dir, "G", excl), kbig(out_dir, "C", excl)
        K[name] = RL.paired(g, c) if B.complete(g) and B.complete(c) else None
        L.append("   " + RL.sentence(f"knowledge kbig G - C, {name}", K[name]))
    res["knowledge"] = {"label": RL.knowledge(K["unexposed (the reading)"]), "K": K, "n_exposed": len(exp["strict"])}
    L.insert(1, f"Knowledge (G vs C, kbig441 unexposed, {len(exp['strict'])} exposed items left out): "
                f"{res['knowledge']['label']}")
    import analyze_e004_b as B4
    for tag in [f"{a}{s}" for a in ("C", "G", "P") for s in B.SEEDS]:
        k = B4.knowledge(out_dir, B.MODEL, tag)
        if k:
            L.append(f"   knowledge {tag} vs base: " + "; ".join(
                f"{n} d {v.get('d', v.get('diff', '-'))} gained {v.get('gained')} lost {v.get('lost')} "
                f"McNemar p {B5.p_str(v.get('mcnemar_p'))}" for n, v in k.items() if "error" not in v))
    res["blind"] = blind_sample(tr_dir)
    return res, L


def blind_sample(tr_dir, n=20):
    """20 random turns per arm (C, G) in shuffled order with neutral labels; the key is returned separately."""
    pool = []
    for arm in ("C", "G"):
        turns = [(arm, s, c["id"], i, t["user"], t["assistant"]) for s in B.SEEDS
                 for c in (transcript(tr_dir, f"{arm}{s}") or []) for i, t in enumerate(c["turns"])]
        pool += random.Random(6006 + (arm == "G")).sample(turns, min(n, len(turns)))
    random.Random(6006).shuffle(pool)
    return {"items": [{"k": j, "user": u, "reply": r} for j, (_, _, _, _, u, r) in enumerate(pool)],
            "key": [{"k": j, "arm": a, "seed": s, "conv": c, "turn": i} for j, (a, s, c, i, _, _) in enumerate(pool)]}


def q_device(out_dir, exp_dir, tr_dir=None):
    exps = os.path.dirname(exp_dir)
    e5, e4 = os.path.join(exps, "E005_alias_eot"), os.path.join(exps, "E004_general_updating")
    mac = {s: f"s{s}" for s in B.SEEDS}
    L, n_ci, n_clear = ["Q-device (stated, not ruled): C minus E005 (Mac records) unless named"], 0, 0
    items = [("e004", f, p, r) for f in RU.PASS + ["ID"] for p in ("LIK", "GEN") for r in ("plain", "chat")]
    items += [("al", f"AL{i}", "LIK", r) for i in range(1, 5) for r in ("plain", "chat")]
    items += [(s, "all", "LIK", "plain") for s in ("new", "extra", "old", "uprobe", "cross", "kbig", "khard")]
    for set_name, fam, part, render in items:
        x = B.arm(out_dir, {s: f"C{s}" for s in B.SEEDS}, set_name, fam, render, part)
        if set_name == "al":
            y = B.arm(os.path.join(e5, "al", "out"), {s: f"mac_al:e005_s{s}" for s in B.SEEDS}, "al", fam, render, part)
        else:
            y = B.arm(os.path.join(e5, "out"), mac, set_name, fam, render, part)
        r = RL.paired(x, y) if B.complete(x) and B.complete(y) else None
        n_ci += r is not None
        n_clear += r is not None and (r[1] > 0 or r[2] < 0)
        L.append("   " + RL.sentence(f"{set_name} {fam} {part} {render}", r))
    _, cr, cc = chat_side(tr_dir or os.path.join(exp_dir, "transcripts"), "C")
    _, mr, mc = chat_side(os.path.join(e5, "transcripts"), "s")
    for k in ("TF", "TF2", "LOOP"):
        r = RL.chat_boot(cr, mr, k) if len(cr) == 5 and len(mr) == 5 else None
        n_ci += r is not None
        n_clear += r is not None and (r[1] > 0 or r[2] < 0)
        L.append("   " + RL.sentence(f"chat probe {k}", r))
    r = RL.checks_boot(cc, mc) if len(cc) == 5 and len(mc) == 5 else None
    n_ci += r is not None
    n_clear += r is not None and (r[1] > 0 or r[2] < 0)
    L.append("   " + RL.sentence("chat probe CHECKS (per model, of 73)", r))
    L.append(f"   {n_clear} of {n_ci} intervals exclude 0 (about {0.05 * n_ci:.1f} expected by chance)")
    for fam in ("H5", "C_noupd", "C_twoslot"):
        for part in ("LIK", "GEN"):
            x = B.arm(out_dir, {s: f"C{s}" for s in B.SEEDS}, "big", fam, "plain", part)
            y = B.arm(out_dir, {s: f"e005w{s}" for s in B.SEEDS}, "big", fam, "plain", part)
            L.append("   " + RL.sentence(f"BIG {fam} {part}: C - e005w (training device)",
                                         RL.paired(x, y) if B.complete(x) and B.complete(y) else None))
    for s in B.SEEDS:
        for render in ("plain", "chat"):
            a = A.load(B.rec_path(out_dir, f"e005w{s}", f"e004__{render}")) or []
            b = A.load(B.rec_path(os.path.join(e5, "out"), f"s{s}", f"e004__{render}")) or []
            if a and b:
                fl = B5.flips_by_hash(a, b)
                L.append(f"   scoring device e005w{s} (CUDA) vs E005 s{s} (Mac) e004/{render}: LIK flips {fl['flips']} of "
                         f"{fl['matched']}, max |score diff| {fl['max_abs_score_diff']}")
            ga = A.load(B.rec_path(out_dir, f"e005w{s}", f"gen_e004__{render}")) or []
            gb = A.load(B.rec_path(os.path.join(e5, "out"), f"s{s}", f"gen_e004__{render}")) or []
            if ga and gb and len(ga) == len(gb):
                L.append(f"   scoring device e005w{s} vs E005 s{s} GEN {render}: identical replies "
                         f"{sum(x['reply'] == y['reply'] for x, y in zip(ga, gb))} of {len(ga)}")
    for fl_label, other in (("base (CUDA) vs E004's untouched records", os.path.join(e4, "out")),):
        for k, v in B5.repro_flips(out_dir, B.MODEL, other).items():
            L.append(f"   {fl_label} {k}: flips {v['flips']} of {v['matched']}, max |diff| {v['max_abs_score_diff']}")
    bt = os.path.join(os.path.dirname(out_dir), "transcripts", f"{B.SLUG}__base__greedy.jsonl")
    if os.path.exists(bt):
        L.append(f"   base transcript sha256 {hashlib.sha256(open(bt, 'rb').read()).hexdigest()[:8]} "
                 f"(E004's untouched {BASE_TRANSCRIPT_SHA}; the E006 file adds weights_sha256 and tag to each record)")
    for s in (1, 2):
        for fam in ("H5", "C_noupd", "C_twoslot"):
            x = B.arm(out_dir, {s: f"C{s}t"}, "big", fam)
            y = B.arm(out_dir, {s: f"C{s}"}, "big", fam)
            L.append("   " + RL.sentence(f"numerics twin C{s}t (TF32) - C{s}, BIG {fam} LIK",
                                         RL.paired(x, y) if x and y else None))
    return {"n_ci": n_ci, "n_clear": n_clear}, L
