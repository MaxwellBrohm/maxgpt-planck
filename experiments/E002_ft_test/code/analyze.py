"""Summarize E002 runs: per-seed pass rule, controls, lock-in, knowledge change (paired vs the
untouched model), and the model verdict. Writes ../results.json and ../logs/tables.txt.
usage: python analyze.py
"""
import glob, json, os, re, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
OUT = os.path.join(EXP, "out")
sys.path.insert(0, HERE)
import metrics_ft as MF

MODELS = ["HuggingFaceTB/SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct"]


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def runs_for(model):
    slug = model.replace("/", "__")
    tags = sorted({re.match(rf"{re.escape(slug)}__(.+?)__run\.json$", os.path.basename(p)).group(1)
                   for p in glob.glob(os.path.join(OUT, f"{slug}__*__run.json"))})
    return slug, [t for t in tags if not t.startswith("dry")]


def run_sets(slug, tag):
    sets = {}
    for p in glob.glob(os.path.join(OUT, f"{slug}__{tag}__*.jsonl")):
        name = os.path.basename(p)[len(f"{slug}__{tag}__"):-len(".jsonl")]
        sets[name] = load(p)
    return sets


def acc_of(recs, pred):
    xs = [MF.right(r["scores"]) for r in recs if pred(r)]
    return MF.acc(xs)


def summarize_run(sets, meta):
    new = sets.get("new__plain", []) + sets.get("new__chat", [])
    extra = sets.get("extra__plain", []) + sets.get("extra__chat", [])
    s = {"primary_plain": MF.primary(new, "plain"), "primary_chat": MF.primary(new, "chat")}
    ctl = {}
    for render in ("plain", "chat"):
        for var in ("same_k1", "same_k2", "same_k3", "pos", "keyorig", "keycorr", "neutral", "noupd", "twoslot"):
            for d in (0, 4, 10):
                ctl[f"{render}|{var}|d{d}"] = MF.cell(new, render, (var,), d)
        for d in (0, 4, 10):
            ctl[f"{render}|noupd_incid|d{d}"] = MF.cell(extra, render, ("noupd_incid",), d)
    s["controls"] = ctl
    det = MF.detail_new(new)
    s["all5"] = {k: v for k, v in det.items() if "PAIR_all5" in k}
    s["margins_d10"] = {k: v.get("margin") for k, v in det.items() if k.endswith("|10") and "|all|" in k and "PAIR" not in k}
    s["picks_d10"] = {k: v.get("picks") for k, v in det.items() if k.endswith("|10") and "|all|" in k and "PAIR" not in k}
    cross = sets.get("cross__plain", []) + sets.get("cross__chat", [])
    if cross:
        cr = {}
        for render in ("plain", "chat"):
            for d in (0, 4, 10):
                cr[f"{render}|cross_A|d{d}"] = MF.cell(cross, render, ("cross_A",), d)
                cr[f"{render}|cross_B|d{d}"] = MF.cell(cross, render, ("cross_B",), d)
                by = {}
                for r in cross:
                    if r["render"] == render and r["d"] == d:
                        by.setdefault(r["sid"], []).append(MF.right(r["scores"]))
                cr[f"{render}|cross_pair|d{d}"] = MF.acc([all(v) and len(v) == 2 for v in by.values()])
        s["cross_posthoc"] = cr
    s["old"] = MF.old_summary(sets.get("old__plain", []))
    s["uprobe"] = MF.uprobe_summary(sets.get("uprobe__plain", []))
    s["khard_open"] = acc_of(sets.get("khard__plain", []), lambda r: r["task"] == "K4_open")
    if meta:
        s["train"] = {k: meta.get(k) for k in ("steps", "lr", "bs", "accum", "eff_batch", "max_len", "grad_ckpt", "train_s",
                                               "eval_s", "total_s", "peak_mps_driver_gb", "peak_mps_alloc_gb", "lock_in_step",
                                               "loss_selfcheck")}
        L = meta.get("losses") or []
        s["train"]["loss_first10"] = round(sum(L[:10]) / max(1, len(L[:10])), 4) if L else None
        s["train"]["loss_last50"] = round(sum(L[-50:]) / max(1, len(L[-50:])), 4) if L else None
        s["train"]["stats"] = meta.get("train_stats")
        s["probe"] = meta.get("probe")
    return s


def knowledge(base_sets, sets):
    out = {}
    kb_b, kb_a = base_sets.get("kbig__plain", []), sets.get("kbig__plain", [])
    if kb_b and kb_a:
        out["kbig_all"] = MF.paired(kb_b, kb_a)
        out["khard40"] = MF.paired([r for r in kb_b if r.get("cat") == "khard40"], [r for r in kb_a if r.get("cat") == "khard40"])
        out["kbig_new"] = MF.paired([r for r in kb_b if r.get("cat") != "khard40"], [r for r in kb_a if r.get("cat") != "khard40"])
    ob, oa = base_sets.get("old__plain", []), sets.get("old__plain", [])
    if ob and oa:
        out["K_closedbook60"] = MF.paired([r for r in ob if r["task"] == "K_closedbook"], [r for r in oa if r["task"] == "K_closedbook"])
    return out


def fmt(c):
    return "  -  " if c is None else f"{c['acc']:.2f}"


def main():
    res = {"models": {}}
    lines = []
    for model in MODELS:
        slug, tags = runs_for(model)
        if not tags:
            continue
        base_sets = run_sets(slug, "base") if "base" in tags else {}
        base_meta = json.load(open(os.path.join(OUT, f"{slug}__base__run.json"))) if "base" in tags else None
        M = {"baseline": summarize_run(base_sets, base_meta) if base_sets else None, "seeds": {}}
        seeds = sorted([t for t in tags if re.fullmatch(r"s\d+", t)], key=lambda t: int(t[1:]))
        for t in seeds:
            sets = run_sets(slug, t)
            meta = json.load(open(os.path.join(OUT, f"{slug}__{t}__run.json")))
            if not meta.get("eval_counts") or "kbig/plain" not in meta["eval_counts"]:
                M["seeds"][t] = {"incomplete": True}
                continue
            s = summarize_run(sets, meta)
            s["knowledge"] = knowledge(base_sets, sets) if base_sets else None
            M["seeds"][t] = s
        done = [t for t in seeds if not M["seeds"][t].get("incomplete")]
        M["verdict"] = MF.model_verdict([M["seeds"][t]["primary_plain"]["pass"] for t in done])
        M["verdict_chat_render"] = MF.model_verdict([M["seeds"][t]["primary_chat"]["pass"] for t in done])
        M["lock_in_steps"] = {t: M["seeds"][t]["train"]["lock_in_step"] for t in done}
        res["models"][model] = M
        # ---- tables
        lines.append(f"== {model}  (seeds: {', '.join(done)})")
        lines.append("run    | LW10  TS10  NU10  pass | chat: LW10 TS10 NU10 | incid10 keyorig10 keycorr10 pos10 neutral10 all5@10 | "
                     "oldU@4-10 owner2@10 twohop2@10 persp2@10 | lockin | kbig dAcc [95% CI]        dMargin [95% CI]")
        rows = [("base", M["baseline"])] + [(t, M["seeds"][t]) for t in done]
        for t, s in rows:
            if s is None:
                continue
            p, c, k = s["primary_plain"], s["primary_chat"], s["controls"]
            old = s["old"]
            kn = (s.get("knowledge") or {}).get("kbig_all")
            kns = (f"{kn['d_acc']:+.3f} [{kn['d_acc_ci95'][0]:+.3f},{kn['d_acc_ci95'][1]:+.3f}]  {kn['d_margin']:+.3f} "
                   f"[{kn['d_margin_ci95'][0]:+.3f},{kn['d_margin_ci95'][1]:+.3f}]") if kn else "(baseline)"
            a5 = s["all5"].get("plain|PAIR_all5|all|10")
            li = s.get("train", {}).get("lock_in_step") if t != "base" else "-"
            lines.append(f"{t:6s} | {fmt(p['LW10'])}  {fmt(p['TS10'])}  {fmt(p['NU10'])}  {'PASS' if p['pass'] else 'fail'} | "
                         f"{fmt(c['LW10'])} {fmt(c['TS10'])} {fmt(c['NU10'])} | {fmt(k['plain|noupd_incid|d10'])}    "
                         f"{fmt(k['plain|keyorig|d10'])}      {fmt(k['plain|keycorr|d10'])}      {fmt(k['plain|pos|d10'])}  "
                         f"{fmt(k['plain|neutral|d10'])}      {fmt(a5)}    | {fmt(old.get('latest_wins_d4-10'))}      "
                         f"{fmt(old.get('owner_pair_d10'))}      {fmt(old.get('twohop_pair_d10'))}       {fmt(old.get('persp_pair_d10'))}     | "
                         f"{str(li):6s} | {kns}")
        v = M["verdict"]
        lines.append(f"verdict (plain, pass rule): {v['passing']}/{v['seeds']} seeds pass -> {'PASS' if v['pass'] else 'FAIL'}; "
                     f"lock-in rate {v['lock_in_rate']}")
        vc = M["verdict_chat_render"]
        lines.append(f"same rule on the chat-template render: {vc['passing']}/{vc['seeds']} seeds")
        lines.append("")
        # knowledge detail
        for t in done:
            kn = M["seeds"][t].get("knowledge") or {}
            parts = [f"{name} n={d['n']} {d['acc_before']:.3f}->{d['acc_after']:.3f} dAcc {d['d_acc']:+.3f} "
                     f"[{d['d_acc_ci95'][0]:+.3f},{d['d_acc_ci95'][1]:+.3f}] dMargin {d['d_margin']:+.3f} "
                     f"[{d['d_margin_ci95'][0]:+.3f},{d['d_margin_ci95'][1]:+.3f}]"
                     for name, d in kn.items()]
            lines.append(f"  knowledge {t}: " + " | ".join(parts))
        lines.append("")
        # probe trajectories
        for t in done:
            pr = M["seeds"][t].get("probe") or []
            lines.append(f"  probe {t}: " + " ".join(f"{x['step']}:{x['LW']:.2f}/{x['TS']:.2f}/{x['NU']:.2f}" for x in pr))
        lines.append("")
        # per-d controls for each seed
        for t, s in rows:
            if s is None:
                continue
            k = s["controls"]
            cells = []
            for var in ("same_k1", "same_k2", "same_k3", "twoslot", "noupd", "noupd_incid", "keyorig", "keycorr", "pos", "neutral"):
                cells.append(f"{var} " + "/".join(fmt(k[f'plain|{var}|d{d}']) for d in (0, 4, 10)))
            lines.append(f"  plain d0/d4/d10 {t}: " + "; ".join(cells))
            if s.get("cross_posthoc"):
                c = s["cross_posthoc"]
                lines.append(f"  POST-HOC crossed {t}: " + "; ".join(
                    f"{render} {v} " + "/".join(fmt(c[f'{render}|{v}|d{d}']) for d in (0, 4, 10))
                    for render in ("plain", "chat") for v in ("cross_A", "cross_B", "cross_pair")))
        lines.append("")
    txt = "\n".join(lines)
    print(txt)
    json.dump(res, open(os.path.join(EXP, "results.json"), "w"), indent=1)
    os.makedirs(os.path.join(EXP, "logs"), exist_ok=True)
    open(os.path.join(EXP, "logs", "tables.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
