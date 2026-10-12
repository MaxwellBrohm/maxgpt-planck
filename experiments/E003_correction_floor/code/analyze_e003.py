"""Summarize E003 (CPU only, loads no model): per model the untouched baseline, the LR search on the dev draw,
the scored seeds (pass rule, controls, E002's post-hoc crossed control, old battery families, uprobe, khard,
closed-book knowledge change paired vs the untouched model), the verdict, and the size ladder.
Writes ../results.json and ../logs/tables.txt.   usage: python analyze_e003.py
"""
import glob, json, os, re, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
OUT = os.path.join(EXP, "out")
sys.path.insert(0, HERE)
import metrics_ft as MF
import params as PR

MODELS = PR.E003_MODELS
E002_RESULTS = os.path.join(os.path.dirname(EXP), "E002_ft_test", "results.json")
CONTROL_VARS = ("same_k1", "same_k2", "same_k3", "pos", "keyorig", "keycorr", "neutral", "noupd", "twoslot")


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def slug(m):
    return m.replace("/", "__")


def tags_for(model):
    s = slug(model)
    return sorted({re.match(rf"{re.escape(s)}__(.+?)__run\.json$", os.path.basename(p)).group(1)
                   for p in glob.glob(os.path.join(OUT, f"{s}__*__run.json"))})


def run_sets(model, tag):
    s = slug(model)
    sets = {}
    for p in glob.glob(os.path.join(OUT, f"{s}__{tag}__*.jsonl")):
        name = os.path.basename(p)[len(f"{s}__{tag}__"):-len(".jsonl")]
        sets[name] = load(p)
    return sets


def acc_of(recs, pred):
    return MF.acc([MF.right(r["scores"]) for r in recs if pred(r)])


def cross_summary(cross):
    cr = {}
    for d in (0, 4, 10):
        cr[f"cross_A|d{d}"] = MF.cell(cross, "plain", ("cross_A",), d)
        cr[f"cross_B|d{d}"] = MF.cell(cross, "plain", ("cross_B",), d)
        by = {}
        for r in cross:
            if r["render"] == "plain" and r["d"] == d:
                by.setdefault(r["sid"], []).append(MF.right(r["scores"]))
        cr[f"cross_pair|d{d}"] = MF.acc([len(v) == 2 and all(v) for v in by.values()])
    return cr


def length_split(new, ctx):
    """DIAGNOSTIC (pre-registered for models whose pretraining context is shorter than some items): the three
    pass-rule cells split by whether the scored sequence fits in the pretraining context."""
    out = {}
    for name, vars_ in (("LW10", MF.LW_VARS), ("TS10", ("twoslot",)), ("NU10", ("noupd",))):
        rs = [r for r in new if r["render"] == "plain" and r["var"] in vars_ and r["d"] == 10]
        out[f"{name}|fits<={ctx}"] = MF.acc([MF.right(r["scores"]) for r in rs if r["seq_len"] <= ctx])
        out[f"{name}|over{ctx}"] = MF.acc([MF.right(r["scores"]) for r in rs if r["seq_len"] > ctx])
    fits = [out[f"{n}|fits<={ctx}"] for n in ("LW10", "TS10", "NU10")]
    out["pass_on_fitting_items"] = all(c is not None and c["acc"] >= MF.THRESH for c in fits)
    return out


def summarize_run(sets, meta):
    new = sets.get("new__plain", [])
    extra = sets.get("extra__plain", [])
    s = {"primary_plain": MF.primary(new, "plain")}
    ctl = {}
    for var in CONTROL_VARS:
        for d in (0, 4, 10):
            ctl[f"plain|{var}|d{d}"] = MF.cell(new, "plain", (var,), d)
    for d in (0, 4, 10):
        ctl[f"plain|noupd_incid|d{d}"] = MF.cell(extra, "plain", ("noupd_incid",), d)
    s["controls"] = ctl
    det = MF.detail_new(new) if new else {}
    s["all5"] = {k: v for k, v in det.items() if "PAIR_all5" in k}
    s["margins_d10"] = {k: v.get("margin") for k, v in det.items() if k.endswith("|10") and "|all|" in k and "PAIR" not in k}
    s["picks_d10"] = {k: v.get("picks") for k, v in det.items() if k.endswith("|10") and "|all|" in k and "PAIR" not in k}
    if sets.get("cross__plain"):
        s["cross_posthoc"] = cross_summary(sets["cross__plain"])
    s["old"] = MF.old_summary(sets.get("old__plain", []))
    s["uprobe"] = MF.uprobe_summary(sets.get("uprobe__plain", []))
    s["khard_open"] = acc_of(sets.get("khard__plain", []), lambda r: r["task"] == "K4_open")
    if sets.get("dev__plain"):
        dv = sets["dev__plain"]
        s["dev"] = {"LW10": MF.cell(dv, "plain", MF.LW_VARS, 10), "TS10": MF.cell(dv, "plain", ("twoslot",), 10),
                    "NU10": MF.cell(dv, "plain", ("noupd",), 10)}
    ctx = (meta or {}).get("trained_ctx")
    if ctx and new and max(r["seq_len"] for r in new) > ctx:
        s["length_diagnostic"] = length_split(new, ctx)
    cp = (s.get("cross_posthoc") or {}).get("cross_pair|d10")
    s["pass_and_cross"] = bool(s["primary_plain"]["pass"] and cp is not None and cp["acc"] >= MF.THRESH)
    if meta:
        s["train"] = {k: meta.get(k) for k in ("steps", "lr", "bs", "accum", "eff_batch", "max_len", "grad_ckpt", "train_s",
                                               "s_per_step", "eval_s", "total_s", "peak_mps_driver_gb", "peak_mps_alloc_gb",
                                               "lock_in_step", "loss_selfcheck", "loss_finite")}
        L = meta.get("losses") or []
        s["train"]["loss_first10"] = round(sum(L[:10]) / max(1, len(L[:10])), 4) if L else None
        s["train"]["loss_last50"] = round(sum(L[-50:]) / max(1, len(L[-50:])), 4) if L else None
        s["train"]["stats"] = meta.get("train_stats")
        s["probe"] = meta.get("probe")
        s["checks"] = meta.get("checks")
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


def complete(meta):
    return bool(meta) and not meta.get("fatal") and "kbig/plain" in (meta.get("eval_counts") or {}) \
        and "new/plain" in (meta.get("eval_counts") or {})


def model_block(model):
    tags = tags_for(model)
    M = {"params": PR.count(PR.load_config(model)), "baseline": None, "lr_search": None, "seeds": {}}
    metas = {t: json.load(open(os.path.join(OUT, f"{slug(model)}__{t}__run.json"))) for t in tags}
    base_sets = run_sets(model, "base") if "base" in tags and complete(metas["base"]) else {}
    if base_sets:
        M["baseline"] = summarize_run(base_sets, metas["base"])
    for f in ("", "_final"):
        p = os.path.join(EXP, "logs", f"lr_pick_{slug(model)}{f}.json")
        if os.path.exists(p):
            M["lr_search" if not f else "lr_search_final"] = json.load(open(p))
    pick = M.get("lr_search_final") or M.get("lr_search")
    M["chosen_lr"] = pick.get("chosen") if pick else None
    seeds = sorted([t for t in tags if re.fullmatch(r"s\d+", t)], key=lambda t: int(t[1:]))
    for t in seeds:
        if not complete(metas[t]):
            M["seeds"][t] = {"incomplete": True}
            continue
        sets = run_sets(model, t)
        s = summarize_run(sets, metas[t])
        s["knowledge"] = knowledge(base_sets, sets) if base_sets else None
        M["seeds"][t] = s
    done = [t for t in seeds if not M["seeds"][t].get("incomplete")]
    M["verdict"] = MF.model_verdict([M["seeds"][t]["primary_plain"]["pass"] for t in done])
    # secondary labels (pre-registered in notes.txt; NOT the verdict)
    M["verdict_pass_and_cross"] = MF.model_verdict([M["seeds"][t]["pass_and_cross"] for t in done])
    if any("length_diagnostic" in M["seeds"][t] for t in done):
        M["verdict_fitting_items"] = MF.model_verdict([M["seeds"][t].get("length_diagnostic", {}).get("pass_on_fitting_items", False)
                                                       for t in done])
    M["lock_in_steps"] = {t: M["seeds"][t]["train"]["lock_in_step"] for t in done}
    M["done_seeds"] = done
    return M


def fmt(c):
    return "  -  " if c is None else f"{c['acc']:.2f}"


def mean_cell(M, key):
    xs = [M["seeds"][t]["primary_plain"][key]["acc"] for t in M["done_seeds"] if M["seeds"][t]["primary_plain"][key]]
    return round(sum(xs) / len(xs), 3) if xs else None


def tables(res):
    lines = ["E003 correction floor: size ladder (body = non-embedding parameters; pass rule and seeds as E002)", ""]
    lines.append(f"{'model':30s} {'body':>11s} {'total':>12s} | base LW/TS/NU     | LR     | seeds pass | verdict | "
                 f"mean LW10 TS10 NU10 | cross_pair@10 | kbig dAcc")
    for model in MODELS:
        M = res["models"].get(model)
        if not M:
            continue
        b = M["baseline"]["primary_plain"] if M["baseline"] else None
        bs = f"{fmt(b['LW10'])}/{fmt(b['TS10'])}/{fmt(b['NU10'])}" if b else "  (none)       "
        v = M["verdict"]
        cp = [M["seeds"][t].get("cross_posthoc", {}).get("cross_pair|d10") for t in M["done_seeds"]]
        cp = [c["acc"] for c in cp if c]
        kb = [(M["seeds"][t].get("knowledge") or {}).get("kbig_all") for t in M["done_seeds"]]
        kb = [k["d_acc"] for k in kb if k]
        verdict = ("PASS" if v["pass"] else "FAIL") if v["seeds"] else "  -  "
        lines.append(f"{model:30s} {M['params']['body']:11,d} {M['params']['total']:12,d} | {bs} | {str(M['chosen_lr']):6s} | "
                     f"{v['passing']}/{v['seeds']}        | {verdict:7s} | {str(mean_cell(M, 'LW10')):5s} {str(mean_cell(M, 'TS10')):5s} "
                     f"{str(mean_cell(M, 'NU10')):5s} | {str(round(sum(cp)/len(cp), 3)) if cp else '-':13s} | "
                     f"{('%+.3f' % (sum(kb)/len(kb))) if kb else '-'}")
    if res.get("e002_reference"):
        r = res["e002_reference"]
        lines.append(f"{'(E002) SmolLM2-135M-Instruct':30s} {106203456:11,d} {134515008:12,d} | (see E002)        | 5e-05  | "
                     f"{r['passing']}/{r['seeds']}        | {'PASS' if r['pass'] else 'FAIL':7s} |")
    lines.append("")
    for model in MODELS:
        M = res["models"].get(model)
        if not M:
            continue
        lines.append(f"== {model}  body {M['params']['body']:,}  total {M['params']['total']:,}  (seeds: {', '.join(M['done_seeds']) or 'none'})")
        for key in ("lr_search", "lr_search_final"):
            ls = M.get(key)
            if ls:
                lines.append(f"  {key}: " + "; ".join(f"{r['lr']:.0e} dev LW/TS/NU {r['LW']:.2f}/{r['TS']:.2f}/{r['NU']:.2f} "
                                                     f"(min {r['min']:.2f}) loss {r['loss_first10']}->{r['loss_last50']}"
                                                     for r in ls["rows"])
                             + (f"; ineligible {ls['ineligible']}" if ls["ineligible"] else "") + f" -> {ls['line']}")
        lines.append("  run    | LW10  TS10  NU10  pass | incid10 keyorig10 keycorr10 pos10 neutral10 all5@10 | "
                     "oldU@4-10 owner2@10 twohop2@10 persp2@10 | lockin | kbig dAcc [95% CI]        dMargin [95% CI]")
        rows = [("base", M["baseline"])] + [(t, M["seeds"][t]) for t in M["done_seeds"]]
        for t, s in rows:
            if s is None:
                continue
            p, k, old = s["primary_plain"], s["controls"], s["old"]
            kn = (s.get("knowledge") or {}).get("kbig_all")
            kns = (f"{kn['d_acc']:+.3f} [{kn['d_acc_ci95'][0]:+.3f},{kn['d_acc_ci95'][1]:+.3f}]  {kn['d_margin']:+.3f} "
                   f"[{kn['d_margin_ci95'][0]:+.3f},{kn['d_margin_ci95'][1]:+.3f}]") if kn else "(baseline)"
            a5 = s["all5"].get("plain|PAIR_all5|all|10")
            li = s.get("train", {}).get("lock_in_step") if t != "base" else "-"
            lines.append(f"  {t:6s} | {fmt(p['LW10'])}  {fmt(p['TS10'])}  {fmt(p['NU10'])}  {'PASS' if p['pass'] else 'fail'} | "
                         f"{fmt(k['plain|noupd_incid|d10'])}    {fmt(k['plain|keyorig|d10'])}      {fmt(k['plain|keycorr|d10'])}      "
                         f"{fmt(k['plain|pos|d10'])}  {fmt(k['plain|neutral|d10'])}      {fmt(a5)}    | "
                         f"{fmt(old.get('latest_wins_d4-10'))}      {fmt(old.get('owner_pair_d10'))}      "
                         f"{fmt(old.get('twohop_pair_d10'))}       {fmt(old.get('persp_pair_d10'))}     | {str(li):6s} | {kns}")
        v = M["verdict"]
        lines.append(f"  verdict (plain, pass rule): {v['passing']}/{v['seeds']} seeds pass -> "
                     f"{('PASS' if v['pass'] else 'FAIL') if v['seeds'] else 'no scored seeds'}; lock-in rate {v['lock_in_rate']}")
        vc = M.get("verdict_pass_and_cross") or {}
        if vc.get("seeds"):
            lines.append(f"  secondary: pass rule AND crossed pair@d10 >= 0.8 (per-object tracking): {vc['passing']}/{vc['seeds']} seeds")
        vf = M.get("verdict_fitting_items")
        if vf and vf.get("seeds"):
            lines.append(f"  secondary (DIAGNOSTIC, items within the pretraining context only): {vf['passing']}/{vf['seeds']} seeds")
        for t in M["done_seeds"]:
            kn = M["seeds"][t].get("knowledge") or {}
            lines.append(f"  knowledge {t}: " + " | ".join(
                f"{name} n={d['n']} {d['acc_before']:.3f}->{d['acc_after']:.3f} dAcc {d['d_acc']:+.3f} "
                f"[{d['d_acc_ci95'][0]:+.3f},{d['d_acc_ci95'][1]:+.3f}] dMargin {d['d_margin']:+.3f} "
                f"[{d['d_margin_ci95'][0]:+.3f},{d['d_margin_ci95'][1]:+.3f}]" for name, d in kn.items()))
        for t in M["done_seeds"]:
            pr = M["seeds"][t].get("probe") or []
            lines.append(f"  probe {t}: " + " ".join(f"{x['step']}:{x['LW']:.2f}/{x['TS']:.2f}/{x['NU']:.2f}" for x in pr))
        for t, s in rows:
            if s is None:
                continue
            k = s["controls"]
            lines.append(f"  plain d0/d4/d10 {t}: " + "; ".join(
                f"{var} " + "/".join(fmt(k[f'plain|{var}|d{d}']) for d in (0, 4, 10))
                for var in ("same_k1", "same_k2", "same_k3", "twoslot", "noupd", "noupd_incid", "keyorig", "keycorr", "pos", "neutral")))
            if s.get("cross_posthoc"):
                c = s["cross_posthoc"]
                lines.append(f"  POST-HOC (E002) crossed {t}: " + "; ".join(
                    f"{v} " + "/".join(fmt(c[f'{v}|d{d}']) for d in (0, 4, 10)) for v in ("cross_A", "cross_B", "cross_pair")))
            u = s.get("uprobe") or {}
            if u:
                lines.append(f"  uprobe {t}: " + "; ".join(f"{kk} {fmt(vv)}" for kk, vv in sorted(u.items())) +
                             f"; khard open {fmt(s.get('khard_open'))}")
            if s.get("length_diagnostic"):
                lines.append(f"  DIAGNOSTIC length split {t}: " + "; ".join(f"{kk} {fmt(vv)} (n={vv['n'] if vv else 0})"
                                                                            for kk, vv in s["length_diagnostic"].items()
                                                                            if kk != "pass_on_fitting_items")
                             + f"; pass on fitting items: {s['length_diagnostic']['pass_on_fitting_items']}")
        lines.append("")
    return "\n".join(lines)


def main():
    res = {"models": {}}
    for model in MODELS:
        if tags_for(model) or os.path.exists(os.path.join(EXP, "logs", f"lr_pick_{slug(model)}.json")):
            res["models"][model] = model_block(model)
    if os.path.exists(E002_RESULTS):
        try:
            e2 = json.load(open(E002_RESULTS))["models"]["HuggingFaceTB/SmolLM2-135M-Instruct"]["verdict"]
            res["e002_reference"] = e2
        except Exception:
            pass
    txt = tables(res)
    print(txt)
    json.dump(res, open(os.path.join(EXP, "results.json"), "w"), indent=1)
    os.makedirs(os.path.join(EXP, "logs"), exist_ok=True)
    open(os.path.join(EXP, "logs", "tables.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
