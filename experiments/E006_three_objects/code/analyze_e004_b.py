"""E004 analysis, part b (notes.txt (e) collateral, continuity, the chat render, length splits); no model.
  format     free-generation replies (e004 GEN + continuity GEN, plain): bare (<= 2 words), >= 4 words, mean words,
             clause-7 (user's voice), cap; FLAG when a seed's bare share is > 10 points above the untouched model
  knowledge  paired with the untouched model on the same items: kbig 441, khard40, K_closedbook60 (old set):
             accuracy change + bootstrap 95% CI (metrics_ft.paired), margin change with a t interval using the
             real t quantile, exact two-sided McNemar p on gained / lost
  continuity LIK accuracy per E002 set (new per variant at d10, uprobe per task at d10, rest pooled), continuity
             GEN strict per set / variant; untouched SmolLM2 vs E002's base: decision flips (must be 0)
  chat       SmolLM2 chat render of the E004 eval draw (reported, not ruled) and the 53-conversation chat probe
             (transcripts/<slug>__<tag>__greedy.jsonl; per category, mean of the run's own hardened checks)
  length     LIK per family split at 512 and 768 tokens (the TinyStories positional caveat; diagnostic only)"""
import math, os
from collections import defaultdict

import metrics_ft as MF
import rules_e004 as RU
import analyze_e004 as A

E002_OUT = os.path.join(os.path.dirname(A.EXP), "E002_ft_test", "out")
CONT_SETS = ("new", "extra", "old", "uprobe", "cross")


def tq975(df):
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except Exception:  # Cornish-Fisher expansion, good to 1e-3 for df >= 3
        z = 1.959964
        return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)


def mcnemar_p(gained, lost):
    n = gained + lost
    if n == 0:
        return 1.0
    k = min(gained, lost)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(before, after):
    try:
        out = MF.paired(before, after, n_boot=2000)
    except (ValueError, ZeroDivisionError) as e:
        return {"error": str(e)}
    dm = []
    b, a = {r["id"]: r for r in before}, {r["id"]: r for r in after}
    for i in sorted(b):
        mb, ma = MF.margin(b[i]["scores"]), MF.margin(a[i]["scores"])
        if mb is not None and ma is not None:
            dm.append(ma - mb)
    if len(dm) > 1:
        m = sum(dm) / len(dm)
        sd = math.sqrt(sum((x - m) ** 2 for x in dm) / (len(dm) - 1))
        half = tq975(len(dm) - 1) * sd / math.sqrt(len(dm))
        out["d_margin_ci95"] = [round(m - half, 4), round(m + half, 4)]
        out["d_margin_ci_method"] = f"t quantile, df={len(dm) - 1}"
    out["mcnemar_p"] = round(mcnemar_p(out["gained"], out["lost"]), 5)
    return out


def recs(out_dir, model, tag, name):
    return A.load(os.path.join(out_dir, f"{A.slug(model)}__{tag}__{name}.jsonl")) or []


def fmt_stats(rs):
    if not rs:
        return None
    n = len(rs)
    return {"n": n, "bare": round(sum(r["bare"] for r in rs) / n, 4), "long": round(sum(r["long"] for r in rs) / n, 4),
            "mean_words": round(sum(r["n_words"] for r in rs) / n, 2), "voice": round(sum(r["voice"] for r in rs) / n, 4),
            "capped": round(sum(r["capped"] for r in rs) / n, 4)}


def knowledge(out_dir, model, tag):
    kb_b, kb_a = recs(out_dir, model, "base", "kbig__plain"), recs(out_dir, model, tag, "kbig__plain")
    ob, oa = recs(out_dir, model, "base", "old__plain"), recs(out_dir, model, tag, "old__plain")
    out = {}
    if kb_b and kb_a:
        out["kbig441"] = paired(kb_b, kb_a)
        out["khard40"] = paired([r for r in kb_b if r.get("cat") == "khard40"], [r for r in kb_a if r.get("cat") == "khard40"])
    if ob and oa:
        out["K_closedbook60"] = paired([r for r in ob if r.get("task") == "K_closedbook"],
                                       [r for r in oa if r.get("task") == "K_closedbook"])
    return out


def continuity(out_dir, model, tag):
    out = {}
    for s in CONT_SETS:
        rs = recs(out_dir, model, tag, f"{s}__plain")
        if not rs:
            continue
        out[s] = A.acc([MF.right(r["scores"]) for r in rs])
        if s in ("new", "uprobe"):
            key = "var" if s == "new" else "task"
            cells = defaultdict(list)
            for r in rs:
                if r.get("d") == 10:
                    cells[f"{s}:{r.get(key)}" + (f":k{r['k']}" if s == "new" and r.get("k") else "")].append(
                        MF.right(r["scores"]))
            out.update({k: A.acc(v) for k, v in sorted(cells.items())})
    g = recs(out_dir, model, tag, "gen_cont__plain")
    cells = defaultdict(list)
    for r in g:
        cells[f"GEN {r['set']}:{r.get('var') or r.get('task')}"].append(bool(r["strict"]))
    out.update({k: A.acc(v) for k, v in sorted(cells.items())})
    return out


def e002_flips(out_dir, model):
    """untouched SmolLM2 continuity: E004 base vs E002 base, same items in the same order -> decision flips."""
    out = {}
    for s in CONT_SETS:
        for render in ("plain", "chat"):
            a = recs(out_dir, model, "base", f"{s}__{render}")
            b = A.load(os.path.join(E002_OUT, f"{A.slug(model)}__base__{s}__{render}.jsonl")) or []
            if a and b:
                ok = len(a) == len(b) and all(x.get("h") == y.get("h") for x, y in zip(a, b))
                out[f"{s}/{render}"] = (sum(MF.right(x["scores"]) != MF.right(y["scores"]) for x, y in zip(a, b))
                                        if ok else f"items differ ({len(a)} vs {len(b)})")
    return out


def length_split(out_dir, model, tag):
    rs = recs(out_dir, model, tag, "e004__plain")
    out = {}
    for f in A.FAMILIES:
        fr = [r for r in rs if r.get("family") == f]
        if fr:
            out[f] = {f"<={c}": [A.acc([MF.right(r["scores"]) for r in fr if r["seq_len"] <= c]),
                                 sum(r["seq_len"] <= c for r in fr)] for c in (512, 768)}
            out[f].update({f">{c}": [A.acc([MF.right(r["scores"]) for r in fr if r["seq_len"] > c]),
                                     sum(r["seq_len"] > c for r in fr)] for c in (512, 768)})
    return out


def chat_probe(model, tag):
    rs = A.load(os.path.join(A.EXP, "transcripts", f"{A.slug(model)}__{tag}__greedy.jsonl"))
    if not rs:
        return None
    cats = defaultdict(list)
    for r in rs:
        for v in (r.get("checks") or {}).values():
            if v is not None:
                cats[r["cat"]].append(float(v))
    return {"n_conv": len(rs), **{c: A.acc(v) for c, v in sorted(cats.items())}}


def collateral(model, n_planned, out_dir):
    tags = ["base"] + [f"s{s}" for s in range(1, n_planned + 1)]
    fm = {t: fmt_stats(recs(out_dir, model, t, "gen_e004__plain") + recs(out_dir, model, t, "gen_cont__plain"))
          for t in tags}
    b = (fm.get("base") or {}).get("bare")
    flags = [t for t in tags[1:] if fm[t] and b is not None and fm[t]["bare"] > b + 0.10]
    out = {"format": fm, "bare_flag": flags,
           "knowledge": {t: knowledge(out_dir, model, t) for t in tags[1:]},
           "continuity": {t: continuity(out_dir, model, t) for t in tags},
           "length_split": {t: length_split(out_dir, model, t) for t in tags},
           "chat_render": {t: A.run_cells(out_dir, model, t, "chat") for t in tags},
           "chat_probe": {t: chat_probe(model, t) for t in tags}}
    if "SmolLM2" in model:
        out["e002_repro_flips"] = e002_flips(out_dir, model)
    return out


def table_collateral(c):
    L = []
    for t, s in c["format"].items():
        if s:
            L.append(f"   format {t}: " + ", ".join(f"{k}={v}" for k, v in s.items()))
    if c["bare_flag"]:
        L.append(f"   FLAG bare-value share > untouched + 10 points: {', '.join(c['bare_flag'])}")
    for t, k in c["knowledge"].items():
        for name, p in k.items():
            if "error" not in p:
                L.append(f"   knowledge {t} {name}: {p['acc_before']:.3f} -> {p['acc_after']:.3f} d={p['d_acc']:+.3f} "
                         f"CI{p['d_acc_ci95']} margin {p['d_margin']:+.3f} CI{p['d_margin_ci95']} McNemar p={p['mcnemar_p']}")
    for t, cc in c["continuity"].items():
        if cc:
            L.append(f"   continuity {t}: " + ", ".join(f"{k}={A.f2(v)}" for k, v in cc.items() if ":" not in k))
    if "e002_repro_flips" in c:
        L.append(f"   untouched SmolLM2 vs E002 base, decision flips: {c['e002_repro_flips']}")
    for t, cells in c["chat_render"].items():
        if cells:
            L.append(f"   chat render {t} (not ruled): LIK " + " ".join(A.f2(cells['LIK'][f]) for f in RU.PASS)
                     + " | GEN " + " ".join(A.f2(cells['GEN'][f]) for f in RU.PASS))
    for t, cp in c["chat_probe"].items():
        if cp:
            L.append(f"   chat probe {t}: " + ", ".join(f"{k}={v}" for k, v in cp.items()))
    for t, ls in c["length_split"].items():
        if ls:
            L.append(f"   LIK by length {t}: " + "; ".join(
                f"{f} {A.f2(v['<=512'][0])}/{A.f2(v['>512'][0])} ({v['>512'][1]}>512, {v['>768'][1]}>768)"
                for f, v in ls.items()))
    return L
