"""RC-12 STEP 14N tables (rc12/notes.txt): the dev panel and every extra side by side, and the likelihood rows,
formatted from the committed results files (logs/panel_20261011.jsonl, logs/extras14_20261011.jsonl,
logs/lik_rows_20261011.jsonl).
No model, no runs read. test_numbers14.py checks that notes.txt carries these lines verbatim.
  python3 -B tables14.py   (prints every table)"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs")
FILES = dict(panel="panel_20261011.jsonl", extras="extras14_20261011.jsonl", lik="lik_rows_20261011.jsonl")
FAMS = ["RECALL", "CORR", "BIND", "TWOHOP", "PERSIST", "OWN", "TOPIC", "ROLE", "LOOP"]
LIKF = ["RECALL", "CORR", "BIND", "TWOHOP", "OWN", "TOPIC", "ROLE", "LOOKUP", "COMPOSE"]
SHORT = {"Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B", "Falcon-H1-Tiny-90M-Instruct": "Falcon-H1-Tiny-90M",
         "SmolLM2-135M-Instruct": "SmolLM2-135M", "SmolLM2-360M-Instruct": "SmolLM2-360M",
         "Doge-160M-Instruct": "Doge-160M", "gemma-3-270m-it": "gemma-3-270m",
         "Veyra2-Blueberry-5M-Instruct": "Blueberry-5M", "BananaMind-2-Nano-Chat": "BananaMind-2-Nano",
         "Veyra2-Blueberry-10M-Instruct": "Blueberry-10M", "Vertex-0.6-15M-Instruct": "Vertex-15M",
         "Vertex-0.6-15M-Instruct@card": "Vertex-15M card row", "Veyra2-Mango-15M-Instruct": "Mango-15M",
         "BananaMind-2-Mini-Chat": "BananaMind-2-Mini", "Veyra2-Mango-30M-Instruct": "Mango-30M",
         "cRia-LM-75M-Instruct": "cRia-75M", "Falcon-H1-Tiny-90M-Instruct-Curriculum": "F90M-Curriculum",
         "Falcon-H1-Tiny-90M-Instruct-pre-DPO": "F90M-pre-DPO",
         "Falcon-H1-Tiny-90M-Instruct-Curriculum-pre-DPO": "F90M-Curr-pre-DPO", "Falcon-H1-Tiny-90M-Base": "F90M-Base"}


def load(which, logs=LOGS):
    with open(os.path.join(logs, FILES[which])) as f:
        return [json.loads(x) for x in f if x.strip()]


def short(m):
    m = m.split("/")[-1]
    return SHORT.get(m, m)


def f3(x):
    return "  -  " if x is None else f"{x:.3f}"


def dev_rows(render, logs=LOGS):
    """mean-of-3 rows of the panel then the extras (panel order, then the extras' size order), one render."""
    p = [r for r in load("panel", logs) if r["kind"] == "mean3" and r["render"] == render]
    x = [r for r in load("extras", logs) if r["kind"] == "mean3" and r["render"] == render]
    return p + x


def dev_table(render, logs=LOGS):
    head = f"{'model':20s} {'R':>5s} {'S7':>5s} " + " ".join(f"{f[:5]:>5s}" for f in FAMS) + "  LKUP  T0    loop"
    out = [head]
    for r in dev_rows(render, logs):
        out.append(f"{short(r['model']):20s} {r['R']:5.2f} {r['S7']:5.2f} " + " ".join(f3(r['F_' + f]) for f in FAMS)
                   + f"  {f3(r['F_LOOKUP'])} {f3(r['T0'])} {f3(r['loop'])}")
    return out


def extras_table(logs=LOGS):
    out = [f"{'extra':20s} {'R':>5s} {'CI vs Qwen2.5':>15s} met  {'S7 (95% CI)':>18s} PERSIST LOOKUP  P     X"]
    for c in load("extras", logs):
        if c["kind"] != "ci":
            continue
        lk = c["lookup"]
        out.append(f"{short(c['model']):20s} {c['R']:5.2f} {c['ci']['lo']:+6.2f} to {c['ci']['hi']:+5.2f} "
                   f"{'yes' if c['ci_part_met'] else 'no ':3s}  {c['S7']:5.2f} ({c['s7_ci']['lo']:4.2f} to "
                   f"{c['s7_ci']['hi']:4.2f}) {c['s12_clause']['PERSIST']:7.3f} {lk['value']:6.3f} {lk['P']:.3f} "
                   f"{lk['X']:.3f}")
    return out


def lik_table(render, logs=LOGS):
    out = [f"{'model':20s} " + " ".join(f"{k[:6]:>9s}" for k in LIKF) + " pairs"]
    for r in load("lik", logs):
        if r["render"] != render:
            continue
        fam = r["families"]
        out.append(f"{short(r['model']):20s} " + " ".join(
            f"{fam[k]['acc_matched']:.2f}/{fam[k]['chance_matched']:.2f}" for k in LIKF)
            + f" {fam['BIND']['pairs']['acc']:.2f}")
    return out


def lik_vs_qwen(render, logs=LOGS):
    out = [f"{'model':20s} " + " ".join(f"{k[:6]:>9s}" for k in LIKF)]
    for r in load("lik", logs):
        if r["render"] == render and r["vs_qwen"]:
            c = r["vs_qwen"]
            out.append(f"{short(r['model']):20s} " + " ".join(f"{c[k]['diff']:+.2f}({c[k]['n_common']:2d})"
                                                            for k in LIKF))
    return out


def all_tables(logs=LOGS):
    return dict(dev_template=dev_table("template", logs), dev_plain=dev_table("plain", logs),
                extras_ci=extras_table(logs), lik_template=lik_table("template", logs),
                lik_plain=lik_table("plain", logs), lik_vs_qwen_template=lik_vs_qwen("template", logs),
                lik_vs_qwen_plain=lik_vs_qwen("plain", logs))


if __name__ == "__main__":
    for name, lines in all_tables().items():
        print(f"[{name}]")
        print("\n".join(lines))
