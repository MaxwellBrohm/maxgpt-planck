"""E006 analysis (notes.txt PRE-REGISTERED RULES AND READINGS, PER-ARM READINGS, EXTRA REPORTS). CPU only, no model.
Reads E006's out/ and transcripts/ (and, read-only, E005's and E004's records for Q-device) and writes ONE reading:
  ../logs/readings/<YYYYmmdd-HHMMSS>__<trigger>.txt and .json, opened exclusively (an existing name is an error,
  never overwritten), plus a line in ../logs/readings/index.txt; ../logs/tables.txt is rewritten each time and is
  never the record of a reading. The blinded chat sample goes to <stamp>__blind.json and its key to
  <stamp>__blind_key.json (read the sample before opening the key or the Q-chat label).
Sections: per-arm readings (C, P, G; e005w and e004w rescored on CUDA): pass rule, any-seed variant, leave-one-seed-out,
fewest flips, alias reading, stopping; Q-H5; Q-chat and knowledge; cost of P and G vs C; Q-device; training traces.
usage: python -B analyze_e006.py --trigger T [--out-dir D] [--tr-dir D] [--readings-dir D]"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import analyze_e006_b as B
import analyze_e006_c as QC
import analyze_e006_d as QD
import passrule_e006 as PR


def arm_lines(name, rd):
    if rd["label"] == "NOT_SCORED":
        return [f"== {name}: no complete scored seed"]
    L = [f"== {name}: pass rule {rd['label']} ({rd.get('detail')})"]
    if rd["any_seed"] != rd["label"]:
        L.append(f"   any-seed variant: {rd['any_seed']}")
    L.append(f"   leave one seed out: {rd['loo']}")
    L.append(f"   fewest item flips that change the label: {rd['flips'][0]} {rd['flips'][1]}")
    L += [f"   sub-reading: {s}" for s in rd.get("sub", [])]
    a = rd["alias"]
    L.append(f"   alias reading: {a['label']} (HIGH {a['n_high']}, LOW {a['n_low']}); stopping learned: "
             f"{rd['stopping']['learned']} {rd['stopping']['seeds_stop']}")
    for i, c in enumerate(rd["cells"]):
        if c:
            L.append(f"   s{i + 1} LIK " + " ".join(f"{f}:{c['LIK'][f]:.3f}" for f in c["LIK"]) + f" ID:{c['ID_LIK']:.3f}")
            L.append(f"   s{i + 1} GEN " + " ".join(f"{f}:{c['GEN'][f]:.3f}" for f in c["GEN"]) + f" ID:{c['ID_GEN']:.3f}")
        else:
            L.append(f"   s{i + 1} missing or incomplete (counts as failing)")
    return L


def traces(out_dir):
    L = ["Training traces (lock-in step, s/step, G's replay use)"]
    for arm in ("C", "P", "G"):
        for s in B.SEEDS:
            p = os.path.join(out_dir, f"{B.SLUG}__{arm}{s}__run.json")
            if not os.path.exists(p):
                continue
            m = json.load(open(p))
            x = f"   {arm}{s}: lock-in {m.get('lock_in_step')} s/step {m.get('s_per_step')} loss_finite {m.get('loss_finite')} " \
                f"update digest {(m.get('update_digest') or {}).get('sha256', '-')[:16]} n {(m.get('update_digest') or {}).get('n')}"
            if arm == "G":
                pc = os.path.join(out_dir, f"{B.SLUG}__C{s}__run.json")
                cd = (json.load(open(pc)).get("update_digest") or {}) if os.path.exists(pc) else {}
                x += f"; update part == C{s}'s: {bool(cd) and cd == m.get('update_digest')}"
                rs = m.get("replay_stats") or {}
                x += (f"; replay threads {len(m.get('replay_threads') or [])} used, drops {rs.get('dropped')}, cut "
                      f"{rs.get('cut')}, labelled/thread {rs.get('labelled', 0) / max(1, rs.get('used', 1)):.0f}; mean loss "
                      f"update {sum(m.get('losses_update') or [0]) / max(1, len(m.get('losses_update') or [])):.4f} replay "
                      f"{sum(m.get('losses_replay') or [0]) / max(1, len(m.get('losses_replay') or [])):.4f}")
            L.append(x)
    return L


def analyze(out_dir, tr_dir):
    res, L = {}, [f"E006 reading (analyze_e006.py), {time.strftime('%F %T')}", ""]
    for arm, prefix in (("C", "C"), ("P", "P"), ("G", "G"), ("e005w (E005 weights, CUDA)", "e005w"),
                        ("e004w (E004 weights, CUDA)", "e004w")):
        rd = PR.arm_readings(out_dir, [f"{prefix}{s}" for s in B.SEEDS])
        res[f"arm {prefix}"] = {k: v for k, v in rd.items() if k not in ("runs", "cells")}
        L += arm_lines(arm, rd) + [""]
    r, lines = QC.q_h5(out_dir)
    res["Q-H5"], L = r, L + lines + [""]
    r, lines = QD.q_chat(out_dir, tr_dir, EXP, res["arm G"])
    res["Q-chat"], L = {k: v for k, v in r.items() if k != "blind"}, L + lines + [""]
    blind = r["blind"]
    for arm in ("G", "P"):
        c, lines = QC.cost(out_dir, arm)
        res[f"cost {arm}"] = {" ".join(k): v for k, v in c.items()}
        L += [f"Cost {arm} - C (COST: d <= -0.05 and CI high < 0; LOWER: CI high < 0)"] + lines + [""]
    L.append("Companion reading: the pass rule's H5 and control cells on BIG (reported, not ruled)")
    for arm in ("C", "P", "G", "e005w", "e004w"):
        L += QC.companion(out_dir, arm)
    L.append("")
    r, lines = QD.q_device(out_dir, EXP, tr_dir)
    res["Q-device"], L = r, L + lines + [""]
    L += traces(out_dir)
    return res, L, blind


def headline(res):
    q = res["Q-H5"]["label"]
    return (f"C {res['arm C'].get('label')} | P {res['arm P'].get('label')} | G {res['arm G'].get('label')} | Q-H5 "
            f"{q['label']}{' [' + ', '.join(q['flags']) + ']' if q['flags'] else ''} | Q-chat "
            f"{res['Q-chat']['label']['label']} | knowledge {res['Q-chat']['knowledge']['label']}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--trigger", required=True)
    ap.add_argument("--out-dir", default=os.path.join(EXP, "out"))
    ap.add_argument("--tr-dir", default=os.path.join(EXP, "transcripts"))
    ap.add_argument("--readings-dir", default=os.path.join(EXP, "logs", "readings"))
    a = ap.parse_args(argv)
    res, lines, blind = analyze(a.out_dir, a.tr_dir)
    os.makedirs(a.readings_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = os.path.join(a.readings_dir, f"{stamp}__{a.trigger}")
    with open(base + ".txt", "x") as f:
        f.write("\n".join(lines) + "\n")
    with open(base + ".json", "x") as f:
        json.dump(res, f, indent=1, default=str)
    with open(os.path.join(a.readings_dir, f"{stamp}__blind.json"), "x") as f:
        json.dump(blind["items"], f, indent=1)
    with open(os.path.join(a.readings_dir, f"{stamp}__blind_key.json"), "x") as f:
        json.dump(blind["key"], f, indent=1)
    with open(os.path.join(a.readings_dir, "index.txt"), "a") as f:
        f.write(f"{stamp}\t{a.trigger}\t{headline(res)}\n")
    tables = os.path.join(os.path.dirname(a.readings_dir), "tables.txt")
    with open(tables, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"READING {stamp}__{a.trigger}: {headline(res)}")


if __name__ == "__main__":
    main()
