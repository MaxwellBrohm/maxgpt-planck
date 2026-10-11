"""SCREENS.txt STAGE 2 VERDICTS (2026-10-11): every decimal number written in that entry, in the four RESULT blocks
(S004-S007), in the three C6 READINGS blocks (S001-S003) and in E3's PART 4d RESULT is one the committed outputs give
(results.json, stage2_verdicts.json, c6_readings.json, base_replay.json, results_part4d.json) or a named derivation of
them below, at the precision written; key phrases are pinned to their values. A typo in the prose fails here. No model.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_stage2_verdicts_numbers.py
"""
import os, re, statistics as st, sys  # noqa: E401


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
from s2v_lib import EXP, STAGE2, SX, flat, jload, res, text  # noqa: E402

NUM = re.compile(r"(?<![\w.])[+-]?\d+\.\d+(?:e[+-]\d+)?%?")
ALLOWED = {"250.0", "0.0", "0.7"}   # C1's 250.0M slots; S007's alpha range bounds as written ("between 0.0 and 0.7")


def nums(x):
    if isinstance(x, bool):
        return
    if isinstance(x, (int, float)):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from nums(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from nums(v)


def forms(x):
    for n in range(1, 7):
        yield from (f"{x:.{n}f}", f"{x:+.{n}f}", f"{abs(x):.{n}f}")
    for n in range(1, 5):
        yield from (f"{100 * x:.{n}f}%", f"{100 * x:+.{n}f}%")
    yield from (f"{x:.2g}", f"{x:.3g}")


def derived():
    """Sums, ratios and summaries the prose states, each from the committed outputs."""
    out = []
    for sid in STAGE2:
        r = res(sid)
        for c in r["contrasts"].values():
            for x in c["readings"].values():
                out += [abs(x["dbar"]) + x["thr"], x["dbar"] / x["base_mean"]]
    t4 = res("S004")["tok_s"]
    out.append((t4["base_s101"] + t4["base_s102"]) / (t4["s004_canonac_g1_s101"] + t4["s004_canonac_g1_s102"]))
    c6 = jload(SX, "c6_readings.json")
    for x in c6["reported"]["modules"]["S005.forget"].values():
        hl = [h for layer in x["by_layer"] for h in layer["half_life"]]
        out += [min(hl), max(hl), st.median(hl)]
    for n in ("base_s101", "base_s102"):
        x = c6["reported"]["runs"][n]
        g = [v for layer in x["gate_mean"] for v in layer]
        out += [st.fmean(g), min(g), max(g), st.fmean(x["gate_mean"][0]),
                min(v for layer in x["gate_p10"] for v in layer), max(v for layer in x["gate_p90"] for v in layer)]
    return out


def known():
    srcs = [res(s) for s in STAGE2] + [jload(SX, f) for f in ("stage2_verdicts.json", "c6_readings.json",
                                                             "base_replay.json")]
    srcs.append(jload(EXP, "E3_seed_noise", "results_part4d.json"))
    return {f for v in [v for s in srcs for v in nums(s)] + derived() for f in forms(v)} | ALLOWED


def check(txt, ok):
    txt = re.sub(r"(?<=\d),(?=\d{3})", "", txt)          # 1,652.4 is 1652.4
    bad = sorted({t for t in NUM.findall(txt) if t not in ok})
    assert not bad, bad


def blocks():
    # the entry and E3's block end at the next blank line: later entries (BATCH 1 CLOSED, E3 PART 3 REOPENED) are not theirs
    out = {"entry": text("SCREENS.txt")[text("SCREENS.txt").index("\nSTAGE 2 VERDICTS (2026-10-11"):].split("\n\n", 1)[0]}
    for d in STAGE2.values():
        out[d] = text(f"{d}/notes.txt").split("\nRESULT (2026-10-11", 1)[1]
    for d in ("S001_attn_gate", "S002_block_ablations", "S003_adamw"):
        out[d] = text(f"{d}/notes.txt").split("\nC6 READINGS (2026-10-11", 1)[1]
    out["E3"] = text("E3_seed_noise/notes.txt").split("\nPART 4d RESULT (2026-10-11", 1)[1].split("\n\n", 1)[0]
    return out


def test_every_number_is_computed():
    ok = known()
    for name, t in blocks().items():
        check(t, ok)


def test_key_phrases_carry_their_values():
    b = {k: flat(v) for k, v in blocks().items()}
    v = jload(SX, "stage2_verdicts.json")
    c6, br = jload(SX, "c6_readings.json"), jload(SX, "base_replay.json")
    on = v["onsets"]
    assert f"sigma_seed {on['sigma']['sigma_seed']:.3f} steps (df 4)" in b["entry"]
    for k, x in v["contrasts"].items():
        if k[:4] in STAGE2:
            r = x["readings"]["CHAT"]
            assert f"{k} SIA {x['verdict']} {x['holm']} {r['d'][0]:+.5f} / {r['d'][1]:+.5f} {r['dbar']:+.5f} {r['thr']:.5f}" in b["entry"]
    p = c6["metrics"]["HD_asst"]["noise_checks"]["pooled_sia"]
    assert f"the pooled SIA check failed (SD {p['sd']:.5f} against the limit {p['limit']:.5f}, df 7)" in b["entry"]
    d = br["readings"]["CHAT"]
    assert f"F(CHAT) D {d['D']['value']:+.5f} (t {d['D']['t']:+.2f}), H {d['H']['value']:+.5f} (t {d['H']['t']:+.2f})" in b["entry"]
    lv = br["base_level_101_105"]["CHAT"]
    assert f"F(CHAT) gap {lv['n5']['gap']:+.5f} ({100 * lv['n5']['gap_rel']:+.2f}%), t {lv['n5']['t']:.2f}" in b["entry"]
    s6 = on["contrasts"]["S006.mtp"]
    assert f"d -153 / +765 steps, dbar +{s6['dbar']:.1f}, thr {s6['thr']:.1f}: UNRESOLVED" in b["S006_mtp_aux"]
    f5 = c6["metrics"]["HD_asst"]["contrasts"]["S005.forget"]
    assert f"HD_asst d {f5['d'][0]:+.5f} / {f5['d'][1]:+.5f} bpb against its own BASE" in b["S005_forget_gate"]
    assert "LOWER at the new-init SD" in b["S005_forget_gate"] and f5["label"] == "LOWER" and f5["class"] == "new init"
    for sid in STAGE2:
        (k, c), = res(sid)["contrasts"].items()
        x = c["readings"]["CHAT"]
        assert f"dbar {x['dbar']:+.5f} ({100 * x['dbar'] / x['base_mean']:+.2f}% of" in b[STAGE2[sid]], sid
    med = [st.median([h for layer in x["by_layer"] for h in layer["half_life"]])
           for x in c6["reported"]["modules"]["S005.forget"].values()]
    assert f"median {med[0]:.1f} / {med[1]:.1f};" in b["S005_forget_gate"]
    hd = c6["metrics"]["HD_asst"]
    reg, po = hd["sensitivity"]["at_registered_sd"]["readings"], hd["sensitivity"]["at_pooled_sd"]["readings"]
    x4 = hd["contrasts"]["S004.canonac"]
    assert (f"the new-init SD (sqrt(2) x sigma_seed = {x4['sd_ref']:.5f}; PART 4d's rho on HD_asst is "
            f"{hd['part4d_noise']['rho']:.3f}) is below the SIA SD whose check failed (u x SD_d = "
            f"{x4['sd_ref_registered']:.5f}) and far below the observed pooled SD ({p['sd']:.5f})") in b["entry"]
    assert (f"at the SIA SD (thr {reg['S004.canonac']['thr']:.5f}) the LOWER and UNRESOLVED labels are the same, but Holm "
            f"over the 9 holds only for S001 (S005 p {reg['S005.forget']['p_two_sided']:.3g}") in b["entry"]
    assert (f"at the observed pooled SD (thr {po['S004.canonac']['thr']:.5f}) every SIA arm reads UNRESOLVED (S005 p "
            f"{po['S005.forget']['p_two_sided']:.3g})") in b["entry"]
    assert (f"at the SIA SD (thr {reg['S004.canonac']['thr']:.5f}) it is still LOWER but Holm does not hold (p "
            f"{reg['S004.canonac']['p_two_sided']:.3g})") in b["S004_canon"]
    assert (f"but Holm does not hold there (p {reg['S005.forget']['p_two_sided']:.3g}); at the observed pooled SD (thr "
            f"{po['S005.forget']['thr']:.5f}) it reads UNRESOLVED (p {po['S005.forget']['p_two_sided']:.3g})") in b["S005_forget_gate"]
    assert (f"Holm holds for none of the three (p novres {reg['S002.novres']['p_two_sided']:.3g}, nonormscale "
            f"{reg['S002.nonormscale']['p_two_sided']:.3g}, noqknorm {reg['S002.noqknorm']['p_two_sided']:.3g})") in b["S002_block_ablations"]
    a3 = c6["metrics"]["HD_asst"]["contrasts"]["S003.adamw"]
    assert f"HD_asst d {a3['d'][0]:+.5f} / {a3['d'][1]:+.5f} bpb, thr {a3['thr']:.5f}: {a3['label']}" in b["S003_adamw"]
    p4 = jload(EXP, "E3_seed_noise", "results_part4d.json")["metrics"]["IND"]["part1_5M"]["noise"]
    assert f"mean d {p4['mean_d']:+.5f} (t {p4['t']:.2f}, p {p4['p_two_sided']:.3g}:" in b["E3"]
    assert f"IND mean A {p4['mean_a']:.5f}, sigma_seed {p4['sigma_seed']:.5f}, SD_d {p4['SD_d']:.5f}" in flat(
        b["E3"].replace("IND       mean A", "IND mean A"))
