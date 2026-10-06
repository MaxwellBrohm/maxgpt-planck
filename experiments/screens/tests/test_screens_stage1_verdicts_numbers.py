"""SCREENS STAGE 1 VERDICTS (SCREENS.txt, 2026-10-06): every decimal number written in that entry and in the three
notes.txt RESULT blocks is one that the screens' results.json gives (or a named derivation of it below), at the
precision written; so a typo in the prose fails here. No model.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_stage1_verdicts_numbers.py
"""
import datetime as dt, os, re, statistics as st, sys  # noqa: E401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import screens_hours as H  # noqa: E402
import screens_lib as L  # noqa: E402
from test_screens_s003_stage_c import section  # noqa: E402
from test_screens_stage1_verdicts import DATE, SIDS, res, text  # noqa: E402

NUM = re.compile(r"(?<![\w.])[+-]?\d+\.\d+(?:e[+-]\d+)?%?")


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
    for n in range(1, 6):
        yield from (f"{x:.{n}f}", f"{x:+.{n}f}", f"{abs(x):.{n}f}")
    for n in range(1, 4):
        yield from (f"{100 * x:.{n}f}%", f"{100 * x:+.{n}f}%")
    yield f"{x:.2g}"


def derived():
    """Numbers the prose states that are sums, means or differences of results.json values."""
    out, hours = [], {}
    for sid in SIDS:
        r = res(sid)
        for c in r["contrasts"].values():
            for x in c["readings"].values():
                out += [abs(x["dbar"]) + x["thr"], x["dbar"] / x["base_mean"]]
        c6 = r["c6"]
        hours.update(c6["hours_lock_train"])
        tr = c6.get("loss_60_160", {})
        for n in tr:
            b = tr.get("base_s" + n[-3:])
            if not n.startswith("base"):
                d = [tr[n][k] - b[k] for k in b]
                out += [st.fmean(d), max(abs(v) for v in d)]
    lock = [v[0] for v in hours.values()]
    out += [sum(lock), sum(v[1] for v in hours.values()), min(lock), max(lock)]
    out += [sum(v[0] for n, v in hours.items() if n.endswith(s)) for s in ("101", "102")]
    src = res("S001")["source"]
    wall = dt.datetime.fromisoformat(src["mark_time"]) - dt.datetime.fromisoformat(src["queue_start"])
    out += [wall.total_seconds() / 3600, L.params()["u"], H.run_h(L.STEPS), 12 * H.run_h(L.STEPS)]
    return out


ALLOWED = {"250.0", "1.0", "0.05"}    # C1's 250.0M slots; the 1.0 gradient clip; Holm's family alpha


def known():
    vals = [v for sid in SIDS for v in nums(res(sid))] + derived()
    return {f for v in vals for f in forms(v)} | ALLOWED


def check(txt, ok):
    bad = sorted({t for t in NUM.findall(txt) if t not in ok})
    assert not bad, bad


def test_every_number_in_the_notes_result_blocks_is_computed():
    ok = known()
    for sid in SIDS:
        t = text(sid, "notes.txt")
        check(t[t.index(f"\nRESULT ({DATE}"):], ok)


def test_every_number_in_the_screens_entry_is_computed():
    sec = section("STAGE 1 VERDICTS")
    stage_c = section("S003 STAGE C RESULT")
    assert "over 2.019 h" in stage_c                # the seed set 101 threshold this entry compares against
    check(sec.replace("2.019 h", "h"), known())


def test_key_phrases_carry_their_own_values():
    """The bag-of-numbers checks above pass a typo that lands on another computed value (a 0.651 written 0.615, a gate
    head's value, passed them); these phrases pin each value to its place."""
    from test_screens_stage1_verdicts import flat, notes_result
    blocks = {sid: notes_result(sid)[1] for sid in SIDS}
    for sid in SIDS:
        for c in res(sid)["contrasts"].values():
            ch, pr = c["readings"]["CHAT"], c["readings"]["PROSE"]
            assert f"CHAT d {ch['d'][0]:+.5f} / {ch['d'][1]:+.5f}" in blocks[sid]
            assert f"dbar {ch['dbar']:+.5f}" in blocks[sid] and f"dbar {pr['dbar']:+.5f}" in blocks[sid]
    r = res("S001")["c6"]
    cv = [r["gate_base"][f"base_s{s}"]["curve_mean_min_max"] for s in (101, 102)]
    l0 = [v for s in (101, 102) for v in r["gate_base"][f"base_s{s}"]["final_per_layer_head"][0]]
    ind = r["ind_final"]["nogate"]
    for p in (f"went from {cv[0][0][1]:.3f} / {cv[1][0][1]:.3f} at step 153 to {cv[0][-1][1]:.3f} / {cv[1][-1][1]:.3f} at 7,630",
              f"layer 0's three heads ended at {min(l0):.3f} to {max(l0):.3f}",
              f"gate off {ind['arm'][0]:.3f} / {ind['arm'][1]:.3f}, BASE {ind['base'][0]:.3f} / {ind['base'][1]:.3f} "
              f"(d {ind['d'][0]:+.3f} / {ind['d'][1]:+.3f}"):
        assert p in blocks["S001"], p
    d = res("S002")["c6"]["ind_final"]
    assert "final IND lower than BASE at both seeds in every arm (d " + ", ".join(
        f"{a} {d[a]['d'][0]:+.3f} / {d[a]['d'][1]:+.3f}" for a in ("novres", "noqknorm", "nonormscale")) in blocks["S002"]
    x = res("S003")["c6"]["ind_final"]["adamw"]
    assert (f"Final IND AdamW {x['arm'][0]:.3f} / {x['arm'][1]:.3f}, BASE {x['base'][0]:.3f} / {x['base'][1]:.3f} "
            f"(d {x['d'][0]:+.3f} / {x['d'][1]:+.3f})") in blocks["S003"]
    assert flat(section("STAGE 1 VERDICTS")).count("(+3.651 / +1.875)") == 1 and "+3.651 / +1.875" in flat(blocks["S001"])
