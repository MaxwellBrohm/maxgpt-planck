"""SCREENS GPU hours and ORDER's cap (SCREENS.txt GPU HOURS, CAP AND CUT RULE; STAGE 2 READINESS 2026-10-05; 28 h and
a flag to Max instead of a cut since AMENDMENT BASE-DIAG and AMENDMENT CAP-FLAG).

  python screens_hours.py measured QUEUE_LOG STATUS_JSONL > measured_hours.tsv   (the PC queue's own two files)
  python screens.py hours [--measured-file TSV] [--s003-rb R]                    report(): estimate and cap check

Every registered run (GPU HOURS at E3's k per class) is one SLOT. A slot whose run has a measured line counts its
measured hours and leaves the queue; every other slot counts its estimate. A measured run with no slot (a C3 or
S003 extension, a matched-LR IND run) counts its measured hours on top. So no run is counted twice (before
2026-10-05 hours --measured added stage 1's measured hours on top of their still-queued estimates).
Estimates: GPU HOURS' method, 250.0M slots at 255,660 slots/s eager plus 15%, times the arm's factor = the shared
BASE smoke's tok/s over the arm's own smoke tok/s (SMOKE, each value at the file and line SRC names), x1.0 for the
config-only arms. An S003 search run (one E2 stage arm = trunk + 3 branches, 9 scorings) is estimated at the mean
measured hours of stage A's runs of the same tag: the +15% rule gave an arm 0.359 h, measured 0.406 to 0.409 h.
OTHER: GPU hours spent for this batch outside its runs (the C1-a gate, the eager smokes). They count as measured:
the conservative reading of the cap's "measured hours so far" (SCREENS.txt STAGE 2 READINESS).
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import sys

import screens_lib as L

EAGER = 255_660           # GPU HOURS: slots/s, 5M docmask B16 eager (harness/notes.txt SPEED INTEGRATE 2)
SMOKE = {                 # median log.jsonl tok/s of the eager smokes, 300 steps, the PC 5070, 2026-10-04
    "BASE": 253_691, ("S004", "canonac"): 184_569, ("S005", "base"): 192_092, ("S005", "forget"): 85_113,
    ("S006", "mtp"): 215_563, ("S007", "smear"): 247_073}
SRC = {                   # where each SMOKE value is logged: file, and the text on its line
    "BASE": ("experiments/SCREENS.txt", "BASE shared (varlen, 16 x 1)   5,010,133  8.881 -> 4.733   253,691"),
    ("S004", "canonac"): ("experiments/S004_canon/notes.txt", "tok/s: 184,569 median, against the shared BASE smoke's 253,691"),
    ("S005", "base"): ("experiments/S005_forget_gate/notes.txt", "192,092 tok/s median"),
    ("S005", "forget"): ("experiments/S005_forget_gate/notes.txt", "85,113 tok/s median"),
    ("S006", "mtp"): ("experiments/S006_mtp_aux/notes.txt", "tok/s: 215,563 median, against the shared BASE smoke's 253,691"),
    ("S007", "smear"): ("experiments/S007_smeared_key/notes.txt", "tok/s: 247,073 median, against the shared BASE smoke's 253,691")}
ETA_A = 3e-3              # S003 stage A pick (STAGE 1 SELECTION A RESULT); stage B runs r in E2's grid at it
STAGE_B_R, STAGE_C_G, TAGS = (0.5, 2.0, 4.0, 8.0), (0.5, 2.0), ("trunk", "b62M", "b125M", "b250M")
OTHER = [                 # name, GPU hours, source
    ("c1a_gate", 2.011, "SCREENS.txt VERIFICATION OF ORDER 0 (2): the C1-a gate, 31 gpu.lock holds from the PC gate logs"),
    ("eager_smokes", 0.109, "SCREENS.txt SMOKES RECORDED: the smokes' gpu.lock holds, the failed 10 s hold included")]
TSV = os.path.join(L.HERE, "measured_hours.tsv")
FMT = "%Y-%m-%d %H:%M:%S"


def run_h(steps: int, mult: float = 1.0, rate: float = EAGER) -> float:
    """GPU HOURS' registered per-run estimate: steps x 32,768 slots at the rate, times the factor, plus 15%."""
    return steps * L.SLOTS / rate * mult * 1.15 / 3600


def mult(sid: str | None, arm: str | None) -> float:
    """The arm's time factor against the shared BASE, from the eager smokes; 1.0 for an arm with no flag smoke."""
    return SMOKE["BASE"] / SMOKE[(sid, arm)] if (sid, arm) in SMOKE else 1.0


TS = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d ")
UNLOCKED = ("waiting for ", "PAUSE present", "STOP present", "disk: ", "GPU temperature [", "queue stops at: ",
            "plan finished: ", "plan says stop", "MARK ", "seen ", "skip ", "bad plan line: ")   # logged after gpu_unlock
RESTART = ("queue start: ", "boot resume")          # a new queue process (launch, waiter, boot.d/40_screens.sh)
SRC_EARLIER = "queue_screens.txt; an earlier hold of this run, ends at the queue's next line after gpu_unlock"
SRC_CUT = "queue_screens.txt; an earlier hold cut off by a queue restart, ends at the cut queue's last line (a lower bound)"


def parse_holds(queue: str, status: str) -> list[tuple[str, str, str, str]]:
    """(run, start, end, source) per gpu.lock hold that trained a run with a status line (SCREENS DEVIATION 1).
    A hold starts at "gpu.lock held" and ends at the first of: the run's status.jsonl line (its last hold: training
    plus scoring, SCREENS.txt STAGE 1 SELECTION A RESULT); a line the queue logs only after gpu_unlock (UNLOCKED: a
    heat stop's cool-down wait, a STOP); or a new queue process (RESTART: the hold was cut off by a power cut or a
    kill), where it ends at the cut queue's last line, a lower bound. A crash inside a hold (queue_lib.sh train_run
    resumes once under the same lock) logs a second train line in that hold: still one hold. A hold whose run had
    saved its final checkpoint before a cut logs no train line, only its score lines, which name it. Every hold of a run
    with a status line counts (a GAP that a restart counted, SCREENS DEVIATION 1, writes its status outside any
    hold); a run with no status line (running, or stopped and not yet resumed) is left out. NUL bytes a power cut
    leaves are skipped."""
    end = {r["run"]: r["time"].replace("T", " ") for r in map(json.loads, filter(str.strip, status.splitlines()))}
    holds, cur, last = [], None, None

    def close(t, src):
        if cur and cur["run"]:
            holds.append((cur["run"], cur["start"], t, src))
    for ln in queue.splitlines():
        ln = ln.lstrip("\x00")
        if not TS.match(ln):
            continue
        t, msg = ln[:19], ln[20:]
        if msg.startswith(RESTART):
            close(last, SRC_CUT)
            cur = None
        elif msg.startswith("gpu.lock held"):
            close(last, SRC_CUT)
            cur = {"start": t, "run": None}
        elif msg.startswith(UNLOCKED):
            close(t, SRC_EARLIER)
            cur = None
        elif cur and (m := re.match(r"(?:train (\S+)\.yaml|score (\S+)/)", msg)):
            cur["run"] = cur["run"] or m[1] or m[2]      # a run whose final was saved before a cut logs no train line
        last = t
    close(None, None)                                              # a hold still open at the log's end
    out = []
    for run, a, b, src in holds:
        e = end.get(run)
        if e is None or e < a:
            continue
        if b is None or e <= b:                                    # the hold the run's status line falls in
            out.append((run, a, e, "queue_screens.txt"))
        else:                    # an earlier hold (a GAP the restart counted has its status outside any hold)
            out.append((run, a, b, src))
    return out


def parse_queue(queue: str, status: str) -> list[tuple[str, str, str]]:
    """(run, start, end) per gpu.lock hold, parse_holds without the source. Before SCREENS DEVIATION 1 (2026-10-10)
    every train line got its own (held, status) pair, so a crash-resumed run was listed twice ("measured twice")."""
    return [(r, a, b) for r, a, b, _ in parse_holds(queue, status)]


def hours_between(a: str, b: str) -> float:
    return (dt.datetime.strptime(b, FMT) - dt.datetime.strptime(a, FMT)).total_seconds() / 3600


def tsv_text(queue: str, status: str, note: str = "") -> str:
    lines = ["# SCREENS measured GPU hours (screens_hours.py measured). run: gpu.lock held to the run's status line",
             f"# (training plus scoring), from the PC queue's queue_screens.txt and status.jsonl{note}.",
             "# other: GPU hours of this batch outside its runs (OTHER in screens_hours.py).",
             "kind\tname\thours\tstart\tend\tsource"]
    lines += [f"run\t{r}\t{hours_between(a, b):.5f}\t{a}\t{b}\t{src}" for r, a, b, src in parse_holds(queue, status)]
    lines += [f"other\t{n}\t{h}\t\t\t{s}" for n, h, s in OTHER]
    return "\n".join(lines) + "\n"


def load(path: str = TSV) -> tuple[dict, dict]:
    """({run: measured h}, {other: h}). A run's hours come from its start and end, never from the rounded column. A
    run resumed in a later gpu.lock hold has one line per hold (parse_holds): each hold counts once and they add up;
    the same hold listed twice, or two holds of one run that overlap, is refused."""
    runs, other, holds = {}, {}, {}
    for ln in open(path):
        if ln.startswith("#") or ln.startswith("kind\t") or not ln.strip():
            continue
        kind, name, h, a, b, _ = ln.rstrip("\n").split("\t")
        if kind == "run":
            seen = holds.setdefault(name, [])
            assert (a, b) not in seen, f"{name} measured twice (the hold {a} to {b} is listed twice)"
            assert all(b <= a2 or b2 <= a for a2, b2 in seen), f"{name}: two holds overlap"
            seen.append((a, b))
            runs[name] = runs.get(name, 0.0) + hours_between(a, b)
        else:
            other[name] = float(h)
    return runs, other


def slot(name: str) -> tuple[str, tuple | None]:
    """(screen, slot key) of a run name; key None = no registered slot (an extension)."""
    m = L.NAME_RE.match(name)
    assert m, f"{name}: not a SCREENS run name"
    sid = m["sid"].upper() if m["sid"] else None
    if m["tag"]:                                       # S003 search run (E2 stage arm)
        e, r = float(m["e"]) * 1e-3, float(m["r"])
        if r == 1 and any(math.isclose(e, x) for x in L.S003_STAGE_A):
            pt = ("A", min(L.S003_STAGE_A, key=lambda x: abs(x - e)))
        elif math.isclose(e, ETA_A) and r in STAGE_B_R:
            pt = ("B", r)
        elif r != 1 and any(math.isclose(e, g * ETA_A) for g in STAGE_C_G):
            pt = ("C", next(g for g in STAGE_C_G if math.isclose(e, g * ETA_A)))
        else:
            return "S003", None
        return "S003", ("S003", "search", *pt, m["tag"])
    seed = int(m["seed"])
    if m["diag"]:                                      # a BASE replay or hook-off run (BASE-REPLAY): on top, no slot
        return sid or "BASE", None
    if m["arm"] == "base":
        return sid or "BASE", (sid or "BASE", "base", seed)
    if m["e"]:
        return "S003", ("S003", "adamw", "seed", seed)
    g = float(m["g"])
    if seed == 1:
        return sid, ((sid, m["arm"], "g", g) if g in (0.5, 1.0, 2.0) else None)
    return sid, (sid, m["arm"], "seed", seed)


def s003_tag_hours(runs: dict) -> dict:
    """Mean measured hours per tag over S003 stage A's runs (the 5 arms at r 1)."""
    out = {}
    for tag in TAGS:
        hs = [runs[n] for e in L.S003_STAGE_A if (n := f"s003_adamw_e{L.num(e * 1e3)}_r1_{tag}") in runs]
        assert len(hs) == len(L.S003_STAGE_A), f"S003 stage A {tag}: {len(hs)} measured runs"
        out[tag] = sum(hs) / len(hs)
    return out


def registered(p: dict, runs: dict, r_b: float | None = None) -> dict:
    """{slot key: (screen, estimated h)} for every registered run (GPU HOURS at E3's k per class). S003: 11 search
    arms (stage A 5, stage B 4, stage C 2; none for stage C if r_B = 1, whose points are stage A runs) + seeds."""
    full = lambda sid, arm: run_h(L.STEPS, mult(sid, arm))  # noqa: E731
    reg = {("BASE", "base", s): ("BASE", full(None, None)) for s in max(p["seeds"].values(), key=len)}
    for sid, s in L.SCREENS.items():
        seeds = p["seeds"][s["cls"]]
        if sid == "S003":
            tag_h = s003_tag_hours(runs)
            pts = [("A", e) for e in L.S003_STAGE_A] + [("B", r) for r in STAGE_B_R]
            pts += [] if r_b == 1 else [("C", g) for g in STAGE_C_G]
            reg.update({("S003", "search", *pt, t): ("S003", tag_h[t]) for pt in pts for t in TAGS})
            reg.update({("S003", "adamw", "seed", x): ("S003", full(sid, "adamw")) for x in seeds})
            continue
        for a in s["arms"]:
            reg.update({(sid, a, "g", g): (sid, full(sid, a)) for g in (0.5, 1.0, 2.0)})
            reg.update({(sid, a, "seed", x): (sid, full(sid, a)) for x in seeds})
        if L.engine(sid):
            reg.update({(sid, "base", x): (sid, full(sid, "base")) for x in seeds})
    return reg


def tally(p: dict, runs: dict, r_b: float | None = None) -> dict:
    """Per screen: measured h (every measured run), queued h and runs (registered slots with no measured run)."""
    reg, seen = registered(p, runs, r_b), set()
    meas, extra, queued, nq = {}, {}, {}, {}
    for name, h in sorted(runs.items()):
        scr, key = slot(name)
        if key in reg and key not in seen:
            seen.add(key)
        else:
            extra[name] = h
        meas[scr] = meas.get(scr, 0.0) + h
    for key, (scr, est) in reg.items():
        if key not in seen:                            # a measured run's estimate leaves the queue
            queued[scr] = queued.get(scr, 0.0) + est
            nq[scr] = nq.get(scr, 0) + 1
    return {"measured": meas, "extra": extra, "queued": queued, "n_queued": nq, "n_registered": len(reg),
            "seen": seen, "registered": reg}


def report(p: dict, path: str = TSV, r_b: float | None = None, cap: float | None = None) -> dict:
    """The table, the conditional hours and ORDER's cap check. cap None = analyze_lib.CAP (28 h since AMENDMENT
    BASE-DIAG); the earlier entries' checks pass their own cap (25 h, 27 h). Over the cap the line says FLAG TO MAX
    (AMENDMENT CAP-FLAG): nothing is cut; "cap" keeps the cut order's arithmetic, the proposal the flag carries."""
    import analyze_lib as AL
    c = AL.CAP if cap is None else cap
    runs, other = load(path)
    t = tally(p, runs, r_b)
    print(f"{'screen':<8}{'measured runs':>14}{'h':>8}{'queued runs':>13}{'h':>8}{'total h':>9}")
    for scr in ["BASE"] + list(L.SCREENS):
        nm = sum(1 for n in runs if slot(n)[0] == scr)
        m, q = t["measured"].get(scr, 0.0), t["queued"].get(scr, 0.0)
        print(f"{scr:<8}{nm:>14}{m:>8.3f}{t['n_queued'].get(scr, 0):>13}{q:>8.3f}{m + q:>9.3f}")
    mr, mo, qs = sum(runs.values()), sum(other.values()), sum(t["queued"].values())
    print(f"runs measured {len(runs)} ({mr:.3f} h; {len(t['extra'])} with no registered slot: "
          f"{', '.join(t['extra']) or 'none'}); queued {sum(t['n_queued'].values())} of "
          f"{t['n_registered']} registered ({qs:.3f} h)")
    for n, h in other.items():
        print(f"other measured: {n} {h:.3f} h")
    factors = ", ".join(f"{k[0]} {k[1]} x{mult(*k):.4f}" for k in SMOKE if k != "BASE")
    tag_h = s003_tag_hours(runs)
    print(f"factors (eager smokes, shared BASE {SMOKE['BASE']:,} tok/s): {factors}; S003 search arm "
          f"{sum(tag_h.values()):.4f} h measured (" + ", ".join(f"{k} {v:.4f}" for k, v in tag_h.items()) + ")")
    full = lambda sid, a: run_h(L.STEPS, mult(sid, a))  # noqa: E731
    open_ = set(t["registered"]) - t["seen"]           # registered slots with no measured run
    ext = sum(2 * full(sid, a) for sid, s in L.SCREENS.items() if sid != "S003" for a in s["arms"]
              if any((sid, a, "g", g) in open_ for g in (0.5, 1.0, 2.0)))
    ext += sum(2 * sum(tag_h.values()) for st in ("B", "C") if any(k[1:3] == ("search", st) for k in open_))
    ind = 2 * (full("S006", "mtp") + full("S007", "smear"))
    print(f"conditional, counted only once queued (ORDER): up to {ext:.2f} h of extensions (2 per g-check arm not yet "
          f"run, 2 per S003 stage B and C); up to {ind:.2f} h of matched-LR IND runs (S006, S007 at g = 1, k = 2)")
    cap = AL.cap_cut(t["queued"], mr + mo, cap=c)
    flag = AL.cap_check(t["queued"], mr + mo, cap=c)
    narrow = AL.cap_cut(t["queued"], mr, cap=c)
    head = (f"ORDER cap (rule input, OTHER counted): measured {mr + mo:.3f} h + queued {qs:.3f} h = "
            f"{cap['total_before']:.3f} h against {cap['cap']:.0f} h: ")
    if not flag["flag_to_max"]:
        print(head + f"fits, nothing cut; headroom {cap['cap'] - cap['total_after']:.3f} h")
    else:
        print(head + f"OVER by {flag['over_by']:.3f} h: FLAG TO MAX, nothing is cut or queued without his decision "
              f"(AMENDMENT CAP-FLAG); the cut order would take " +
              (f"{', '.join(cap['cut'])} -> {cap['total_after']:.3f} h" if cap["cut"] else "nothing") +
              ("" if cap["fits"] else " (still over: nothing left that the order may take)"))
    print(f"runs only (reported, not the rule input): {narrow['total_before']:.3f} h, cut order would take "
          f"{narrow['cut'] or 'none'}")
    return {"tally": t, "cap": cap, "flag": flag, "narrow": narrow, "measured_runs": mr, "other": mo}


def main(argv=None) -> int:
    a = argv if argv is not None else sys.argv[1:]
    if len(a) != 3 or a[0] != "measured":
        print(__doc__)
        return 2
    sha = [L.sha256(f)[:16] for f in a[1:]]
    sys.stdout.write(tsv_text(open(a[1]).read(), open(a[2]).read(), f" (sha256 {sha[0]}... and {sha[1]}...)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
