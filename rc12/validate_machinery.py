"""RC-12 machinery checks for runner.py, render.py, score.py, score_stats.py and hf_responder.py (no model).
Each check returns a list of failure strings; main() writes logs/e2e_machinery.txt and exits 1 on any failure.
  render     plain render string, the plain stop cut, the template stub, fit() and own_reply()
  history    HISTCHECK sees exactly the user turns plus its OWN earlier replies (stripped), plain and template,
             with and without truncation; IDEAL_ALT scores OWN 1.00 only because its replies are fed back
  stops      CONTINUER: plain render cuts the fake user turn (stop "role", IDEAL scores); template render keeps it
             (LEAK on every reply, R = 0); CAPPER: RUNAWAY on every reply, loop rate 0, R = 0
  trunc      ctx 556 (proxy budget 300): some probes fail "trunc"; a probe fails iff a needed turn was dropped;
             ctx 4096 gives the same rows as no limit
  seeds      SAMPLER: runs keep their seed; sampling seeds are scored by default, greedy only on request
  scoring    T0 / K / COMPOSE excluded from R; BIND unit = pair; nested seed means; Level A comparator and
             claimability; K gap
  stats      headroom floor / ceiling, paired bootstrap and Level R, PERSIST base rates, sensitivity row
  hf         hf_responder imports without torch; runner refuses it without --hf-untested-ok
  owncf      the OWN counterfactual-history diagnostic (step 5 audit, own_cf.py); padded, blank-bulleted and
             U+2028 lists are swapped too (LISTPAD, LISTBLANK, LISTLS: the list rule is G-DYN's)
  owngate    OD1 (b): OWN counts in R only if right in the own-history AND the --own-cf run (score.py GATE); NEXT
             (right only on the cf history) is 0; a cf run of another responder or render is no twin; seeds come
             from the own-history rows; random per-unit outcomes with shuffled cf rows match a per-unit AND made
             here; a rewrite that did not take (cf_unswapped) scores 0
  ackrep     OD6 (iii): ACKER ("Got it." on every statement turn) has no LOOP, meets the Level A loop criterion,
             and its ack-repeat rates equal a count made here from the records; IDEAL has none; --own-cf rows do
             not move the rates; PARROT (its last short answer again on statement turns) has no LOOP either, and
             its answer-repeat rate equals a count made here
Without --own-cf rows R is None (OD1 b), so the checks of other machinery read R_ungated (the same composite with
the own-history OWN); c_stats adds each fake's --own-cf OWN rows so the bootstrap and the sensitivity row run."""
import argparse
import os
import sys

import fakes_family as FF
import grade_text as T
import own_cf as OC
import render as RD
import runner as RN
import score as S
import score_stats as ST

HERE = os.path.dirname(os.path.abspath(__file__))


def run(recs, fake, render="plain", seeds=(None,), ctx=None, train_seed=0, own_cf=False):
    return RN.run(recs, FF.make(fake), render, list(seeds), ctx, None, fake, train_seed, own_cf)


def check(cond, msg, out):
    if not cond:
        out.append(msg)


def c_render(recs):
    out = []
    m = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}, {"role": "user", "content": "u2"}]
    check(RD.plain(m) == "User: u1\nAssistant: a1\nUser: u2\nAssistant:", "plain render string", out)
    check(RD.template_stub(m).endswith("<|im_start|>assistant\n") and RD.template_stub(m).count("<|im_end|>") == 3,
          "template stub", out)
    for raw, want in [("Sure.\nUser: hi", "Sure."), ("Sure.\nuser: hi", "Sure."), ("Sure.\n  Human: x", "Sure."),
                      ("A\nAssistant: B", "A"), ("A\nB", "A\nB"), ("The user: said hi", "The user: said hi"),
                      ("A\nThe user: x", "A\nThe user: x")]:
        check(RD.cut_plain(raw)[0] == want, f"cut_plain({raw!r})", out)
    msgs = [{"role": "user" if k % 2 == 0 else "assistant", "content": "w " * 10} for k in range(9)]
    kept, n = RD.fit(msgs, 60, RD.proxy_tokens)
    check(n > 0 and kept == msgs[2 * n:] and RD.proxy_tokens(kept) <= 60 and kept[0]["role"] == "user",
          "fit drops whole oldest pairs until it fits", out)
    check(RD.fit(msgs, 1, RD.proxy_tokens)[0] == msgs[-1:], "fit keeps the current user turn", out)
    check(RD.fit(msgs, None, RD.proxy_tokens) == (msgs, 0), "fit without a budget", out)
    h = msgs[4:]
    check(RD.first_turn(h, 5) == 3 and RD.own_reply(h, 5, 4) == h[3]["content"] and RD.own_reply(h, 5, 2) is None
          and RD.own_reply(h, 5, 5) is None, "own_reply index", out)
    return out


def c_history(recs):
    out = []
    for render, ctx in (("plain", None), ("template", None), ("plain", 556), ("template", 556)):
        f = FF.make("HISTCHECK")
        bad, rows = [], []
        for r in recs:
            rows.append(RN.run_one(r, f, render, None, ctx))
            bad += f.violations
        check(not bad, f"HISTCHECK {render} ctx={ctx}: {len(bad)} bad histories {bad[:3]}", out)
        if ctx is None:
            check(S.summarize(rows)["R_ungated"] == 100, f"HISTCHECK {render}: padded IDEAL replies do not score 100",
                  out)
    s = S.summarize(run(recs, "IDEAL_ALT"))
    check(s["keys"]["OWN"] == 1.0, "IDEAL_ALT OWN != 1.00 (gold not taken from the model's own reply)", out)
    return out


def c_stops(recs):
    out = []
    rows = run(recs, "CONTINUER", "plain")
    s = S.summarize(rows)
    check(s["R_ungated"] == 100 and all(t["stop"] == "role" and t["raw"] for r in rows for t in r["turns"]),
          "CONTINUER plain: fake user turn not cut", out)
    s = S.summarize(run(recs, "CONTINUER", "template"))
    check(s["R_ungated"] == 0 and s["degenerate_rates"]["LEAK"] == 1.0, "CONTINUER template: leak not flagged", out)
    s = S.summarize(run(recs, "CAPPER"))
    check(s["R_ungated"] == 0 and s["loop_rate"] == 0 and s["degenerate_rates"]["RUNAWAY"] == 1.0,
          f"CAPPER: R {s['R_ungated']}, loop {s['loop_rate']}, runaway {s['degenerate_rates']['RUNAWAY']}", out)
    return out


def c_trunc(recs):
    out = []
    f, rows, n_trunc, bad = FF.make("FORGETFUL"), [], 0, []
    for r in recs:
        row = RN.run_one(r, f, "plain", None, 556)
        rows.append(row)
        for p in row["probes"]:
            n_trunc += "trunc" in p["fails"]
            lost, forgot = bool(p.get("lost")), p["turn"] in f.forgot
            if (not p["ok"]) != lost or lost != ("trunc" in p["fails"]) or lost != forgot:
                bad.append((row["id"], p["turn"]))
    check(n_trunc >= 50, f"ctx 556: only {n_trunc} truncated probes", out)
    check(not bad, f"ctx 556: verdict, runner's lost turns and the fake's history disagree {bad[:3]}", out)
    check(S.summarize(rows)["R_ungated"] < 100, "ctx 556: truncation does not lower the unit scores", out)
    rows = run(recs, "IDEAL", ctx=556)
    stale = [r["id"] for r in rows if (r["unit"] < 1) != any(not p["ok"] for p in r["probes"])]
    check(any("trunc" in p["fails"] for r in rows for p in r["probes"]) and not stale,
          f"ctx 556: units not recomputed after truncation {stale[:3]}", out)
    a, b = run(recs, "IDEAL", ctx=4096), run(recs, "IDEAL")
    check(a == b, "ctx 4096 differs from no limit", out)
    return out


def c_seeds(recs):
    out = []
    rows = run(recs, "SAMPLER", seeds=(None, 1))
    check(sorted({str(r["seed"]) for r in rows}) == ["1", "None"] and len(rows) == 2 * len(recs), "seed labels", out)
    first = S.summarize(run(recs, "FIRST"))["R_ungated"]
    check(abs(S.summarize(rows)["R_ungated"] - first) < 1e-9, "default scoring did not select the sampling seed", out)
    check(S.summarize(rows, seeds=[None])["R_ungated"] == 100, "greedy selection", out)
    return out


def c_scoring(recs):
    out = []
    s = S.summarize(run(recs, "T0WRONG"))
    k = s["keys"]
    check(s["R_ungated"] == 100 and k["T0"] == 0 and k["TWOHOP"] == 1 and k["TWOHOP:COMPOSE"] == 0 and k["K"] == 0,
          f"T0/K/COMPOSE leak into R or TWOHOP (R {s['R_ungated']}, TWOHOP {k['TWOHOP']})", out)
    check(not s["t0"]["met"] and len(s["t0"]["failures"]) == 48, "T0 gate / failure list", out)
    s = S.summarize(run(recs, "BINDHALF"))
    check(s["families"]["BIND"] == 0 and s["families"]["CORR"] == 1 and not s["level_a"]["BIND"]["met"],
          f"BIND pair unit (got {s['families']['BIND']})", out)
    check(S.nested({(0, 1): 1.0, (0, 2): 0.0, (1, 1): 1.0}) == 0.75, "nested seed means", out)
    ideal = run(recs, "IDEAL")
    si = S.summarize(ideal)
    rep = S.summarize(run(recs, "L_REPEAT"), comparator=si)
    check(S.summarize(ideal, comparator=rep)["level_a_met"] and not rep["level_a"]["LOOP_vs_comparator"]["met"],
          "loop-vs-comparator criterion", out)
    check(not si["level_a_met"], "Level A met without a comparator", out)
    check(si["R"] is None and si["families"]["OWN"] is None and si["R_ungated"] == 100
          and "OWN_GATED" not in si["keys"],
          f"no --own-cf rows: R {si['R']}, OWN {si['families']['OWN']} (OD1 b: missing, never the ungated value)", out)
    three = [dict(r, train_seed=ts) for ts in (0, 1, 2) for r in ideal]
    check(not S.summarize(ideal, comparator=si)["level_a_claimable"] and
          S.summarize(three, comparator=si)["level_a_claimable"], "claimable needs 3 training seeds", out)
    check(si["k"]["gap"] == 0 and si["own_source_fail"] == 0, "K gap / OWN source-fail rate", out)
    k = S.summarize(run(recs, "WORDING"))["keys"]
    check(k["CORR:U-diff"] != k["CORR:U-same"] and k["CORR:U"] == (k["CORR:U-diff"] + k["CORR:U-same"]) / 2,
          "CORR:U is not U-diff and U-same pooled", out)
    n = {x["key"]: x["n_units"] for x in S.score_lines(ideal)}
    check(n["CORR:C_noupd"] == 16 and n["CORR:U"] == 32 and n["BIND"] == 30 and n["TWOHOP"] == 48,
          f"unit counts {n}", out)
    return out


def c_stats(recs):
    out = []
    names = ["IDEAL", "IDEAL_ALT", "ECHO", "ABSTAIN", "DEFLECT", "FIRST"]
    own = [r for r in recs if r["family"] == "OWN"]
    base = {n: run(recs, n) for n in names}
    rows = {n: base[n] + run(own, n, own_cf=True) for n in names}       # + the --own-cf OWN run (OD1 b)
    sm = {n: S.summarize(rows[n]) for n in names}
    check(sm["IDEAL"]["R"] == 100 and sm["IDEAL_ALT"]["R"] == 100, "IDEAL / IDEAL_ALT R != 100 with the cf rows", out)
    h = ST.headroom({n: sm[n] for n in ("IDEAL", "IDEAL_ALT")})
    check(all(v["reason"] == "ceiling" for v in h.values()), "headroom ceiling", out)
    h = ST.headroom({n: sm[n] for n in ("ECHO", "ABSTAIN")})
    check(all(v["reason"] == "floor" for k, v in h.items() if k != "LOOP") and not h["LOOP"]["fail"],
          "headroom floor", out)
    check(not any(v["fail"] for v in ST.headroom({n: sm[n] for n in ("IDEAL", "ECHO")}).values()),
          "headroom spread", out)
    h = ST.headroom({n: sm[n] for n in ("ECHO", "DEFLECT")})
    check(h["PERSIST"]["reason"] == "floor" and 0 < sm["DEFLECT"]["families"]["PERSIST"] <= 0.05
          and not h["LOOP"]["fail"], "headroom floor bar (DEFLECT PERSIST sits between 0 and 0.05)", out)
    ci = ST.bootstrap_diff(rows["IDEAL"], rows["IDEAL"], n=200)
    check(ci["D"] == 0 and ci["lo"] == 0 and ci["hi"] == 0, "bootstrap D(IDEAL, IDEAL)", out)
    ci = ST.bootstrap_diff(rows["FIRST"], rows["FIRST"], n=200)
    check(ci["lo"] == 0 and ci["hi"] == 0, f"bootstrap is not paired: D(FIRST, FIRST) CI {ci}", out)
    ci = ST.bootstrap_diff(rows["IDEAL"], rows["FIRST"], n=200)
    check(ci["lo"] > 0 and ci["lo"] <= ci["D"] <= ci["hi"] and ST.level_r(ci), f"bootstrap IDEAL-FIRST {ci}", out)
    ci = ST.bootstrap_diff(rows["FIRST"], rows["IDEAL"], n=200)
    check(ci["hi"] < 0 and not ST.level_r(ci), f"bootstrap FIRST-IDEAL {ci}", out)
    rates = ST.persist_base_rates(rows["IDEAL"], ST.persist_rules(recs))
    check(rates == ST.persist_base_rates(base["IDEAL"], ST.persist_rules(recs)), "cf rows in PERSIST base rates", out)
    check(rates["first_word:Greetings"] == 0 and rates["one_sentence:None"] > ST.PERSIST_DROP,
          f"PERSIST base rates {rates}", out)
    check(ST.persist_drops({"IDEAL": rates}) == ["one_sentence:None"], "PERSIST drop list", out)
    sens = ST.sensitivity(sm["IDEAL"], sm["FIRST"])
    check(sorted(sens["dropped"]) == ["BIND", "LOOKUP", "OWN", "PERSIST"] and sens["R"] == 100, "sensitivity", out)
    check(ST.sensitivity(sm["IDEAL"], sm["ECHO"])["R"] is None, "sensitivity with an all-floor comparator", out)
    check(ST.sensitivity(S.summarize(base["IDEAL"]), sm["FIRST"])["R"] is None, "sensitivity without the cf rows", out)
    try:
        ST.bootstrap_diff(base["IDEAL"], base["IDEAL"], n=10)
        out.append("bootstrap ran without the --own-cf rows (OD1 b)")
    except Exception as e:  # noqa: BLE001 (the refusal must be the stated assertion, not a crash)
        check(isinstance(e, AssertionError) and "['OWN']" in str(e), f"bootstrap refusal {type(e).__name__}: {e}", out)
    return out


def c_hf(recs):
    out = []
    import hf_responder as H
    check(H.HF_TESTED is False and "torch" not in sys.modules and "transformers" not in sys.modules,
          "hf_responder import is not light or claims to be tested", out)
    try:
        RN.make_responder("hf:none", argparse.Namespace(hf_untested_ok=False, render="template", dtype="bfloat16",
                                                        device="cpu"))
        out.append("runner accepted the untested HF responder without the flag")
    except SystemExit as e:
        check("UNTESTED" in str(e), "runner refusal message", out)
    except Exception as e:  # noqa: BLE001 (it went on to load a model)
        out.append(f"runner went past the refusal to load a model ({type(e).__name__})")
    return out


def c_owncf(recs):
    """--own-cf (own_cf.py): readers of their own history keep OWN 1.00, the CONSIST re-deriver (1.00 without it)
    drops to 0.00, every OWN Q reply is swapped, and nothing outside OWN changes."""
    out = []
    own = [r for r in recs if r["family"] == "OWN"]
    mean = lambda rows: sum(r["unit"] for r in rows) / len(rows)  # noqa: E731
    for fake in ("CONSIST", "LISTPAD", "LISTBLANK", "LISTLS"):
        check(mean(run(own, fake)) == 1.0, f"{fake} no longer passes OWN without --own-cf (update notes)", out)
    for fake, want in (("IDEAL", 1.0), ("IDEAL_ALT", 1.0), ("CONSIST", 0.0), ("LISTPAD", 0.0), ("LISTBLANK", 0.0),
                       ("LISTLS", 0.0)):
        rows = run(own, fake, own_cf=True)
        check(mean(rows) == want, f"{fake} OWN under --own-cf = {mean(rows):.2f}, want {want}", out)
        check(all(t["raw"] is not None for r in rows for t in r["turns"] if t["kind"] == "Q"),
              f"{fake}: an OWN Q reply was not swapped under --own-cf", out)
    other = [r for r in recs if r["family"] in ("RECALL", "PERSIST", "TOPIC")][:30]
    check(run(other, "IDEAL", own_cf=True) == [dict(x, own_cf=True) for x in run(other, "IDEAL")],
          "--own-cf changed a conversation outside OWN", out)
    return out


def c_owngate(recs):
    """OD1 (b), ruled 2026-09-25: an OWN unit counts in R only if it is right in the own-history run AND the
    --own-cf run. IDEAL and IDEAL_ALT stay 1.00, CONSIST (1.00 on its own history) drops to 0.00, and so does NEXT
    (0.00 on its own history, 1.00 on the cf one: the gate needs both, not the cf run alone); OWN (own history) and
    OWN_CF (cf run alone) are reported beside; the bootstrap and headroom use the gated units; a missing or partial
    cf run, or one of another responder or render, makes the gated OWN missing, never the ungated value; cf rows
    count in nothing else and never choose the seeds; heterogeneous units are joined per conversation (verifier
    2026-09-25)."""
    out = []
    own = [r for r in recs if r["family"] == "OWN"]
    sums = {}
    for fake, h_want, cf_want in (("IDEAL", 1.0, 1.0), ("IDEAL_ALT", 1.0, 1.0), ("CONSIST", 1.0, 0.0),
                                  ("NEXT", 0.0, 1.0), ("LISTPAD", 1.0, 0.0)):
        want = min(h_want, cf_want)
        hist, cf = run(own, fake), run(own, fake, own_cf=True)
        s = sums[fake] = S.summarize(hist + cf)
        check(s["families"]["OWN"] == want and s["own_gate"] == dict(OWN=h_want, OWN_CF=cf_want, OWN_GATED=want,
                                                                     cf_unswapped=0)
              and s["keys"].get("OWN_GATED:pick") == want and s["keys"].get("OWN_GATED:list") == want,
              f"{fake}: gated OWN {s['families']['OWN']} {s['own_gate']}, want OWN_GATED {want}", out)
        h = S.summarize(hist)
        check(s["replies"] == h["replies"] and s["keys"]["OWN"] == h["keys"]["OWN"] and s["t0"] == h["t0"],
              f"{fake}: --own-cf rows counted outside the gate (replies {s['replies']} vs {h['replies']}, OWN "
              f"{s['keys']['OWN']} vs {h['keys']['OWN']})", out)
        slot = ST.unit_table(hist + cf)[0][None]["OWN"]
        check(len(slot) == len(own) and set(slot.values()) == {want}, f"{fake}: bootstrap OWN slot not gated", out)
    hr = ST.headroom({n: sums[n] for n in ("IDEAL", "CONSIST")})["OWN"]
    check(not hr["fail"] and hr["scores"] == {"IDEAL": 1.0, "CONSIST": 0.0}, f"headroom OWN not gated {hr}", out)
    hist, cf = run(own, "IDEAL"), run(own, "IDEAL", own_cf=True)
    for label, rows in (("no cf run", hist), ("cf rows for half the OWN units", hist + cf[:len(cf) // 2]),
                        ("cf rows for 1 of 2 training seeds", hist + cf + [dict(r, train_seed=1) for r in hist]),
                        ("cf run of another responder", run(own, "CONSIST") + cf),
                        ("cf run in another render", hist + run(own, "IDEAL", "template", own_cf=True))):
        s = S.summarize(rows)
        check(s["families"]["OWN"] is None and s["own_gate"]["OWN_GATED"] is None and s["own_gate"]["OWN"] == 1.0
              and not any(k.startswith("OWN_GATED") for k in s["keys"]),
              f"{label}: gated OWN not missing ({s['own_gate']})", out)
    s = S.summarize(hist + run(own, "IDEAL", seeds=(1,), own_cf=True))
    check(s["replies"] == S.summarize(hist)["replies"] and s["own_gate"]["OWN"] == 1.0
          and s["own_gate"]["OWN_GATED"] is None, f"greedy history + sampled cf rows: seeds taken from the cf rows "
          f"(replies {s['replies']}, {s['own_gate']})", out)
    out += owngate_units(hist, cf)
    saved = OC._exchange
    OC._exchange = lambda text, a, b: text          # a pick rewrite that does not take (own_cf.UNSWAPPED)
    try:
        cf_bad = run(own, "IDEAL", own_cf=True)
    finally:
        OC._exchange = saved
    s = S.summarize(hist + cf_bad)
    picks = [r for r in cf_bad if r["cell"] == "pick"]
    check(all(r["cf_unswapped"] == (r["cell"] == "pick") for r in cf_bad)
          and all(t.get("cf_unswapped") and t["raw"] is None for r in picks for t in r["turns"] if t["kind"] == "Q")
          and s["keys"].get("OWN_CF:pick") == 1.0 and s["keys"].get("OWN_GATED:pick") == 0.0
          and s["keys"].get("OWN_GATED:list") == 1.0 and s["own_gate"]["cf_unswapped"] == len(picks) > 0,
          f"unswapped cf rows not scored 0 in OWN_GATED ({s['own_gate']}, pick {s['keys'].get('OWN_GATED:pick')})", out)
    return out


def owngate_units(hist, cf):
    """per-unit AND on random outcomes (2 training seeds x 2 sampling seeds, cf rows shuffled), against a nested
    mean of a per-(training seed, seed, id) AND computed here; mixed cells catch a misaligned join and a product of
    rates that uniform fakes cannot (verifier 2026-09-25)."""
    import random
    rng, rows, want = random.Random(1212), [], {}
    for tr in (0, 1):
        for sd in (1, 2):
            h = [dict(r, train_seed=tr, seed=sd, unit=float(rng.random() < 0.6)) for r in hist]
            c = [dict(r, train_seed=tr, seed=sd, unit=float(rng.random() < 0.6), cf_unswapped=rng.random() < 0.1)
                 for r in cf]
            byid = {r["id"]: r for r in c}
            for cell in ("pick", "list"):
                ands = [float(r["unit"] == 1.0 and byid[r["id"]]["unit"] == 1.0 and not byid[r["id"]]["cf_unswapped"])
                        for r in h if r["cell"] == cell]
                want.setdefault(cell, {}).setdefault(tr, []).append(sum(ands) / len(ands))
            rows += h + c
    rng.shuffle(rows)
    ks = S.summarize(rows)["keys"]
    fails = []
    for cell, per_tr in want.items():
        w = sum(sum(v) / len(v) for v in per_tr.values()) / len(per_tr)
        got = ks.get(f"OWN_GATED:{cell}")
        if got is None or abs(got - w) > 1e-12:
            fails.append(f"random units: OWN_GATED:{cell} {got}, per-unit AND {w:.4f}")
    return fails


def c_ackrep(recs):
    """OD6 (iii), ruled 2026-09-25: an identical reply to a statement turn is an ack-repeat, not a LOOP. Expected
    counts come from the records (every statement turn after a conversation's first repeats "Got it."), not from
    grade_loop."""
    out = []
    stated = [sum(t["kind"] in "SLCIOT" for t in r["turns"]) for r in recs]
    n_turns = sum(r["n_turns"] for r in recs)
    hits = sum(max(0, n - 1) for n in stated)
    s = S.summarize(run(recs, "ACKER"))
    check(s["R_ungated"] == 100 and s["loop_rate"] == 0 and not any(s["degenerate_rates"].values())
          and s["level_a"]["LOOP"]["met"], f"ACKER: R {s['R_ungated']}, loop {s['loop_rate']}, "
          f"degenerate {s['degenerate_rates']}, Level A loop criterion met {s['level_a']['LOOP']['met']}", out)
    check(s["replies"] == n_turns and s["ack_repeat"] == hits / n_turns and s["ack_repeat_of_statements"]
          == hits / sum(stated), f"ACKER ack-repeat {s['ack_repeat']} / {s['ack_repeat_of_statements']}, want "
          f"{hits / n_turns} / {hits / sum(stated)}", out)
    check(s["ack_repeat_of_answers"] == 0, f"ACKER answer-repeat {s['ack_repeat_of_answers']}, want 0", out)
    acker = run(recs, "ACKER")
    for fake in ("ACKER", "PARROT"):       # PARROT: its OWN cf rows hold answer repeats, so they would move the rate
        rows = acker if fake == "ACKER" else run(recs, fake)
        a = S.summarize(rows)
        c = S.summarize(rows + run([r for r in recs if r["family"] == "OWN"], fake, own_cf=True))
        check([c[k] for k in ACK_KEYS] == [a[k] for k in ACK_KEYS], f"{fake}: --own-cf rows moved the ack-repeat "
              f"rates: {[c[k] for k in ACK_KEYS]} vs {[a[k] for k in ACK_KEYS]}", out)
    i = S.summarize(run(recs, "IDEAL"))
    check(i["ack_repeat"] == 0 and i["ack_repeat_of_statements"] == 0 and i["ack_repeat_of_answers"] == 0,
          f"IDEAL ack-repeat {i['ack_repeat']} / {i['ack_repeat_of_answers']}", out)
    rows = [{k: v for k, v in r.items() if k != "ack_repeat"} for r in run(recs[:5], "ACKER")]
    check(S.ack_stats(rows) == (None, None), "rows without the ack_repeat record not reported as None", out)
    rows = [{k: v for k, v in r.items() if k != "ack_of_answer"} for r in acker[:5]]
    check(S.answer_repeat_stats(rows) is None, "rows without the ack_of_answer record not reported as None", out)
    # PARROT (verifier 2026-09-25): its last short (< 12 words) answer again on each later statement turn. Under OD6
    # (iii) as ruled that is an ack-repeat, not a LOOP; the answer-repeat rate reports it (count made here).
    hits = sum(any(u["kind"] in "PXQD" and len(T.lwords(T.norm(u["ideal"]))) < 12 for u in r["turns"][:t["i"] - 1])
               for r in recs for t in r["turns"] if t["kind"] in "SLCIOT")
    p = S.summarize(run(recs, "PARROT"))
    check(p["loop_rate"] == 0 and p["level_a"]["LOOP"]["met"] and p["ack_repeat"] == hits / n_turns
          and p["ack_repeat_of_answers"] == p["ack_repeat_of_statements"] == hits / sum(stated) and hits > 0,
          f"PARROT loop {p['loop_rate']}, answer-repeat {p['ack_repeat_of_answers']}, want {hits / sum(stated)}", out)
    return out


ACK_KEYS = ("ack_repeat", "ack_repeat_of_statements", "ack_repeat_of_answers")


CHECKS = [c_render, c_history, c_stops, c_trunc, c_seeds, c_scoring, c_stats, c_hf, c_owncf, c_owngate, c_ackrep]


def checks(recs=None):
    recs = recs or RN.load()
    return [f"{c.__name__[2:]}: {m}" for c in CHECKS for m in c(recs)]


def main():
    fails = checks()
    text = "RC-12 machinery checks (validate_machinery.py), no model.\n" + (
        "\n".join(f"FAIL {f}" for f in fails) if fails else f"ALL {len(CHECKS)} CHECK GROUPS PASS") + "\n"
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    with open(os.path.join(HERE, "logs", "e2e_machinery.txt"), "w") as f:
        f.write(text)
    print(text)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
