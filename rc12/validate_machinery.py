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
  owncf      the OWN counterfactual-history diagnostic (step 5 audit, own_cf.py)"""
import argparse
import os
import sys

import fakes_family as FF
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
            check(S.summarize(rows)["R"] == 100, f"HISTCHECK {render}: padded IDEAL replies do not score 100", out)
    s = S.summarize(run(recs, "IDEAL_ALT"))
    check(s["families"]["OWN"] == 1.0, "IDEAL_ALT OWN != 1.00 (gold not taken from the model's own reply)", out)
    return out


def c_stops(recs):
    out = []
    rows = run(recs, "CONTINUER", "plain")
    s = S.summarize(rows)
    check(s["R"] == 100 and all(t["stop"] == "role" and t["raw"] for r in rows for t in r["turns"]),
          "CONTINUER plain: fake user turn not cut", out)
    s = S.summarize(run(recs, "CONTINUER", "template"))
    check(s["R"] == 0 and s["degenerate_rates"]["LEAK"] == 1.0, "CONTINUER template: leak not flagged", out)
    s = S.summarize(run(recs, "CAPPER"))
    check(s["R"] == 0 and s["loop_rate"] == 0 and s["degenerate_rates"]["RUNAWAY"] == 1.0,
          f"CAPPER: R {s['R']}, loop {s['loop_rate']}, runaway {s['degenerate_rates']['RUNAWAY']}", out)
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
    check(S.summarize(rows)["R"] < 100, "ctx 556: truncation does not lower the unit scores", out)
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
    first = S.summarize(run(recs, "FIRST"))["R"]
    check(abs(S.summarize(rows)["R"] - first) < 1e-9, "default scoring did not select the sampling seed", out)
    check(S.summarize(rows, seeds=[None])["R"] == 100, "greedy selection", out)
    return out


def c_scoring(recs):
    out = []
    s = S.summarize(run(recs, "T0WRONG"))
    k = s["keys"]
    check(s["R"] == 100 and k["T0"] == 0 and k["TWOHOP"] == 1 and k["TWOHOP:COMPOSE"] == 0 and k["K"] == 0,
          f"T0/K/COMPOSE leak into R or TWOHOP (R {s['R']}, TWOHOP {k['TWOHOP']})", out)
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
    rows = {n: run(recs, n) for n in names}
    sm = {n: S.summarize(rows[n]) for n in names}
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
    check(rates["first_word:Greetings"] == 0 and rates["one_sentence:None"] > ST.PERSIST_DROP,
          f"PERSIST base rates {rates}", out)
    check(ST.persist_drops({"IDEAL": rates}) == ["one_sentence:None"], "PERSIST drop list", out)
    sens = ST.sensitivity(sm["IDEAL"], sm["FIRST"])
    check(sorted(sens["dropped"]) == ["BIND", "LOOKUP", "OWN", "PERSIST"] and sens["R"] == 100, "sensitivity", out)
    check(ST.sensitivity(sm["IDEAL"], sm["ECHO"])["R"] is None, "sensitivity with an all-floor comparator", out)
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
    check(mean(run(own, "CONSIST")) == 1.0, "CONSIST no longer passes OWN without --own-cf (update notes)", out)
    for fake, want in (("IDEAL", 1.0), ("IDEAL_ALT", 1.0), ("CONSIST", 0.0)):
        rows = run(own, fake, own_cf=True)
        check(mean(rows) == want, f"{fake} OWN under --own-cf = {mean(rows):.2f}, want {want}", out)
        check(all(t["raw"] is not None for r in rows for t in r["turns"] if t["kind"] == "Q"),
              f"{fake}: an OWN Q reply was not swapped under --own-cf", out)
    other = [r for r in recs if r["family"] in ("RECALL", "PERSIST", "TOPIC")][:30]
    check(run(other, "IDEAL", own_cf=True) == [dict(x, own_cf=True) for x in run(other, "IDEAL")],
          "--own-cf changed a conversation outside OWN", out)
    return out


CHECKS = [c_render, c_history, c_stops, c_trunc, c_seeds, c_scoring, c_stats, c_hf, c_owncf]


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
