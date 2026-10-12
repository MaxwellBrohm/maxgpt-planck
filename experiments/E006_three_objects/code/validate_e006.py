"""E006 stream acceptance and purity on the KEPT streams (notes.txt WHAT STAYS IDENTICAL 2-4, STREAM ACCEPTANCE AND
PURITY). No model; the SmolLM2 tokenizer is loaded (tokshim_e006.load: "hf" on the PC, E006_TOK=shim elsewhere).
One row per arm x seed; exit 0 only if every check passes.
  E004  reference position rates R4 (GL2), DELTA4, X4 (GLX2): pooled over E004's first 6,404 kept, seeds 1-5
  C     kept stream through e006_sets == through e005_sets (sha256 of ids and labels); stats == E005's run.json
        train_stats (n, rejected, mean and max length, every drawn/kept share); E005's gates; 0 eval 5-grams
  P     E005's gates; 0 eval 5-grams; checks_e006p.all_problems_p empty on every kept example; the position gates
        (whole-stream GL2, ALIAS GL2, IND GL2 <= R4; Delta <= DELTA4; GLX2 <= X4); identity 3: block and render
        sequence and the E004-block examples equal C's by drawn index; reported: the share of P's kept examples that are
        byte-identical to one of C's kept examples
  G     identity 4 (G's update source is C's; digests equal); the replay pool's checks (validate_e006_b.replay_checks)
usage: python -B validate_e006.py [--seeds 1 2 3 4 5] [--pool ../replay/replay_pool.jsonl] [--no-replay]"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import e006_sets as S6
import position_e006 as PO
import checks_e006p as CP
import validate_e006_b as VB
import tokshim_e006 as TK

N_KEPT = 6404
E005_RUN = os.path.join(os.path.dirname(EXP), "E005_alias_eot", "out",
                        "HuggingFaceTB__SmolLM2-135M-Instruct__s{}__run.json")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    ap.add_argument("--pool", default=os.path.join(EXP, "replay", "replay_pool.jsonl"))
    ap.add_argument("--no-replay", action="store_true")
    a = ap.parse_args(argv)
    import json
    kind = os.environ.get("E006_TOK", "hf")
    tok = TK.load(kind)
    S6.S5.eot_id(tok)
    if kind == "hf":
        import lik
        cand = lik.cand_ids
    else:
        cand = S6.cand_ids_notorch
    fails = []
    eg = VB.PU.eval_alias_grams()
    e4 = [PO.stats(VB.e004_kept(tok, s, N_KEPT, cand)) for s in (1, 2, 3, 4, 5)]
    ref = PO.pooled(e4)
    print(f"E004 kept, pooled s1-5: GL2 {ref['GL2']:.4f} GL1 {ref['GL1']:.4f} Delta {ref['Delta']:.4f} "
          f"GLX2 {ref['GLX2']:.4f} (n2 {ref['n2']})", flush=True)
    for s in a.seeds:
        kc, stc, ndc = S6.kept(tok, "C", s, N_KEPT, cand)
        d6 = S6.digest([(x, y) for _, x, y, _ in kc])
        side = S6.S5.new_stats()
        it = S6.S5.example_stream(tok, s, 768, side, cand)
        d5 = S6.digest([(x, y) for x, y, _ in (next(it) for _ in range(N_KEPT))])
        want = json.load(open(E005_RUN.format(s)))["train_stats"]
        got = S6.S5.stream_summary(stc)
        same = all(got[k] == want[k] for k in ("n", "rejected_long", "mean_len", "len_max", "shares"))
        g, info = VB.gates_e005([x[0] for x in kc], stc, eg)
        f5 = VB.fivegram_overlap([x[0] for x in kc])
        row = [f"C s{s}: drawn {ndc}, digest {d6[:16]} (e005_sets {d5[:16]}), stats == E005 run.json {same}; {info}; "
               f"eval 5-grams {len(f5)}; {PO.fmt(PO.stats([x[0] for x in kc]), ('all',))}"]
        fails += [f"C s{s}: {x}" for x in g] + ([f"C s{s}: digest differs from e005_sets"] if d6 != d5 else []) + \
            ([f"C s{s}: stats differ from E005's run.json"] if not same else []) + \
            ([f"C s{s}: eval 5-grams {f5[:3]}"] if f5 else [])
        kp, stp, ndp = S6.kept(tok, "P", s, N_KEPT, cand)
        pex = [x[0] for x in kp]
        g, info = VB.gates_e005(pex, stp, eg)
        f5 = VB.fivegram_overlap(pex)
        tab = sum(1 for ex in pex if CP.all_problems_p(ex))
        pst = PO.stats(pex)
        pg = PO.gates(pst, ref)
        drawn_p = S6.TP.take_p(s, ndp)
        drawn_c = S6.T5.take(s, ndp)
        seq = all((x["block"], x["render"]) == (y["block"], y["render"]) for x, y in zip(drawn_p, drawn_c))
        e4same = all(x == y for x, y in zip(drawn_p, drawn_c) if x["block"] == "e004")
        cset = {json.dumps(x[0], sort_keys=True) for x in kc}
        same_pos = sum(json.dumps(x[0], sort_keys=True) in cset for x in kp) / N_KEPT
        fails += [f"P s{s}: {x}" for x in g + pg] + ([f"P s{s}: eval 5-grams {f5[:3]}"] if f5 else []) + \
            ([f"P s{s}: {tab} kept examples fail the per-example table"] if tab else []) + \
            ([f"P s{s}: block/render sequence or E004 block differs from C's"] if not (seq and e4same) else [])
        row.append(f"P s{s}: drawn {ndp}, {info}; eval 5-grams {len(f5)}; per-example table failures {tab}; "
                   f"position gates {'PASS' if not pg else pg}; sequence and E004 block == C {seq and e4same}; kept "
                   f"examples byte-identical to one of C's kept examples {same_pos:.4f}; changed {sum('p_change' in ex for ex in pex)}")
        row.append("    P position " + PO.fmt(pst) + f" | GLX2 {pst['all']['GLX2']:.4f} GL2 {pst['all']['GL2']:.4f}")
        if S6.source("G", s) is not S6.source("C", s):
            fails.append(f"G s{s}: G's update source is not C's")
        row.append(f"G s{s}: update part = C's kept stream (source {S6.source('G', s)}, digest {d6[:16]})")
        print("\n".join(row), flush=True)
    if not a.no_replay:
        pool, bj = VB.load_pool(a.pool)
        g, L = VB.replay_checks(tok, pool, bj, a.seeds, 1604)
        fails += g
        print("\n".join(L), flush=True)
    for f in fails:
        print("FAIL ", f)
    print("RESULT: " + ("ALL PASS" if not fails else f"{len(fails)} FAILED"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
