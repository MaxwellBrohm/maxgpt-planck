"""RC-12 dev-subset evaluation during training (OFF by default).

Config (a run's config.yaml; absent or every: 0 = off, and train.py then never imports rc12):

  eval:
    rc12:
      every: 500                 # optimizer steps between evals (runs at step % every == 0)
      per_family: 2              # conversations per family, first in file order (BIND: whole twin pairs)
      families: [RECALL, CORR]   # default: all 12 families of the dev file
      seeds: [greedy]            # greedy and/or integer sampling seeds (T 0.6, SPEC s2)
      render: template           # template (Planck role tokens, scored protocol) | plain (diagnostic)
      data: path/to/rc12.jsonl   # default rc12/dev/rc12_dev.jsonl
      tokenizer: path            # default data.tokenizer; relative paths resolve against the config's dir
      out: rc12                  # under out_dir: step_<8 digits>/transcripts.jsonl + scores.jsonl

Each eval runs rc12/runner.run on the live model through rc12/planck_responder.PlanckResponder (ctx = the model's
seq_len, the runner's truncation rule), grades with rc12's graders, scores with rc12/score.summarize, and appends
one line to out_dir/rc12_eval.jsonl: step, n_conv, seconds, and per seed kind (greedy, sampled) R (None: the OD1 b
OWN gate needs an --own-cf run, which the hook does not make), R_ungated (None unless all 10 composite families are
in the subset), families (OWN None for the same reason), loop_rate, the OD6 ack-repeat rates (ack_repeat,
ack_repeat_of_statements, ack_repeat_of_answers), degenerate rates, t0, k. The eval never changes
training: no_grad, the model's train/eval mode restored, sampling from a private generator (the global torch RNG
is untouched), the loader and optimizer are not read. test_rc12_eval.py checks a run with the hook on is bitwise
identical to the same run with it off. An exception inside an eval is logged to rc12_eval.jsonl (error) and
printed; training continues.
"""
from __future__ import annotations

import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
RC12 = os.path.join(os.path.dirname(HERE), "rc12")


def use_rc12():
    if RC12 not in sys.path:
        sys.path.append(RC12)


def subset(recs, families=None, per_family=None):
    """first per_family records of each family in file order; BIND counts twin PAIRS (score.py needs both)."""
    out, seen = [], {}
    for r in recs:
        f = r["family"]
        if families and f not in families:
            continue
        key = r["meta"].get("pair_id") if f == "BIND" else r["id"]
        got = seen.setdefault(f, [])
        if key not in got:
            if per_family is not None and len(got) >= per_family:
                continue
            got.append(key)
        out.append(r)
    return out


def _seed(s):
    return None if s in (None, "greedy") else int(s)


class RC12Eval:
    def __init__(self, ecfg: dict, run_cfg: dict, base_dir: str, out_dir: str, device: str, amp, seq_len: int):
        use_rc12()
        import planck_responder as PR
        import runner as RN
        from data import load_tokenizer
        self.PR, self.RN = PR, RN
        self.every = int(ecfg.get("every", 0))
        self.render = ecfg.get("render", "template")
        self.seeds = [_seed(s) for s in ecfg.get("seeds", ["greedy"])]
        dcfg = run_cfg.get("data", {})
        tok_path = ecfg.get("tokenizer") or dcfg.get("tokenizer")
        assert tok_path, "eval.rc12 needs a tokenizer (eval.rc12.tokenizer or data.tokenizer)"
        self.tok = load_tokenizer(PR.resolve(tok_path, base_dir))
        self.tmpl = PR.template_for(dcfg, self.tok)
        self.eos = PR.eos_for(dcfg, self.tok)
        data = PR.resolve(ecfg["data"], base_dir) if ecfg.get("data") else RN.DEV
        self.recs = subset(RN.load(data), ecfg.get("families"), ecfg.get("per_family"))
        assert self.recs, "eval.rc12 selected no conversations"
        self.dir = os.path.join(out_dir, ecfg.get("out", "rc12"))
        self.log_path = os.path.join(out_dir, "rc12_eval.jsonl")
        self.device, self.amp, self.seq_len = device, amp, int(seq_len)

    def responder(self, model):
        return self.PR.PlanckResponder(model, self.tok, self.tmpl, self.seq_len, self.eos, self.device, self.amp,
                                       self.render)

    def evaluate(self, model, step: int) -> dict:
        import score as S
        t0 = time.time()
        resp = self.responder(model)
        rows = self.RN.run(self.recs, resp, self.render, self.seeds, self.seq_len,
                           os.path.join(self.dir, f"step_{step:08d}"), f"step_{step}")
        rec = {"step": step, "n_conv": len(rows), "render": self.render, "short_ctx_replies": resp.short}
        kinds = [("greedy", [None])] if None in self.seeds else []
        if any(s is not None for s in self.seeds):
            kinds.append(("sampled", [s for s in self.seeds if s is not None]))
        for name, seeds in kinds:
            s = S.summarize(rows, seeds=seeds)
            rec[name] = {"R": s["R"], "R_ungated": s["R_ungated"], "families": s["families"],
                         "own_gate": s["own_gate"], "loop_rate": s["loop_rate"],
                         "ack_repeat": s["ack_repeat"], "ack_repeat_of_statements": s["ack_repeat_of_statements"],
                         "ack_repeat_of_answers": s["ack_repeat_of_answers"],
                         "degenerate": s["degenerate_rates"], "t0": s["t0"]["score"], "k": s["k"]}
        rec["seconds"] = round(time.time() - t0, 2)
        return rec

    def __call__(self, trainer):
        if not self.every or trainer.step % self.every:
            return None
        import runio
        try:
            rec = self.evaluate(trainer.model, trainer.step)
        except Exception as e:     # an eval bug must not kill a long run; it is logged, not hidden
            rec = {"step": trainer.step, "error": f"{type(e).__name__}: {e}"}
            traceback.print_exc()
        runio.append_jsonl(self.log_path, rec)
        return rec


def make_hook(run_cfg: dict, base_dir: str, out_dir: str, device: str, amp, seq_len: int):
    """-> RC12Eval, or None when eval.rc12 is absent or every is 0 (the default)."""
    ecfg = (run_cfg.get("eval") or {}).get("rc12") or {}
    if not int(ecfg.get("every", 0)):
        return None
    return RC12Eval(ecfg, run_cfg, base_dir, out_dir, device, amp, seq_len)
