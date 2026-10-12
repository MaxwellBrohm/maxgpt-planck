"""RC-12 lockstep runner (draft s4: "conversations advance in lockstep batches, turn t of every conversation in one
batch"; notes STEP 9).

Every conversation of a run (x every seed) is a runner.play_steps() generator: the SAME loop runner.play() drives,
so the history fitting, the plain stop cut, the strip, the --own-cf rewrite (own_cf.swap, per conversation, before
its next turn is requested) and the transcript fields are one code path. play_all() advances all of them one turn
at a time: it collects the pending (history, turn) of every live conversation, makes ONE engine.reply_batch() call,
and sends each reply back to its own generator. A conversation with fewer turns simply leaves the batch early.
rows() then grades and builds each row with runner.make_row, in runner.run's order (for s in seeds for r in recs).
For a deterministic responder the rows are identical to runner.run's (test_lockstep.py, every fake).

Engines (anything with count(messages, render) and reply_batch(reqs) -> [(raw, stop)], reqs = [(k, rec, seed,
history, turn)], k = the conversation's index in the pass):
  vllm_responder.VLLMResponder  native batch: one llm.generate call per turn index, a per-request seed
  PerConv(factory, render)      one fresh responder per conversation, start()ed on its first turn (the fakes keep
                                per-conversation state in start, so one shared instance would mix conversations)
  Serial(responder, render)     HFResponder / PlanckResponder: start() + reply() per request in lockstep order.
                                No batching (their start() only sets the conversation id and seed); batched HF
                                sampling with per-reply seeds is NOT built.
engine(responder, render) picks: reply_batch -> itself; HF / Planck -> Serial; anything else is refused (a fake must
come wrapped in PerConv, which runner.make_responder does under --lockstep).
Stats: play_all returns the transcripts; engine.batches (when the engine has that list) gets one size per call."""
import runner as R


class PerConv:
    def __init__(self, factory, render):
        self.factory, self.render, self.live = factory, render, {}
        self.proto = factory()
        self.ctx = getattr(self.proto, "ctx", None)
        self.batches = []

    def count(self, messages, render=None):
        return self.proto.count(messages, render)

    def reply_batch(self, reqs):
        self.batches.append(len(reqs))
        out = []
        for k, rec, seed, history, i in reqs:
            if k not in self.live:
                self.live[k] = self.factory()
                self.live[k].start(rec, seed, self.render)
            out.append(self.live[k].reply(history, i))
        return out

    def end(self, k):
        self.live.pop(k, None)


class Serial:
    def __init__(self, responder, render):
        self.r, self.render, self.batches = responder, render, []
        self.ctx = getattr(responder, "ctx", None)

    def count(self, messages, render=None):
        return self.r.count(messages, render)

    def reply_batch(self, reqs):
        self.batches.append(len(reqs))
        out = []
        for _, rec, seed, history, i in reqs:
            self.r.start(rec, seed, self.render)
            out.append(self.r.reply(history, i))
        return out


def engine(responder, render):
    if hasattr(responder, "reply_batch"):
        return responder
    import hf_responder as HR
    import planck_responder as PR
    if isinstance(responder, (HR.HFResponder, PR.PlanckResponder)):
        return Serial(responder, render)
    raise TypeError(f"{type(responder).__name__} has no reply_batch; wrap a fake in lockstep.PerConv")


def play_all(convs, eng, render="plain", ctx=None, own_cf=False):
    """convs: [(rec, seed)]; returns their transcripts in the same order (runner.play's output for each)."""
    gens, played, pend = [], [None] * len(convs), {}

    def advance(k, reply):
        try:
            pend[k] = gens[k].send(reply)
        except StopIteration as done:
            played[k] = done.value
            if hasattr(eng, "end"):
                eng.end(k)

    for k, (rec, _) in enumerate(convs):
        gens.append(R.play_steps(rec, eng, render, ctx, own_cf))
        advance(k, None)
    while pend:
        keys = sorted(pend)
        reqs = [(k, convs[k][0], convs[k][1]) + tuple(pend.pop(k)) for k in keys]
        outs = eng.reply_batch(reqs)
        if len(outs) != len(reqs):
            raise RuntimeError(f"reply_batch returned {len(outs)} replies for {len(reqs)} requests")
        for k, reply in zip(keys, outs):
            advance(k, tuple(reply))
    return played


def rows(recs, responder, render="plain", seeds=(None,), ctx=None, name="?", train_seed=0, own_cf=False):
    convs = [(r, s) for s in seeds for r in recs]
    played = play_all(convs, engine(responder, render), render, ctx, own_cf)
    return [R.make_row(rec, pl, render, seed, name, train_seed, own_cf) for (rec, seed), pl in zip(convs, played)]
