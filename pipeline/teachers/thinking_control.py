"""Positive control for the thinking-off check (DRY; nothing here is training data). PC only, gpu.lock held:

    flock -w 7200 ~/planck/locks/gpu.lock python thinking_control.py --teacher qwen3.5-9b --out OUT.dry.json

The pilot passes a teacher when no thought marker appears in its outputs. An absence check is only evidence if
the detector fires when thinking is ON, so control() runs the same 3 reasoning prompts with the same seeds:
  off      serve.generate, the pilot path (thinking off);
  on       the teacher's own chat template with enable_thinking=True. Gemma: '<|think|>' system turn, no pre-filled
           channel, so a thought OPENS in the output ('<|channel>', id 100). Qwen: the template pre-fills
           '<think>\\n', so only a CLOSING '</think>' can show, and only if the thought ends inside max_tokens;
  on_open  Qwen only: the 'on' render with the pre-filled '<think>\\n' removed, so the model has to write the
           OPENING tag itself (id 248068); a few tokens are enough.
Pass: no marker in any 'off' output, and a marker in at least 2 of 3 outputs of the probe that can show an opening
tag (Gemma 'on', Qwen 'on_open'). Qwen's 'on' is reported, not scored: 2 of 3 thoughts ran past 1500 tokens on
2026-09-26 05:06, which failed the earlier version of this bar. Ministral 3 Instruct has no thinking mode, so it
has no ON render and no control."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import serve  # noqa: E402

PROMPTS = [
    "What is 17 times 23? Think it through step by step before you answer.",
    "A bat and a ball cost 1.10 dollars in total. The bat costs one dollar more than the ball. How much is the ball? "
    "Reason carefully.",
    "Plan a three day trip to Lisbon. First think about what matters, then give the plan.",
]
SEEDS = [201, 202, 203]
QWEN_PREFILL = "<think>\n"
SCORED = {"gemma-4-12b": "on", "qwen3.5-9b": "on_open"}
MAX_TOKENS = {"gemma-4-12b": {"off": 400, "on": 400}, "qwen3.5-9b": {"off": 400, "on": 1900, "on_open": 16}}


def render_on(t, prompt, open_tag=False):
    """-> (ids, text) of the official chat template with thinking ON; open_tag drops Qwen's pre-filled '<think>\\n'."""
    text = t.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=True)
    if open_tag:
        if not text.endswith(QWEN_PREFILL):
            raise serve.RenderError(f"ON render does not end in the think prefill: {text[-30:]!r}")
        text = text[: -len(QWEN_PREFILL)]
    return t.tok.encode(text, add_special_tokens=False), text


def control(t, max_tokens=None):
    """run the control on a loaded teacher -> dict with rows, counts and pass."""
    from vllm.inputs import TokensPrompt
    mt = {**MAX_TOKENS[t.name], **(max_tokens or {})}
    off = serve.generate(t, PROMPTS, {"max_tokens": mt["off"]}, seeds=SEEDS)
    rows = [{"prompt_head": p[:60], "off_thought": o["thought"], "off_n_out": o["n_out"], "off_finish": o["finish"],
             "off_head": o["raw"][:200]} for p, o in zip(PROMPTS, off)]
    probes = ["on"] + (["on_open"] if t.name == "qwen3.5-9b" else [])
    for probe in probes:
        todo = []
        for p, s in zip(PROMPTS, SEEDS):
            ids, text = render_on(t, p, open_tag=probe == "on_open")
            sp, _ = serve._params(t, {"max_tokens": mt[probe]}, len(ids), s)
            todo.append((TokensPrompt(prompt_token_ids=ids), sp, text[-40:]))
        res = t.llm.generate([x[0] for x in todo], [x[1] for x in todo], use_tqdm=False)
        for row, x, r in zip(rows, todo, res):
            c = r.outputs[0]
            row.update({f"{probe}_prompt_tail": x[2], f"{probe}_thought": serve.thought_in(t, c.text, list(c.token_ids)),
                        f"{probe}_n_out": len(c.token_ids), f"{probe}_finish": c.finish_reason,
                        f"{probe}_head": c.text[:200], f"{probe}_first_ids": list(c.token_ids[:4])})
    n_off = sum(1 for r in rows if r["off_thought"])
    fired = {p: sum(1 for r in rows if r[f"{p}_thought"]) for p in probes}
    scored = SCORED[t.name]
    return {"rows": rows, "off_with_marker": n_off, "fired": fired, "scored_probe": scored, "max_tokens": mt,
            "pass": n_off == 0 and fired[scored] >= 2}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, choices=sorted(SCORED))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = {"status": "dry", "what": "thinking-check positive control; not training data", "teacher": a.teacher,
           "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    t = serve.load(a.teacher, max_num_seqs=8)
    try:
        res.update(control(t))
        res["load_s"] = t.load_s
        print(f"[control] {a.teacher}: off {res['off_with_marker']}/3 with a marker, fired {res['fired']} "
              f"-> pass {res['pass']}", flush=True)
    finally:
        serve.unload(t)
        res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0 if res.get("pass") else 1


if __name__ == "__main__":
    sys.exit(main())
