"""RC-12 item 14 (notes STEP 11): Loom-Spark-3.2 ships no Hugging Face chat template; its card gives the exact format
("Prompt format is exact: <tools:off>\\n<user>\\n{message}\\n<|eot|>\\n<loom>\\n. For more turns, append
{reply}<|eot|>\\n<user>\\n{next message}\\n<|eot|>\\n<loom>\\n"; tools off, as item 14 says). engines.json carries
it as a Jinja chat template. This checks, with real jinja2 under transformers' chat-template settings (trim_blocks,
lstrip_blocks, raise_exception), that it renders exactly the card's string on dev histories of 1 to 12 user turns
(replies with newlines and trailing spaces included), that add_generation_prompt changes nothing (the card's prompt
already ends in the assistant tag), and that a system turn is refused (RC-12 passes none). No model, no tokenizer.
Run: python3 -B test_loom_template.py   (exit 1 on any failure)"""
import json
import sys

import jinja2

import dev_batch as DB
import runner as R

LOOM = "textilelabs/Loom-Spark-3.2"


def render(template, messages, gen=True):
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)
    env.globals["raise_exception"] = raise_exception
    return env.from_string(template).render(messages=messages, add_generation_prompt=gen)


def card(msgs):
    p = f"<tools:off>\n<user>\n{msgs[0]['content']}\n<|eot|>\n<loom>\n"
    for a, u in zip(msgs[1::2], msgs[2::2]):
        p += f"{a['content']}<|eot|>\n<user>\n{u['content']}\n<|eot|>\n<loom>\n"
    return p


def main():
    fails = []
    T = json.load(open(DB.ENGINES))["models"].get(LOOM, {}).get("chat_template")
    if not isinstance(T, str):
        print("FAIL engines.json has no chat_template for", LOOM)
        return 1
    n = 0
    for rec in R.load()[::37]:
        turns = sorted(rec["turns"], key=lambda x: x["i"])
        for k in (1, 2, 6, 12):
            msgs = []
            for t in turns[:k]:
                msgs += [{"role": "user", "content": t["text"]},
                         {"role": "assistant", "content": f"r{t['i']} line one\nline two "}]
            msgs = msgs[:-1]
            n += 1
            if render(T, msgs) != card(msgs) or render(T, msgs, gen=False) != card(msgs):
                fails.append(f"{rec['id']} k={k}: {render(T, msgs)[-80:]!r} != {card(msgs)[-80:]!r}")
    try:
        render(T, [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
        fails.append("a system turn is accepted")
    except jinja2.exceptions.TemplateError:
        pass
    if n < 40:
        fails.append(f"only {n} histories checked")
    for f in fails:
        print("FAIL", f)
    print(f"test_loom_template: {n} histories,", "PASS" if not fails else f"{len(fails)} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
