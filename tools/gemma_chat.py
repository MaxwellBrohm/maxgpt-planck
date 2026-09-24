"""Terminal chat with a Gemma 4 model served by LM Studio, with thinking truly OFF.

LM Studio's "Enable Thinking" switch did not reach the model, so this builds the Gemma 4 prompt
by hand and sends it to the raw completions endpoint. With thinking off, the official template
(2026-07-09) pre-fills an empty thought block after the model turn opens; we do the same.

    python3 gemma_chat.py                      # default model google/gemma-4-12b-qat
    python3 gemma_chat.py --model google/gemma-4-26b-a4b-qat --system "Keep replies short."

Commands inside the chat: /reset (new conversation), /quit
Needs: LM Studio server running (lms server start) and the model loaded (lms load <model>).
Standard library only.
"""
import argparse
import json
import time
import urllib.request

URL = "http://localhost:1234/v1/completions"


def build_prompt(system, history, user_msg):
    # History keeps final answers only (no thought blocks), as the Gemma 4 template does.
    parts = []
    if system:
        parts.append("<|turn>system\n" + system + "<turn|>\n")
    for u, a in history:
        parts.append("<|turn>user\n" + u + "<turn|>\n")
        parts.append("<|turn>model\n" + a + "<turn|>\n")
    parts.append("<|turn>user\n" + user_msg + "<turn|>\n")
    parts.append("<|turn>model\n<|channel>thought\n<channel|>")
    return "".join(parts)


def stream_reply(model, prompt, temperature, max_tokens):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature,
            "top_p": 0.95, "stop": ["<turn|>", "<|turn>"], "stream": True}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    out, n, t0 = [], 0, time.time()
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            piece = json.loads(data)["choices"][0].get("text", "")
            if piece:
                out.append(piece)
                n += 1
                print(piece, end="", flush=True)
    dt = time.time() - t0
    print(f"\n  [{n} chunks in {dt:.1f}s]")
    return "".join(out).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-12b-qat")
    ap.add_argument("--system", default="")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=800)
    args = ap.parse_args()

    history = []
    print(f"Chatting with {args.model} (thinking off). /reset for a new chat, /quit to exit.")
    while True:
        try:
            user_msg = input("\nYou: ").strip()
        except EOFError:
            break
        if not user_msg:
            continue
        if user_msg == "/quit":
            break
        if user_msg == "/reset":
            history = []
            print("  [new conversation]")
            continue
        print("Gemma: ", end="", flush=True)
        reply = stream_reply(args.model, build_prompt(args.system, history, user_msg),
                             args.temperature, args.max_tokens)
        history.append((user_msg, reply))


if __name__ == "__main__":
    main()
