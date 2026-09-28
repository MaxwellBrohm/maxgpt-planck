"""Accepted tokens in Planck's tokenizer for one drive.py run (DRY; runs where the tokenizers package is, the PC).

    python count_planck.py --texts accepted_texts.dry.jsonl --tok tok_v0_8k.json --md5 EXPECTED --elapsed-s S \
        [--json OUT.dry.json]

--texts comes from drive_report.py --export (turn text and author of every accepted chat). Counts turn text only,
no role or template tokens, with encode_special_tokens on as harness/data.load_tokenizer does; teacher-written turns
(author teacher:*) are also counted alone. --elapsed-s is the run's wall time summed over driver sessions
(drive_report elapsed_s_all_sessions, as yld_all.sh passes it; elapsed_s is the LAST session only and inflates tokens
per hour after a resume), so tokens per hour are accepted tokens over the whole run at its concurrency, retries and
queueing included. The
tokenizer file must match --md5 (the repo's tokenizer/v0/tok_v0_8k.json), so a stale copy cannot be counted with."""
import argparse
import hashlib
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--md5", required=True)
    ap.add_argument("--elapsed-s", type=float, required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    with open(a.tok, "rb") as f:
        got = hashlib.md5(f.read()).hexdigest()
    if got != a.md5:
        raise SystemExit(f"tokenizer md5 {got} != expected {a.md5}")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tok)
    if hasattr(tok, "encode_special_tokens"):
        tok.encode_special_tokens = True
    chats = allt = teach = 0
    with open(a.texts, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = json.loads(ln)
            chats += 1
            for t in r["turns"]:
                n = len(tok.encode(t["text"]).ids)
                allt += n
                teach += n if str(t["author"]).startswith("teacher:") else 0
    out = {"status": "dry", "tokenizer_md5": got, "chats": chats, "accepted_all_turns": allt,
           "accepted_teacher_turns": teach, "per_chat": round(allt / chats, 1) if chats else None,
           "elapsed_s": a.elapsed_s, "accepted_tokens_per_hour": round(allt / a.elapsed_s * 3600),
           "accepted_teacher_tokens_per_hour": round(teach / a.elapsed_s * 3600),
           "counted": "turn text only, no role or template tokens"}
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
