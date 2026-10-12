"""RC-12 item 14 card row (notes STEP 11, ITEM 14; NOVELTY_2026-10 s6): engines.json "decode" (dev_batch.decode_override
/ apply_decode), with the stub transformers / torch / vllm of test_chat_template.py (no model, no GPU).
  dev_batch  a vllm entry's decode (top_p, repetition_penalty) reaches hf_responder.DECODE, so meta.json "decode" and
             the vLLM sampling kwargs (sampled and greedy) carry it; no decode leaves DECODE as it was; decode on hf /
             hfb, an unknown key, a non-dict, a string value or an out-of-range value is refused before anything
             loads, and DECODE is left unchanged
  verify     verify_dev_runs --engines E checks a run against E's decoding: a card run passes C1's decode and vLLM
             kwargs checks under the card file and fails them under a file without decode, and a plain run fails
             them under the card file
  files      the repo's engines.json has no decode anywhere (s4: plain decoding for every scored row); every
             engines_card.json entry is its engines.json entry plus a valid decode (engine, dtype, family, parity,
             template and load options equal)
Run: python3 -B test_decode_variant.py   (exit 1 on any failure)"""
import contextlib
import io
import json
import os
import sys
import tempfile

import test_chat_template as TC        # installs the stubs at import

DB, VD = TC.DB, TC.VD
HR = VD.HR
PLAIN = dict(HR.DECODE)
CARD = {"top_p": 0.9, "repetition_penalty": 1.3}      # Vertex-0.6-15M-Instruct's card at the pinned revision
fails = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(cond, what):
    if not cond:
        fails.append(what)


def reset():
    HR.DECODE.clear()
    HR.DECODE.update(PLAIN)


def audited_fake(spec, ns):
    """test_chat_template's fake engine (the real responder is built through the stubs, IDEAL replies are played),
    keeping the real responder's audit(), so meta.json records the vLLM kwargs it would pass."""
    eng, name = TC.REAL_MAKE(spec, ns)
    TC.BUILT.append((type(eng).__name__,))
    fake = TC.LS.PerConv(lambda: TC.FF.make("IDEAL"), ns.render)
    if hasattr(eng, "audit"):
        fake.audit = eng.audit
    return fake, name


TC.fake_engine = audited_fake


def dev(base, kind, entry):
    reset()
    try:
        return TC.run_dev(base, kind, entry)
    finally:
        after = dict(HR.DECODE)
        reset()
        dev.after = after


def c_dev_batch(b):
    rc, meta = dev(os.path.join(b, "card"), "vllm", {"engine": "vllm", "decode": CARD})
    a = (meta or {}).get("audit") or {}
    want = dict(PLAIN, **CARD)
    check(rc == 0 and meta is not None and meta.get("decode") == want,
          f"card run: rc {rc} decode {meta and meta.get('decode')}")
    sp, gr = a.get("sampled", {}), a.get("greedy", {})
    check(sp.get("top_p") == 0.9 and sp.get("repetition_penalty") == 1.3 and gr.get("repetition_penalty") == 1.3,
          f"card run: vLLM kwargs {sp} {gr}")
    rc, meta = dev(os.path.join(b, "plain"), "vllm", {"engine": "vllm"})
    check(rc == 0 and meta is not None and meta.get("decode") == PLAIN and dev.after == PLAIN,
          f"plain run: decode {meta and meta.get('decode')}")
    for name, kind, d in (("hfb", "hfb", CARD), ("hf", "hf", CARD), ("key", "vllm", {"temperature": 1.0}),
                          ("list", "vllm", ["top_p"]), ("str", "vllm", {"top_p": "0.9"}), ("p0", "vllm", {"top_p": 0}),
                          ("p_big", "vllm", {"top_p": 1.5}), ("rp_low", "vllm", {"repetition_penalty": 0.9}),
                          ("bool", "vllm", {"top_p": True})):
        rc, meta = dev(os.path.join(b, "bad_" + name), kind, {"engine": kind, "decode": d})
        check("decode" in str(rc) and not TC.CALLS and not TC.BUILT and meta is None and dev.after == PLAIN,
              f"decode {name} on {kind} refused: rc {rc} calls {TC.CALLS} after {dev.after}")
    return os.path.join(b, "card", "root"), os.path.join(b, "plain", "root")


def verify(root, decode):
    reset()
    b = tempfile.mkdtemp(prefix="rc12_dec_ver_")
    engines = os.path.join(b, "engines.json")
    entry = {"engine": "vllm"}
    if decode:
        entry["decode"] = decode
    json.dump({"models": {"org/M": entry}}, open(engines, "w"))
    VD.FAILS.clear()
    old = sys.argv
    sys.argv = ["verify_dev_runs.py", "--root", root, "--model", "org/M", "--render", "template", "--engines", engines,
                "--no-prompt"]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            VD.main()
    finally:
        sys.argv = old
        reset()
    return [f for f in VD.FAILS if "decode" in f or "vLLM kwargs" in f]


def c_verify(card_root, plain_root):
    got = verify(card_root, CARD)
    check(not got, f"card run under the card file: {got}")
    got = verify(card_root, None)
    check(any("decode" in f for f in got) and any("vLLM kwargs" in f for f in got), f"card run, plain file: {got}")
    got = verify(plain_root, CARD)
    check(any("decode" in f for f in got) and any("vLLM kwargs" in f for f in got), f"plain run, card file: {got}")
    got = verify(plain_root, None)
    check(not got, f"plain run under a plain file: {got}")


def c_files():
    main = json.load(open(os.path.join(HERE, "engines.json")))
    with_decode = [m for m, e in main["models"].items() if "decode" in e]
    check(not with_decode, f"engines.json carries decode for {with_decode}")
    card = json.load(open(os.path.join(HERE, "engines_card.json")))
    check(bool(card["models"]), "engines_card.json has no entry")
    for m, e in card["models"].items():
        base = main["models"].get(m)
        same = base is not None and {k: v for k, v in e.items() if k not in ("decode", "reason")} == \
            {k: v for k, v in base.items() if k != "reason"}
        check(same, f"engines_card.json {m} differs from engines.json beyond decode")
        try:
            d = DB.decode_override(m, os.path.join(HERE, "engines_card.json"))
        except ValueError as err:
            d = str(err)
        check(isinstance(d, dict) and d and e.get("engine") == "vllm", f"engines_card.json {m} decode {d}")
    vx = card["models"].get("VertexResearch/Vertex-0.6-15M-Instruct", {})
    check(vx.get("decode") == CARD, f"engines_card.json Vertex decode {vx.get('decode')} is not its card's {CARD}")


def main():
    b = tempfile.mkdtemp(prefix="rc12_dec_")
    card_root, plain_root = c_dev_batch(b)
    c_verify(card_root, plain_root)
    c_files()
    print("\n".join(f"FAIL {f}" for f in fails) + "\n" * bool(fails) + "test_decode_variant: " +
          ("PASS" if not fails else f"{len(fails)} FAILURES"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
