"""Accepted and reject records (SPEC 11) and the per-attempt yield keys.

accepted: ids, pass R0, system text, turns [{role, text, mask, author, spans}], slots, events with golds and turn
  indices, gold notes, provenance (skeleton + teacher + prompt), check codes (report-only ones and parser drift),
  heldout gate hash, rc12 status, token estimate, and trainable=false with the blocking reasons. Nothing is
  trainable in this build: every bank, pool and the teacher are FAKE (FAKE_PROVENANCE), the teacher license must be
  Apache-2.0 (D8, LICENSE), and RC-12 decontamination has not run (DECON_PENDING).
reject: skel_id, attempt, primary code, all codes, the first failing line, hits, teacher, finish reason, the raw
  teacher text (for measuring checker false rejects in the pilot)."""
import time

import golds

PASS = "R0"
TOKENS_PER_WORD = 1.3
TOKEN_FLAG = "words x 1.3 (no Planck tokenizer yet)"
OK_LICENSES = {"Apache-2.0"}


def est_tokens(turns):
    return round(sum(len(t["text"].split()) for t in turns) * TOKENS_PER_WORD)


def spans(text, slots):
    out = []
    for sid, s in slots.items():
        if not s.get("value"):
            continue
        for m in golds.value_re(s["value"]).finditer(text):
            out.append({"start": m.start(), "end": m.end(), "slot_id": sid, "type": s["type"]})
    return sorted(out, key=lambda x: (x["start"], x["end"]))


def author(t, teacher_model):
    if t["role"] == "tool" or t.get("lookup_call"):
        return "program"
    if t["mode"] == "exact":
        return f"bank:{t.get('bank_ref') or 'line'}"
    return f"teacher:{teacher_model}"


def blocked(skel, built, teacher):
    out = []
    if skel.get("provenance", {}).get("fake") or built.get("provenance") == "FAKE" or teacher["license"] == "FAKE":
        out.append("FAKE_PROVENANCE")
    elif teacher["license"] not in OK_LICENSES:
        out.append("LICENSE")
    if skel.get("rc12", "pending") != "clean":
        out.append("DECON_PENDING")
    return out


def teacher_meta(client, call, built, seed):
    return {"model": (call or {}).get("model") or client.model, "license": client.license, "mode": client.mode,
            "stub": client.is_stub, "prompt_variant": built["variant"], "prompt_sha256": built["prompt_sha256"],
            "render_seed": seed}


def accepted(skel, built, res, call, attempt, teacher, turns, dedup_keys=None):
    """turns = checker.record_turns(skel, res) (exact lines stored as the bank line itself)."""
    by_i = {t["i"]: t for t in skel["turns"]}
    out_turns = []
    for i, t in enumerate(turns):
        st = by_i[i]
        out_turns.append({"role": t["role"], "text": t["text"], "mask": t["mask"],
                          "author": author(st, teacher["model"]), "spans": spans(t["text"], skel["slots"])})
    notes = {str(t["i"]): t["gold_note"] for t in skel["turns"] if t.get("gold_note")}
    reasons = blocked(skel, built, teacher)
    return {
        "conv_id": f"{skel['skel_id']}.a{attempt}", "skel_id": skel["skel_id"], "attempt": attempt, "pass": PASS,
        "parent": None, "register": skel["register"], "slice": skel["slice"], "topic_path": skel["topic_path"],
        "user": {"style": skel["user"].get("style"), "age_band": skel["user"].get("age_band")},
        "system": skel["assistant"].get("system_text"), "turns": out_turns,
        "slots": {k: {"type": v["type"], "value": v["value"], "owner": v.get("owner"), "nonce": v.get("nonce")}
                  for k, v in skel["slots"].items()},
        "events": [{"id": e["id"], "kind": e["kind"], "params": e["params"], "gold": e.get("gold"),
                    "turns": e.get("turns")} for e in skel["events"]],
        "gold_notes": notes, "provenance": {"skeleton": skel.get("provenance"), "teacher": teacher,
                                            "gen_version": skel.get("gen_version")},
        "check": {"codes": res["codes"], "drift": res.get("drift", []), "finish": (call or {}).get("finish")},
        "heldout_gate": skel.get("heldout_gate"), "rc12": skel.get("rc12", "pending"),
        "est_tokens": est_tokens(out_turns), "est_tokens_flag": TOKEN_FLAG,
        "trainable": not reasons, "blocked": reasons, "time": round(time.time(), 3),
        "dedup": {"exact": dedup_keys[0]} if dedup_keys else None,
    }


def _first_line(skel, res):
    for code, i, detail in res.get("hits", []):
        if i is not None and i in res.get("turns", {}):
            return {"turn": i, "text": res["turns"][i], "detail": detail}
    h = res.get("hits") or [(None, None, None)]
    return {"turn": None, "text": None, "detail": h[0][2]}


def reject(skel, attempt, primary, codes, teacher, call=None, res=None, raw=None, extra=None):
    res = res or {}
    return {"skel_id": skel["skel_id"], "attempt": attempt, "primary": primary, "codes": list(codes),
            "first_line": _first_line(skel, res) if res else None,
            "hits": [[c, i, str(d)[:300]] for c, i, d in res.get("hits", [])][:20],
            "teacher": teacher, "finish": (call or {}).get("finish"), "usage": (call or {}).get("usage"),
            "requests": (call or {}).get("requests"), "raw": raw, "time": round(time.time(), 3), **(extra or {})}


def dbin(d):
    if d is None:
        return "-"
    return "d1-2" if d <= 2 else ("d3-5" if d <= 5 else "d6-10")


def cells(skel):
    """yield cells of one attempt: (kind, variant, d bin) per event, so yield is logged per kind x variant x d."""
    return [(e["kind"], e["params"].get("variant", "-"), dbin(e["params"].get("d"))) for e in skel["events"]]
