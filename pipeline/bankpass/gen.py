"""Run a bank plan against a teacher client (BANKPASS s2b, s2i, s6). One JSONL call record per finished call (fsync);
a restart skips done call ids; a hold stops at 33 minutes so the next hold resumes.

DRY BY DEFAULT. Bank items are training-text material, so a non-dry run needs Max's OK (BANKPASS s8 item 2): mode
"real" is refused unless max_ok names him and the date, and every record carries it. A dry run must write a
.dry.jsonl file under a "dry" directory; no run may write under planck/data or any sealed/ directory.

client protocol: client.generate(prompt, decode_spec, seed) -> {"text": str, "model": str}; the model must be the
planned teacher (a mismatch is an error, never a silent relabel)."""
import os
import time

from bankpass import prompts, store

HOLD_S = 33 * 60


class Refused(RuntimeError):
    pass


def check_out_path(path, mode, max_ok=None):
    """the output guard. Raises Refused."""
    p = os.path.abspath(os.path.expanduser(path))
    parts = p.split(os.sep)
    if "sealed" in parts:
        raise Refused("never write under sealed/")
    if any(parts[i] == "planck" and i + 1 < len(parts) and parts[i + 1] == "data" for i in range(len(parts))):
        raise Refused("bank pass output never goes under planck/data")
    if mode == "dry":
        if not p.endswith(".dry.jsonl") or "dry" not in parts[:-1]:
            raise Refused("a dry run writes <...>/dry/<name>.dry.jsonl")
    elif mode == "real":
        if not (isinstance(max_ok, dict) and max_ok.get("by") == "Max" and max_ok.get("date")):
            raise Refused("a non-dry bank generation needs Max's OK (BANKPASS s8 item 2)")
        if "dry" in parts:
            raise Refused("a real run does not write under a dry directory")
    else:
        raise Refused(f"unknown mode {mode!r}")
    return p


def literals_of(call):
    """the value each line must hold literally (plant, corr, twin: the value), and the values a query must not say."""
    role = call["bank"].split(".")[2] if call["class"] == "K" else None
    if role in ("plant", "corr", "twin"):
        return [f["v"] for f in call["fills"]], []
    if role in ("query", "bait"):
        return [None] * call["n"], [f["v"] for f in call["fills"]]
    return [None] * call["n"], []


def run_call(client, call):
    """one call -> its call record (lines parsed; nothing templatized or gated here)."""
    prompt = prompts.prompt_for(call)
    lits, banned = literals_of(call)
    verb = bool(call.get("verbalized")) or call["class"] == "T"
    spec = prompts.decode_spec(call["teacher"], lits, verbalized=verb, banned=banned)
    t0 = time.time()
    out = client.generate(prompt, spec, call["seed"])
    if out.get("model") != call["teacher"]:
        raise RuntimeError(f"{call['call_id']}: client answered as {out.get('model')!r}, planned {call['teacher']!r}")
    lines, likes, problem = prompts.parse_lines(out.get("text"), call["n"], verb)
    return {"schema": store.CALL_SCHEMA, "call_id": call["call_id"], "bank": call["bank"], "class": call["class"],
            "author": store.teacher_author(call["teacher"]), "prompt_id": f"{prompts.PROMPT_VERSION}:{call['bank']}",
            "prompt_sha256": store.sha256_text(prompt), "decode_sha256": spec["sha256"], "seed": call["seed"],
            "sampling": getattr(client, "sampling", None), "raw": out.get("text"),
            "raw_sha256": store.sha256_text(out.get("text") or ""), "lines": lines, "likelihoods": likes,
            "problem": problem, "time": round(t0, 3), "seconds": round(time.time() - t0, 3)}


def run(calls, client_for, out_path, mode="dry", max_ok=None, teacher=None, hold_s=HOLD_S, clock=time.monotonic):
    """run the calls of one teacher (or all) in plan order, skipping done ids, until done or the hold cap.
    -> {"done": n new, "skipped": n done before, "stopped": None | "HOLD_CAP", "problems": n, "not_ready": n}."""
    path = check_out_path(out_path, mode, max_ok)
    done = store.done_ids(path)
    start, stats = clock(), {"done": 0, "skipped": 0, "stopped": None, "problems": 0, "not_ready": 0}
    for call in calls:
        if teacher and call["teacher"] != teacher:
            continue
        if not call.get("ready", True):
            stats["not_ready"] += 1            # no prompt written for this bank yet: counted, never sent
            continue
        if call["call_id"] in done:
            stats["skipped"] += 1
            continue
        if clock() - start >= hold_s:
            stats["stopped"] = "HOLD_CAP"
            break
        rec = run_call(client_for(call["teacher"]), call)
        rec["mode"] = mode
        if mode == "real":
            rec["max_ok"] = max_ok
        else:
            rec["dry"] = True
        store.append_jsonl(path, rec)
        stats["done"] += 1
        stats["problems"] += rec["problem"] is not None
    return stats
