"""Bank pass store (BANKPASS s2h, s2i, s0 "Store"): one provenance record per item, kept AND dropped, as JSONL with
an fsync per append; call records for restart (a restart skips done call ids, as driver_state does); the per-bank
manifest (sha256, counts, author shares, gate hash) and the "<bank>@<sha>:<author mix>" ref a skeleton records.

Author kinds: teacher (one of the three pinned Apache-2.0 teachers), human (an open source with url, license,
retrieval date, file sha256, row), computed (code sha256 + input manifest sha256), program (closed lists), fake
(Claude-written test stand-ins; admit refuses them in any non-fixture bank)."""
import hashlib
import json
import os
import re
import unicodedata

SCHEMA = "bankpass-item-v1"
CALL_SCHEMA = "bankpass-call-v1"
AUTHOR_KINDS = ("teacher", "human", "computed", "program", "fake")
STATUSES = ("kept", "dropped")
# hf_pins.json (2026-09-25): official repo and revision; the served weights are the pinned 4-bit quants
TEACHERS = {
    "gemma-4-12b": dict(repo="google/gemma-4-12B-it", revision="707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
                        quant="google/gemma-4-12B-it-qat-w4a16-ct", license="apache-2.0"),
    "ministral-3-8b": dict(repo="mistralai/Ministral-3-8B-Instruct-2512",
                           revision="5b26027e7b19eeb4b7352e1fed3926375dd2cb4d",
                           quant="cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-4bit", license="apache-2.0"),
    "qwen3.5-9b": dict(repo="Qwen/Qwen3.5-9B", revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
                       quant="cyankiwi/Qwen3.5-9B-AWQ-4bit", license="apache-2.0"),
}
SHORT = {"gemma-4-12b": "g", "ministral-3-8b": "m", "qwen3.5-9b": "q"}
# D8 audit table: the licenses an item may carry, by author kind
HUMAN_LICENSES = ("public-domain", "cc0", "cc-by-4.0", "cc-by-3.0", "apache-2.0")
QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "ʼ": "'",
                        "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"'})
_WS = re.compile(r"\s+")


def straight(text):
    """NFKC, straight quotes, one space between words (the round 1 curly-quote finding: text is STORED straight)."""
    return _WS.sub(" ", unicodedata.normalize("NFKC", text or "").translate(QUOTES)).strip()


def norm_key(text):
    """exact-dedup key: straight text, lowercase, holes normalized to {h}, end punctuation dropped."""
    t = re.sub(r"\{\w+\}", "{h}", straight(text).lower())
    return re.sub(r"[.!?]+$", "", t).strip()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_text(s):
    return sha256_bytes(s.encode("utf-8"))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def item_id(cls, bank, n):
    return f"{cls}.{bank}.{n}"


def teacher_author(model, quant=True):
    if model not in TEACHERS:
        raise ValueError(f"not a pinned teacher: {model!r}")
    t = TEACHERS[model]
    return {"kind": "teacher", "model": model, "repo": t["repo"], "revision": t["revision"],
            "quant": t["quant"] if quant else None, "license": t["license"]}


def make_item(cls, bank, n, text, author, status="kept", drop=None, holes=None, features=None, call=None,
              seeds=None, judges=None, gates=None, dedup=None, source=None, extra=None):
    """one provenance record (s2h). text is stored straight; holes are the {names} in it."""
    if status not in STATUSES:
        raise ValueError(status)
    text = straight(text)
    rec = {"schema": SCHEMA, "id": item_id(cls, bank, n), "class": cls, "bank": bank, "text": text,
           "holes": sorted(set(re.findall(r"\{(\w+)\}", text))) if holes is None else sorted(holes),
           "features": features or {}, "status": status, "drop": drop, "author": author, "call": call or {},
           "seeds": seeds or {}, "judges": judges or [], "gates": gates or {}, "dedup": dedup or {}}
    if source is not None:
        rec["source"] = source
    if extra:
        rec.update(extra)
    return rec


def author_problems(a):
    """what is missing from an author block for its kind (the s2h fields)."""
    if not isinstance(a, dict) or a.get("kind") not in AUTHOR_KINDS:
        return ["author kind"]
    k, out = a["kind"], []
    if k == "teacher":
        t = TEACHERS.get(a.get("model"))
        if not t:
            return ["teacher not pinned"]
        if a.get("revision") != t["revision"]:
            out.append("teacher revision")
        if a.get("license") != "apache-2.0":
            out.append("teacher license")
    elif k == "human":
        src = a.get("source") or {}
        out += [f"human {f}" for f in ("url", "license", "retrieved", "file_sha256", "row") if src.get(f) in (None, "")]
        if src.get("license") not in HUMAN_LICENSES:
            out.append("human license")
    elif k == "computed":
        out += [f"computed {f}" for f in ("code_sha256", "input_sha256") if not a.get(f)]
    return out


# ---- JSONL ----------------------------------------------------------------------------------------------------------
def append_jsonl(path, rec):
    """append one record and fsync, so a crash loses at most the line being written."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    line = json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path, tolerate_torn_tail=True):
    """records of a JSONL file. A last line without its newline (a torn write) is skipped when tolerated, else an
    error; a bad line anywhere else is always an error."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    lines, out = raw.split("\n"), []
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        last = i == len(lines) - 1
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            if last and tolerate_torn_tail:
                break
            raise ValueError(f"{path}: bad JSON on line {i + 1}")
    return out


def done_ids(path, key="call_id"):
    return {r[key] for r in read_jsonl(path) if r.get(key) is not None}


def write_jsonl(path, recs):
    """write a whole file (a frozen bank), sorted by id, then fsync; returns its sha256."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in sorted(recs, key=lambda r: r["id"]):
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return sha256_file(path)


# ---- author mix, refs and the manifest --------------------------------------------------------------------------
def author_label(a):
    if a.get("kind") == "teacher":
        return a["model"]
    if a.get("kind") == "human":
        return "human:" + str((a.get("source") or {}).get("name") or "source")
    return a.get("kind", "unknown")


def author_shares(recs):
    kept = [r for r in recs if r["status"] == "kept"]
    n = len(kept) or 1
    out = {}
    for r in kept:
        lab = author_label(r["author"])
        out[lab] = out.get(lab, 0) + 1
    return {k: round(v / n, 4) for k, v in sorted(out.items())}


def mix_string(shares):
    """the provenance suffix: 'FAKE' if any share is fake, else 'g34m33q33' style for teachers, other kinds named."""
    if "fake" in shares:
        return "FAKE"
    if shares and all(lab in SHORT for lab in shares):
        return "".join(f"{SHORT[lab]}{round(100 * s)}" for lab, s in sorted(shares.items()))
    return ",".join(f"{lab}={round(100 * s)}" for lab, s in sorted(shares.items()))


def bank_ref(bank, sha, shares):
    return f"{bank}@{sha[:12]}:{mix_string(shares)}"


def build_manifest(bank_dir, version, gate_hash, rc12, banks, status="frozen", code_sha256=None, fixture=False,
                   aux=None, extra=None):
    """banks: {bank: {"class", "kind", "target"}} with <bank>.jsonl present in bank_dir; aux: the same for files the
    skeleton never installs (prompt-side seeds, rubric lists), checked by admit but not loaded; extra: more top-level
    fields (sources, code hashes). Writes manifest.json."""
    out = dict(extra or {}, schema="bankpass-manifest-v1", version=version, status=status, fixture=bool(fixture),
               gate_hash=gate_hash, rc12=rc12, code_sha256=code_sha256, banks={})
    if set(banks) & set(aux or {}):
        raise ValueError(f"listed as both bank and aux: {sorted(set(banks) & set(aux))}")
    if aux:
        out["aux"] = {}
    for bank, meta in sorted(list(banks.items()) + list((aux or {}).items())):
        path = os.path.join(bank_dir, bank + ".jsonl")
        recs = read_jsonl(path, tolerate_torn_tail=False)
        shares = author_shares(recs)
        drops = {}
        for r in recs:
            if r["status"] == "dropped":
                drops[r["drop"]] = drops.get(r["drop"], 0) + 1
        sha = sha256_file(path)
        part = "aux" if bank in (aux or {}) else "banks"
        out[part][bank] = dict(meta, file=bank + ".jsonl", sha256=sha, kept=sum(r["status"] == "kept" for r in recs),
                               dropped=sum(r["status"] == "dropped" for r in recs), drops=drops,
                               authors=shares, ref=bank_ref(bank, sha, shares))
    with open(os.path.join(bank_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    return out
