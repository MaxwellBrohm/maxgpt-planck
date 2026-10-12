#!/usr/bin/env python3
"""Planck model registry: one JSON line per model in models/registry.jsonl. Policy and field meanings: MODELS.md.
  python3 models/registry.py validate
  python3 models/registry.py list [--experiment E] [--status S] [--kind K] [--id GLOB] [--json]
  python3 models/registry.py add --json FILE        (one JSON object, or JSON lines; every field explicit)
  python3 models/registry.py add --from-run WEIGHTS_DIR --id ID --set key=JSON ...   (an E002-E006 style HF run)
  python3 models/registry.py verify [--where mac|pc] [--id GLOB] [--root LOCPREFIX=DIR] [--fill]
  python3 models/registry.py hash FILE ...
Stdlib only. Nothing here imports torch or loads weights: parameter counts come from safetensors headers.
"""
import argparse, fnmatch, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import regtools as T  # noqa: E402
REGISTRY = os.path.join(HERE, "registry.jsonl")
N = type(None)
TYPES = {"schema": int, "id": str, "experiment": str, "arm": (str, N), "role": str, "seed": (int, N), "kind": str,
         "parent": str, "step": int, "base_model": (dict, N), "params_total": (int, N), "params_body": (int, N),
         "params_source": (str, N), "tokenizer": dict, "data": dict, "code": dict, "config_sha256": (str, N),
         "config_source": (str, N), "hparams": dict, "result": str, "status": str, "run": dict, "files": list,
         "locations": list, "created": (str, N), "release": str, "notes": str}
OPTIONAL = {"parent", "step", "hparams", "params_source", "config_source"}
SUBKEYS = {"base_model": {"id", "revision", "revision_source"}, "tokenizer": {"id", "sha256"},
           "data": {"identity", "sha256"}, "code": {"git_commit", "relation", "digest_sha256", "digest_source"},
           "run": {"state", "started", "ended", "log"}}
KINDS = {"finetune", "from_scratch", "checkpoint"}
STATUSES = {"success", "failure", "partial", "diagnostic", "not_saved"}
RUN_STATES = {"completed", "killed", "dry", "in_progress", "unknown"}
RELEASES = {"with_writeup", "never", "undecided"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{7,40}$")
ISO = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d$")
ID_RE = re.compile(r"^[A-Za-z0-9]+(/[A-Za-z0-9_.=+-]+)+$")
LOC = re.compile(r"^(mac:(?!/)(?!.*\.\.)\S+|pc:(~/planck|/mnt/d/planck-archive)(/\S*)?|hf:[\w.-]+/[\w.-]+(@\w+)?)$")
PRIVATE = re.compile(r"/Users/|/home/|\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|[\w.-]+@[\w-]+\.(com|net|org|edu)")
# a null in one of these must be explained in notes, by naming the field
REASONED = ["params_total", "params_body", "config_sha256", "created", "tokenizer.id", "tokenizer.sha256",
            "data.identity", "data.sha256", "base_model.revision"]
_ABSENT = object()


def _get(e, dotted):
    for part in dotted.split("."):
        if not isinstance(e, dict) or part not in e:
            return _ABSENT
        e = e[part]
    return e


def validate_entry(e):
    if not isinstance(e, dict):  # -> list of error strings (empty = valid)
        return ["entry is not a JSON object"]
    err = [f"missing {k}" for k in TYPES if k not in OPTIONAL and k not in e]
    err += [f"unknown key {k}" for k in e if k not in TYPES]
    err += [f"{k} has type {type(v).__name__}" for k, v in e.items() if k in TYPES and
            (not isinstance(v, TYPES[k]) or (isinstance(v, bool) and TYPES[k] is not bool))]
    if err:
        return err
    for k, want in SUBKEYS.items():
        if isinstance(e.get(k), dict) and set(e[k]) != want:
            err.append(f"{k} keys {sorted(e[k])} != {sorted(want)}")
    if err:
        return err
    if e["schema"] != 1:
        err.append("schema must be 1")
    if not ID_RE.match(e["id"]) or not e["id"].startswith(e["experiment"] + "/"):
        err.append(f"bad id {e['id']!r} (EXPERIMENT/part/..., starting with the experiment)")
    for k, allowed in (("kind", KINDS), ("status", STATUSES), ("release", RELEASES)):
        if e[k] not in allowed:
            err.append(f"{k} {e[k]!r} not in {sorted(allowed)}")
    if e["run"]["state"] not in RUN_STATES:
        err.append(f"run.state {e['run']['state']!r} not in {sorted(RUN_STATES)}")
    if e["kind"] == "finetune" and not (e["base_model"] and e["base_model"]["id"]):
        err.append("a finetune needs base_model.id")
    if e["kind"] == "from_scratch" and e["base_model"] is not None:
        err.append("a from_scratch model has base_model null")
    if e["kind"] == "checkpoint" and not ("parent" in e and "step" in e):
        err.append("a checkpoint needs parent and step")
    for k in ("tokenizer.sha256", "data.sha256", "code.digest_sha256", "config_sha256"):
        v = _get(e, k)
        if v is not None and v is not _ABSENT and not HEX64.match(str(v)):
            err.append(f"{k} is not a sha256 hex digest")
    c = e["code"]
    if c["git_commit"] is not None and not COMMIT.match(str(c["git_commit"])):
        err.append("code.git_commit is not a commit hash")
    if (c["git_commit"] is None) != (c["relation"] is None) or c["relation"] not in (None, "pre_run", "post_run"):
        err.append("code.relation is pre_run or post_run when code.git_commit is set, else null")
    if c["git_commit"] is None and c["digest_sha256"] is None and "code.git_commit" not in e["notes"]:
        err.append("no code commit and no code digest: notes must say why (name code.git_commit)")
    for k in ("created", "run.started", "run.ended"):
        v = _get(e, k)
        if v is not None and not ISO.match(str(v)):
            err.append(f"{k} is not ISO 8601 with an offset")
    paths = []
    for f in e["files"]:
        if not isinstance(f, dict) or set(f) != {"path", "bytes", "sha256"}:
            err.append(f"file entry {f!r} needs exactly path, bytes, sha256")
            continue
        p = f["path"]
        if not isinstance(p, str) or not p or p.startswith("/") or ".." in p.split("/"):
            err.append(f"file path {p!r} must be relative to its location")
        paths.append(p)
        if f["bytes"] is not None and (not isinstance(f["bytes"], int) or f["bytes"] < 0):
            err.append(f"{p}: bytes must be a non-negative int or null")
        if f["sha256"] is not None and not HEX64.match(str(f["sha256"])):
            err.append(f"{p}: sha256 is not a hex digest")
        if (f["sha256"] is None or f["bytes"] is None) and "files.sha256" not in e["notes"]:
            err.append(f"{p}: unhashed file needs a reason in notes (name files.sha256)")
    if len(paths) != len(set(paths)):
        err.append("duplicate file paths")
    saved = e["status"] != "not_saved"
    if saved != bool(e["files"]) or saved != bool(e["locations"]):
        err.append("status not_saved <=> files == [] and locations == []")
    for loc in e["locations"]:
        if not isinstance(loc, str) or not LOC.match(loc):
            err.append(f"bad location {loc!r} (mac:<repo path>, pc:~/planck/..., pc:/mnt/d/planck-archive/..., hf:)")
    if len(e["locations"]) != len(set(map(str, e["locations"]))):
        err.append("duplicate locations")
    for k in REASONED:
        if _get(e, k) is None and k not in e["notes"]:
            err.append(f"{k} is null: notes must say why (name {k})")
    if e["release"] == "never" and "release" not in e["notes"]:
        err.append("release never: notes must give the rule (name release)")
    if not e["result"].strip():
        err.append("result is empty (write 'not scored')")
    if PRIVATE.search(json.dumps(e)):
        err.append("private path, address or email in the entry")
    return err


def load(path=REGISTRY):
    out = []
    for i, line in enumerate(open(path, encoding="utf-8") if os.path.exists(path) else [], 1):
        try:
            out.append(json.loads(line)) if line.strip() else None
        except json.JSONDecodeError as x:
            raise SystemExit(f"{path}:{i}: not JSON ({x})")
    return out


def validate_all(entries):
    err, seen = [], set()
    for i, e in enumerate(entries, 1):
        eid = e.get("id") if isinstance(e, dict) and isinstance(e.get("id"), str) else None  # a list id is unhashable
        err += [f"line {i} ({eid or '?'}): {m}" for m in validate_entry(e)]
        if eid is not None and eid in seen:
            err.append(f"line {i}: duplicate id {eid}")
        seen.add(eid)
    return err + T.cross_check(entries)


def dumps(e):
    return json.dumps({k: e[k] for k in TYPES if k in e}, separators=(", ", ": "))


def write_all(entries, path=REGISTRY):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(dumps(e) + "\n" for e in entries)
    os.replace(tmp, path)


def append(new, path=REGISTRY):
    """Validate the new entries against the schema and the existing ids, then append them."""
    err = validate_all(load(path) + list(new))
    if err:
        raise ValueError("\n".join(err))
    with open(path, "a", encoding="utf-8") as f:
        f.writelines(dumps(e) + "\n" for e in new)


def set_field(e, dotted, value):
    *head, last = dotted.split(".")
    for part in head:
        e = e.setdefault(part, {})
    e[last] = value


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=REGISTRY)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    ls = sub.add_parser("list")
    [ls.add_argument("--" + k) for k in ("experiment", "status", "kind")]
    ls.add_argument("--id", default="*")
    ls.add_argument("--json", action="store_true")
    ad = sub.add_parser("add")
    [ad.add_argument(k) for k in ("--json", "--from-run", "--id")]
    ad.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    ve = sub.add_parser("verify")
    ve.add_argument("--where", default="mac", choices=["mac", "pc", "hf"])
    ve.add_argument("--id", default="*")
    ve.add_argument("--root", action="append", default=[], metavar="LOCPREFIX=DIR")
    ve.add_argument("--fill", action="store_true")
    sub.add_parser("hash").add_argument("files", nargs="+")
    a = ap.parse_args(argv)
    entries = load(a.registry) if a.cmd != "hash" else []
    if a.cmd == "validate":
        err = validate_all(entries)
        print("\n".join(err) if err else f"OK: {len(entries)} entries, schema 1, unique ids")
        return 1 if err else 0
    if a.cmd == "list":
        rows = [e for e in entries if fnmatch.fnmatch(e["id"], a.id) and a.experiment in (None, e["experiment"])
                and a.status in (None, e["status"]) and a.kind in (None, e["kind"])]
        size = lambda es: sum(f["bytes"] or 0 for e in es for f in e["files"])
        for e in rows:
            print(dumps(e) if a.json else f"{e['id']:<44} {e['status']:<10} {e['kind']:<12} "
                  f"{e['params_total'] or '-':>11} {size([e]) / 1e6:>9.1f} MB  {' '.join(e['locations']) or '-'}")
        if not a.json:
            print(f"{len(rows)} entries, {sum(1 for e in rows if e['files'])} with files, {size(rows) / 1e9:.3f} GB")
        return 0
    if a.cmd == "add":
        if bool(a.json) == bool(a.from_run):
            ap.error("add needs exactly one of --json FILE or --from-run WEIGHTS_DIR")
        if a.json:
            text = open(a.json, encoding="utf-8").read().strip()
            one = text.startswith("{") and "\n{" not in text
            new = [json.loads(text)] if one else [json.loads(l) for l in text.splitlines() if l.strip()]
        else:
            new = [dict(T.from_run(a.from_run), id=a.id)]
        for kv in a.set:
            k, v = kv.split("=", 1)
            [set_field(e, k, json.loads(v)) for e in new]
        try:
            append(new, a.registry)
        except ValueError as x:
            print(f"NOT ADDED:\n{x}")
            return 1
        print(f"added {len(new)}: {', '.join(e['id'] for e in new)}")
        return 0
    if a.cmd == "verify":
        bad, filled = T.verify(entries, a.where, dict(r.rsplit("=", 1) for r in a.root), a.fill, a.id)
        err = validate_all(entries) if filled else []
        if err:
            print("NOT WRITTEN (the fill made the registry invalid):\n" + "\n".join(err))
            return 1
        if filled:
            write_all(entries, a.registry)
        print(f"{'FAIL' if bad else 'OK'}: {bad} missing or mismatched, {filled} filled")
        return 1 if bad else 0
    for p in a.files:
        print(f"{T.sha256_file(p)}  {os.path.getsize(p)}  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
