"""Run readers and file tools for the Planck model registry (registry.py). Stdlib only; never loads weights."""
import fcntl, fnmatch, glob, hashlib, json, os, re, struct, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
EMBED = re.compile(r"(embed_tokens|embed_in|embed_out|lm_head|\bwte\b|\bwpe\b|tok_emb|pos_emb|out_head)")
# the E002-E006 training scripts; eval jobs (gen_probe.py, run_chat.py, e005_al_ft_test.py) reuse the tag
TRAIN_SCRIPT = re.compile(r"^(e\d{3}_)?ft_test\.py$")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path, chunk=8 << 20):
    """sha256 of a file; on macOS the read bypasses the page cache (F_NOCACHE), so hashing GBs costs no RAM."""
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        if hasattr(fcntl, "F_NOCACHE"):
            fcntl.fcntl(f.fileno(), fcntl.F_NOCACHE, 1)
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def safetensors_numel(path):
    """{tensor name: element count} from the file's JSON header only (no tensor data is read)."""
    with open(path, "rb") as f:
        head = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    out = {}
    for k, v in head.items():
        if k != "__metadata__":
            n = 1
            for d in v["shape"]:
                n *= d
            out[k] = n
    return out


def param_counts(numel):
    """-> (total, body); body = total minus token/position embeddings and an untied output head."""
    total = sum(numel.values())
    return total, total - sum(v for k, v in numel.items() if EMBED.search(k))


def local_iso(stamp):
    """'2026-09-25 14:55:40' (this machine's local time, as the guards write it) -> ISO 8601 with offset."""
    t = time.mktime(time.strptime(stamp, "%Y-%m-%d %H:%M:%S"))
    off = time.localtime(t).tm_gmtoff
    return stamp.replace(" ", "T") + ("+" if off >= 0 else "-") + time.strftime("%H:%M", time.gmtime(abs(off)))


def hf_revision(model_id, before=None, hf_home=None):
    """-> (revision, source) from the local HF cache, or (None, reason). Accepted only when exactly one snapshot
    exists and (with before=epoch) it predates the run; the runs are HF_HUB_OFFLINE=1."""
    root = os.path.join(hf_home or os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"),
                        "hub", "models--" + model_id.replace("/", "--"), "snapshots")
    snaps = sorted(os.listdir(root)) if os.path.isdir(root) else []
    if len(snaps) != 1:
        return None, f"{len(snaps)} snapshots of {model_id} in the local HF cache"
    if before is not None and os.path.getmtime(os.path.join(root, snaps[0])) > before:
        return None, "the only cached snapshot is newer than the run"
    return snaps[0], "the only snapshot in the local HF cache, older than the run (runs use HF_HUB_OFFLINE=1)"


def run_args(cmd):
    """Guard cmd -> (script, args): everything after the first *.py (the interpreter and its flags dropped)."""
    i = next((i for i, a in enumerate(cmd) if a.endswith(".py")), None)
    return (None, []) if i is None else (os.path.basename(cmd[i]), cmd[i + 1:])


def config_sha256(script, args):
    return hashlib.sha256(json.dumps({"script": script, "args": args}, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def run_tag(args):
    """The tag an E002-E006 *_ft_test.py run writes under: --tag, else (dry_)base|s<seed> (ft_test.py's default)."""
    val = lambda k, d: args[args.index(k) + 1] if k in args else d
    return val("--tag", "") or (("dry_" if "--dry" in args else "") +
                                ("base" if val("--steps", "") == "0" else "s" + val("--seed", "0")))


def find_guard(exp_dir, model, tag):
    """The finished (exit 0, not killed) guard record of the training run that wrote <model>__<tag>, newest first."""
    hits = []
    for p in glob.glob(os.path.join(exp_dir, "logs", "*.guard.json")):
        try:
            g = read_json(p)
        except (OSError, ValueError):
            continue
        script, args = run_args(g.get("cmd") or [])
        if (TRAIN_SCRIPT.match(script or "") and args and args[0] == model and run_tag(args) == tag
                and g.get("exit") == 0 and not g.get("killed")):
            hits.append((g.get("ended") or "", p, g))
    if not hits:
        raise SystemExit(f"no finished guard record for {model} {tag} in {exp_dir}/logs")
    return sorted(hits)[-1][1:]


def cross_check(entries):
    """Errors between entries (registry.validate_all): one file at one location registered by two entries, or a
    checkpoint whose parent is not a registered id."""
    err, held = [], {}
    ids = {e.get("id") for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)}
    for e in (e for e in entries if isinstance(e, dict)):     # malformed fields are validate_entry's to report
        locs, files = (e.get(k) if isinstance(e.get(k), list) else [] for k in ("locations", "files"))
        for key in {f"{l}/{f.get('path')}" for l in locs for f in files if isinstance(l, str) and isinstance(f, dict)}:
            if key in held:
                err.append(f"{e.get('id')}: {key} is already registered by {held[key]}")
            held.setdefault(key, e.get("id"))
        if isinstance(e.get("parent"), str) and e["parent"] not in ids:
            err.append(f"{e.get('id')}: parent {e['parent']!r} is not a registered id")
    return err


def rel(path, repo=REPO):
    return os.path.relpath(os.path.abspath(path), repo)


def from_run(wdir, repo=REPO, hf_home=None):
    """A draft entry for a saved E002-E006 style run (weights/<slug>__<tag>); the caller sets id, arm, role,
    result, status, data, code, release and notes. Hashes every file in the directory."""
    wdir = os.path.abspath(wdir)
    exp_dir = os.path.dirname(os.path.dirname(wdir))
    stem = os.path.basename(wdir)
    tag = stem.rsplit("__", 1)[1]
    rj = read_json(os.path.join(exp_dir, "out", stem + "__run.json"))
    gpath, g = find_guard(exp_dir, rj["model"], tag)
    script, args = run_args(g["cmd"])
    started = time.mktime(time.strptime(g["started"], "%Y-%m-%d %H:%M:%S"))
    rev, rsrc = hf_revision(rj["model"], started, hf_home)
    files = [{"path": n, "bytes": os.path.getsize(os.path.join(wdir, n)), "sha256": sha256_file(os.path.join(wdir, n))}
             for n in sorted(os.listdir(wdir)) if os.path.isfile(os.path.join(wdir, n)) and not n.startswith(".")]
    pc = (rj.get("checks") or {}).get("params_config")
    total, body = param_counts(safetensors_numel(os.path.join(wdir, "model.safetensors")))
    psrc = "safetensors header (body = total minus embedding and untied head tensors)"
    if pc and pc.get("total") == total:
        body, psrc = pc.get("body"), "run.json checks.params_config (total equals the safetensors header)"
    tok = next((f["sha256"] for f in files if f["path"] == "tokenizer.json"), None)
    hp = {k: rj[k] for k in ("steps", "lr", "bs", "accum", "eff_batch", "max_len", "dtype", "device", "torch") if k in rj}
    exp = os.path.basename(exp_dir).split("_")[0]
    return {"schema": 1, "id": None, "experiment": exp, "arm": None, "role": "scored", "seed": rj.get("seed"),
            "kind": "finetune", "base_model": {"id": rj["model"], "revision": rev, "revision_source": rsrc},
            "params_total": total, "params_body": body, "params_source": psrc,
            "tokenizer": {"id": rj["model"] + ("@" + rev if rev else ""), "sha256": tok},
            "data": {"identity": rj.get("train_stream"), "sha256": None},
            "code": {"git_commit": None, "relation": None, "digest_sha256": None, "digest_source": None},
            "config_sha256": config_sha256(script, args), "config_source": "mac:" + rel(gpath, repo) + " cmd",
            "hparams": hp, "result": "not scored", "status": "diagnostic",
            "run": {"state": "completed", "started": local_iso(g["started"]), "ended": local_iso(g["ended"]),
                    "log": "mac:" + rel(gpath, repo)},
            "files": files, "locations": ["mac:" + rel(wdir, repo)], "created": local_iso(g["ended"]),
            "release": "undecided", "notes": ""}


def resolve(loc, roots, repo=REPO):
    """A location -> a local directory, or None when it is not on this machine. roots maps a location prefix to a
    local directory (e.g. {"pc:/mnt/d/planck-archive": "/Volumes/D/planck-archive"}); the longest prefix wins."""
    for pre in sorted(roots, key=len, reverse=True):
        if loc == pre or loc.startswith(pre.rstrip("/") + "/"):
            return roots[pre] + loc[len(pre.rstrip("/")):]
    kind, path = loc.split(":", 1)
    if kind == "mac":
        return os.path.join(repo, path)
    if kind == "pc" and "microsoft" in os.uname().release.lower():   # on the PC (WSL)
        return os.path.expanduser(path)
    return None


def verify(entries, where, roots=None, fill=False, idglob="*", repo=REPO, out=print):
    """Re-hash every file at each `where` location on this machine. -> (n_bad, filled). fill writes only null
    sha256/bytes; a location this machine cannot reach is reported NOT HERE and not counted."""
    bad = filled = 0
    for e in entries:
        if not fnmatch.fnmatch(e["id"], idglob):
            continue
        for loc in (l for l in e["locations"] if l.split(":", 1)[0] == where):
            d = resolve(loc, roots or {}, repo)
            if d is None:
                out(f"NOT HERE {e['id']}  {loc}  (not on this machine; map it with --root)")
                continue
            for f in e["files"]:
                p = os.path.join(d, f["path"])
                if not os.path.isfile(p):
                    out(f"MISSING  {e['id']}  {loc}/{f['path']}")
                    bad += 1
                    continue
                size, digest = os.path.getsize(p), sha256_file(p)
                if f["sha256"] is None and fill:
                    f["sha256"], f["bytes"], filled = digest, size, filled + 1
                    out(f"FILLED   {e['id']}  {loc}/{f['path']}")
                elif f["sha256"] is None:
                    out(f"UNHASHED {e['id']}  {loc}/{f['path']}  sha256 {digest}")
                elif (size, digest) != (f["bytes"], f["sha256"]):
                    out(f"MISMATCH {e['id']}  {loc}/{f['path']}")
                    bad += 1
                else:
                    out(f"OK       {e['id']}  {loc}/{f['path']}")
    return bad, filled
