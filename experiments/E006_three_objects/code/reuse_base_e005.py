"""E005: reuse E004's untouched-model records instead of scoring the untouched model again (notes.txt UNCHANGED: "its
records are copied read-only with sha256 and not rerun"). No model is loaded; the SmolLM2 tokenizer is (chat prompts).
Every check must hold. Otherwise nothing is copied, the exit code is 1, and the queue scores the untouched model
itself (e005_ft_test.py --steps 0: the same calls as e004_ft_test.py, i.e. load, load_checks, score_set over
lik_sets, run_gen over gen_sets; with --steps 0 only the train branch and the eot check differ, and both are skipped).
  1 code     the 64 copies and their E004 sources match copied_from_e004_sha256.txt; every E004 .py file is older
             than the start of E004's base run and of its untouched chat probe (guard.json "started");
             e005_sets.lik_sets / gen_sets ARE the copied e004_sets functions
  2 runs     both guard verdicts: exit 0, killed None, the expected command line; base run.json: model, steps 0,
             tag base, dry False, chat True, sets cont/eval/know, max_new = gen_run.MAX_NEW, no fatal
  3 files    E004's base files are exactly the ones the plan writes (15 sets + run.json); eval_counts and line
             counts equal the plan's set sizes; the chat-probe transcript has 53 lines
  4 items    every set of the plan, built in two processes (one per code directory, python -B, PYTHONHASHSEED 0),
             hashes the same; E004's code directory listing is unchanged by its process
  5 records  every LIK record's h equals the prompt hash of the same E005 item (plain transcript, or the tokenizer's
             chat template through lik.render); every GEN record's report keys equal the item's
  6 weights  exactly one snapshot of the model in the HF cache, refs/main points at it, its files are older than
             E004's base run (the queue runs offline); snapshot id and model.safetensors sha256 logged
Then shutil.copy2 of the 16 base files into ../out and the chat-probe transcript into ../transcripts, and
../logs/base_reuse_sha256.txt (source sha256 = copy sha256 per file). An existing copy with other bytes: refused.
usage: <venv python> -B reuse_base_e005.py [--check-only] > ../logs/reuse_base_e005.stdout"""
import argparse, datetime, glob, hashlib, json, os, shutil, subprocess, sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
E4 = os.path.join(os.path.dirname(EXP), "E004_general_updating")
E4CODE, E4OUT = os.path.join(E4, "code"), os.path.join(E4, "out")
sys.path.insert(0, HERE)
SMOL = "HuggingFaceTB/SmolLM2-135M-Instruct"
SLUG = SMOL.replace("/", "__")
SETS = {"eval", "cont", "know"}
BASE_CMD = ["e004_ft_test.py", SMOL, "--steps", "0", "--sets", "eval,cont,know", "--chat", "--tag", "base"]
CHAT_CMD = ["run_chat.py", SMOL, "--mode", "greedy"]
TRANSCRIPT = f"{SLUG}__base__greedy.jsonl"
GEN_KEYS = ("family", "cell", "vtype", "k", "d", "var", "fam", "task", "idx", "latest_ref", "h")
SNIPPET = r"""
import hashlib, json, e004_sets as S
out = []
for kind, fn in (("lik", S.lik_sets), ("gen", S.gen_sets)):
    for name, render, its in fn({"eval", "cont", "know"}, True, False):
        blob = json.dumps(its, sort_keys=True, default=str).encode()
        out.append([kind, name, render, len(its), hashlib.sha256(blob).hexdigest()])
print(json.dumps(out))
"""


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ts(s):
    return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()


def lines(p):
    return [json.loads(x) for x in open(p) if x.strip()]


def listing(d):
    return sorted((f, os.path.getsize(os.path.join(d, f)), os.path.getmtime(os.path.join(d, f))) for f in os.listdir(d))


def set_hashes(code_dir):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    r = subprocess.run([sys.executable, "-B", "-c", SNIPPET], cwd=code_dir, capture_output=True, text=True, env=env,
                       timeout=300)
    if r.returncode:
        raise SystemExit(f"set build failed in {code_dir}: {r.stderr[-600:]}")
    return json.loads(r.stdout)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args(argv)
    fail = []
    ok = lambda c, m: (print(("PASS  " if c else "FAIL  ") + m), None if c else fail.append(m))
    import e004_sets as S4, e005_sets as S5, gen_run as R
    # 1 code
    rows = [ln.split() for ln in open(os.path.join(EXP, "logs", "copied_from_e004_sha256.txt")).read().splitlines()[3:]
            if ln.strip()]
    bad = [f for h, f in rows if sha(os.path.join(HERE, f)) != h or sha(os.path.join(E4CODE, f)) != h]
    ok(len(rows) == 64 and not bad, f"1 copies and E004 sources match the step-1 sha256 list: {64 - len(bad)}/{len(rows)}")
    gb = json.load(open(os.path.join(E4, "logs", "base_135m.guard.json")))
    gc = json.load(open(os.path.join(E4, "logs", "chat_135m_base.guard.json")))
    t0 = min(ts(gb["started"]), ts(gc["started"]))
    newer = [f for f in os.listdir(E4CODE) if f.endswith(".py") and os.path.getmtime(os.path.join(E4CODE, f)) >= t0]
    ok(not newer, f"1 every E004 .py file predates {min(gb['started'], gc['started'])}: newer {newer} (queue .sh files "
       f"are not imported)")
    ok(S5.lik_sets is S4.lik_sets and S5.gen_sets is S4.gen_sets, "1 e005_sets.lik_sets/gen_sets are e004_sets'")
    # 2 runs
    for g, cmd, name in ((gb, BASE_CMD, "base_135m"), (gc, CHAT_CMD, "chat_135m_base")):
        tail = g["cmd"][2:]
        ok(g.get("exit") == 0 and g.get("killed") is None and all(x in tail for x in cmd) and tail[0] == cmd[0],
           f"2 {name}: exit {g.get('exit')} killed {g.get('killed')} cmd {' '.join(tail)}")
    run = json.load(open(os.path.join(E4OUT, f"{SLUG}__base__run.json")))
    want = dict(model=SMOL, steps=0, tag="base", dry=False, chat=True, sets=sorted(SETS), max_new=R.MAX_NEW)
    ok(all(run.get(k) == v for k, v in want.items()) and "fatal" not in run,
       f"2 base run.json {({k: run.get(k) for k in want})} fatal={run.get('fatal')}")
    # 3 files
    lik, gen = S4.lik_sets(SETS, True, False), S4.gen_sets(SETS, True, False)
    plan = {f"{n}/{r}": len(i) for n, r, i in lik} | {f"gen_{n}/{r}": len(i) for n, r, i in gen}
    files = {f"{SLUG}__base__{k.replace('/', '__')}.jsonl": n for k, n in plan.items()}
    have = {os.path.basename(p) for p in glob.glob(os.path.join(E4OUT, f"{SLUG}__base__*"))}
    ok(have == set(files) | {f"{SLUG}__base__run.json"}, f"3 base files = plan: {len(have)} files ({len(plan)} sets + run.json)")
    ok(run.get("eval_counts") == plan, "3 run.json eval_counts = plan set sizes")
    cnt = {f: sum(1 for _ in open(os.path.join(E4OUT, f))) for f in files if f in have}
    ok(all(cnt.get(f) == n for f, n in files.items()), f"3 line counts = plan ({sum(cnt.values())} records)")
    tr = os.path.join(E4, "transcripts", TRANSCRIPT)
    ok(os.path.exists(tr) and sum(1 for _ in open(tr)) == 53, "3 untouched chat-probe transcript: 53 conversations")
    # 4 items, built once per code directory
    before = listing(E4CODE)
    h5, h4 = set_hashes(HERE), set_hashes(E4CODE)
    ok(listing(E4CODE) == before, "4 E004 code directory unchanged by its set-building process")
    ok(h5 == h4 and len(h5) == len(plan), f"4 set hashes E005 = E004: {sum(x == y for x, y in zip(h5, h4))}/{len(h4)}")
    # 5 records
    from transformers import AutoTokenizer
    import e004_core as C, lik as L
    tok = AutoTokenizer.from_pretrained(SMOL)
    nb = nl = 0
    for n, r, its in lik:
        fp = os.path.join(E4OUT, f"{SLUG}__base__{n}__{r}.jsonl")
        recs = lines(fp) if os.path.exists(fp) else []
        nb += abs(len(its) - len(recs))  # a missing or short file counts every unmatched item
        for i, (it, rec) in enumerate(zip(its, recs)):
            p = L.render(it, r, tok, SMOL)[0] if "turns" in it else it["prompt"]
            same = rec["id"] == i and rec["set"] == n and rec["render"] == r and rec["h"] == C.phash(p) and all(
                json.loads(json.dumps(it[k])) == rec.get(k) for k in C.REC_KEYS if k in it)
            nb += not same
            nl += 1
    ok(nb == 0, f"5 LIK records whose prompt hash or keys differ from the E005 item: {nb} of {nl}")
    nb = ng = 0
    for n, r, its in gen:
        fp = os.path.join(E4OUT, f"{SLUG}__base__gen_{n}__{r}.jsonl")
        recs = lines(fp) if os.path.exists(fp) else []
        nb += abs(len(its) - len(recs))
        for i, (it, rec) in enumerate(zip(its, recs)):
            same = rec["id"] == i and rec["render"] == r and rec["set"] == it["set"] and all(
                json.loads(json.dumps(it[k])) == rec.get(k) for k in GEN_KEYS if k in it)
            nb += not same
            ng += 1
    ok(nb == 0, f"5 GEN records whose report keys differ from the E005 item: {nb} of {ng}")
    # 6 weights
    from huggingface_hub import constants
    repo = os.path.join(constants.HF_HUB_CACHE, "models--" + SMOL.replace("/", "--"))
    snaps = sorted(os.listdir(os.path.join(repo, "snapshots")))
    ref = open(os.path.join(repo, "refs", "main")).read().strip()
    sdir = os.path.join(repo, "snapshots", snaps[0]) if snaps else ""
    late = [f for f in os.listdir(sdir) if os.path.getmtime(os.path.realpath(os.path.join(sdir, f))) >= t0] if sdir else ["?"]
    wsha = sha(os.path.join(sdir, "model.safetensors")) if sdir else None
    ok(len(snaps) == 1 and ref == snaps[0] and not late,
       f"6 one snapshot {snaps} = refs/main {ref[:12]}, files newer than the base run: {late}; model.safetensors {wsha}")
    if fail:
        print(f"\nREUSE REFUSED ({len(fail)} failed): nothing copied")
        return 1
    if a.check_only:
        print("\nREUSE OK (check only; nothing copied)")
        return 0
    pairs = [(os.path.join(E4OUT, f), os.path.join(EXP, "out", f)) for f in sorted(have)]
    pairs.append((tr, os.path.join(EXP, "transcripts", TRANSCRIPT)))
    for src, dst in pairs:
        if os.path.exists(dst) and sha(dst) != sha(src):
            print(f"\nREUSE REFUSED: {dst} exists with other bytes")
            return 1
    out = [f"E005 base reuse, {datetime.datetime.now():%Y-%m-%d %H:%M:%S}: E004's untouched-model records copied read-only "
           f"(shutil.copy2). Checks: logs/reuse_base_e005.stdout. Weights snapshot {snaps[0]}, model.safetensors {wsha}.",
           "sha256  source (E004)  copy (E005)"]
    for src, dst in pairs:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        hs, hd = sha(src), sha(dst)
        if hs != hd:
            print(f"\nREUSE REFUSED: copy of {src} differs")
            return 1
        out.append(f"{hs}  {os.path.relpath(src, os.path.dirname(EXP))}  {os.path.relpath(dst, os.path.dirname(EXP))}")
    open(os.path.join(EXP, "logs", "base_reuse_sha256.txt"), "w").write("\n".join(out) + "\n")
    print(f"\nREUSE OK: {len(pairs)} files copied, sha256 source = copy (logs/base_reuse_sha256.txt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
