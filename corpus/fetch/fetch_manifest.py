"""fetch_manifest.py: download every file in a manifest (make_starter_manifest.py) with resume, check
its sha256 against the Hugging Face LFS hash, and write <dest>/FETCHED.json. Runs on the PC, never on
the Mac (PLAN: downloads go to the PC).

  python corpus/fetch/fetch_manifest.py corpus/fetch/starter_manifest.json ~/planck/data/raw/starter
"""
import hashlib, json, os, subprocess, sys, time


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    man, dest = sys.argv[1], os.path.expanduser(sys.argv[2])
    m = json.load(open(man))
    done, bad = [], []
    for i, e in enumerate(m["files"], 1):
        out = os.path.join(dest, e["dataset"].replace("/", "__"), e["path"])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if os.path.exists(out) and os.path.getsize(out) == e["size"] and sha256(out) == e["sha256"]:
            done.append(e["path"]); continue
        for attempt in range(1, 6):
            r = subprocess.run(["curl", "-sSfL", "--retry", "5", "-C", "-", "-o", out, e["url"]])
            if r.returncode == 0 and os.path.getsize(out) == e["size"]:
                break
            if r.returncode == 33 or (os.path.exists(out) and os.path.getsize(out) > e["size"]):
                os.remove(out)   # a range resume that cannot work: start over
            time.sleep(10 * attempt)
        ok = os.path.exists(out) and os.path.getsize(out) == e["size"] and sha256(out) == e["sha256"]
        (done if ok else bad).append(e["path"])
        print(f"[{i}/{len(m['files'])}] {'ok ' if ok else 'BAD'} {e['size'] / 1e6:8.1f} MB {e['dataset']}/{e['path']}", flush=True)
    json.dump(dict(manifest=os.path.abspath(man), n_ok=len(done), bad=bad,
                   finished=time.strftime("%Y-%m-%dT%H:%M:%S")), open(os.path.join(dest, "FETCHED.json"), "w"), indent=1)
    print(f"done: {len(done)} ok, {len(bad)} bad {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
