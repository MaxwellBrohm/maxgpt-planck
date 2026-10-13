"""RC-12 sealed-session kit, part 5: the S5 pack (prereg s13 S5). Stdlib, no model.

  python3 -B rc12/sealed_pack.py WORK --seed-file F --log LOG --oodh O --ngrams F.gz --out DIR

Writes DIR/RC-12-sealed.tar (deterministic) and DIR/SEALED_HASHES.txt (public). It refuses unless: the work copy
equals its commit (sealed_kit check); GATES.txt is a full, clean S4 table whose recorded inputs (items, content files,
expect.json, seed) are the files being packed; the seed rebuilds the items byte for byte; the draw log's last run
accepted this commit's seed; OVERLAP.txt is the S1 check of these items and passed (a dry-run split may pack a failed
S1, and says so in S5)."""
import argparse
import gzip
import io
import json
import os
import re
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sealed_crypto as CR                   # noqa: E402
import sealed_kit as K                       # noqa: E402


def files_for(work, seed_file, log, oodh, ngram_file):
    """(archive name, bytes) of every sealed file, in archive order."""
    def read(path):
        return open(path, "rb").read()
    out = [("rc12_sealed.jsonl", read(os.path.join(work, K.ITEMS)))]
    out += [(f"content_{m}.py", read(os.path.join(work, "rc12", f"content_{m}.py"))) for m in K.CONTENT_MODULES]
    out += [("expect.json", read(os.path.join(work, "expect.json"))), ("generator_seed.txt", read(seed_file)),
            ("GATES.txt", read(os.path.join(work, "GATES.txt"))),
            ("OVERLAP.txt", read(os.path.join(work, "OVERLAP.txt"))),
            ("SEED_ATTEMPTS.txt", read(log)), (CR.NGRAMS, read(ngram_file)), (CR.OODH, read(oodh))]
    return out


def det_tar(files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as t:
        for name, data in files:
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode, info.uid, info.gid, info.uname, info.gname = \
                len(data), 0, 0o644, 0, 0, "", ""
            t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def rebuild_items(work, seed_file, split):
    """the items the seed builds in the work copy (None if the build fails)."""
    p = subprocess.run([sys.executable, "-B", "build_dev.py", "--stdout"], cwd=os.path.join(work, "rc12"),
                       env=K.env_for(seed_file, split), capture_output=True)
    return None if p.returncode else p.stdout


def refusals(work, seed_file, log, files, split):
    """why these files may not be packed (an empty list: they may)."""
    bad, gates = [], open(os.path.join(work, "GATES.txt")).read()
    if not K.gates_clean(gates):
        bad.append("GATES.txt is not a full, clean S4 gate table (every S4 gate PASS)")
    recorded = dict(reversed(m.groups()) for m in re.finditer(r"^input sha256 ([0-9a-f]{64})  (\S+)", gates, re.M))
    data = dict(files)
    for name in ["rc12_sealed.jsonl", "expect.json", "generator_seed.txt"] + [f"content_{m}.py" for m in
                                                                             K.CONTENT_MODULES]:
        if recorded.get(name) != CR.sha(data[name]):
            bad.append(f"{name} is not the file the S4 gates ran on (GATES.txt input hashes)")
    commit = K.commit_of(work)
    if f"commit {commit} split {split}" not in gates.split("\n", 1)[0]:
        bad.append("GATES.txt was not written for this commit and split")
    if rebuild_items(work, seed_file, split) != data["rc12_sealed.jsonl"]:
        bad.append("the seed does not rebuild the items byte for byte")
    runs = open(log).read().split("\ncommit ")
    last = "commit " + runs[-1] if len(runs) > 1 else runs[0]
    accepted = re.findall(r"^accepted draw \d+; seed sha256 ([0-9a-f]{64})$", last, re.M)
    if not last.startswith(f"commit {commit} split {split}:") or len(accepted) != 1 or \
            accepted[0] != CR.sha(open(seed_file, "rb").read().strip()):
        bad.append("the draw log's last run is not one accepted run of this commit and split")
    s1 = data["OVERLAP.txt"].decode()
    if f"items sha256 {CR.sha(data['rc12_sealed.jsonl'])}" not in s1:
        bad.append("OVERLAP.txt is not the S1 check of these items")
    elif "S1 overall: PASS" not in s1 and split == "sealed":
        bad.append("S1 did not pass (OVERLAP.txt)")
    return bad


def pack(work, seed_file, log, oodh, ngram_file, out, repo=K.REPO):
    if K.check(work, repo=repo):
        K.die("the work copy differs from its commit; run sealed_kit.py check")
    files = files_for(work, seed_file, log, oodh, ngram_file)
    recs = [json.loads(x) for x in files[0][1].decode().splitlines()]
    split = recs[0]["split"]
    bad = refusals(work, seed_file, log, files, split)
    if bad:
        K.die("not packed: " + "; ".join(bad))
    tar = det_tar(files)
    K.check_place(out, repo)                        # the plaintext tar goes under sealed/ or outside the repo
    os.makedirs(out, exist_ok=True)
    CR.write_private(os.path.join(out, CR.TAR), tar)
    s1 = "S1 overall: PASS" in dict(files)["OVERLAP.txt"].decode()
    lines = ["RC-12 sealed split: S5 hashes (prereg s13). Public; the files are only in RC-12-sealed.tar.enc.",
             f"commit {K.commit_of(work)} (the work copy's code: sealed_kit check)",
             f"split {split}; records {len(recs)}; twelve-turn {sum(r['n_turns'] == 12 for r in recs)}; "
             f"S1 {'PASS' if s1 else 'NOT PASSED (a dry-run split)'}"]
    lines += [f"sha256 {CR.sha(data)}  {name}" for name, data in files]
    lines.append(f"sha256 {CR.sha(open(seed_file, 'rb').read().strip())}  generator seed (the hex string)")
    head = gzip.decompress(open(ngram_file, "rb").read()).split(b"\n", 1)[0].decode()
    count = re.search(r"; (\d+) hashes$", head).group(1)
    lines.append(f"decon13 hashes {count} (the S6 list's count; its sha256 is the decon13_sha256.txt.gz line)")
    lines.append(f"sha256 {CR.sha(tar)}  {CR.TAR} (the plaintext archive, deterministic)")
    with open(os.path.join(out, "SEALED_HASHES.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    for name in ("--seed-file", "--log", "--oodh", "--ngrams", "--out"):
        ap.add_argument(name, required=True)
    a = ap.parse_args()
    work = os.path.abspath(a.work)
    K.check_place(work)
    pack(work, a.seed_file, os.path.abspath(a.log), a.oodh, a.ngrams, a.out)


if __name__ == "__main__":
    main()
