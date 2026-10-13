"""RC-12 sealed-session kit, part 2: seed, key, hashes, 13-grams, encryption and the decryption check (prereg s13
S3, S5-S8). Stdlib plus the openssl command line (OpenSSL 3; the Mac's Homebrew openssl and WSL's both work).

  seed F                    a fresh generator seed (32 random bytes, hex) into F; refuses an existing F
  keygen KEY                a random 256-bit key (openssl rand -hex 32) into KEY, mode 600, outside any git work
                            tree; prints only a fingerprint (sha256 of the key file, first 16 hex) to compare copies
  ngrams WORK --oodh O --out F.gz
                            S6: sha256 of every normalized word 13-gram of sealed user text and OOD-H Part 1 text
  (pack, S5: sealed_pack.py)
  encrypt TAR --key KEY --out RC-12-sealed.tar.enc --hashes SEALED_HASHES.txt
                            S7: openssl enc -aes-256-cbc -salt -pbkdf2 -iter 600000 -md sha256 -pass file:KEY; the
                            ciphertext's sha256 is appended to the S5 file
  verify ARCHIVE --key KEY --hashes SEALED_HASHES.txt
                            decrypts into a temporary directory, checks every file against S5, deletes it
The key never enters a repository, a notes file, a transcript or a command line (only its path does)."""
import argparse
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sealed_kit as K                      # noqa: E402

ENC = ["-aes-256-cbc", "-salt", "-pbkdf2", "-iter", "600000", "-md", "sha256"]
TAR = "RC-12-sealed.tar"
OODH = "oodh_part1.jsonl"
ENC_NAME = "RC-12-sealed.tar.enc"
NGRAMS = "decon13_sha256.txt.gz"


def die(msg):
    sys.exit(f"sealed_crypto: {msg}")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def in_git_tree(path):
    d = os.path.dirname(os.path.abspath(path)) or "."
    while not os.path.isdir(d):
        d = os.path.dirname(d)
    p = subprocess.run(["git", "-C", d, "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True)
    return p.returncode == 0 and p.stdout.strip() == "true"


def write_private(path, data):
    if os.path.exists(path):
        die(f"{path} exists; never overwritten")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)


def seed(path):
    write_private(path, os.urandom(32).hex().encode() + b"\n")
    print(f"seed written to {path} (not printed)")


def keygen(path):
    if in_git_tree(path):
        die(f"{path} is inside a git work tree; the key lives outside every repository (S7, S8)")
    key = subprocess.run(["openssl", "rand", "-hex", "32"], capture_output=True, check=True).stdout
    if not re.fullmatch(rb"[0-9a-f]{64}\n", key):
        die("openssl rand gave an unexpected key format")
    write_private(path, key)
    print(f"key written to {path}; fingerprint {sha(key)[:16]} (sha256 of the key file, first 16 hex)")


def ngrams(work, oodh, out, n=13):
    items = os.path.join(work, K.ITEMS)
    texts = [t["text"] for line in open(items, encoding="utf-8") for t in json.loads(line)["turns"]]   # user text
    for line in open(oodh, encoding="utf-8"):       # OOD-H records use the RC-12 format (oodh/build.py): the user
        for t in json.loads(line)["turns"]:          # turns, probe turns and the thread's own assistant replies,
            texts += [t["text"], t.get("ideal") or ""]   # which every model sees in its history
    hashes = set()
    for t in texts:
        w = re.findall(r"[a-z0-9']+", t.lower())
        hashes.update(sha(" ".join(w[i:i + n]).encode()) for i in range(len(w) - n + 1))
    head = (f"# RC-12 sealed + OOD-H Part 1: sha256 of every word {n}-gram, words = re.findall(r\"[a-z0-9']+\", "
            f"text.lower()) joined by one space; {len(hashes)} hashes\n")
    write_private(out, gzip.compress((head + "\n".join(sorted(hashes)) + "\n").encode(), mtime=0))
    print(f"{len(hashes)} {n}-gram hashes written to {out}")


def openssl(args, key):
    """runs openssl enc; True when it succeeded (a wrong key fails decryption: "bad decrypt")."""
    if not os.path.isfile(key):
        die(f"no key file at {key}")
    return subprocess.run(["openssl", "enc", *args, *ENC, "-pass", f"file:{key}"], capture_output=True).returncode == 0


def encrypt(tar, key, out, hashes):
    """S7; then the ciphertext's sha256 is appended to the S5 file, so the public archive is tied to S5."""
    if os.path.exists(out):
        die(f"{out} exists")
    if not re.search(rf"^sha256 {sha(open(tar, 'rb').read())}  {TAR} ", open(hashes).read(), re.M):
        die(f"{tar} is not the plaintext archive {hashes} names")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if not openssl(["-in", tar, "-out", out], key) or open(out, "rb").read(8) != b"Salted__":
        die("encryption failed or the output lacks the openssl salt header")
    with open(hashes, "a") as f:
        f.write(f"sha256 {sha(open(out, 'rb').read())}  {ENC_NAME} (the ciphertext)\n")
    print(f"encrypted {tar} -> {out}; its sha256 appended to {hashes}")


def verify(archive, key, hashes):
    text = open(hashes).read()
    want = {m.group(2): m.group(1) for m in re.finditer(r"^sha256 ([0-9a-f]{64})  (\S+)", text, re.M)}
    tmp = tempfile.mkdtemp(prefix="rc12v_")
    try:
        plain = os.path.join(tmp, TAR)
        if not openssl(["-d", "-in", archive, "-out", plain], key):
            print("verify: FAIL (decryption failed: wrong key or damaged archive)")
            return 1
        if sha(open(plain, "rb").read()) != want.get(TAR):
            print(f"verify: FAIL (the decrypted {TAR} is not the one S5 names)")
            return 1
        with tarfile.open(plain) as t:
            names = t.getnames()
            bad = [m.name for m in t.getmembers()
                   if not m.isfile() or want.get(m.name) != sha(t.extractfile(m).read())]
        bad += [n for n in want if n not in names and n not in (TAR, "generator", ENC_NAME)]
    finally:
        shutil.rmtree(tmp)
    print(f"verify: {len(names)} files, {'PASS' if not bad else 'FAIL ' + ', '.join(bad)}")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["seed", "keygen", "ngrams", "encrypt", "verify"])
    ap.add_argument("path")
    ap.add_argument("--oodh")
    ap.add_argument("--out")
    ap.add_argument("--key")
    ap.add_argument("--hashes")
    a = ap.parse_args()
    if a.cmd == "seed":
        seed(a.path)
    elif a.cmd == "keygen":
        keygen(a.path)
    elif a.cmd == "ngrams":
        ngrams(os.path.abspath(a.path), a.oodh, a.out)
    elif a.cmd == "encrypt":
        encrypt(a.path, a.key, a.out, a.hashes)
    else:
        sys.exit(verify(a.path, a.key, a.hashes))


if __name__ == "__main__":
    main()
