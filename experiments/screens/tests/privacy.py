"""Private-identifier check for files this repo publishes (SCREENS.txt C9: "No machine address, user name or key path").
No personal name is written in this file. The patterns come from three places:
  generic  committed shapes: home directories (/Users/x, /home/x, C:\\Users\\x, /mnt/<drive>/), ssh key paths, IPv4
           addresses, e-mail addresses and the names of the PC access variables;
  runtime  this machine's user name and home directory, and, when pc/local.env exists (git-ignored by the repo's
           .gitignore), the PC's user, address and key path read from it;
  local    one regex per line of tests/private/patterns.txt when it exists (git-ignored: .gitignore "private/"), for
           names no runtime value gives (a first name, a host name). Lines starting with # are skipped.
hits() returns the labels of the patterns that match, never the matched text, so a failing test prints no identifier.
"""
from __future__ import annotations

import getpass
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
LOCAL = os.path.join(HERE, "private", "patterns.txt")
PC_ENV = os.path.join(ROOT, "pc", "local.env")
GENERIC = {
    "home /Users": r"/Users/[^/\s\"']+", "home /home": r"/home/[^/\s\"']+", "wsl drive": r"/mnt/[a-z]/",
    "windows home": r"\b[A-Za-z]:\\{1,2}Users\\{1,2}", "ssh dir": r"\.ssh\b", "ssh key": r"\bid_(?:rsa|ed25519|ecdsa)\b",
    "ipv4": r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])", "email": r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}\b",
    "pc env name": r"PLANCK_PC_(?:HOST|KEY)", "windows user var": r"%USERNAME%"}


def _runtime() -> dict:
    vals = {"mac user": getpass.getuser(), "mac home": os.path.expanduser("~")}
    if os.path.isfile(PC_ENV):
        env = dict(ln.strip().split("=", 1) for ln in open(PC_ENV) if "=" in ln and not ln.lstrip().startswith("#"))
        host = env.get("PLANCK_PC_HOST", "").strip("\"'")
        key = env.get("PLANCK_PC_KEY", "").strip("\"'")
        user, _, addr = host.rpartition("@")
        vals.update({"pc user": user, "pc address": addr, "pc key": key, "pc key file": os.path.basename(key)})
    return {k: re.escape(v) for k, v in vals.items() if len(v) >= 3}


def _local() -> dict:
    if not os.path.isfile(LOCAL):
        return {}
    lines = [ln.strip() for ln in open(LOCAL, encoding="utf-8")]
    return {f"local {i}": ln for i, ln in enumerate(lines, 1) if ln and not ln.startswith("#")}


def patterns() -> list[tuple[str, str, re.Pattern]]:
    """(source, label, compiled) for every pattern: generic, runtime, local."""
    out = [("generic", k, re.compile(v, re.I)) for k, v in GENERIC.items()]
    out += [("runtime", k, re.compile(v, re.I)) for k, v in _runtime().items()]
    out += [("local", k, re.compile(v, re.I)) for k, v in _local().items()]
    return out


def hits(text: str, pats=None) -> list[str]:
    """Labels of the patterns found in text (empty = clean)."""
    return [f"{src} {label}" for src, label, rx in (pats or patterns()) if rx.search(text)]
