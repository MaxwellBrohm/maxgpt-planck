"""Raw-file side of stream_core.py: raw paths, checksums, resumable downloads, the disk guard and
the prefetch threads.

- A raw file lives at RAW/<dataset with / as __>/<path> (the layout fetch_manifest.py uses, so a
  file it prefetched is picked up and re-verified). Starter files (manifest starter_local) are read
  in place from STARTER/<same layout> when their checksum matches, and are never deleted.
- Downloads go to <file>.part with curl -C - (resume), and are renamed only after the size and the
  manifest checksum match (sha256 for Hugging Face LFS files; sha1, else md5, for archive.org). A
  .part that is too long, or a finished file with a wrong checksum, is deleted and fetched again.
- Mirrors: cfg["url_rewrite"] ({prefix: replacement}, stream_core --url-rewrite FROM=TO) swaps a
  URL prefix before curl, e.g. archive.org/download/ (which returned HTTP 503 for the Stack
  Exchange item on 2026-09-26) for the item's own server; the manifest keeps the canonical URL and
  the file's checksum decides whether the copy is good.
- Disk guard: a download starts only if, after it, free space stays at or above min_free plus
  reserve (room for the extraction running meanwhile). disk_free() is the one place free space is
  read (tests replace it).
- Prefetch: `threads` downloader threads walk the pending files in manifest order, keeping at most
  window bytes of raw files downloaded (or downloading) but not yet processed; the first pending
  file may always start, so a file larger than the window cannot deadlock the stream.
"""
import hashlib
import os
import shutil
import subprocess
import threading
import time

GIB = 1 << 30
EXPAND = {".gz": 4.0, ".7z": 6.0}   # text bytes per raw byte, upper-end guess for the disk guard


class FetchError(RuntimeError):
    pass


def disk_free(path) -> int:
    p = os.path.abspath(path)
    while not os.path.exists(p):
        p = os.path.dirname(p)
    return shutil.disk_usage(p).free


def raw_rel(entry) -> str:
    return entry["dataset"].replace("/", "__") + "/" + entry["path"]


def expand(entry) -> float:
    return next((v for k, v in EXPAND.items() if entry["path"].endswith(k)), 1.0)


def checksum(entry):
    """-> (algorithm, hex digest) from the manifest entry; sha256 preferred."""
    for alg in ("sha256", "sha1", "md5"):
        if entry.get(alg):
            return alg, entry[alg]
    raise FetchError(f"{entry['path']}: no checksum in the manifest")


def verify(path, entry) -> bool:
    if not os.path.exists(path) or os.path.getsize(path) != entry["size"]:
        return False
    alg, want = checksum(entry)
    h = hashlib.new(alg)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest() == want


def source_url(entry, cfg) -> str:
    """The URL to fetch: the manifest's, with the first matching --url-rewrite prefix swapped."""
    url = entry["url"]
    for old, new in (cfg.get("url_rewrite") or {}).items():
        if url.startswith(old):
            return new + url[len(old):]
    return url


def check_disk_config(c, floor_gb):
    """The disk guard's settings: limits at or over the hard floor, a reserve >= 0, a window."""
    if (min(c["min_free_gb"], c["floor_gb"]) < floor_gb or c["reserve_gb"] < 0
            or c["window_gb"] <= 0):
        raise ValueError(f"free-space limits below {floor_gb} GiB, a negative reserve or no "
                         "download window are not allowed")


def _room(entry, have, root, cfg) -> bool:
    need = entry["size"] - have
    return disk_free(root) - need >= (cfg["min_free_gb"] + cfg["reserve_gb"]) * GIB


def fetch(item, cfg, procs, stop):
    """Make item["path"] a verified copy of item["entry"]; -> fetch stats. Raises FetchError, or
    returns {"blocked": True} while the disk guard forbids starting."""
    e, dest = item["entry"], item["path"]
    t0 = time.time()
    if item["local"]:
        if verify(dest, e):
            return {"cached": True, "local": True, "fetched_bytes": 0, "seconds": 0.0}
        item["local"], dest = False, item["dl_path"]     # starter copy differs: download it
        item["path"] = dest
    if os.path.exists(dest):
        if verify(dest, e):
            return {"cached": True, "fetched_bytes": 0, "seconds": round(time.time() - t0, 1)}
        os.remove(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part, fetched, err, url = dest + ".part", 0, "", source_url(e, cfg)
    for attempt in range(1, cfg["curl_attempts"] + 1):
        if stop():
            raise FetchError("stopped")
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have > e["size"]:
            os.remove(part)
            have = 0
        if not _room(e, have, cfg["raw"], cfg):
            return {"blocked": True}
        if have < e["size"]:
            p = subprocess.Popen(["curl", "-sSfL", "--retry", "3", "--connect-timeout", "30",
                                  "--speed-limit", "65536", "--speed-time", "120", "-C", "-",
                                  "-o", part, "-w", "%{size_download}", url],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            procs.add(p)
            out, errs = p.communicate()
            procs.discard(p)
            fetched += int(float(out.strip() or 0))
            err = f"curl exit {p.returncode}: {errs.strip()[-300:]}"
            if p.returncode in (33, 36):             # the server cannot resume: start over
                os.remove(part)
        if os.path.exists(part) and os.path.getsize(part) == e["size"]:
            if verify(part, e):
                os.replace(part, dest)
                return {"cached": False, "fetched_bytes": fetched, "attempts": attempt,
                        "seconds": round(time.time() - t0, 1),
                        **({"url": url} if url != e["url"] else {})}
            os.remove(part)
            err = f"{checksum(e)[0]} mismatch"
        time.sleep(min(60, cfg["retry_sleep"] * attempt))
    raise FetchError(f"{e['path']}: gave up after {cfg['curl_attempts']} attempts ({err})")


class Prefetcher:
    """Background downloads for items [{"entry", "path", "local", ...}] in list order. Item states:
    wait, run, blocked (disk guard), ready, failed, consumed."""

    def __init__(self, items, cfg, log=print):
        self.items, self.cfg, self.log = items, cfg, log
        self.cond, self.procs, self.stopped, self.threads = threading.Condition(), set(), False, []
        for it in items:
            it["state"] = "wait"

    def start(self):
        for _ in range(max(1, self.cfg["dl_threads"])):
            t = threading.Thread(target=self._loop, daemon=True)
            t.start()
            self.threads.append(t)

    def _inflight(self):
        return sum(it["entry"]["size"] for it in self.items
                   if it["state"] in ("run", "ready") and not it["local"])

    def _loop(self):
        while True:
            with self.cond:
                it = next((x for x in self.items if x["state"] in ("wait", "blocked")), None)
                if self.stopped or it is None:
                    return
                first = not any(x["state"] in ("run", "ready") for x in self.items)
                if not first and self._inflight() + it["entry"]["size"] > self.cfg["window_gb"] * GIB:
                    self.cond.wait(2)
                    continue
                it["state"] = "run"
            try:
                r = fetch(it, self.cfg, self.procs, lambda: self.stopped)
                new = "blocked" if r.get("blocked") else "ready"
                it["fetch"] = r
            except Exception as e:  # noqa: BLE001 (recorded as file_failed by the runner)
                new, it["error"] = "failed", f"{type(e).__name__}: {e}"
            with self.cond:
                it["state"] = new
                if new == "blocked":
                    it.setdefault("blocked_since", time.time())
                else:
                    it.pop("blocked_since", None)
                self.cond.notify_all()
                if new == "blocked":
                    self.cond.wait(5)

    def consumed(self, it):
        with self.cond:
            it["state"] = "consumed"
            self.cond.notify_all()

    def stop(self):
        with self.cond:
            self.stopped = True
            self.cond.notify_all()
        for p in list(self.procs):
            p.terminate()
        for t in self.threads:
            t.join(timeout=30)
