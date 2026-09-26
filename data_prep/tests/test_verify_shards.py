"""verify_shards.py passes on a correct output and catches a doc changed under a re-hashed manifest."""
from __future__ import annotations

import json
import os
import shutil

import numpy as np

import prep_common as C
import verify_shards as V


def test_clean_output_verifies(shards, corpus, capsys):
    assert V.main([shards, corpus["input"], "--heldout", corpus["heldout"], "--sample", "1000"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["ok"] and all(r["sample_checked"] == r["docs"] for r in rep["sources"])


def test_changed_token_is_caught_even_with_matching_hash(shards, corpus, tmp_path, capsys):
    bad = str(tmp_path / "bad")
    shutil.copytree(shards, bad)
    mp = os.path.join(bad, "web", "manifest.json")
    man = C.read_json(mp)
    p = os.path.join(bad, man["shards"][0]["path"])
    a = np.fromfile(p, dtype="<u2")
    a[0] = a[0] + 1 if a[0] + 1 < 8192 else a[0] - 1
    a.tofile(p)
    man["shards"][0]["sha256"] = C.sha256_file(p)
    C.write_json(mp, man)
    assert V.main([bad, corpus["input"], "--sample", "1000", "--sources", "web"]) == 1
    rep = json.loads(capsys.readouterr().out)
    assert any(f.startswith("round trip") for f in rep["sources"][0]["fails"])


def test_heldout_leak_is_caught(shards, corpus, tmp_path, capsys):
    held = str(tmp_path / "held")
    os.makedirs(held)
    kept = [r for r in corpus["records"]["web"] if r["id"] not in corpus["excluded_ids"]["web"]][:1]
    with open(os.path.join(held, "web.jsonl"), "w") as f:
        f.write(json.dumps(kept[0]) + "\n")
    assert V.main([shards, corpus["input"], "--heldout", held, "--sources", "web"]) == 1
    assert "share a key with the held-out split" in capsys.readouterr().out
