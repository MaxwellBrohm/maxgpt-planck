"""stream_loop.wait_chunk: a chunk (one heavy-lock hold) may mix the sources of one tier, never two
tiers, and still honours each source's worker cap. Tier 1 has ten sources, several with one file,
so one-source chunks cost one lock hold per source while the lock is contended."""
import threading
import time
import types

import stream_loop as L


def item(src, tier, state="ready"):
    return {"source": src, "state": state, "entry": {"tier": tier}}


ITEMS = [item("dolly", 1), item("irc", 1), item("stackexchange", 1), item("stackexchange", 1),
         item("gutenberg", 2), item("gutenberg", 2)]


def chunk(tmp_path, monkeypatch, pos=0, **kw):
    monkeypatch.setattr(L, "mem_cap", lambda cfg: 20)
    c = dict({"workers": 16, "source_workers": {}, "batch_wait": 2.0, "out": str(tmp_path)}, **kw)
    t0 = time.time()
    got = L.wait_chunk(c, types.SimpleNamespace(cond=threading.Condition()), ITEMS, pos)
    return [it["source"] for it in got], time.time() - t0


def test_sources_of_one_tier_share_a_chunk_and_a_tier_boundary_starts_it(tmp_path, monkeypatch):
    got, took = chunk(tmp_path, monkeypatch)
    assert got == ["dolly", "irc", "stackexchange", "stackexchange"]
    assert took < 1.0                              # the next tier's file does not make it wait
    assert chunk(tmp_path, monkeypatch, pos=4)[0] == ["gutenberg", "gutenberg"]


def test_worker_and_source_caps_still_bound_a_chunk(tmp_path, monkeypatch):
    assert chunk(tmp_path, monkeypatch, workers=3)[0] == ["dolly", "irc", "stackexchange"]
    got, took = chunk(tmp_path, monkeypatch, source_workers={"irc": 2})
    assert got == ["dolly", "irc"] and took < 1.0
    assert chunk(tmp_path, monkeypatch, source_workers={"stackexchange": 3})[0] == [
        "dolly", "irc", "stackexchange"]
