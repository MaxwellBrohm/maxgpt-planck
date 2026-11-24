"""kpreflight.py (K3 round 16, 2026-11-21): K's rules in place of E2 preflight's whole-list data rule, on planted code
and data roots; E2's check_one itself is E2's (tested there) and is stubbed here."""
import json
import os

import pytest

import check_k
import gen
import kcommon as K
import kpreflight as KP

PROBE_SHA = "ab" * 32
LANG = [(n, k) for n, k, _ in gen.LANG]


def roots(tmp_path, passed=True, code_sha=None, probe_sha=PROBE_SHA):
    code, data = tmp_path / "code", tmp_path / "data"
    sm = code / "planck_root" / "data" / "shards" / "starter_v0_8k"
    sm.mkdir(parents=True)
    hd = [{"name": n, "kind": k, **({"eot_id": 1} if k == "tokens" else {}),
           "paths": [f"{n}/{n}-*.{'bin' if k == 'tokens' else 'jsonl'}"]} for n, k in LANG]
    (sm / "manifest.json").write_text(json.dumps({"harness_data": {"sources": hd}}))
    (data / "evals").mkdir(parents=True)
    (data / "evals" / "manifest.json").write_text(json.dumps({"sets": {"k_probe": {"sha256": probe_sha}}}))
    mine = {f: K.sha256_file(os.path.join(K.HERE, f)) for f in check_k.CHECK_CODE}
    (data / "check.json").write_text(json.dumps({"pass": passed, "fails": [] if passed else ["x"],
                                                 "code_sha256": code_sha or mine}))
    return str(code), str(data)


def cfg(arm="FH", probe=True):
    c = {"provenance": {"shard_manifest": "data/shards/starter_v0_8k/manifest.json"},
         "data": {"sources": gen.data_block(arm)["sources"]}}
    if probe:
        c["k"] = {"probe": {"every": 153, "files": [{"file": "x", "sha256": PROBE_SHA}]}}
    return c


def test_a_k_config_on_a_checked_data_root_passes(tmp_path):
    code, data = roots(tmp_path)
    for arm in ("FH", "F0", "L0", "CAP", "CONT"):
        assert KP.k_rules(cfg(arm), code, data) == [], arm


def test_lang_sources_must_be_the_shard_manifests(tmp_path):
    code, data = roots(tmp_path)
    c = cfg()
    lang = [i for i, s in enumerate(c["data"]["sources"]) if s["name"] in dict(LANG)]
    c["data"]["sources"][lang[0]], c["data"]["sources"][lang[1]] = c["data"]["sources"][lang[1]], c["data"]["sources"][lang[0]]
    assert any(x.startswith("lang:") for x in KP.k_rules(c, code, data))          # order
    c = cfg()
    del c["data"]["sources"][lang[-1]]
    assert any(x.startswith("lang:") for x in KP.k_rules(c, code, data))          # one missing
    c = cfg()
    c["data"]["sources"] = [s for s in c["data"]["sources"] if s["name"] not in dict(LANG)]
    assert any("no LANG source" in x for x in KP.k_rules(c, code, data))         # all missing, not CAP


def test_the_data_roots_check_must_have_passed_with_this_code(tmp_path):
    for i, (kw, msg) in enumerate(((dict(passed=False), "did not pass"),
                                   (dict(code_sha={"check_k.py": "0" * 64}), "other gate code"),
                                   (dict(probe_sha="cd" * 32), "K-PROBE sha256"))):
        code, data = roots(tmp_path / str(i), **kw)
        assert any(msg in x for x in KP.k_rules(cfg(), code, data)), (kw, msg)
    code, data = roots(tmp_path / "n")
    os.unlink(os.path.join(data, "check.json"))
    assert any("no " in x and "check.json" in x for x in KP.k_rules(cfg(), code, data))
    assert KP.k_rules(cfg(probe=False), *roots(tmp_path / "p", probe_sha="cd" * 32)) == []   # no probe, no pin


def test_only_e2s_whole_list_data_rule_is_dropped(tmp_path, monkeypatch):
    code, data = roots(tmp_path)
    p = tmp_path / "c.yaml"
    p.write_text(json.dumps(cfg()))

    class PF:
        @staticmethod
        def harness_reads(h):
            return set()

        @staticmethod
        def check_one(path, c, strict, reads):
            assert strict
            return {"name": "x", "ok": False, "refusals": [KP.DROPPED + ": [...] vs [...]", "prereg: not committed"]}
    monkeypatch.setattr(KP, "e2_preflight", lambda: PF)
    rep = KP.run(str(p), code, data)
    assert rep["refusals"] == ["prereg: not committed"] and not rep["ok"]
    PF.check_one = staticmethod(lambda path, c, strict, reads: {"name": "x", "ok": False,
                                                                "refusals": [KP.DROPPED + ": a vs b"]})
    assert KP.run(str(p), code, data)["ok"]
    with pytest.raises(SystemExit):
        KP.main([str(p)])
