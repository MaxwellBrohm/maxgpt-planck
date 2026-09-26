"""checks.py end to end: train a tiny family through train_bpe's CLI, run every check, read the files."""
from __future__ import annotations

import json
import os

import pytest

import checks
import synth
import train_bpe
from checks_report import to_markdown


def _write(path, texts):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for k, t in enumerate(texts):
            f.write(json.dumps({"id": f"d{k}", "text": t}, ensure_ascii=False) + "\n")


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("chk")
    sample, fam, out = str(root / "sample"), str(root / "fam"), str(root / "out")
    docs = synth.corpus(seed=3, n_docs=1200)
    _write(os.path.join(sample, "train", "web.jsonl"), docs[:900])
    _write(os.path.join(sample, "train", "oasst2.jsonl"), synth.chat_lines(seed=4, n=600))
    _write(os.path.join(sample, "heldout", "web.jsonl"), docs[900:])
    _write(os.path.join(sample, "heldout", "oasst2.jsonl"), synth.chat_lines(seed=5, n=150))
    assert train_bpe.main(["nested", "--sample", sample, "--out", fam, "--top", "700", "--sizes", "300,512",
                           "--prefix", "tok_c"]) == 0
    assert train_bpe.main(["separate", "--sample", sample, "--out", fam, "--size", "512",
                           "--prefix", "tok_c_sep"]) == 0
    assert checks.main(["--family", fam, "--sample", sample, "--out", out, "--prefix", "tok_c",
                        "--base-size", "512", "--superword-n", "20,60", "--mine-mb", "0.02"]) == 0
    return {"root": str(root), "sample": sample, "fam": fam, "out": out}


def test_outputs(run_dir):
    assert sorted(os.listdir(run_dir["out"])) == ["tok_c_checks.json", "tok_c_checks.md"]
    res = json.load(open(os.path.join(run_dir["out"], "tok_c_checks.json")))
    assert res["sizes"] == [300, 512, 700] and res["sources"] == ["oasst2", "web"]
    assert set(res["bytes_per_token"]) == {"300", "512", "700"}
    assert set(res["bytes_per_token"]["512"]) == {"oasst2", "web", "ALL"}
    p98 = res["P-098"]
    assert p98["size"] == 512 and p98["identity"]["merges_identical"] and p98["pooled_rel_diff_pct"] == 0.0
    p101 = res["P-101"]
    assert p101["chat_sources"] == ["oasst2"] and p101["base_vocab"] == 512
    assert [(a["n"], a["ranking"]) for a in p101["arms"]] == [(20, "freq"), (20, "saved"), (60, "freq"), (60, "saved")]
    assert 0 < p101["mined_from"]["oasst2"] < 600, "--mine-mb caps the mined training docs"
    assert set(res["P-097"]) == {"300", "512", "700"} and res["P-097"]["700"]["all"]["n"] == 200
    assert res["P-100"]["status"] == "out_of_scope"
    prov = res["provenance"]
    assert set(prov["tokenizer_files"]) == {"tok_c_300.json", "tok_c_512.json", "tok_c_700.json", "tok_c_sep_512.json"}
    assert set(prov["heldout_files"]) == {"oasst2", "web"}


def test_no_local_paths_and_no_em_dashes(run_dir):
    for name in ("tok_c_checks.json", "tok_c_checks.md"):
        text = open(os.path.join(run_dir["out"], name), encoding="utf-8").read()
        assert run_dir["root"] not in text and os.path.basename(run_dir["root"]) not in text
        assert "\u2014" not in text


def test_markdown(run_dir):
    md = open(os.path.join(run_dir["out"], "tok_c_checks.md"), encoding="utf-8").read()
    for head in ("## Bytes per token", "## P-098", "## P-101", "## P-097", "## P-100"):
        assert head in md
    res = json.load(open(os.path.join(run_dir["out"], "tok_c_checks.json")))
    assert to_markdown(res) == md, "the markdown renders the same from the JSON read back"
    assert ("): passes." if res["P-101"]["passes_bar"] else "): fails.") in md
    assert "'truncated 8k compresses about as well'): holds (identical files)." in md
    row = next(l for l in md.splitlines() if l.startswith("| ALL |"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[3:] == [f"{res['bytes_per_token'][v]['ALL']['bpt']:.3f}" for v in ("300", "512", "700")]


def test_p101_skipped_without_chat_or_base(run_dir, tmp_path):
    assert checks.main(["--family", run_dir["fam"], "--sample", run_dir["sample"], "--out", str(tmp_path),
                        "--prefix", "tok_c", "--base-size", "8192"]) == 0
    res = json.load(open(tmp_path / "tok_c_checks.json"))
    assert res["P-101"]["status"] == "not_run" and "8192" in res["P-101"]["reason"]
    assert checks.main(["--family", run_dir["fam"], "--sample", run_dir["sample"], "--out", str(tmp_path),
                        "--prefix", "tok_c", "--base-size", "512", "--chat-sources", "dolly"]) == 0
    res = json.load(open(tmp_path / "tok_c_checks.json"))
    assert res["P-101"]["status"] == "not_run"
    assert "not run" in to_markdown(res).lower()


def test_find_files_splits_family_and_separate(run_dir):
    fam, sep = checks.find_files(run_dir["fam"], "tok_c")
    assert sorted(fam) == [300, 512, 700] and sorted(sep) == [512]
    assert checks.main(["--family", run_dir["fam"], "--sample", run_dir["sample"], "--out", run_dir["out"],
                        "--prefix", "nope"]) == 2


def test_markdown_reports_the_real_added_token_gain(run_dir):
    md = open(os.path.join(run_dir["out"], "tok_c_checks.md"), encoding="utf-8").read()
    res = json.load(open(os.path.join(run_dir["out"], "tok_c_checks.json")))
    assert "| real added-token gain, pooled (%) |" in md
    a = res["P-101"]["arms"][0]
    row = next(l for l in md.splitlines() if l.startswith(f"| {a['n']} | {a['ranking']} |"))
    assert f"{a['real_gain_pct']['ALL']:.2f}" in row
    verdict = "passes" if res["P-101"]["passes_bar_real"] else "fails"
    assert f"gain is {res['P-101']['best_real_gain_pct_freq']:.2f}%: {verdict}." in md
