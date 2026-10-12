"""Tests for models/registry.py and regtools.py on tiny fake files (no model, no torch). Run: python3 models/tests/
test_registry.py (or pytest). REGISTRY_UNDER_TEST=DIR runs them against copies of the two modules (mutation checks)."""
import contextlib, copy, hashlib, io, json, os, struct, sys, tempfile, time, unittest
SRC = os.environ.get("REGISTRY_UNDER_TEST") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
import registry as R  # noqa: E402
import regtools as T  # noqa: E402

SHA = "ab" * 32
REV = "0123456789abcdef0123456789abcdef01234567"


def good():
    return {"schema": 1, "id": "E999/tiny/s1", "experiment": "E999", "arm": "tiny", "role": "scored", "seed": 1,
            "kind": "finetune", "base_model": {"id": "org/m", "revision": REV, "revision_source": "test"},
            "params_total": 96, "params_body": 16, "tokenizer": {"id": "org/m@" + REV, "sha256": SHA},
            "data": {"identity": "stream(seed)", "sha256": None},
            "code": {"git_commit": "abc1234", "relation": "pre_run", "digest_sha256": None, "digest_source": None},
            "config_sha256": SHA, "result": "not scored", "status": "success",
            "run": {"state": "completed", "started": "2026-09-25T09:00:00-04:00", "ended": None, "log": None},
            "files": [{"path": "model.safetensors", "bytes": 10, "sha256": SHA}],
            "locations": ["mac:experiments/E999/weights/x", "pc:/mnt/d/planck-archive/E999/x"],
            "created": "2026-09-25T10:00:00-04:00", "release": "undecided", "notes": "data.sha256: not recorded"}


def put(path, data):
    with open(path, "wb" if isinstance(data, bytes) else "w") as f:
        f.write(data)


def write_safetensors(path, shapes):
    head, off = {"__metadata__": {"format": "pt"}}, 0
    for name, shape in shapes.items():
        n = 4
        for d in shape:
            n *= d
        head[name] = {"dtype": "F32", "shape": shape, "data_offsets": [off, off + n]}
        off += n
    raw = json.dumps(head).encode()
    put(path, struct.pack("<Q", len(raw)) + raw + b"\0" * off)


class Validator(unittest.TestCase):
    def errs(self, change):
        e = good()
        change(e)
        return " | ".join(R.validate_entry(e))

    def test_good_entry_is_valid(self):
        self.assertEqual(R.validate_entry(good()), [])

    def test_each_rule_fires(self):
        cases = [
            (lambda e: e.pop("status"), "missing status"),
            (lambda e: e.update(colour="red"), "unknown key colour"),
            (lambda e: e.update(seed=True), "seed has type bool"),
            (lambda e: e.update(status="great"), "status 'great'"),
            (lambda e: e.update(kind="merge"), "kind 'merge'"),
            (lambda e: e.update(release="maybe"), "release 'maybe'"),
            (lambda e: e.update(schema=2), "schema must be 1"),
            (lambda e: e.update(id="E998/tiny/s1"), "bad id"),
            (lambda e: e.update(base_model=None), "needs base_model.id"),
            (lambda e: e.update(kind="from_scratch"), "base_model null"),
            (lambda e: e.update(kind="checkpoint"), "needs parent and step"),
            (lambda e: e["tokenizer"].pop("sha256"), "tokenizer keys"),
            (lambda e: e["tokenizer"].update(sha256="xyz"), "tokenizer.sha256 is not"),
            (lambda e: e.update(config_sha256="ABC"), "config_sha256 is not"),
            (lambda e: e["code"].update(git_commit="not-a-hash"), "not a commit hash"),
            (lambda e: e["code"].update(relation=None), "code.relation"),
            (lambda e: e["code"].update(relation="later"), "code.relation"),
            (lambda e: e["code"].update(git_commit=None, relation=None), "name code.git_commit"),
            (lambda e: e.update(created="2026-09-25 10:00:00"), "created is not ISO"),
            (lambda e: e["run"].update(started="yesterday"), "run.started is not ISO"),
            (lambda e: e["run"].update(state="exploded"), "run.state"),
            (lambda e: e["files"][0].update(path="/abs/model.safetensors"), "must be relative"),
            (lambda e: e["files"][0].update(path="../model.safetensors"), "must be relative"),
            (lambda e: e["files"][0].update(bytes=-1), "non-negative"),
            (lambda e: e["files"][0].update(sha256=None), "name files.sha256"),
            (lambda e: e["files"].append(dict(e["files"][0])), "duplicate file paths"),
            (lambda e: e["files"][0].pop("bytes"), "exactly path, bytes, sha256"),
            (lambda e: e.update(status="not_saved"), "not_saved <=>"),
            (lambda e: e.update(files=[]), "not_saved <=>"),
            (lambda e: e.update(locations=[]), "not_saved <=>"),
            (lambda e: e.update(locations=["mac:/abs/path"]), "bad location"),
            (lambda e: e.update(locations=["mac:experiments/../x"]), "bad location"),
            (lambda e: e.update(locations=["pc:/tmp/x"]), "bad location"),
            (lambda e: e.update(locations=["mac:a", "mac:a"]), "duplicate locations"),
            (lambda e: e.update(params_body=None), "params_body is null"),
            (lambda e: e["data"].update(identity=None), "data.identity is null"),
            (lambda e: e["base_model"].update(revision=None), "base_model.revision is null"),
            (lambda e: e.update(release="never"), "release never"),
            (lambda e: e.update(result="  "), "result is empty"),
            (lambda e: e.update(notes="data.sha256 lives in /Users/someone/x"), "private"),
            (lambda e: e.update(notes="data.sha256 copied from 203.0.113.7"), "private"),
            (lambda e: e.update(notes="data.sha256 ask someone@example.com"), "private"),
        ]
        for change, want in cases:
            with self.subTest(want=want):
                self.assertIn(want, self.errs(change))

    def test_valid_variants(self):
        e, n = good(), good()
        e.update(params_body=None, notes=e["notes"] + "; params_body: unknown, the run log has no split")
        n.update(status="not_saved", files=[], locations=[])
        self.assertEqual(R.validate_entry(e) + R.validate_entry(n), [])

    def test_duplicate_ids_rejected(self):
        self.assertIn("duplicate id", " ".join(R.validate_all([good(), good()])))


class Files(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name

        self.addCleanup(self.tmp.cleanup)

    def test_sha256_matches_hashlib(self):
        p, data = os.path.join(self.d, "blob"), os.urandom(3 * 1024 * 1024 + 17)
        put(p, data)
        self.assertEqual(T.sha256_file(p, chunk=65536), hashlib.sha256(data).hexdigest())

    def test_safetensors_counts_and_body(self):
        p = os.path.join(self.d, "m.safetensors")
        write_safetensors(p, {"model.embed_tokens.weight": [10, 4], "model.layers.0.mlp.w": [4, 4],
                              "lm_head.weight": [10, 4], "transformer.wpe.weight": [2, 4]})
        self.assertEqual(T.param_counts(T.safetensors_numel(p)), (104, 16))       # __metadata__ skipped

    def test_run_tag_defaults(self):
        got = [T.run_tag(a) for a in (["m", "--seed", "3"], ["m", "--dry"], ["m", "--steps", "0"], ["m", "--tag", "x"])]
        self.assertEqual(got, ["s3", "dry_s0", "base", "x"])

    def test_config_sha_depends_on_args(self):
        a, b, c = (T.config_sha256("ft_test.py", ["m", "--lr", x]) for x in ("1e-3", "1e-3", "3e-3"))
        self.assertTrue(a == b != c)

    def test_local_iso_offset(self):
        old = os.environ.get("TZ")
        try:
            for tz, want in (("America/New_York", "-04:00"), ("UTC", "+00:00")):
                os.environ["TZ"] = tz
                time.tzset()
                self.assertEqual(T.local_iso("2026-09-25 14:55:40"), "2026-09-25T14:55:40" + want)
        finally:
            os.environ.pop("TZ") if old is None else os.environ.update(TZ=old)
            time.tzset()

    def fake_run(self, extra_snapshot=False, root="a"):
        exp = os.path.join(self.d, root, "repo", "experiments", "E999_fake")
        w = os.path.join(exp, "weights", "org__m__s1")
        for sub in (w, os.path.join(exp, "out"), os.path.join(exp, "logs")):
            os.makedirs(sub)
        write_safetensors(os.path.join(w, "model.safetensors"), {"embed_in.weight": [8, 2], "h.0.w": [2, 3]})
        put(os.path.join(w, "tokenizer.json"), '{"v": 1}')
        put(os.path.join(exp, "out", "org__m__s1__run.json"),
            json.dumps({"model": "org/m", "seed": 1, "steps": 4, "lr": 1e-3, "params": 22}))
        cmd = ["/x/python", "-B", "ft_test.py", "org/m", "--seed", "1", "--save", "1"]
        for name, ex, killed, ended in (("ft_s1", 0, None, "2026-09-25 10:00:00"),
                                        ("ft_s1_b", 1, "swap_grew_1000MB", "2026-09-25 11:00:00")):
            put(os.path.join(exp, "logs", name + ".guard.json"), json.dumps(
                {"name": name, "cmd": cmd, "started": "2026-09-25 09:00:00", "ended": ended, "exit": ex,
                 "killed": killed}))
        hf = os.path.join(self.d, root, "hf")
        for rev in [REV] + (["f" * 40] if extra_snapshot else []):
            snap = os.path.join(hf, "hub", "models--org--m", "snapshots", rev)
            os.makedirs(snap)
            os.utime(snap, (1e9, 1e9))
        return os.path.join(self.d, root, "repo"), w, hf

    def test_from_run_builds_a_valid_entry(self):
        repo, w, hf = self.fake_run()
        e = T.from_run(w, repo=repo, hf_home=hf)
        self.assertEqual((e["params_total"], e["params_body"]), (22, 6))
        self.assertEqual(e["base_model"]["revision"], REV)
        self.assertEqual(e["locations"], ["mac:experiments/E999_fake/weights/org__m__s1"])
        self.assertEqual(e["run"]["log"], "mac:experiments/E999_fake/logs/ft_s1.guard.json")
        tok = next(f for f in e["files"] if f["path"] == "tokenizer.json")
        self.assertEqual(tok["sha256"], hashlib.sha256(b'{"v": 1}').hexdigest())
        self.assertEqual(e["tokenizer"]["sha256"], tok["sha256"])
        self.assertEqual(e["config_sha256"], T.config_sha256("ft_test.py", ["org/m", "--seed", "1", "--save", "1"]))
        e.update(id="E999/m/s1", notes="data.identity data.sha256: fake; code.git_commit: fake")
        self.assertEqual(R.validate_entry(e), [])

    def test_from_run_refuses_ambiguous_or_newer_snapshot(self):
        repo, w, hf = self.fake_run(extra_snapshot=True)
        self.assertIsNone(T.from_run(w, repo=repo, hf_home=hf)["base_model"]["revision"])
        repo, w, hf = self.fake_run(root="b")
        self.assertIsNone(T.hf_revision("org/m", before=0, hf_home=hf)[0])       # snapshot newer than the run
        self.assertEqual(T.hf_revision("org/m", before=2e9, hf_home=hf)[0], REV)

    def test_find_guard_needs_a_finished_run(self):
        repo, w, hf = self.fake_run()
        exp = os.path.dirname(os.path.dirname(w))
        self.assertTrue(T.find_guard(exp, "org/m", "s1")[0].endswith("/ft_s1.guard.json"))
        os.remove(os.path.join(exp, "logs", "ft_s1.guard.json"))
        with self.assertRaises(SystemExit):
            T.find_guard(exp, "org/m", "s1")

    def test_verify_ok_mismatch_missing_and_fill(self):
        repo, w, hf = self.fake_run()
        e = dict(T.from_run(w, repo=repo, hf_home=hf), id="E999/m/s1")
        quiet = lambda *a: None
        self.assertEqual(T.verify([e], "mac", repo=repo, out=quiet), (0, 0))
        p = os.path.join(w, "tokenizer.json")
        put(p, '{"v": 2}')              # same size, different bytes
        self.assertEqual(T.verify([e], "mac", repo=repo, out=quiet), (1, 0))
        os.remove(p)
        self.assertEqual(T.verify([e], "mac", repo=repo, out=quiet), (1, 0))
        e2 = copy.deepcopy(e)
        st = next(f for f in e2["files"] if f["path"] == "model.safetensors")
        known = st["sha256"]
        e2["files"] = [st, {"path": "extra.bin", "bytes": None, "sha256": None}]
        put(os.path.join(w, "extra.bin"), b"xyz")
        self.assertEqual(T.verify([e2], "mac", repo=repo, fill=True, out=quiet), (0, 1))
        self.assertEqual(e2["files"][1], {"path": "extra.bin", "bytes": 3,
                                          "sha256": hashlib.sha256(b"xyz").hexdigest()})
        self.assertEqual(e2["files"][0]["sha256"], known)
        e2["locations"].append("pc:~/planck/e999/org__m__s1"); seen = []
        self.assertEqual(T.verify([e2], "pc", repo=repo, roots={"pc:~/planck/e999": os.path.dirname(w)},
                                  out=seen.append), (0, 0))
        self.assertEqual([l.split()[0] for l in seen], ["OK", "OK"])
        self.assertEqual(T.verify([e2], "pc", repo=repo, out=seen.append), (0, 0))       # unmapped: NOT HERE
        self.assertEqual(seen[-1].split()[:2], ["NOT", "HERE"])

    def test_cli_add_list_validate(self):
        reg, src = os.path.join(self.d, "registry.jsonl"), os.path.join(self.d, "entry.json")
        put(src, json.dumps(good(), indent=1))
        run = lambda *a: R.main(["--registry", reg, *a])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run("add", "--json", src), 0)
            self.assertEqual(run("add", "--json", src), 1)        # duplicate id: nothing written
            s2 = ["--set", 'id="E999/tiny/s2"', "--set", "seed=2", "--set", 'locations=["mac:x/s2"]']  # own files
            self.assertEqual(run("add", "--json", src, *s2), 0)
            self.assertEqual(run("add", "--json", src, "--set", 'id="E999/tiny/s3"', "--set", 'status="x"'), 1)
            self.assertEqual(run("validate"), 0)
            self.assertEqual(run("list", "--status", "success"), 0)
        self.assertEqual([e["seed"] for e in R.load(reg)], [1, 2])


class RealRegistry(unittest.TestCase):
    def test_committed_registry_is_valid(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "registry.jsonl")
        if not os.path.exists(path):
            self.skipTest("no registry.jsonl yet")
        self.assertEqual(R.validate_all(R.load(path)), [])


if __name__ == "__main__":
    unittest.main()
