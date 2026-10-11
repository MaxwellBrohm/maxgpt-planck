"""Shared by the stage 2 verdict tests (SCREENS.txt STAGE 2 VERDICTS, 2026-10-11): rebuild a run tree from what the repo
holds (each screen's out/<run>/ copies, screens/base_out/ and screens/c6_out/, the seed-1 runs' recorded final bpb)
and the committed outputs. Data and helpers only.
"""
import contextlib, io, json, os, shutil, sys  # noqa: E401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import analyze  # noqa: E402
import screens_lib as L  # noqa: E402
from s2select_record import VALS as S2SEL  # noqa: E402
from s2seeds_record import EXT_VALS  # noqa: E402
from test_screens_s003_stage_c import STAGE_B  # noqa: E402
from test_screens_stage1_next import RECORDED as STAGE_1A  # noqa: E402
from test_screens_stage1_seeds import STAGE_C, write_runs  # noqa: E402

EXP = os.path.dirname(L.HERE)
SX = L.HERE
STAGE2 = {"S004": "S004_canon", "S005": "S005_forget_gate", "S006": "S006_mtp_aux", "S007": "S007_smeared_key"}
KEYS = {"S004": "S004.canonac", "S005": "S005.forget", "S006": "S006.mtp", "S007": "S007.smear"}
VERDICT = {"S004.canonac": "NOMINATE", "S005.forget": "NOMINATE", "S006.mtp": "REJECT", "S007.smear": "TIE"}
STAGE1 = {"S001.nogate": "TIE", "S002.novres": "REJECT", "S002.noqknorm": "TIE", "S002.nonormscale": "TIE",
          "S003.adamw": "REJECT"}
HOLDS = {"S003.adamw", "S006.mtp", "S002.novres", "S005.forget", "S004.canonac"}
SEED1 = {**STAGE_1A, **STAGE_B, **STAGE_C, **S2SEL, "s006_mtp_g4_s1": EXT_VALS}


def jload(*p):
    return json.load(open(os.path.join(*p)))


def res(sid):
    return jload(EXP, STAGE2[sid], "results.json")


def text(rel):
    return open(os.path.join(EXP, rel)).read()


def flat(s):
    return " ".join(s.split())


def build(root) -> str:
    """The run tree: every seed run's copied files, the C6 JSON under <run>/c6/, the seed-1 runs' final bpb."""
    root = str(root)
    write_runs(__import__("pathlib").Path(root), SEED1)
    srcs = [os.path.join(EXP, d, "out") for d in ("S001_attn_gate", "S002_block_ablations", "S003_adamw",
                                                  *STAGE2.values())] + [os.path.join(SX, "base_out")]
    for s in srcs:
        for run in os.listdir(s):
            shutil.copytree(os.path.join(s, run), os.path.join(root, run), dirs_exist_ok=True)
    for run in os.listdir(os.path.join(SX, "c6_out")):
        shutil.copytree(os.path.join(SX, "c6_out", run), os.path.join(root, run, "c6"), dirs_exist_ok=True)
    return root


def verdicts(root) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        assert analyze.main(["verdicts", "--runs", root, "--out", os.path.join(root, "v.json")]) == 0
    return jload(root, "v.json")


def roundtrip(x):
    return json.loads(json.dumps(x, sort_keys=True, default=str))
