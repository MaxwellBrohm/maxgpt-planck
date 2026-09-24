"""Collect E001 outputs into ../results.json: likelihood summaries, chat-probe regrades,
grader/item self-test and mutation-test results, and every guard verdict (memory, exits)."""
import glob, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
PY = sys.executable


def run(cmd):
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    return {"cmd": " ".join(cmd[1:]), "exit": r.returncode, "tail": (r.stdout + r.stderr).strip().splitlines()[-3:]}


def main():
    out = {"experiment": "E001_battery_and_probes", "date": "2026-09-24"}
    out["self_tests"] = [run([PY, "battery.py"]), run([PY, "mutation_test_graders.py"]),
                         run([PY, "validate_items.py"]), run([PY, "mutation_test_items.py"])]
    bat = json.load(open(os.path.join(EXP, "battery_results.json")))
    out["likelihood"] = {m: {k: v for k, v in d.items() if k in ("attribution", "runs", "khard", "khard_own_format", "new_items_n_records")}
                         for m, d in bat.items() if not m.startswith("_")}
    out["likelihood_old_item_table"] = bat.get("_old_item_table")
    out["likelihood_full_summaries_file"] = "battery_results.json"
    chat = json.load(open(os.path.join(EXP, "chat_results.json")))
    out["chat"] = {src: {m: {mode: {k: v for k, v in s.items() if k != "flips"} for mode, s in modes.items()}
                         for m, modes in d.items()} for src, d in chat.items() if src != "table_source"}
    out["chat_table_source"] = chat.get("table_source")
    out["chat_flips_file"] = "chat_results.json"
    out["guard_verdicts"] = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(EXP, "logs", "*.guard.json")))]
    json.dump(out, open(os.path.join(EXP, "results.json"), "w"), indent=1)
    print("wrote results.json;", [(t["cmd"], t["exit"]) for t in out["self_tests"]])


if __name__ == "__main__":
    main()
