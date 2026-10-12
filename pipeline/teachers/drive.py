"""pipeline/driver.py with a --serve flag: render through a teacher loaded by serve.py on the PC (DRY prep).

    python3 -B pipeline/teachers/drive.py --serve --allow-real-teacher --endpoint http://127.0.0.1:18763 \
        --out RUN_DIR --n 200 --shard-seed teachdry --concurrency 16 --timeout 3600 [every other driver.py flag]

Without --serve this is driver.main unchanged (the fake stub, or an OpenAI-compatible server with
--allow-real-teacher). With --serve, driver.main builds its client as serve_client.ServeClient instead of
teacher_client.TeacherClient; nothing else in the driver changes (sample -> gate -> prompt -> render -> parse -> check
-> dedup -> write, one retry with a new render seed, resume from the shards). --serve still requires
--allow-real-teacher, the endpoint must be loopback (serve_http.py reached through an SSH tunnel), and the teacher's
model id and license come from its pins, so --model and --license are ignored. Run dirs of served renders are dry:
keep them out of the repo. When the served teacher goes away (serve_client.ServerGone) the run stops after a yield
snapshot, instead of recording TEACHER_ERROR for every remaining attempt; the same command resumes it.

Decoding controls and repair (2026-09-27, driver.py flags, all off by default and recorded in every record):
    --structured labels|labels_exact   structured output of each skeleton's planned label lines plus END
    --ban-dashes                       the dash-token ban (serve_http must run with --dash-ban)
    --ban-phrases                      the AI-ism phrase ban, vLLM bad_words (serve_http must offer it). ON BY
                                       DEFAULT with --serve since 2026-10-04 (round 4, aiism-v4: the service filler
                                       family; Max: service filler stays rejected); --no-ban-phrases turns it off (a
                                       control arm, or resuming a run dir pinned before it)
    --preset card|shared|yld0926       a D4 sampling preset (serve.TEACHERS; Gemma has no "shared")
    --repair                           after a near miss the retry prompt names what failed (attempt kind "repair")
serve_http offers them (its GET /planck-serve lists presets, structured backend, line separator and dash ban); the
client refuses a server that lacks what the run asks for, and a row whose "decode" disagrees is a TEACHER_ERROR."""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import driver  # noqa: E402
import serve_client  # noqa: E402
import teacher_client as TC  # noqa: E402


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--serve" not in argv:
        return driver.main(argv)
    argv.remove("--serve")
    if "--no-ban-phrases" in argv:               # 10-04: the phrase ban is on for every served run unless turned off
        argv.remove("--no-ban-phrases")
    elif "--ban-phrases" not in argv:
        argv.append("--ban-phrases")
    if "--allow-real-teacher" not in argv:
        raise SystemExit("drive.py --serve renders with a real teacher: pass --allow-real-teacher too")
    TC.TeacherClient = serve_client.ServeClient      # driver.main builds its client through this name
    driver.Driver.run = _run_snapshot_on_gone
    return driver.main(argv)


_RUN = driver.Driver.run


def _run_snapshot_on_gone(self):
    """Driver.run, plus a yield snapshot (stopped 'server gone') when the served teacher goes away, so the session's
    wall time and counts are in yield.jsonl for drive_report; the error still ends the run (a rerun resumes)."""
    try:
        return _RUN(self)
    except serve_client.ServerGone:
        self.stopped = "server gone"
        self.snapshot(final=False)
        raise


if __name__ == "__main__":
    sys.exit(main())
