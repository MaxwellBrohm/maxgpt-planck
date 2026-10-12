"""The two write-time checks of the 80 GiB floor in stream_core.py (the extraction worker's and the
shard writer's) stop the run cleanly with disk_low, and a restart with space gives the output of an
uninterrupted run. Also: se_html drops <script> and <style> content."""
import os
import sys

import pytest

import stream_core as C
import stream_fetch as F
import stream_io as SIO
import stream_fixtures as X
from se_html import html_to_text
from test_stream_core import cfg, events

LOW, PLENTY = 79 * F.GIB, 10 ** 13


def in_shard_write():
    f = sys._getframe(2)
    while f is not None:
        if f.f_code.co_name == "_write" and f.f_globals.get("__name__") == "stream_io":
            return True
        f = f.f_back
    return False


FREE = {"tmp": lambda p: LOW if f"{os.sep}_state{os.sep}tmp{os.sep}" in os.path.abspath(p) + os.sep
        else PLENTY,                                        # the worker's scratch output
        "out": lambda p: LOW if in_shard_write() else PLENTY}  # only the shard writer's check


@pytest.mark.parametrize("where", ["tmp", "out"])
def test_the_write_time_floor_stops_the_run_and_a_restart_finishes_it(tmp_path, monkeypatch,
                                                                     where):
    clean, root = str(tmp_path / "clean"), str(tmp_path / "low")
    for r in (clean, root):
        os.makedirs(r)
        X.make_world(r)
    monkeypatch.setattr(F, "disk_free", lambda p: PLENTY)
    assert C.run(cfg(clean))["reason"] == "done"
    monkeypatch.setattr(SIO, "CHECK_EVERY", 64)            # check after every 64 bytes written
    monkeypatch.setattr(F, "disk_free", FREE[where])
    c = cfg(root)
    assert C.run(c)["reason"] == "disk_low"
    assert events(c["out"], "run_stop")[-1]["reason"] == "disk_low"
    if where == "tmp":
        fail = events(c["out"], "file_failed")[-1]
        assert fail["stage"] == "extract" and fail["error"].startswith("disk_low")
    monkeypatch.setattr(F, "disk_free", lambda p: PLENTY)
    assert C.run(c)["reason"] == "done"
    assert X.output_ids(c["out"], c["codec"]) == X.oracle(root, c["manifest"])
    assert X.tree_hashes(c["out"]) == X.tree_hashes(os.path.join(clean, "out"))


def test_script_and_style_content_is_dropped():
    html = ("<p>Keep this.</p><script>var x = 1; alert('no');</script>"
            "<style>p { color: red }</style><p>And this.</p>")
    t = html_to_text(html)
    assert "Keep this." in t and "And this." in t
    assert "alert" not in t and "color" not in t


@pytest.mark.parametrize("bad", [dict(min_free_gb=79), dict(floor_gb=79), dict(reserve_gb=-60),
                                 dict(window_gb=0)])
def test_disk_settings_that_would_undercut_the_floor_are_refused(tmp_path, bad):
    with pytest.raises(ValueError, match="not allowed"):
        C.check_config(cfg(str(tmp_path), **bad))
    C.check_config(cfg(str(tmp_path)))
