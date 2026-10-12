"""Shared fixtures for the PC kit tests (CPU only; nothing touches a GPU).

Run from the repo root:  <python> -m pytest -q -p no:cacheprovider pc/tests
"""
import json
import os
import stat
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "wsl"))

IDLE = {"temp_c": 45, "mem_used_mib": 600, "mem_total_mib": 12227, "util_pct": 0,
        "power_w": 18, "throttle": "0x0000000000000001"}


class Smi:
    """Handle on the fake nvidia-smi: smi.set(temp_c=95) changes what the next query sees."""

    def __init__(self, path):
        self.path = path
        self.state = dict(IDLE)
        self._write()

    def _write(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f)
        os.replace(tmp, self.path)

    def set(self, **kw):
        self.state.update(kw)
        self._write()


@pytest.fixture
def smi(tmp_path, monkeypatch):
    import gpuguard
    wrapper = tmp_path / "nvidia-smi"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{HERE}/fake_nvidia_smi.py" "$@"\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    s = Smi(str(tmp_path / "smi_state.json"))
    monkeypatch.setenv("PLANCK_NVIDIA_SMI", str(wrapper))
    monkeypatch.setenv("FAKE_SMI_STATE", s.path)
    gpuguard._throttle_field.clear()
    yield s
    gpuguard._throttle_field.clear()
