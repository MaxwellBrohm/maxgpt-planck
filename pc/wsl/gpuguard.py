"""nvidia-smi sampling and the GPU guard for the Planck PC queue runner (stdlib only).

The runner samples the GPU every few seconds while a job runs and feeds each sample to a
Guard. The guard trips (and the runner stops the job) when:
  temp      GPU temperature >= max_temp_c for temp_strikes samples in a row
  gpu_mem   GPU memory used > max_mem_mib for mem_strikes samples in a row (runaway memory)
  host_mem  WSL MemAvailable < min_host_avail_mib for host_strikes samples in a row
  smi       nvidia-smi failed, or reported no temperature, smi_fail_strikes times in a row
            (a job the guard cannot see is a job it cannot protect)

nvidia-smi path: $PLANCK_NVIDIA_SMI, else "nvidia-smi" on PATH. In WSL2 that is
/usr/lib/wsl/lib/nvidia-smi. If WSL's copy ever reports [N/A] for temperature, point
PLANCK_NVIDIA_SMI at the Windows one: /mnt/c/Windows/System32/nvidia-smi.exe.
Tests use tests/fake_nvidia_smi.py through the same variable.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass, fields

FIELDS = ["temperature.gpu", "memory.used", "memory.total", "utilization.gpu", "power.draw"]
KEYS = ["temp_c", "mem_used_mib", "mem_total_mib", "util_pct", "power_w"]
# Newer drivers renamed clocks_throttle_reasons to clocks_event_reasons; asking for an
# unknown field fails the whole query, so throttle state is a separate, optional query.
THROTTLE_FIELDS = ["clocks_event_reasons.active", "clocks_throttle_reasons.active"]
THROTTLE_BITS = {0x1: "idle", 0x2: "app_clocks", 0x4: "sw_power_cap", 0x8: "hw_slowdown",
                 0x10: "sync_boost", 0x20: "sw_thermal", 0x40: "hw_thermal",
                 0x80: "hw_power_brake", 0x100: "display_clocks"}
_throttle_field: list[str | None] = []      # cache: the field name that worked, or None


def smi_path() -> str:
    return os.environ.get("PLANCK_NVIDIA_SMI", "nvidia-smi")


def _smi(query: str, timeout: float) -> str | None:
    cmd = [smi_path(), f"--query-gpu={query}", "--format=csv,noheader,nounits", "-i", "0"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    return r.stdout.strip().splitlines()[0]


def _num(s: str) -> float | None:
    s = s.strip()
    if not s or s.startswith("[") or s.upper() in ("N/A", "NOT SUPPORTED"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def decode_throttle(mask: str | None) -> list[str]:
    if not mask:
        return []
    try:
        v = int(mask.strip(), 16)
    except ValueError:
        return []
    return [name for bit, name in THROTTLE_BITS.items() if v & bit]


def throttle(timeout: float = 10.0) -> str | None:
    """Active clock-event (throttle) bitmask as nvidia-smi prints it, or None."""
    if _throttle_field and _throttle_field[0] is None:
        return None
    for f in (_throttle_field or THROTTLE_FIELDS):
        line = _smi(f, timeout)
        if line is not None and line.strip().lower().startswith("0x"):
            if not _throttle_field:
                _throttle_field.append(f)
            return line.strip()
    if not _throttle_field:
        _throttle_field.append(None)
    return None


def query(timeout: float = 10.0, with_throttle: bool = False) -> dict | None:
    """One GPU sample, or None when nvidia-smi is missing, hangs or fails."""
    line = _smi(",".join(FIELDS), timeout)
    if line is None:
        return None
    parts = line.split(",")
    if len(parts) != len(FIELDS):
        return None
    s = {k: _num(v) for k, v in zip(KEYS, parts)}
    if with_throttle:
        m = throttle(timeout)
        s["throttle"] = m
        s["throttle_reasons"] = decode_throttle(m)
    return s


@dataclass
class GuardConfig:
    max_temp_c: float = 87.0         # RTX 5070 slows itself near 90 C; stop before that
    temp_strikes: int = 3            # consecutive hot samples before the stop
    resume_temp_c: float = 75.0      # nothing starts until the GPU is at or below this
    max_mem_mib: float = 11600.0     # 12 GB card reports ~12227 MiB total; above this = runaway
    mem_strikes: int = 3
    start_max_mem_mib: float = 3000.0  # refuse to start while something else holds this much
    min_host_avail_mib: float = 1024.0  # WSL MemAvailable floor (the 24 GB .wslconfig cap)
    host_strikes: int = 3
    smi_fail_strikes: int = 6
    require_temp: bool = True

    @classmethod
    def from_dict(cls, d: dict | None) -> "GuardConfig":
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        bad = set(d) - known
        if bad:
            raise ValueError(f"unknown guard keys: {sorted(bad)}")
        return cls(**d)

    def merged(self, over: dict | None) -> "GuardConfig":
        return GuardConfig.from_dict({**asdict(self), **(over or {})})


class Guard:
    """Feed one sample per poll; check() returns a trip reason or None."""

    def __init__(self, cfg: GuardConfig):
        self.cfg = cfg
        self.strikes = {"temp": 0, "gpu_mem": 0, "host_mem": 0, "smi": 0}
        self.peak = {"temp_c": None, "mem_used_mib": None}

    def _count(self, key: str, bad: bool, limit: int) -> bool:
        self.strikes[key] = self.strikes[key] + 1 if bad else 0
        return self.strikes[key] >= limit

    def check(self, sample: dict | None, host_avail_mib: float | None = None) -> str | None:
        c = self.cfg
        temp = None if sample is None else sample.get("temp_c")
        mem = None if sample is None else sample.get("mem_used_mib")
        for k, v in (("temp_c", temp), ("mem_used_mib", mem)):
            if v is not None and (self.peak[k] is None or v > self.peak[k]):
                self.peak[k] = v
        blind = sample is None or (c.require_temp and temp is None)
        trips = []
        if self._count("smi", blind, c.smi_fail_strikes):
            trips.append("smi")
        if self._count("temp", temp is not None and temp >= c.max_temp_c, c.temp_strikes):
            trips.append("temp")
        if self._count("gpu_mem", mem is not None and mem > c.max_mem_mib, c.mem_strikes):
            trips.append("gpu_mem")
        low = host_avail_mib is not None and host_avail_mib < c.min_host_avail_mib
        if self._count("host_mem", low, c.host_strikes):
            trips.append("host_mem")
        return trips[0] if trips else None


def may_start(sample: dict | None, cfg: GuardConfig) -> tuple[bool, str]:
    """Is the GPU in a state where a new job may start?"""
    if sample is None:
        return False, "nvidia-smi failed"
    temp, mem = sample.get("temp_c"), sample.get("mem_used_mib")
    if temp is None and cfg.require_temp:
        return False, "nvidia-smi reports no temperature (see PLANCK_NVIDIA_SMI)"
    if temp is not None and temp > cfg.resume_temp_c:
        return False, f"GPU at {temp:.0f} C, waiting for <= {cfg.resume_temp_c:.0f} C"
    if mem is not None and mem > cfg.start_max_mem_mib:
        return False, f"GPU already has {mem:.0f} MiB in use (> {cfg.start_max_mem_mib:.0f})"
    return True, "ok"
