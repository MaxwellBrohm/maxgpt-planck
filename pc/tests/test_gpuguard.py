"""gpuguard: nvidia-smi parsing (through the fake tool) and the guard's strike logic."""
import pytest

import gpuguard
from gpuguard import Guard, GuardConfig, decode_throttle, may_start


def test_query_parses_fields_in_order(smi):
    smi.set(temp_c=63, mem_used_mib=4321, util_pct=97, power_w=201.5)
    s = gpuguard.query()
    assert s == {"temp_c": 63.0, "mem_used_mib": 4321.0, "mem_total_mib": 12227.0,
                 "util_pct": 97.0, "power_w": 201.5}


def test_query_na_temperature_is_none(smi):
    smi.set(temp_na=True)
    assert gpuguard.query()["temp_c"] is None


def test_query_failure_and_missing_binary(smi, monkeypatch, tmp_path):
    smi.set(fail=True)
    assert gpuguard.query() is None
    monkeypatch.setenv("PLANCK_NVIDIA_SMI", str(tmp_path / "does-not-exist"))
    assert gpuguard.query() is None


def test_throttle_new_and_old_field_names(smi):
    smi.set(throttle="0x0000000000000024")
    s = gpuguard.query(with_throttle=True)
    assert s["throttle_reasons"] == ["sw_power_cap", "sw_thermal"]
    assert gpuguard._throttle_field == ["clocks_event_reasons.active"]
    gpuguard._throttle_field.clear()
    smi.set(old_driver=True, throttle="0x0000000000000040")
    s = gpuguard.query(with_throttle=True)
    assert s["throttle_reasons"] == ["hw_thermal"]
    assert gpuguard._throttle_field == ["clocks_throttle_reasons.active"]


def test_decode_throttle_garbage():
    assert decode_throttle(None) == [] and decode_throttle("[N/A]") == []
    assert decode_throttle("0x1") == ["idle"]


def hot(t):
    return {"temp_c": t, "mem_used_mib": 1000}


def test_temp_trips_on_third_consecutive_hot_sample():
    g = Guard(GuardConfig(max_temp_c=87, temp_strikes=3))
    assert g.check(hot(87)) is None
    assert g.check(hot(90)) is None
    assert g.check(hot(88)) == "temp"
    assert g.peak["temp_c"] == 90


def test_temp_strikes_reset_by_a_cool_sample():
    g = Guard(GuardConfig(max_temp_c=87, temp_strikes=3))
    for t in (88, 88, 86, 88, 88):
        assert g.check(hot(t)) is None
    assert g.check(hot(88)) == "temp"


def test_below_threshold_never_trips():
    g = Guard(GuardConfig(max_temp_c=87, temp_strikes=1))
    assert all(g.check(hot(86.9)) is None for _ in range(50))


def test_gpu_memory_runaway():
    g = Guard(GuardConfig(max_mem_mib=11600, mem_strikes=2))
    assert g.check({"temp_c": 60, "mem_used_mib": 11600}) is None     # at the cap is allowed
    assert g.check({"temp_c": 60, "mem_used_mib": 11700}) is None
    assert g.check({"temp_c": 60, "mem_used_mib": 11900}) == "gpu_mem"


def test_host_memory_floor():
    g = Guard(GuardConfig(min_host_avail_mib=1024, host_strikes=2))
    assert g.check(hot(50), 900) is None
    assert g.check(hot(50), 800) == "host_mem"
    g2 = Guard(GuardConfig(min_host_avail_mib=1024, host_strikes=1))
    assert g2.check(hot(50), None) is None                        # no /proc/meminfo: no check


def test_blind_guard_trips_smi():
    g = Guard(GuardConfig(smi_fail_strikes=3))
    assert g.check(None) is None and g.check(None) is None
    assert g.check(None) == "smi"
    g2 = Guard(GuardConfig(smi_fail_strikes=2))
    assert g2.check({"temp_c": None, "mem_used_mib": 5}) is None
    assert g2.check({"temp_c": None, "mem_used_mib": 5}) == "smi"   # no temperature = blind
    g3 = Guard(GuardConfig(smi_fail_strikes=1, require_temp=False))
    assert g3.check({"temp_c": None, "mem_used_mib": 5}) is None


def test_may_start():
    c = GuardConfig(resume_temp_c=75, start_max_mem_mib=3000)
    assert may_start({"temp_c": 75, "mem_used_mib": 3000}, c) == (True, "ok")
    assert not may_start({"temp_c": 76, "mem_used_mib": 100}, c)[0]
    assert not may_start({"temp_c": 40, "mem_used_mib": 3001}, c)[0]
    assert not may_start(None, c)[0]
    assert not may_start({"temp_c": None, "mem_used_mib": 1}, c)[0]


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError):
        GuardConfig.from_dict({"max_temp": 80})
    assert GuardConfig().merged({"max_temp_c": 80}).max_temp_c == 80
