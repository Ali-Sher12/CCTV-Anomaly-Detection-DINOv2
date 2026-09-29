import os
import json
import pytest
import time
from pathlib import Path
from watchdog import (
    load_watchdog_config,
    is_heartbeat_stale,
    load_restart_log,
    save_restart_log,
    clean_and_count_restarts,
    should_trigger_loop_guard,
    WatchdogService,
)


def test_config_validation(tmp_path):
    # 1. Valid config
    valid_cfg = {
        "heartbeat_interval": 5,
        "stale_after": 60,
        "check_interval": 20,
        "app_install_path": str(tmp_path),
        "app_entrypoint": ["python", "main.py"]
    }
    cfg_file = tmp_path / "watchdog_config.json"
    cfg_file.write_text(json.dumps(valid_cfg))
    loaded = load_watchdog_config(cfg_file)
    assert loaded["stale_after"] == 60
    assert loaded["heartbeat_interval"] == 5

    # 2. Invalid ratio (stale_after < 3 * heartbeat_interval)
    invalid_ratio = dict(valid_cfg, heartbeat_interval=10, stale_after=25)
    cfg_file.write_text(json.dumps(invalid_ratio))
    with pytest.raises(ValueError, match="must be at least 3x"):
        load_watchdog_config(cfg_file)

    # 3. Missing app_install_path
    missing_path = dict(valid_cfg, app_install_path="")
    cfg_file.write_text(json.dumps(missing_path))
    with pytest.raises(ValueError, match="'app_install_path' is required"):
        load_watchdog_config(cfg_file)

    # 4. Missing app_entrypoint
    missing_entry = dict(valid_cfg, app_entrypoint=[])
    cfg_file.write_text(json.dumps(missing_entry))
    with pytest.raises(ValueError, match="'app_entrypoint' must be a non-empty list"):
        load_watchdog_config(cfg_file)


def test_heartbeat_staleness_classification(tmp_path):
    hb_file = tmp_path / "heartbeat.txt"
    stale_threshold = 60.0

    # 1. Missing file -> stale
    is_stale, msg = is_heartbeat_stale(hb_file, stale_threshold, current_time=1000.0)
    assert is_stale
    assert "does not exist" in msg

    # 2. Fresh heartbeat (written at t=1000.0, checked at t=1020.0, age=20s)
    hb_file.write_text("1000.0\n")
    is_stale, msg = is_heartbeat_stale(hb_file, stale_threshold, current_time=1020.0)
    assert not is_stale
    assert "healthy" in msg

    # 3. Exactly at threshold (age=60s) -> healthy / not stale
    is_stale, msg = is_heartbeat_stale(hb_file, stale_threshold, current_time=1060.0)
    assert not is_stale

    # 4. Just after threshold (age=60.1s) -> stale
    is_stale, msg = is_heartbeat_stale(hb_file, stale_threshold, current_time=1060.1)
    assert is_stale
    assert "stale" in msg


def test_restart_history_persistence_and_corruption(tmp_path):
    log_file = tmp_path / "Assets" / "watchdog_restart_log.json"

    # 1. Missing file -> returns empty list
    assert load_restart_log(log_file) == []

    # 2. Save and reload round-trip
    history = [100.0, 200.5, 300.0]
    save_restart_log(log_file, history)
    loaded = load_restart_log(log_file)
    assert loaded == history

    # 3. Corrupt file -> returns empty list without crashing (WD-12)
    log_file.write_text("{ corrupt json: [ }")
    assert load_restart_log(log_file) == []


def test_restart_loop_guard_window(tmp_path):
    now = 1000.0
    window = 600.0  # 10 minutes (valid range: [400.0, 1000.0])

    past_restarts = [
        300.0,  # Old (outside window: 1000 - 300 = 700 > 600) -> discarded
        399.9,  # Old -> discarded
        400.0,  # Exactly at edge (1000 - 400 = 600) -> kept
        700.0,  # Inside window -> kept
        950.0,  # Inside window -> kept
    ]

    recent = clean_and_count_restarts(past_restarts, window, now)
    assert recent == [400.0, 700.0, 950.0]
    assert len(recent) == 3

    # max_restarts = 3 -> Should trip guard
    assert should_trigger_loop_guard(len(recent), max_restarts=3)
    # max_restarts = 4 -> Should NOT trip
    assert not should_trigger_loop_guard(len(recent), max_restarts=4)


def test_watchdog_service_state_machine(tmp_path, monkeypatch):
    # Setup test workspace
    assets = tmp_path / "Assets"
    assets.mkdir(parents=True, exist_ok=True)
    hb_path = assets / "heartbeat.txt"
    pid_path = assets / "app.pid"

    cfg = {
        "heartbeat_interval": 5,
        "stale_after": 60,
        "check_interval": 20,
        "startup_grace": 0,
        "restart_grace": 10,
        "terminate_timeout": 1,
        "max_restarts": 2,
        "restart_window": 300,
        "app_install_path": str(tmp_path),
        "app_entrypoint": ["echo", "test"],
        "heartbeat_path": "Assets/heartbeat.txt",
        "pid_path": "Assets/app.pid",
        "email": None,
    }

    # Inject mock clock and sleep
    current_sim_time = 1000.0

    def mock_clock():
        return current_sim_time

    def mock_sleep(dt):
        nonlocal current_sim_time
        current_sim_time += dt

    service = WatchdogService(cfg, clock=mock_clock, sleeper=mock_sleep)

    # 1. Initially healthy
    hb_path.write_text(f"{current_sim_time}\n")
    state = service.run_single_check()
    assert state == "HEALTHY"
    assert not service.in_down_episode

    # 2. Advance time past stale_after (age = 70s > 60s)
    current_sim_time += 70.0
    state = service.run_single_check()
    assert state == "RESTARTED"
    assert service.in_down_episode
    # Restart log should have 1 entry
    log = load_restart_log(service.restart_log_path)
    assert len(log) == 1

    # 3. Next check still stale -> second restart allowed
    current_sim_time += 70.0
    state = service.run_single_check()
    assert state == "RESTARTED"
    log = load_restart_log(service.restart_log_path)
    assert len(log) == 2

    # 4. Next check still stale -> max_restarts (2) reached -> Loop guard trips!
    current_sim_time += 70.0
    state = service.run_single_check()
    assert state == "LOOP_GUARD_TRIPPED"
    assert service.loop_guard_tripped

    # 5. Heartbeat restored -> Recovery!
    hb_path.write_text(f"{current_sim_time}\n")
    state = service.run_single_check()
    assert state == "HEALTHY"
    assert not service.in_down_episode
    assert not service.loop_guard_tripped
