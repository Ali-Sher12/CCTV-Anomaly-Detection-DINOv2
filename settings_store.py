"""Persists GUI settings to a JSON file so they survive app restarts.

Reads on import, writes on save_settings(). All values fall back to
the current Globals.py defaults if the file doesn't exist or a key
is missing.
"""

import json
import os
import Globals as gb

SETTINGS_FILE = "Assets/settings.json"

# Keys that map directly to gb.* attributes
_RUNTIME_KEYS = {
    "secondsForOneFrame": float,
    "delay": int,
    "REQUIRED_PERSISTENCE": int,
    "allowed_error": int,
    "anomaly_report_wait": int,
    "auto_update_calibration": bool,
}

_THRESHOLD_KEYS = {
    "TIER_THRESHOLDS_ALERT": float,
    "TIER_THRESHOLDS_CRITICAL": float,
    "TIER_THRESHOLDS_HIGH_PRIORITY_ALERT": float,
    "TIER_THRESHOLDS_HIGH_PRIORITY_CRITICAL": float,
}

_RESTART_KEYS = {
    "totalCalibrationFrames": int,
    "doVideoStream": bool,
    "DINO_ONLY": bool,
    "stream_url": str,
    "EMAIL_SENDER": str,
    "EMAIL_PASSWORD": str,
    "EMAIL_RECEIVER": str,
}


def load_settings():
    """Load saved settings from disk and apply them to Globals."""
    if not os.path.exists(SETTINGS_FILE):
        return

    try:
        with open(SETTINGS_FILE, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return

    # Runtime
    for key, cast in _RUNTIME_KEYS.items():
        if key in data:
            setattr(gb, key, cast(data[key]))

    # Thresholds (flat keys → nested dicts)
    if "TIER_THRESHOLDS_ALERT" in data:
        gb.TIER_THRESHOLDS["ALERT"] = float(data["TIER_THRESHOLDS_ALERT"])
    if "TIER_THRESHOLDS_CRITICAL" in data:
        gb.TIER_THRESHOLDS["CRITICAL"] = float(data["TIER_THRESHOLDS_CRITICAL"])
    if "TIER_THRESHOLDS_HIGH_PRIORITY_ALERT" in data:
        gb.TIER_THRESHOLDS_HIGH_PRIORITY["ALERT"] = float(data["TIER_THRESHOLDS_HIGH_PRIORITY_ALERT"])
    if "TIER_THRESHOLDS_HIGH_PRIORITY_CRITICAL" in data:
        gb.TIER_THRESHOLDS_HIGH_PRIORITY["CRITICAL"] = float(data["TIER_THRESHOLDS_HIGH_PRIORITY_CRITICAL"])

    # Restart-required
    for key, cast in _RESTART_KEYS.items():
        if key == "stream_url":
            continue  # handled separately
        if key in data:
            setattr(gb, key, cast(data[key]))

    if "stream_url" in data and gb.doVideoStream:
        gb.url = data["stream_url"]
    elif not gb.doVideoStream:
        gb.url = 0

    return data  # return raw dict so GUI can pull stream_url etc.


def save_settings(stream_url=""):
    """Snapshot all current gb.* values to the settings file."""
    data = {}

    # Runtime
    for key in _RUNTIME_KEYS:
        data[key] = getattr(gb, key)

    # Thresholds
    data["TIER_THRESHOLDS_ALERT"] = gb.TIER_THRESHOLDS["ALERT"]
    data["TIER_THRESHOLDS_CRITICAL"] = gb.TIER_THRESHOLDS["CRITICAL"]
    data["TIER_THRESHOLDS_HIGH_PRIORITY_ALERT"] = gb.TIER_THRESHOLDS_HIGH_PRIORITY["ALERT"]
    data["TIER_THRESHOLDS_HIGH_PRIORITY_CRITICAL"] = gb.TIER_THRESHOLDS_HIGH_PRIORITY["CRITICAL"]

    # Restart-required
    for key in _RESTART_KEYS:
        if key == "stream_url":
            data["stream_url"] = stream_url
        else:
            data[key] = getattr(gb, key)

    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)
