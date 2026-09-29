"""settings_store.py — Persists app settings to JSON (schema v2).

load_settings() -> returns settings dict if setup_complete is True, else None.
save_settings(data) -> writes the clean dict without obsolete persistence keys.
"""

import json
import os

SETTINGS_FILE = os.path.join("Assets", "settings.json")


def load_settings() -> dict | None:
    """Return the settings dict, or None if wizard must run."""
    if not os.path.exists(SETTINGS_FILE):
        return None
    try:
        with open(SETTINGS_FILE, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    if not isinstance(data, dict) or not data.get("setup_complete", False):
        return None

    # Strip legacy required_persistence if present (PER-03)
    if "cameras" in data and isinstance(data["cameras"], list):
        for cam in data["cameras"]:
            cam.pop("required_persistence", None)

    return data


def save_settings(data: dict):
    """Write settings dict to disk, filtering out obsolete keys."""
    if not isinstance(data, dict):
        return

    # Deep clean cameras list
    if "cameras" in data and isinstance(data["cameras"], list):
        for cam in data["cameras"]:
            cam.pop("required_persistence", None)

    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def default_settings() -> dict:
    """Return a bare-minimum settings skeleton for the wizard."""
    return {
        "setup_complete": False,
        "device_tier": "MID",
        "dino_model_version": "Models/DINO/dinov3-vitb16",
        "yolo_model_version": "Models/YOLO/yolo26m.pt",
        "dino_only": False,
        "total_calibration_frames": 100,
        "inter_camera_delay": 0.5,
        "email": {
            "sender": "",
            "app_password": "",
            "receivers": []
        },
        "default_thresholds": {
            "ALERT": 45.0,
            "CRITICAL": 50.0,
            "HIGH_PRIORITY_ALERT": 35.0,
            "HIGH_PRIORITY_CRITICAL": 40.0
        },
        "cameras": []
    }

