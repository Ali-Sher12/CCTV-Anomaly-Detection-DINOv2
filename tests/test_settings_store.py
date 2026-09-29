import os
import json
import pytest
import settings_store

def test_legacy_settings_persistence_ignored(tmp_path, monkeypatch):
    test_file = str(tmp_path / "settings.json")
    monkeypatch.setattr(settings_store, "SETTINGS_FILE", test_file)

    legacy_data = {
        "setup_complete": True,
        "inter_camera_delay": 0.5,
        "cameras": [
            {
                "id": 1,
                "name": "Camera 1",
                "source": 0,
                "mask_path": "Assets/masks/camera_1_mask.png",
                "required_persistence": 5, # Legacy key
                "allowed_error": 1,
                "anomaly_report_wait": 10
            }
        ]
    }

    with open(test_file, "w") as f:
        json.dump(legacy_data, f)

    loaded = settings_store.load_settings()
    assert loaded is not None
    assert "required_persistence" not in loaded["cameras"][0]

    # Save and verify it is not written back
    settings_store.save_settings(loaded)
    with open(test_file, "r") as f:
        reloaded_raw = json.load(f)
    assert "required_persistence" not in reloaded_raw["cameras"][0]
