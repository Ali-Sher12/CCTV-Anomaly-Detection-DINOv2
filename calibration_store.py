"""calibration_store.py — Per-camera on-disk calibration storage and validation.

Implements CAL-01 to CAL-08:
- Directory structure: Assets/calibration/camera_{id}/calibration.npz
- Metadata: schema_version, camera_id, dino_model_version, grid_h, grid_w, frame_width, frame_height, num_frames, last_updated (ISO 8601 UTC)
- Atomic file writes using temp file + os.replace
- Validates model version, grid size, and frame resolution at startup
- Preserves files on invalid resolution or model mismatch (no auto-deletion)
"""

import os
import json
import tempfile
import datetime
import numpy as np

CALIBRATION_SCHEMA_VERSION = 2
CALIBRATION_BASE_DIR = os.path.join("Assets", "calibration")


def get_camera_calibration_dir(camera_id: int) -> str:
    return os.path.join(CALIBRATION_BASE_DIR, f"camera_{camera_id}")


def get_camera_calibration_path(camera_id: int) -> str:
    return os.path.join(get_camera_calibration_dir(camera_id), "calibration.npz")


def save_calibration(camera_id: int,
                     calibration_store: list[np.ndarray],
                     dino_model_version: str,
                     grid_h: int,
                     grid_w: int,
                     frame_width: int,
                     frame_height: int) -> tuple[bool, str]:
    """Atomically save calibration store and metadata to disk using np.savez_compressed."""
    if not calibration_store:
        return False, "Cannot save empty calibration store"

    cam_dir = get_camera_calibration_dir(camera_id)
    os.makedirs(cam_dir, exist_ok=True)
    target_path = get_camera_calibration_path(camera_id)

    metadata = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "camera_id": camera_id,
        "dino_model_version": dino_model_version,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "num_frames": len(calibration_store),
        "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

    # Stack into numpy array of shape (N, num_patches, embed_dim)
    calibration_array = np.stack(calibration_store, axis=0)
    meta_json = json.dumps(metadata)

    # Write to a temporary file in the exact same directory, then atomic os.replace
    temp_fd, temp_path = tempfile.mkstemp(prefix="calib_tmp_", suffix=".npz", dir=cam_dir)
    os.close(temp_fd)

    try:
        np.savez_compressed(temp_path,
                            calibration_array=calibration_array,
                            metadata=meta_json)
        os.replace(temp_path, target_path)
        return True, ""
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        return False, f"Failed to save calibration: {e}"


def load_calibration(camera_id: int,
                     current_dino_model: str,
                     current_grid_h: int,
                     current_grid_w: int,
                     current_frame_w: int,
                     current_frame_h: int) -> tuple[bool, np.ndarray | None, list[np.ndarray], dict, str]:
    """Loads and validates calibration for camera_id.
    
    Returns (is_valid, calibration_array, calibration_store, metadata, status_reason).
    """
    path = get_camera_calibration_path(camera_id)
    if not os.path.exists(path):
        return False, None, [], {}, "Not calibrated — no saved data"

    try:
        with np.load(path, allow_pickle=False) as data:
            if "calibration_array" not in data or "metadata" not in data:
                return False, None, [], {}, "Calibration corrupt — missing arrays"

            cal_arr = data["calibration_array"]
            meta_str = str(data["metadata"])
            meta = json.loads(meta_str)

        # Validate metadata against current runtime parameters
        if meta.get("dino_model_version") != current_dino_model:
            return False, None, [], meta, f"Calibration invalid — model version changed (saved '{meta.get('dino_model_version')}', now '{current_dino_model}')"

        if meta.get("grid_h") != current_grid_h or meta.get("grid_w") != current_grid_w:
            return False, None, [], meta, f"Calibration invalid — grid dimensions changed (saved {meta.get('grid_w')}x{meta.get('grid_h')}, now {current_grid_w}x{current_grid_h})"

        if current_frame_w > 0 and current_frame_h > 0:
            if meta.get("frame_width") != current_frame_w or meta.get("frame_height") != current_frame_h:
                return False, None, [], meta, f"Calibration invalid — camera resolution changed (saved {meta.get('frame_width')}x{meta.get('frame_height')}, now {current_frame_w}x{current_frame_h})"

        cal_store = [cal_arr[i] for i in range(cal_arr.shape[0])]
        updated_ts = meta.get("last_updated", "")
        # Format human-friendly timestamp
        if "T" in updated_ts:
            date_part, time_part = updated_ts.split("T")
            time_clean = time_part.split(".")[0]
            ts_str = f"{date_part} {time_clean} UTC"
        else:
            ts_str = updated_ts

        status_msg = f"Calibrated — {len(cal_store)} frames, updated {ts_str}"
        return True, cal_arr, cal_store, meta, status_msg

    except Exception as e:
        return False, None, [], {}, f"Calibration read error: {e}"


def delete_camera_calibration(camera_id: int) -> tuple[bool, str]:
    """Permanently delete calibration files for a specific camera."""
    path = get_camera_calibration_path(camera_id)
    if os.path.exists(path):
        try:
            os.remove(path)
            return True, ""
        except Exception as e:
            return False, f"Failed to delete calibration file: {e}"
    return True, ""


def get_calibration_size_bytes(camera_id: int) -> int:
    path = get_camera_calibration_path(camera_id)
    if os.path.exists(path):
        return os.path.getsize(path)
    return 0

