import os
import shutil
import pytest
import numpy as np
import calibration_store

@pytest.fixture(autouse=True)
def setup_teardown_tmp_dir(monkeypatch, tmp_path):
    # Route calibration storage to temp path during tests
    monkeypatch.setattr(calibration_store, "CALIBRATION_BASE_DIR", str(tmp_path / "calibration"))


def test_calibration_roundtrip():
    camera_id = 1
    # Create fake embedding store (150 frames, 196 patches, 384 embed_dim for dinov2-small)
    num_frames = 150
    num_patches = 196
    embed_dim = 384
    fake_store = [np.random.randn(num_patches, embed_dim).astype(np.float32) for _ in range(num_frames)]

    model_ver = "facebook/dinov2-small"
    ok, err = calibration_store.save_calibration(
        camera_id=camera_id,
        calibration_store=fake_store,
        dino_model_version=model_ver,
        grid_h=14, grid_w=14,
        frame_width=640, frame_height=480
    )
    assert ok, f"Save failed: {err}"

    # Load and validate
    is_valid, cal_arr, cal_store, meta, msg = calibration_store.load_calibration(
        camera_id=camera_id,
        current_dino_model=model_ver,
        current_grid_h=14, current_grid_w=14,
        current_frame_w=640, current_frame_h=480
    )

    assert is_valid
    assert len(cal_store) == num_frames
    assert cal_arr.shape == (num_frames, num_patches, embed_dim)
    assert "Calibrated — 150 frames" in msg
    assert np.allclose(cal_arr[0], fake_store[0])

    # Check on-disk size for 150 frames
    size_bytes = calibration_store.get_calibration_size_bytes(camera_id)
    size_mb = size_bytes / (1024 * 1024)
    print(f"\nOn-disk size for 150 frames: {size_mb:.2f} MB ({size_bytes} bytes)")
    assert size_mb < 100.0, f"Size {size_mb} MB exceeds 100MB limit (CAL-08)"


def test_calibration_metadata_mismatch():
    camera_id = 2
    fake_store = [np.random.randn(196, 384).astype(np.float32) for _ in range(20)]
    calibration_store.save_calibration(
        camera_id=camera_id,
        calibration_store=fake_store,
        dino_model_version="facebook/dinov2-small",
        grid_h=14, grid_w=14,
        frame_width=640, frame_height=480
    )

    # 1. Model version changed
    is_valid, _, _, _, msg = calibration_store.load_calibration(
        camera_id=camera_id,
        current_dino_model="facebook/dinov2-base",
        current_grid_h=14, current_grid_w=14,
        current_frame_w=640, current_frame_h=480
    )
    assert not is_valid
    assert "model version changed" in msg

    # 2. Camera resolution changed
    is_valid, _, _, _, msg = calibration_store.load_calibration(
        camera_id=camera_id,
        current_dino_model="facebook/dinov2-small",
        current_grid_h=14, current_grid_w=14,
        current_frame_w=1280, current_frame_h=720
    )
    assert not is_valid
    assert "resolution changed" in msg

    # Verify file was NOT deleted
    assert os.path.exists(calibration_store.get_camera_calibration_path(camera_id))


def test_atomic_write_simulation(monkeypatch, tmp_path):
    camera_id = 3
    fake_store = [np.random.randn(196, 384).astype(np.float32) for _ in range(10)]
    calibration_store.save_calibration(
        camera_id=camera_id,
        calibration_store=fake_store,
        dino_model_version="facebook/dinov2-small",
        grid_h=14, grid_w=14,
        frame_width=640, frame_height=480
    )

    # Replace os.replace with failure function
    orig_replace = os.replace
    def mock_replace(src, dst):
        raise OSError("Simulated power loss / crash before os.replace")

    os.replace = mock_replace
    try:
        # Try saving new frames
        new_store = [np.random.randn(196, 384).astype(np.float32) for _ in range(30)]
        ok, err = calibration_store.save_calibration(
            camera_id=camera_id,
            calibration_store=new_store,
            dino_model_version="facebook/dinov2-small",
            grid_h=14, grid_w=14,
            frame_width=640, frame_height=480
        )
        assert not ok
        assert "Simulated power loss" in err
    finally:
        os.replace = orig_replace

    # Verify old calibration file is still intact and readable
    is_valid, _, store, _, _ = calibration_store.load_calibration(
        camera_id=camera_id,
        current_dino_model="facebook/dinov2-small",
        current_grid_h=14, current_grid_w=14,
        current_frame_w=640, current_frame_h=480
    )
    assert is_valid
    assert len(store) == 10
