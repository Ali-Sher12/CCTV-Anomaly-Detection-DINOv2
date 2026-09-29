import os
from pathlib import Path
import tkinter as tk
import Accessories as Acc
import Globals as gb
from settings_store import load_settings
from DINO import ModelBundle, CameraFeed
from feed_manager import FeedManager
from audio import AudioEngine
from gui import AnomalyDetectionGUI
from loading_screen import LoadingScreen
from capture_registry import CaptureRegistry


def _build_camera_feed(cam_cfg: dict, bundle: ModelBundle,
                       settings: dict) -> CameraFeed:
    email_cfg = settings.get("email", {})
    return CameraFeed(
        camera_id                = cam_cfg["id"],
        name                     = cam_cfg.get("name", f"Camera {cam_cfg['id']}"),
        source                   = cam_cfg["source"],
        mask_path                = cam_cfg["mask_path"],
        model_bundle             = bundle,
        thresholds               = cam_cfg.get("thresholds"),
        allowed_error            = cam_cfg.get("allowed_error", 1),
        anomaly_report_wait      = cam_cfg.get("anomaly_report_wait", 10),
        total_calibration_frames = settings.get("total_calibration_frames", 100),
        dino_only                = settings.get("dino_only", False),
        email_cfg                = email_cfg,
    )


def main():
    # Write PID early at startup (HB-04, HB-05, HB-06)
    base_dir = Path(__file__).resolve().parent
    assets_dir = base_dir / "Assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    pid_path = assets_dir / "app.pid"
    tmp_pid = assets_dir / f"app.pid.{os.getpid()}.tmp"
    try:
        tmp_pid.write_text(str(os.getpid()))
        os.replace(tmp_pid, pid_path)
    except Exception as e:
        print(f"[main] Warning: could not write app.pid: {e}")

    root = tk.Tk()
    root.withdraw()

    # 1. Setup Wizard check
    data = load_settings()
    if data is None:
        from setup_wizard import SetupWizard
        root.deiconify()
        wizard = SetupWizard(root)
        root.wait_window(wizard)
        root.withdraw()
        data = load_settings()
        if data is None:
            root.destroy()
            return

    # 2. Global defaults
    dt = data.get("default_thresholds", {})
    gb.TIER_THRESHOLDS["ALERT"]                  = dt.get("ALERT",    45.0)
    gb.TIER_THRESHOLDS["CRITICAL"]               = dt.get("CRITICAL", 50.0)
    gb.TIER_THRESHOLDS_HIGH_PRIORITY["ALERT"]    = dt.get("HIGH_PRIORITY_ALERT",    35.0)
    gb.TIER_THRESHOLDS_HIGH_PRIORITY["CRITICAL"] = dt.get("HIGH_PRIORITY_CRITICAL", 40.0)

    email_cfg = data.get("email", {})
    if email_cfg.get("sender"):
        gb.EMAIL_SENDER   = email_cfg["sender"]
    if email_cfg.get("app_password"):
        gb.EMAIL_PASSWORD = email_cfg["app_password"]
    receivers = email_cfg.get("receivers", [])
    if receivers:
        gb.EMAIL_RECEIVER = receivers[0]

    # 3. Loading screen
    loading = LoadingScreen(root)

    loading.set_status("Checking hardware specs...")
    Acc.getnSetSpecs()
    if data.get("dino_model_version"):
        gb.DINO_MODEL_VERSION = data["dino_model_version"]
    if data.get("yolo_model_version"):
        gb.YOLO_MODEL_VERSION = data["yolo_model_version"]
    if "dino_only" in data:
        gb.DINO_ONLY = data["dino_only"]

    loading.set_status("Loading DINO & YOLO AI models...")
    bundle = ModelBundle()

    loading.set_status("Building camera feeds...")
    cams_cfg = data.get("cameras", [])
    feeds = []
    for cam_cfg in cams_cfg:
        try:
            feed = _build_camera_feed(cam_cfg, bundle, data)
            feeds.append(feed)
        except Exception as e:
            print(f"[main] Failed to init camera {cam_cfg.get('id')}: {e}")

    if not feeds:
        print("[main] No cameras configured. Exiting.")
        loading.close()
        root.destroy()
        return

    loading.set_status("Initializing audio engine...")
    audio = AudioEngine()

    loading.set_status("Initializing user interface...")
    gui_inst = AnomalyDetectionGUI(root,
                                   cameras=cams_cfg,
                                   settings_data=data)
    gb.gui = gui_inst
    gui_inst.set_feeds(feeds)

    Acc.printSpecs()

    loading.close()
    root.deiconify()

    # 4. Feed manager & Scheduler
    inter_delay = data.get("inter_camera_delay", 0.5)
    manager = FeedManager(feeds, bundle, gui_inst, audio,
                          inter_camera_delay=inter_delay)
    gui_inst.set_feed_manager(manager)
    manager.start(root)

    root.bind("<Escape>", lambda e: gui_inst.on_closing())
    root.mainloop()

    # Clean shutdown
    CaptureRegistry().release_all()


if __name__ == "__main__":
    main()

