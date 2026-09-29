"""feed_manager.py — Staggered multi-camera orchestrator and scheduler.

Integrates:
- StaggeredScheduler (SCH-04, SCH-05, SCH-06, SCH-07)
- InferenceWorker (SCH-08)
- Stale highlight layer compositing onto live video (SCH-09)
- Non-blocking frame display & audio/email triggering on detection finish (SCH-10)
"""

import os
import json
import time
from pathlib import Path
import cv2
import numpy as np
import Globals as gb
from scheduler import StaggeredScheduler
from inference_worker import InferenceWorker, InferenceTask
from calibration_session import SessionState


_OFFLINE_BG = (30, 30, 30)
_OFFLINE_COLOR = (255, 255, 255)


def _make_offline_placeholder(w: int, h: int, camera_id: int, reason: str) -> np.ndarray:
    img = np.full((h, w, 3), _OFFLINE_BG, dtype=np.uint8)
    msg = f"Camera {camera_id}"
    sub = f"Offline — {reason}"
    reconn = "Reconnecting..."
    cv2.putText(img, msg, (20, h // 2 - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, _OFFLINE_COLOR, 2)
    cv2.putText(img, sub, (20, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 1)
    cv2.putText(img, reconn, (20, h // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    return img


class FeedManager:
    """Orchestrates video rendering, scheduler timing, and worker inference."""

    TICK_MS = 30  # ms between display refresh ticks (~33 fps)

    def __init__(self, feeds, model_bundle, gui, audio_engine,
                 inter_camera_delay: float = 0.5):
        self.feeds = feeds                          # list[CameraFeed]
        self.bundle = model_bundle                  # ModelBundle (shared)
        self.gui = gui                              # AnomalyDetectionGUI
        self.audio = audio_engine                   # AudioEngine
        self.worker = InferenceWorker(model_bundle) # Dedicated inference thread
        self.scheduler = StaggeredScheduler(inter_camera_delay=inter_camera_delay)
        self._root = None
        self._last_warning = ""

        # Heartbeat configuration (HB-01, HB-02, HB-03, HB-05, HB-06)
        self._base_dir = Path(__file__).resolve().parent
        self._assets_dir = self._base_dir / "Assets"
        self._assets_dir.mkdir(parents=True, exist_ok=True)
        self._heartbeat_path = self._assets_dir / "heartbeat.txt"
        self._heartbeat_interval = self._load_heartbeat_interval()
        self._last_heartbeat = 0.0

    def _load_heartbeat_interval(self) -> float:
        cfg_path = self._base_dir / "watchdog_config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path, "r") as f:
                    data = json.load(f)
                    return float(data.get("heartbeat_interval", 5.0))
            except Exception:
                pass
        return 5.0

    def _write_heartbeat(self, timestamp: float):
        try:
            self._assets_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = self._assets_dir / f"heartbeat_{os.getpid()}.tmp"
            tmp_path.write_text(f"{timestamp:.6f}\n")
            os.replace(tmp_path, self._heartbeat_path)
            self._last_heartbeat = timestamp
        except Exception:
            pass

    def start(self, root):
        """Kick off main UI loop."""
        self._root = root
        root.after(self.TICK_MS, self.tick)

    def set_inter_camera_delay(self, delay_val: float):
        """Live update delay without restart (SCH-03)."""
        self.scheduler.inter_camera_delay = delay_val

    def tick(self):
        if not self.gui.is_running():
            self.worker.stop()
            return

        # Main-thread heartbeat update (HB-01, HB-02, HB-03)
        now_ts = time.time()
        if now_ts - self._last_heartbeat >= self._heartbeat_interval:
            self._write_heartbeat(now_ts)

        # --------------------------------------------------------------
        # 1. Drain and apply completed results from background worker
        # --------------------------------------------------------------
        completed_results = self.worker.poll_results()
        for res in completed_results:
            feed = self._find_feed(res.camera_id)
            if feed is None:
                continue

            if res.error:
                if gb.gui:
                    gb.gui.log(f"[Cam {feed.id}] Inference error: {res.error}")
                self.scheduler.notify_inference_finished(res.duration)
                continue

            if res.is_calibration:
                # Calibration collection frame
                if feed.calibration_session and feed.calibration_session.state == SessionState.COLLECTING:
                    target_reached = feed.calibration_session.add_frame(res.embeddings)
                    if target_reached:
                        if gb.gui:
                            gb.gui.log(f"[Cam {feed.id}] Calibration target reached ({feed.calibration_session.collected_count} frames).")
                self.scheduler.notify_inference_finished(res.duration)
            else:
                # Detection result
                feed.overall_tier = res.overall_tier
                feed.current_highlight = (res.tier_grid, res.overall_tier)

                # Auto-update calibration if enabled
                if gb.auto_update_calibration and res.frame_eligible and feed.calibration_array is not None:
                    patch_indices = np.arange(res.embeddings.shape[0])
                    feed.calibration_array[res.nearest_slot_per_patch, patch_indices, :] = res.embeddings

                if res.overall_tier in ("ALERT", "CRITICAL"):
                    self.audio.report(feed.id, res.overall_tier)
                    if feed.last_good_frame is not None:
                        feed.handle_anomaly_reporting(feed.last_good_frame, res.overall_tier)

                self.gui.update_camera_status(feed.id, res.overall_tier)
                self.scheduler.notify_inference_finished(res.duration)

                # Check for slow detection warning
                warn = self.scheduler.slow_detection_warning
                if warn and warn != self._last_warning:
                    self._last_warning = warn
                    if gb.gui:
                        gb.gui.log(f"[Scheduler] {warn}")
                elif not warn:
                    self._last_warning = ""

        # --------------------------------------------------------------
        # 2. Render live video frames + stale highlight overlay for ALL feeds
        # --------------------------------------------------------------
        for feed in self.feeds:
            try:
                is_online, live_frame, err = feed.get_fresh_frame()
                if not is_online or live_frame is None:
                    # Render offline placeholder
                    h, w = (480, 640)
                    placeholder = _make_offline_placeholder(w, h, feed.id, err or "Connecting")
                    self.gui.update_camera_frame(feed.id, placeholder)
                    self.gui.update_camera_status(feed.id, "OFFLINE")
                else:
                    # Composite stale highlight overlay onto the live frame
                    display_frame = feed.draw_overlay(live_frame)
                    if not feed.dino_only and getattr(self.bundle, "yolo_model", None):
                        display_frame = self.bundle.yolo_model.draw_overlay(display_frame)

                    self.gui.update_camera_frame(feed.id, display_frame)
                    self.gui.update_camera_status(feed.id, feed.overall_tier)

                # Update calibration UI state
                self.gui.update_calibration_ui(feed.id, feed)

            except Exception as e:
                if gb.gui:
                    gb.gui.log(f"[Cam {feed.id}] Render error: {e}")

        # --------------------------------------------------------------
        # 3. Schedule next slot if worker is free
        # --------------------------------------------------------------
        if not self.worker.is_busy:
            eligible_flags = []
            for feed in self.feeds:
                # A feed is eligible if it is online and has a frame, and:
                # 1. Active calibration session in COLLECTING state, OR
                # 2. Fully calibrated and no active session
                can_run = False
                if feed.is_online and feed.last_good_frame is not None:
                    if feed.calibration_session is not None:
                        if feed.calibration_session.state == SessionState.COLLECTING:
                            can_run = True
                    elif feed.is_calibrated:
                        can_run = True
                eligible_flags.append(can_run)

            next_idx = self.scheduler.get_next_camera_for_slot(eligible_flags)
            if next_idx is not None:
                target_feed = self.feeds[next_idx]
                is_calib = bool(
                    target_feed.calibration_session
                    and target_feed.calibration_session.state == SessionState.COLLECTING
                )

                task = InferenceTask(
                    camera_id=target_feed.id,
                    frame=target_feed.last_good_frame.copy(),
                    mask_grid=target_feed.zone_class_grid,
                    is_calibration=is_calib,
                    calibration_array=target_feed.calibration_array,
                    thresholds=target_feed.get_effective_thresholds(),
                    allowed_error=target_feed.allowed_error,
                    dino_only=target_feed.dino_only,
                    auto_update_calibration=gb.auto_update_calibration
                )

                if self.worker.submit_task(task):
                    self.scheduler.notify_inference_started()

        # --------------------------------------------------------------
        # 4. Loop reschedule
        # --------------------------------------------------------------
        if self._root and self.gui.is_running():
            self._root.after(self.TICK_MS, self.tick)

    def _find_feed(self, camera_id: int):
        for f in self.feeds:
            if f.id == camera_id:
                return f
        return None

