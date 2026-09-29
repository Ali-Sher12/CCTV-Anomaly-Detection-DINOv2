"""DINO.py — Shared ModelBundle and per-camera CameraFeed.

ModelBundle : Loads DINO & YOLO models once; shared across all feeds.
CameraFeed  : Per-camera state machine, unified zone masks, atomic calibration persistence,
              and non-blocking capture registry integration.
"""

import os
import time
import socket
import smtplib
import threading
import datetime
import cv2
import numpy as np
import torch
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from transformers import AutoImageProcessor, AutoModel

import Globals as gb
from YOLO import YOLO_MODEL as yolo
from capture_registry import CaptureRegistry, normalize_source
from mask_utils import build_zone_class_grid, ZONE_IGNORE, ZONE_MEDIUM, ZONE_HIGH
import calibration_store
from calibration_session import CalibrationSession, SessionState


class ModelBundle:
    """Holds DINO processor/model and YOLO model. Load once."""

    def __init__(self):
        # ---- DINO ----
        if gb.DINO_MODEL_VERSION == "facebook/dinov2-small":
            self.processor = AutoImageProcessor.from_pretrained(
                gb.DINO_MODEL_VERSION, cache_dir="Models/DINO", local_files_only=True)
            self.model = AutoModel.from_pretrained(
                gb.DINO_MODEL_VERSION, cache_dir="Models/DINO", local_files_only=True)
        else:
            self.processor = AutoImageProcessor.from_pretrained(
                gb.DINO_MODEL_VERSION, local_files_only=True)
            self.model = AutoModel.from_pretrained(
                gb.DINO_MODEL_VERSION, local_files_only=True)

        self.device = "cpu"
        if gb.useGPU and torch.cuda.is_available():
            self.device = "cuda"
        self.model = self.model.to(self.device)
        self.model.eval()

        self._input_size = self.processor.crop_size["height"]
        self._patch_size = self.model.config.patch_size
        self.grid_h = self.grid_w = self._input_size // self._patch_size

        # ---- YOLO ----
        self.yolo_model = yolo(gb.YOLO_MODEL_VERSION)

    def get_patch_embeddings(self, final_frame: np.ndarray) -> np.ndarray:
        """Run DINO on a 224x224 RGB image; return patch embeddings (num_patches, embed_dim)."""
        inputs = self.processor(images=final_frame, return_tensors="pt",
                                do_resize=False, do_center_crop=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, interpolate_pos_encoding=True)
        patch_embeddings = outputs.last_hidden_state[0, 1:, :]
        return patch_embeddings.cpu().numpy()


class CameraFeed:
    """Per-camera state machine and vision pipeline."""

    def __init__(self, camera_id: int, name: str, source,
                 mask_path: str, model_bundle: ModelBundle,
                 thresholds: dict | None = None,
                 allowed_error: int = 1,
                 anomaly_report_wait: int = 10,
                 total_calibration_frames: int = 100,
                 dino_only: bool = False,
                 email_cfg: dict | None = None):
        self.id = camera_id
        self.name = name
        self.source = normalize_source(source)
        self.mask_path = mask_path
        self.bundle = model_bundle

        # Per-camera configuration
        self.thresholds = thresholds
        self.allowed_error = allowed_error
        self.anomaly_report_wait = anomaly_report_wait
        self.total_calibration_frames = total_calibration_frames
        self.dino_only = dino_only
        self.email_cfg = email_cfg or {}

        # Grid dimensions from bundle
        self.grid_h = model_bundle.grid_h
        self.grid_w = model_bundle.grid_w

        # Live capture via reference-counted CaptureRegistry
        self.reader = CaptureRegistry().acquire(self.source, self.id)

        # Vision and detection state
        self.last_good_frame = None
        self.current_highlight = None  # tuple (tier_grid, overall_tier)
        self.overall_tier = "NORMAL"
        self.mask_warning = ""
        self.zone_class_grid = None

        # Calibration state
        self.calibration_store: list[np.ndarray] = []
        self.calibration_array: np.ndarray | None = None
        self.is_calibrated = False
        self.calibration_status = "Not calibrated"
        self.calibration_session: CalibrationSession | None = None

        # Reporting state
        self.last_report_time = 0.0

        # Initialise mask and load calibration
        self.update_mask()
        self._init_calibration()

    # ------------------------------------------------------------------
    # Capture & Frame Acquisition
    # ------------------------------------------------------------------

    def get_fresh_frame(self) -> tuple[bool, np.ndarray | None, str]:
        """Fetch newest frame from reader without blocking."""
        is_online, frame, err = self.reader.get_latest_frame()
        if is_online and frame is not None:
            flipped = cv2.flip(frame, 1)
            self.last_good_frame = flipped
            return True, flipped, ""
        return False, self.last_good_frame, err

    @property
    def is_online(self) -> bool:
        is_on, _, _ = self.reader.get_latest_frame()
        return is_on

    # ------------------------------------------------------------------
    # Mask Management (MASK-01 to MASK-09)
    # ------------------------------------------------------------------

    def update_mask(self):
        """Build or reload zone_class_grid from mask_path."""
        fw, fh = self.reader.get_resolution()
        grid, warning = build_zone_class_grid(
            self.mask_path, fw, fh, self.grid_w, self.grid_h
        )
        self.zone_class_grid = grid
        self.mask_warning = warning
        if warning and gb.gui:
            gb.gui.log(f"[Cam {self.id}] {warning}")

    # ------------------------------------------------------------------
    # Calibration Storage & Sessions (CAL-01 to CAL-24)
    # ------------------------------------------------------------------

    def _init_calibration(self):
        """Validate on-disk calibration at startup without auto-calibrating."""
        fw, fh = self.reader.get_resolution()
        ok, cal_arr, cal_store, meta, msg = calibration_store.load_calibration(
            self.id, gb.DINO_MODEL_VERSION, self.grid_h, self.grid_w, fw, fh
        )
        if ok and cal_arr is not None:
            self.calibration_array = cal_arr
            self.calibration_store = cal_store
            self.is_calibrated = True
            self.calibration_status = msg
            self.calibration_session = None
        else:
            self.calibration_array = None
            self.calibration_store = []
            self.is_calibrated = False
            self.calibration_status = msg
            # Start initial session in NOT_STARTED state (waiting for user Start)
            self.calibration_session = CalibrationSession(
                target_frames=self.total_calibration_frames, is_initial=True
            )

    def start_calibration_session(self, target_frames: int | None = None):
        """Create or resume a calibration session."""
        if self.calibration_session is None:
            target = target_frames if target_frames is not None else 30
            self.calibration_session = CalibrationSession(target_frames=target, is_initial=not self.is_calibrated)
        
        self.calibration_session.start()
        self.current_highlight = None  # Clear highlight during calibration (CAL-16)

    def pause_calibration_session(self):
        if self.calibration_session:
            self.calibration_session.pause()

    def restart_calibration_session(self):
        if self.calibration_session:
            self.calibration_session.restart()

    def confirm_calibration_session(self) -> tuple[bool, str]:
        """Commit session frames, rebuild array, save to disk atomically."""
        if not self.calibration_session or not self.calibration_session.can_confirm:
            return False, "Cannot confirm session in current state"

        try:
            new_store = self.calibration_session.confirm(self.calibration_store)
            fw, fh = self.reader.get_resolution()
            ok, err = calibration_store.save_calibration(
                camera_id=self.id,
                calibration_store=new_store,
                dino_model_version=gb.DINO_MODEL_VERSION,
                grid_h=self.grid_h,
                grid_w=self.grid_w,
                frame_width=fw,
                frame_height=fh
            )
            if not ok:
                return False, err

            self.calibration_store = new_store
            self.calibration_array = np.stack(self.calibration_store, axis=0)
            self.is_calibrated = True
            self.calibration_session = None
            
            # Reload status string
            _, _, _, _, status_msg = calibration_store.load_calibration(
                self.id, gb.DINO_MODEL_VERSION, self.grid_h, self.grid_w, fw, fh
            )
            self.calibration_status = status_msg
            if gb.gui:
                gb.gui.log(f"[Cam {self.id}] Calibration confirmed ({len(self.calibration_store)} frames saved).")
            return True, ""
        except Exception as e:
            return False, str(e)

    def remove_all_calibration_and_recalibrate(self) -> tuple[bool, str]:
        """Delete saved calibration and start fresh initial session."""
        ok, err = calibration_store.delete_camera_calibration(self.id)
        if not ok:
            return False, err

        self.calibration_store.clear()
        self.calibration_array = None
        self.is_calibrated = False
        self.current_highlight = None
        self.calibration_status = "Not calibrated — calibration deleted"
        self.calibration_session = CalibrationSession(
            target_frames=self.total_calibration_frames, is_initial=True
        )
        if gb.gui:
            gb.gui.log(f"[Cam {self.id}] All calibration deleted. Session reset.")
        return True, ""

    # ------------------------------------------------------------------
    # Threshold Resolution
    # ------------------------------------------------------------------

    def get_effective_thresholds(self) -> dict:
        """Resolve per-camera thresholds with fallback to globals."""
        def get_val(key, is_hp=False):
            fallback = gb.TIER_THRESHOLDS_HIGH_PRIORITY[key.replace("HIGH_PRIORITY_", "")] if is_hp else gb.TIER_THRESHOLDS[key]
            if not self.thresholds:
                return float(fallback)
            v = self.thresholds.get(key)
            if v is None or float(v) <= 0:
                return float(fallback)
            return float(v)

        return {
            "ALERT": get_val("ALERT"),
            "CRITICAL": get_val("CRITICAL"),
            "HIGH_PRIORITY_ALERT": get_val("HIGH_PRIORITY_ALERT", True),
            "HIGH_PRIORITY_CRITICAL": get_val("HIGH_PRIORITY_CRITICAL", True),
        }

    # ------------------------------------------------------------------
    # Overlay Rendering (SCH-09)
    # ------------------------------------------------------------------

    def draw_overlay(self, live_frame: np.ndarray) -> np.ndarray:
        """Composite stale highlight layer onto the live frame."""
        if live_frame is None:
            return None

        if self.current_highlight is None:
            return live_frame

        tier_grid, overall_tier = self.current_highlight
        if overall_tier == "NORMAL":
            return live_frame

        fh, fw = live_frame.shape[:2]
        gh, gw = tier_grid.shape

        color_grid = np.zeros((gh, gw, 3), dtype=np.uint8)
        highlight_mask = np.zeros((gh, gw), dtype=np.uint8)

        # Draw highlight ONLY for monitored cells where tier is ALERT/CRITICAL
        for tier, color in gb.TIER_COLORS.items():
            matches = (tier_grid == tier) & (self.zone_class_grid != ZONE_IGNORE)
            color_grid[matches] = color
            highlight_mask[matches] = 1

        if not np.any(highlight_mask):
            return live_frame

        # Nearest-neighbor resize of overlay to frame dimensions
        color_full = cv2.resize(color_grid, (fw, fh), interpolation=cv2.INTER_NEAREST)
        mask_full = cv2.resize(highlight_mask, (fw, fh), interpolation=cv2.INTER_NEAREST)

        alpha = 0.45
        blended = cv2.addWeighted(live_frame, 1.0 - alpha, color_full, alpha, 0)

        result = live_frame.copy()
        result[mask_full == 1] = blended[mask_full == 1]

        return result

    # ------------------------------------------------------------------
    # Anomaly Reporting (Email & Local Log)
    # ------------------------------------------------------------------

    def handle_anomaly_reporting(self, frame: np.ndarray, overall_tier: str):
        now = time.time()
        if (now - self.last_report_time) < self.anomaly_report_wait:
            return

        self.last_report_time = now
        threading.Thread(target=self._dispatch_report,
                         args=(frame.copy(), overall_tier),
                         daemon=True).start()

    def _dispatch_report(self, frame: np.ndarray, tier: str):
        self.log_anomaly_locally(frame, tier)
        sender = self.email_cfg.get("sender") or gb.EMAIL_SENDER
        password = self.email_cfg.get("app_password") or gb.EMAIL_PASSWORD
        receivers = self.email_cfg.get("receivers") or ([gb.EMAIL_RECEIVER] if gb.EMAIL_RECEIVER else [])

        if sender and password and receivers:
            if not self._is_internet_available():
                if gb.gui:
                    gb.gui.log(f"[Cam {self.id}] Email send failed. No internet.")
                return

            try:
                self._send_email(frame, tier, sender, password, receivers)
                if gb.gui:
                    gb.gui.log(f"[Cam {self.id}] Anomaly email alert sent.")
            except Exception as e:
                if gb.gui:
                    gb.gui.log(f"[Cam {self.id}] Email send failed: {e}")

    def log_anomaly_locally(self, frame: np.ndarray, tier: str):
        try:
            os.makedirs(gb.LOG_DIR, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"cam{self.id}_{tier}_{ts}.jpg"
            path = os.path.join(gb.LOG_DIR, filename)
            cv2.imwrite(path, frame)
        except Exception as e:
            if gb.gui:
                gb.gui.log(f"[Cam {self.id}] Local logging failed: {e}")

    def _is_internet_available(self) -> bool:
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False

    def _send_email(self, frame: np.ndarray, tier: str,
                    sender: str, password: str, receivers: list[str]):
        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = ", ".join(receivers)
        msg["Subject"] = f"[{tier}] Anomaly Detected — Camera {self.id} ({self.name})"

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = (
            f"Anomaly Detection Alert\n"
            f"----------------------\n"
            f"Camera    : {self.id} — {self.name}\n"
            f"Severity  : {tier}\n"
            f"Timestamp : {ts}\n\n"
            f"Please review the attached camera capture.\n"
        )
        msg.attach(MIMEText(body, "plain"))

        ret, buffer = cv2.imencode(".jpg", frame)
        if ret:
            img = MIMEImage(buffer.tobytes())
            img.add_header("Content-Disposition", "attachment",
                           filename=f"cam{self.id}_{tier}_{ts.replace(':', '-')}.jpg")
            msg.attach(img)

        server = smtplib.SMTP(gb.SMTP_SERVER, gb.SMTP_PORT, timeout=10)
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receivers, msg.as_string())
        server.quit()

