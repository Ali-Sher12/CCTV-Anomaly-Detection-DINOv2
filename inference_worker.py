"""inference_worker.py — Dedicated single background worker thread for DINO & YOLO inference.

Implements SCH-08:
- Exactly ONE dedicated background worker thread executes DINO + YOLO
- Inference runs under torch.inference_mode()
- Main thread is never blocked and GUI video display remains 100% fluid
- Results returned to main thread via a thread-safe queue.Queue
"""

import time
import queue
import threading
import torch
import numpy as np


class InferenceTask:
    def __init__(self, camera_id: int, frame: np.ndarray,
                 mask_grid: np.ndarray, is_calibration: bool,
                 calibration_array: np.ndarray | None,
                 thresholds: dict, allowed_error: int,
                 dino_only: bool, auto_update_calibration: bool):
        self.camera_id = camera_id
        self.frame = frame  # BGR numpy frame copy
        self.mask_grid = mask_grid  # zone_class_grid (grid_h, grid_w)
        self.is_calibration = is_calibration
        self.calibration_array = calibration_array
        self.thresholds = thresholds
        self.allowed_error = allowed_error
        self.dino_only = dino_only
        self.auto_update_calibration = auto_update_calibration
        self.submit_time = time.monotonic()


class InferenceResult:
    def __init__(self, camera_id: int, is_calibration: bool,
                 embeddings: np.ndarray | None = None,
                 tier_grid: np.ndarray | None = None,
                 overall_tier: str = "NORMAL",
                 yolo_detections: list | None = None,
                 duration: float = 0.0,
                 error: str = "",
                 nearest_slot_per_patch: np.ndarray | None = None,
                 frame_eligible: bool = False):
        self.camera_id = camera_id
        self.is_calibration = is_calibration
        self.embeddings = embeddings
        self.tier_grid = tier_grid
        self.overall_tier = overall_tier
        self.yolo_detections = yolo_detections or []
        self.duration = duration
        self.error = error
        self.nearest_slot_per_patch = nearest_slot_per_patch
        self.frame_eligible = frame_eligible


class InferenceWorker:
    """Manages the single background model inference thread."""

    def __init__(self, model_bundle):
        self.bundle = model_bundle
        self._task_queue = queue.Queue(maxsize=2)
        self._result_queue = queue.Queue()
        self._running = True
        self._in_flight = False
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    @property
    def is_busy(self) -> bool:
        return self._in_flight or not self._task_queue.empty()

    def submit_task(self, task: InferenceTask) -> bool:
        """Submit a task if not currently busy. Returns True if accepted."""
        try:
            self._task_queue.put_nowait(task)
            self._in_flight = True
            return True
        except queue.Full:
            return False

    def poll_results(self) -> list[InferenceResult]:
        """Drains and returns all available completed results for the main thread."""
        results = []
        while True:
            try:
                res = self._result_queue.get_nowait()
                results.append(res)
                self._in_flight = False
            except queue.Empty:
                break
        return results

    def _worker_loop(self):
        while self._running:
            try:
                task = self._task_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            t_start = time.monotonic()
            try:
                res = self._execute_task(task)
            except Exception as e:
                res = InferenceResult(
                    camera_id=task.camera_id,
                    is_calibration=task.is_calibration,
                    error=str(e),
                    duration=time.monotonic() - t_start
                )

            res.duration = time.monotonic() - t_start
            self._result_queue.put(res)
            self._task_queue.task_done()

    def _execute_task(self, task: InferenceTask) -> InferenceResult:
        # Pre-process frame for DINO
        h, w = task.frame.shape[:2]
        ignored_zone = task.frame.copy()
        
        # Zero out ignored zones for DINO embedding
        # Resize mask_grid to (h, w)
        if task.mask_grid is not None:
            import cv2
            mask_full = cv2.resize(task.mask_grid, (w, h), interpolation=cv2.INTER_NEAREST)
            ignored_zone[mask_full == 0] = 0

        final_frame = cv2.cvtColor(ignored_zone, cv2.COLOR_BGR2RGB)
        final_frame = cv2.resize(final_frame, (224, 224), interpolation=cv2.INTER_AREA)

        with torch.inference_mode():
            embeddings = self.bundle.get_patch_embeddings(final_frame)

        if task.is_calibration:
            return InferenceResult(
                camera_id=task.camera_id,
                is_calibration=True,
                embeddings=embeddings
            )

        # Detection mode
        if task.calibration_array is None or len(task.calibration_array) == 0:
            return InferenceResult(
                camera_id=task.camera_id,
                is_calibration=False,
                overall_tier="NORMAL",
                error="No calibration available"
            )

        # Patch scores vectorised computation
        diffs = task.calibration_array - embeddings[np.newaxis, :, :]
        distances = np.linalg.norm(diffs, axis=2)
        patch_scores = np.min(distances, axis=0)
        nearest_slot_per_patch = np.argmin(distances, axis=0)

        grid_h, grid_w = task.mask_grid.shape
        score_grid = patch_scores.reshape(grid_h, grid_w)

        # Compute tier grid using MASK-01 / MASK-05 (0=IGNORE, 1=MEDIUM, 2=HIGH)
        tier_grid = np.full(score_grid.shape, "NORMAL", dtype=object)
        med_mask = (task.mask_grid == 1)
        high_mask = (task.mask_grid == 2)

        tier_grid[med_mask & (score_grid >= task.thresholds["ALERT"])] = "ALERT"
        tier_grid[med_mask & (score_grid >= task.thresholds["CRITICAL"])] = "CRITICAL"

        tier_grid[high_mask & (score_grid >= task.thresholds["HIGH_PRIORITY_ALERT"])] = "ALERT"
        tier_grid[high_mask & (score_grid >= task.thresholds["HIGH_PRIORITY_CRITICAL"])] = "CRITICAL"

        # Overall tier is maximum severity across monitored cells (IGNORE cells remain NORMAL)
        severity_order = ["NORMAL", "ALERT", "CRITICAL"]
        overall_tier = max(set(tier_grid.flatten()), key=severity_order.index)

        # Frame eligibility for auto-calibration
        flat_tier = tier_grid.flatten()
        flat_mask = task.mask_grid.flatten()
        is_hp = (flat_mask == 2)
        is_med = (flat_mask == 1)
        
        frame_eligible = True
        if np.any(is_hp & (flat_tier != "NORMAL")):
            frame_eligible = False
        elif np.any(is_med & (flat_tier == "CRITICAL")):
            frame_eligible = False
        elif np.sum(is_med & (flat_tier == "ALERT")) >= task.allowed_error:
            frame_eligible = False

        # YOLO hybrid run
        yolo_dets = []
        if not task.dino_only and hasattr(self.bundle, "yolo_model") and self.bundle.yolo_model:
            self.bundle.yolo_model.run_model(task.frame)
            yolo_dets = list(self.bundle.yolo_model.listOfItems)

        return InferenceResult(
            camera_id=task.camera_id,
            is_calibration=False,
            embeddings=embeddings,
            tier_grid=tier_grid,
            overall_tier=overall_tier,
            yolo_detections=yolo_dets,
            nearest_slot_per_patch=nearest_slot_per_patch,
            frame_eligible=frame_eligible
        )

    def stop(self):
        self._running = False

