# AGENT.md — Passive Anomaly Detector: Multi-Camera Upgrade

You are working inside an existing Python/Tkinter/OpenCV anomaly-detection app ("Anomaly Detection" folder). It currently supports **one** camera. Your job is to upgrade it to support **N cameras**, add a **first-run Setup Wizard**, and remove the old "Apply & Restart" panel. Follow this file as your source of truth — the decisions below are final, do not re-litigate them, just build.

Time budget: ~1 hour total. Work in the priority order given in "Task List". Ship P0 completely before touching P1. If you run out of time, P0 alone should still run end-to-end.


## 1. Current architecture (read this first)

- `main.py` — creates the Tk root, loading screen, one `DINO_MODEL` instance, one `cv2.VideoCapture`, and runs a `root.after()` loop calling `dino_model.getFrame()` → `DINO_computation_loop()` → GUI update.

- `Accessories.py` — `getnSetSpecs()` is the **device scan**. It reads RAM / VRAM / CPU cores via `psutil` + `torch`, buckets the machine into `HIGH` / `MID` / `LOW`, and sets `gb.DINO_MODEL_VERSION`, `gb.YOLO_MODEL_VERSION`, `gb.secondsForOneFrame`, `gb.DINO_ONLY`. **Do not** **change this bucketing logic or the model path strings** — only call it and read its results.

- `Globals.py` (`gb`) — a module used as a global settings/state bag.

- `settings_store.py` — flat JSON load/save for `gb.*` values.

- `DINO.py` — one class doing two jobs at once: (a) loading/holding the DINO model + YOLO model, and (b) per-camera state (mask, calibration store, persistence counters, current highlight, email reporting). **This is the** **class you must split** — see §3.

- `YOLO.py` — thin wrapper around `ultralytics.YOLO`, one instance per `DINO_MODEL` today.

- `audio.py` — `pyttsx3`-based TTS, one background thread, drops new speech requests if already speaking.

- `gui.py` — Tkinter UI: one canvas, sliders that write straight to `gb.*` (live), and a "Settings (Require Restart)" panel with staged values that only commit on "Apply & Restart".

- `mask_editor.py` — a Paint-style Toplevel for drawing red/green/black zone masks onto a PNG the same size as the camera frame.


## 2. Non-negotiable constraints

1. **OS-agnostic.** No Windows-only paths, no `os.system("cls")`-type calls, no hardcoded `\`. Use `os.path.join` / `pathlib`. `cv2.VideoCapture(source)` must be called with just the source (int index or URL string) — no OS-specific backend flags (no `cv2.CAP_DSHOW`). Right-click binding must work cross-platform (see §5, gotcha list).

2. **Keep the existing model-path/tier logic exactly as-is** (`Accessories.getnSetSpecs`, `gb.DINO_MODEL_VERSION`, `gb.YOLO_MODEL_VERSION`). Only load the model **once**, shared across all cameras (see §3) — don't load a separate DINO/YOLO copy per camera, that will exhaust RAM/VRAM.

3. **Audio phrase**, exact text, triggered whenever a camera's tier is `ALERT` or `CRITICAL`: `"ANOMALY DETECTED AT CAMERA {n}"` where `{n}` is that camera's number. This fully replaces the old "MAJOR ANOMALY DETECTED" string. Multiple cameras alerting at once must **queue and speak one at a** **time**, not overlap or drop each other.

4. **Delete the "Apply & Restart" staged-settings system entirely** (`_staged_*` attributes, `_check_restart_needed`, `_apply_and_restart`, `btn_apply_restart`, `lbl_restart_hint`, the whole `_build_restart_group`). Anything that used to require a restart (calibration frame count, DINO-only/hybrid, camera sources, email) is now only set in the Setup Wizard.

5. Every camera feed owns **its own** mask, persistence counter array, and threshold values (with global defaults it can override) — never share these across cameras.

6. **A dead/disconnected camera must never crash the app.** If a camera fails to open, or stops returning frames mid-run, that one feed goes into an "offline" state (placeholder shown on its canvas, periodic reconnect attempts) while every other camera keeps running normally. See §3a.

7. **Calibration is per-camera and must pause for confirmation when it** **finishes**, before that camera starts actively flagging anomalies. See §3b. Other cameras are not blocked while one camera is waiting on this confirmation.


## 3. New runtime architecture

Split the current `DINO_MODEL` class into two:

### `ModelBundle` (loaded exactly once, shared by all cameras)

Holds: DINO `processor`/`model`, the `YOLO_MODEL` instance, `device` (cpu/cuda), `grid_h`/`grid_w`/`patch_size`. Exposes one method, e.g. `get_patch_embeddings(frame) -> np.ndarray`, plus access to the YOLO model for `run_model()`/`draw_overlay()`. This is basically today's `DINO_MODEL.__init__`

- `_get_patch_embeddings`, with all per-frame/per-camera state removed.

### `CameraFeed` (one instance per configured camera)

Holds: `id`, `name`, `source` (int or URL string), `cv2.VideoCapture`, `mask_path`, `is_active`/`is_high_priority`/`is_watched`/`new_mask`/ `zone_grid_high_priority`, `persistence_counters`, `calibration_store`, `calibration_array`, `initial_calibration`, `current_calibration_frames_held`, `current_highlight`, `highlighted_frame`, `overall_tier`, its own `thresholds` dict (falls back to global defaults — see §4 schema), `required_persistence`, `allowed_error`, `anomaly_report_wait`, `reporting_active`, `last_report_time`.

Move `camera_init`, `DINO_computation_loop`, `compute_patch_scores`, `compute_tier`, `apply_persistence_filter`, `is_frame_eligible`, `self_fix_calibration`, `draw_overlay`, and the email-reporting methods (`is_internet_available`, `send_email`, `log_anomaly_locally`, `report_anomaly`, `_dispatch_report`, `handle_anomaly_reporting`) onto `CameraFeed`, taking a `ModelBundle` reference for the actual inference calls. The math inside these methods does not need to change — only what object owns the state.

### `FeedManager` (new, orchestrates the staggered schedule)

This directly implements requirement: *"delay between each DINO/YOLO run* *means that if this delay is set to 0.5, at 0s it runs for cam 1, at 0.5s for* *cam 2, at 1s for cam 3, and so on — camera stays live at all times regardless."*

- Config: `inter_camera_delay` (seconds), list of `CameraFeed`s in order.

- Every UI tick (short interval, e.g. every 30–50ms):

  1. **Always** grab a fresh frame from every camera's `VideoCapture` and push it to that camera's canvas — this is what keeps video "live" no matter whose turn it is for inference.

  2. Separately track elapsed time since the *cycle* started. A full cycle length = `num_cameras * inter_camera_delay`. `current_slot = floor(time_since_cycle_start / inter_camera_delay) % num_cameras`. Only the `CameraFeed` at `current_slot` runs `DINO_computation_loop()` this tick; all others just redraw their last `highlighted_frame` (which is how "highlight persists until the next run" already works today via `current_highlight` — keep that behavior, just make it per-camera instead of global).

  3. After a camera runs inference, call `audio_engine.report(camera_id, tier)` and `gui.update_status(camera_id, tier)` for that camera only.

This design also conveniently avoids thread-safety problems: since only one camera ever runs a model forward-pass per tick, you do **not** need multiple model copies or locks around the shared `ModelBundle`.

### 3a. Camera-down fallback (no crashing, ever)

Add to `CameraFeed`: `is_online` (bool), `last_reconnect_attempt` (float timestamp), `reconnect_interval` (seconds, e.g. 3.0).

- **On initial open** (both in the wizard's "Test" button and in `CameraFeed.__init__`/`camera_init`): wrap `cv2.VideoCapture(source)` and the `isOpened()` check in a guard. If it fails, set `is_online = False` and continue constructing the rest of the app — don't raise.

- **On every read**, in `FeedManager.tick()`, wrap each camera's frame grab in `try/except` individually (one `try` per camera, inside the loop over cameras — not one `try` around the whole loop, or a single bad camera aborts everyone else's turn too). If `cap.read()` returns `frameRead == False`, or raises, or `cap is None`/not `isOpened()`:

  - Set `is_online = False`.

  - Push `camera_feed.last_frame` (freeze on the last good frame) or a solid placeholder image to that camera's canvas, with an overlay label like `"Camera {n} offline — reconnecting..."` drawn with `cv2.putText` or a Tk label over the canvas.

  - Skip that camera entirely for its inference turn in the schedule (don't let a dead camera "use up" its slot forever — either skip its turn silently, or shrink the effective cycle to only online cameras; either is acceptable, pick whichever is simpler to implement in the time you have).

  - Do **not** run audio/email reporting for an offline camera.

- **Reconnection**: every `reconnect_interval` seconds, while `is_online is False`, attempt `cv2.VideoCapture(source)` again in the background of the same tick loop (cheap check, not a new thread). If it opens and returns a frame successfully, set `is_online = True`, re-run `camera_init()` for that feed (mask resize etc. depends on frame size, which could differ if it's e.g. a URL stream that reconnects at a different resolution), and resume normal scheduling for it. Do **not** wipe its existing calibration data on reconnect — a brief disconnect shouldn't force recalibration.

- One camera being offline must never prevent the Setup Wizard from finishing, and must never prevent `main.py` from starting the rest of the app — log it and move on.

### 3b. Per-camera calibration confirmation checkpoint

Calibration already happens independently per `CameraFeed` (it collects `total_calibration_frames` samples into `calibration_store`, per the existing `DINO_computation_loop` logic). Add a pause here instead of auto-continuing:

- New `CameraFeed` field: `calibration_awaiting_confirmation` (bool, default `False`).

- In the calibration branch of `DINO_computation_loop` (the `if initial_calibration:` block), when `current_calibration_frames_held` reaches `total_calibration_frames`: **do not** immediately finalize into `calibration_array` and flip `initial_calibration = False`. Instead set `calibration_awaiting_confirmation = True` and stop collecting further frames, but keep displaying the live raw feed for that camera as normal.

- While `calibration_awaiting_confirmation` is `True` for a camera: `FeedManager` should skip that camera's turn in the inference schedule entirely (it's neither calibrating nor detecting — just idling on live video) so it doesn't block other cameras' turns.

- **GUI**: as soon as a camera enters this state, show a small banner/ overlay on that specific camera's canvas (or a lightweight non-modal Tk frame docked under its canvas) reading `"Calibration complete for Camera {n}"` with two buttons:

  - **Continue** → call `_finalize_calibration()` for that camera, set `calibration_array`, `initial_calibration = False`, `calibration_awaiting_confirmation = False`. Camera resumes normal scheduled detection on its next turn.

  - **Restart Calibration** → clear that camera's `calibration_store`, reset `current_calibration_frames_held = 0`, `calibration_awaiting_confirmation = False`, keep `initial_calibration = True`. Camera goes back into the collecting phase from frame 0, independent of every other camera's state.

- This is purely a runtime/per-camera UI event — it has nothing to do with the Setup Wizard (which only sets `total_calibration_frames` as a target number, once, up front).


## 4. Settings schema (v2) — replace `settings_store.py`'s flat format

Old `Assets/settings.json` is single-camera and can be thrown away — don't write migration code, that's wasted time. New shape:

```
{
  "setup_complete": true,
  "device_tier": "MID",
  "dino_model_version": "Models/DINO/dinov3-vitb16",
  "yolo_model_version": "Models/YOLO/yolo26m.pt",
  "dino_only": false,
  "total_calibration_frames": 100,
  "inter_camera_delay": 0.5,
  "email": {
    "sender": "...",
    "app_password": "...",
    "receivers": ["a@example.com", "b@example.com"]
  },
  "default_thresholds": {
    "ALERT": 45.0,
    "CRITICAL": 50.0,
    "HIGH_PRIORITY_ALERT": 35.0,
    "HIGH_PRIORITY_CRITICAL": 40.0
  },
  "cameras": [
    {
      "id": 1,
      "name": "Front Door",
      "source": 0,
      "mask_path": "Assets/masks/camera_1_mask.png",
      "thresholds": null,
      "required_persistence": 1,
      "allowed_error": 1,
      "anomaly_report_wait": 10
    }
  ]
}
```

`thresholds: null` on a camera means "use `default_thresholds`". A non-null object overrides one or more of the four keys for that camera only — this is exactly the "one global threshold per status now, per-feed override later" requirement.

`email.receivers` is a list — wizard should accept a comma-separated string and split/trim it.


## 5. Task list (build in this order)

### P0 — must work for the demo

1. **Settings v2 + loader.** Rewrite `settings_store.py` around the schema in §4 (`load_settings()` returns the dict or `None` if `setup_complete` is missing/false; `save_settings(data)` writes the whole dict).

2. **`ModelBundle` + `CameraFeed` split** as described in §3. Keep the actual math identical to today's `DINO.py` — this is a refactor, not a rewrite of the algorithm.

3. **`FeedManager`** implementing the staggered schedule from §3.

4. **Setup Wizard** (`setup_wizard.py`, new file) — a `Toplevel` (or its own `Tk` root) shown by `main.py` when `load_settings()` returns `None`. Steps, one screen each with Back/Next:

   - **Step 1 – Device scan.** Call `Accessories.getnSetSpecs()`, display `Accessories.specLog1` (tier, RAM/VRAM, chosen models) read-only, Next.

   - **Step 2 – Cameras.** A list with "Add Camera" (name field + source field, source accepts an int like `0`/`1` or a URL like `http://192.168.1.5:8080/video`) and "Remove". A "Test" button per row that does a quick `cv2.VideoCapture(source).isOpened()` check and shows ✅/❌ inline. Require at least 1 camera to proceed.

   - **Step 3 – Email.** Sender address, app password (masked entry), receivers (comma-separated).

   - **Step 4 – Thresholds.** The same 4 sliders as today's "Thresholds" group in `gui.py` (Alert, Critical, HP Alert, HP Critical) — these become `default_thresholds`.

   - **Step 5 – Suggested values / summary.** Based on `device_tier` from step 1 **and** camera count from step 2, pre-fill (editable):

     - `total_calibration_frames`: LOW=60, MID=100, HIGH=150

     - `inter_camera_delay`: LOW=1.5s, MID=0.5s, HIGH=0.2s (this is *per-camera-slot* delay, not affected by camera count — more cameras just means a longer full cycle, which is expected and fine)

     - `dino_only`: pre-set from `gb.DINO_ONLY` (tier result), user can flip it — this is the "YOLO integration" toggle

     - "Finish" writes `settings.json` (schema in §4) with `setup_complete: true`, generates a default all-green mask PNG per camera at `Assets/masks/camera_{id}_mask.png` sized to that camera's actual resolution (open the capture briefly to read `CAP_PROP_FRAME_WIDTH/HEIGHT`, default to 640×480 if it fails), closes the wizard, and lets `main.py` proceed into the normal app.

5. **`gui.py` — multi-canvas layout.** Replace the single `self.canvas` with one canvas per configured camera (simple grid via `tk.Frame` + `grid()`, e.g. 2 columns). Each canvas needs a small title label above it ("Camera 1 — Front Door") and its own status label (NORMAL/ALERT/CRITICAL color-coded, same as today's `lbl_status`).

6. **Right-click → per-camera settings.** Bind right-click on each camera's canvas to open a `Toplevel` scoped to that one `CameraFeed`: reuse the slider code from today's `_build_runtime_group` / `_build_threshold_group` but have the callbacks write to `camera_feed.required_persistence` / `camera_feed.allowed_error` / `camera_feed.anomaly_report_wait` / `camera_feed.thresholds[...]` instead of `gb.*`. Include the existing "Modify Mask..." button, pointed at that camera's own `mask_path`.

7. **Delete the restart-group code** from `gui.py` per §2.4.

8. **`audio.py` — queue + camera-aware message.** Replace the single "already speaking? skip" check with a `queue.Queue`. `report(camera_id, tier)` pushes `f"ANOMALY DETECTED AT CAMERA {camera_id}"` onto the queue only for `ALERT`/`CRITICAL` (never `NORMAL`). One worker thread pulls from the queue and speaks items one at a time so simultaneous multi-camera alerts don't clobber each other.

9. **`main.py` rewrite** — on startup: `data = load_settings()`; if `None`, run `SetupWizard`, then re-load; build one `ModelBundle`, build `CameraFeed` objects from `data["cameras"]`, hand them to `FeedManager`, start the `root.after()` tick loop calling `FeedManager.tick()`.

10. **Camera-down fallback**, per §3a: per-camera `try/except` around every open/read, offline placeholder + reconnect loop, offline cameras skipped for inference/audio/email without touching other cameras' schedule.

11. **Per-camera calibration confirmation**, per §3b: `calibration_awaiting_confirmation` flag, the pause in `DINO_computation_loop`, and the Continue/Restart Calibration banner on that camera's canvas.

### P1 — only if P0 is done and time remains

1. Cross-platform right-click binding: bind **both** `<Button-3>` (Windows/ Linux) and `<Button-2>` (older macOS Tk convention), plus `<Control-Button-1>` as a trackpad-friendly fallback — don't rely on just one.

2. A "Reconfigure..." menu item that re-opens the Setup Wizard pre-filled with current settings and restarts the app on Finish (simplest correct behavior: just re-exec/relaunch rather than trying to hot-swap cameras).

3. Replace `install_dependencies.bat` / `delete_dependencies.bat` with cross-platform equivalents (a `requirements.txt` + `pip install -r requirements.txt`, or a small `setup.py`/`Makefile`), since `.bat` only runs on Windows and the app itself is meant to be OS-agnostic.

4. Refine the calibration-frame suggestion to also measure one real inference pass during the device-scan step and nudge `inter_camera_delay` up if measured inference time exceeds the tier default.


## 6. Gotchas

- **Tkinter is not thread-safe.** Only touch widgets from the main thread. `FeedManager.tick()` should run via `root.after()`, not a background thread. The audio worker thread and email-sending thread (already `daemon=True` in `DINO.py`) are fine as background threads because they never touch Tk widgets directly — only call `gui.log(...)` from the main thread's tick, not from those worker threads.

- **`pyttsx3` engines are not safe to run concurrently** — that's exactly why §5.8 requires a single serialized worker + queue instead of one thread per alert.

- **Don't reload the DINO/YOLO models per camera.** One `ModelBundle`, reused by every `CameraFeed`, or the process will run out of RAM/VRAM with more than ~2 cameras.

- **Mask size must match that camera's own resolution**, not a global constant — read `cap.get(cv2.CAP_PROP_FRAME_WIDTH/HEIGHT)` per camera like the existing `camera_init()` already does for the single camera.

- Camera indices are OS/driver dependent (`0` isn't guaranteed to be the same physical device on every machine) — that's exactly why the wizard's "Test" button in step 2 exists; don't try to auto-detect "the right" camera, let the user confirm.

- **Never let one camera's `try/except` be missing.** The single most likely crash bug here is one unhandled `cv2.error` or `AttributeError` on a dead camera propagating out of `FeedManager.tick()` and killing the whole `root.after()` loop for every camera. Wrap per-camera, not per-tick.

- A camera sitting in `calibration_awaiting_confirmation` is not "idle" from the user's perspective — it's still showing live video, just not being analyzed. Don't accidentally treat it the same as an offline camera (no "reconnecting" placeholder should appear for it).


## 7. Suggested prompts to drive Antigravity, in order

Paste these one at a time, waiting for each to finish before the next:

1. "Read AGENT.md fully. Implement task 1 and 2 (settings\_store.py v2, and split DINO.py into ModelBundle + CameraFeed as specified). Don't touch gui.py yet."

2. "Now implement task 3 (FeedManager) and task 9 (main.py rewrite), wiring ModelBundle + CameraFeed + FeedManager together, but keep gui.py's current single-canvas layout for now so we can test the scheduler with one camera first."

3. "Now implement task 4, the Setup Wizard, in a new setup\_wizard.py, and wire it into main.py's startup check."

4. "Now implement tasks 5, 6, 7 in gui.py: multi-canvas layout, right-click per-camera settings, and deleting the old restart-group code."

5. "Now implement task 8, the audio queue changes in audio.py."

6. "Now implement task 10 (camera-down fallback, §3a) and task 11 (per-camera calibration confirmation banner, §3b)."

7. "Run the app end to end with two cameras (webcam index 0 twice is fine for a demo). Test unplugging/disabling one camera mid-run and confirm the other keeps working. Fix anything that crashes."


## 8. Definition of done for submission

- Deleting `Assets/settings.json` and launching the app shows the Setup Wizard; completing it launches straight into the multi-camera view.

- At least 2 camera feeds shown, both live at all times.

- Right-clicking a feed opens settings scoped to only that feed; changing a threshold there doesn't affect the other feed.

- Forcing an anomaly on one camera speaks "ANOMALY DETECTED AT CAMERA \{n\}" with the correct number.

- No "Apply & Restart" button/panel exists anywhere in the UI.

- Unplugging a camera (or pointing one at a bad/invalid source) shows that feed as offline and keeps every other feed running — the app does not crash or freeze.

- When a camera finishes calibration, it pauses on a "Continue / Restart Calibration" prompt scoped to that camera only; other cameras keep calibrating/detecting undisturbed while it waits.



Lastly, update the install and delete dependancies accordingly.
