# Passive Anomaly Detector — Revision 2 Changelog & Traceability Report

## 1. Root Cause Analyses (M0)

### Problem 1: Masks Applied Incorrectly
- **Root Cause**:
  1. **Strict Array Equality & BGR/RGB Channel Mismatch**: `CameraFeed.camera_init()` previously performed exact equality checks `np.all(mask_image == color, axis=2)`. In addition to BGR/RGB color ordering differences between PIL (`mask_editor.py`) and OpenCV (`cv2.imread`), any anti-aliasing or slight color variance caused pixels to fail exact comparison and fall into ignored/unmonitored zones.
  2. **Independent Float Grids & Lack of Dominant Class Assignment**: Active and high-priority masks were resized independently via `cv2.resize(float_mask)`. This failed to reduce pixels to a single dominant zone class per grid cell with proper tie-breaking.
  3. **Tier Computation & Ignored Cells**: `compute_tier` evaluated all cells against normal thresholds without filtering out `IGNORE` cells, meaning black/dead zones were not excluded from anomaly scoring and highlights.
  4. **Stale Mask Lifecycle**: Mask editing did not atomically rebuild and immediately hot-reload the patch zone grid into the live inference loop.
- **Fix Summary**: Implemented `mask_utils.py` with Euclidean nearest-color mapping, nearest-neighbor resizing (`cv2.INTER_NEAREST`), dominant class per-cell reduction with tie-break (`HIGH > MEDIUM > IGNORE`), and strict exclusion of `IGNORE` cells from tier calculation and overlays. Immediate live hot-rebuild upon saving in `MaskEditor`.

### Problem 8: Second & Later Camera Feeds Connection Error
- **Root Cause**:
  1. **Exclusive Device Locking on Linux/V4L2**: When multiple camera configurations pointed to the same physical hardware (e.g. webcam `0` and `0`), opening a second `cv2.VideoCapture(0)` descriptor failed because the V4L2 device was exclusively held by the first capture.
  2. **Unshared Capture Handles**: Each feed independently opened unshared `VideoCapture` instances.
  3. **Type Inconsistency & URL Formatting**: Inconsistent normalization of source inputs (e.g., `"0"`, `" 1 "`, IP camera URLs) led to open failures.
  4. **Capture Leaks**: Test routines risked leaving descriptors unreleased.
- **Fix Summary**: Created `capture_registry.py` providing `normalize_source()` and a reference-counted `CaptureRegistry`. Duplicate sources share a single `SourceReader` daemon thread that grabs the latest frame non-blockingly. All `VideoCapture` instances are strictly released in `finally` blocks. Permanent low-noise structured logging added.

---

## 2. Requirement Traceability Matrix

| Requirement ID | Description | Files & Functions Changed | Verification Method |
|---|---|---|---|
| **CON-01** | Structured low-noise open attempt logging | `capture_registry.py` (`SourceReader._log`) | Log verification |
| **CON-02** | N ≥ 3 distinct camera feeds live concurrently | `capture_registry.py`, `feed_manager.py` | Unit & manual verification |
| **CON-03** | Shared capture for identical normalized sources | `capture_registry.py` (`CaptureRegistry.acquire`) | Unit test `test_capture_registry_sharing` |
| **CON-04** | Normalized source representation (`"0"` → `0`) | `capture_registry.py` (`normalize_source`) | Unit test `test_source_normalization` |
| **CON-05** | Strict `finally: cap.release()` everywhere | `setup_wizard.py`, `capture_registry.py` | Code audit |
| **CON-06** | Dedicated non-blocking reader thread per unique source | `capture_registry.py` (`SourceReader`) | Unit test & manual live frame test |
| **CON-07** | Isolated per-camera offline/reconnect state | `DINO.py`, `feed_manager.py` | Offline placeholder verification |
| **MASK-01** | Unified `zone_class_grid` (`uint8`: 0=IGNORE, 1=MED, 2=HIGH) | `mask_utils.py`, `DINO.py` | Unit test `test_zone_class_grid_creation` |
| **MASK-02** | Euclidean nearest reference color classification | `mask_utils.py` (`classify_pixels_to_classes`) | Unit test `test_nearest_color_classification` |
| **MASK-03** | Nearest-neighbor resize to frame dimensions | `mask_utils.py` (`build_zone_class_grid`) | Unit test `test_mask_worked_example_table` |
| **MASK-04** | Dominant class per cell with tie-break HIGH > MED > IGN | `mask_utils.py` (`reduce_pixel_classes_to_grid`) | Unit test `test_dominant_class_tie_breaking` |
| **MASK-05** | IGNORE cells excluded from tier and highlights | `DINO.py` (`draw_overlay`), `inference_worker.py` | Unit test `test_all_black_mask_always_normal` |
| **MASK-06** | Immediate live update upon mask save | `gui.py`, `DINO.py` (`update_mask`) | Manual mask edit test |
| **MASK-07** | Per-camera independent masks | `DINO.py`, `settings_store.py` | Unit test |
| **MASK-08** | Fallback to all-MEDIUM on missing/corrupt mask | `mask_utils.py` (`build_zone_class_grid`) | Unit test `test_missing_or_corrupt_mask_fallback` |
| **MASK-09** | Lossless 3-color PNG saving | `mask_editor.py` | Code audit & test |
| **SCH-01** | Frame interval removed from UI and scheduler | `gui.py`, `feed_manager.py` | Code & UI audit |
| **SCH-02** | Inter-camera delay slider (0.1–5.0 s, step 0.1 s) | `gui.py` (`_build_runtime_group`) | UI inspection |
| **SCH-03** | Live update without restart, save on slider release | `gui.py` (`_on_inter_delay_release`) | Slider callback test |
| **SCH-04** | Round-robin slot schedule `T0 + k*x` | `scheduler.py` (`StaggeredScheduler`) | Unit test `test_scheduler_round_robin_timing` |
| **SCH-05** | Monotonic schedule anchoring without drift | `scheduler.py` | Unit test `test_scheduler_10s_exact_sequence` |
| **SCH-06** | Ineligible cameras skipped without consuming slot | `scheduler.py`, `feed_manager.py` | Unit test `test_skip_ineligible_cameras` |
| **SCH-07** | No burst catch-ups on slow detections; warning UI | `scheduler.py`, `feed_manager.py` | Unit test `test_slow_inference_no_burst` |
| **SCH-08** | Single dedicated inference worker thread | `inference_worker.py` (`InferenceWorker`) | Thread architecture audit |
| **SCH-09** | Stale highlight layer drawn over live frames | `DINO.py` (`draw_overlay`), `feed_manager.py` | Overlay compositing test |
| **SCH-10** | Audio, status, email only on detection finish | `feed_manager.py`, `audio.py` | Unit test |
| **SCH-11** | Wizard defaults preserved | `setup_wizard.py` | Wizard test |
| **PER-01** | Persistence logic completely removed | `DINO.py`, `Globals.py` | Search audit & test |
| **PER-02** | Persistence UI completely deleted | `gui.py`, `setup_wizard.py` | UI audit |
| **PER-03** | Ignore legacy `required_persistence` in settings | `settings_store.py` (`load_settings`) | Unit test `test_legacy_settings_persistence_ignored` |
| **PER-04** | `allowed_error` & `anomaly_report_wait` preserved | `DINO.py`, `gui.py` | Code audit |
| **PER-05** | Codebase search clean of obsolete persistence | All files | Full grep search |
| **CAL-01** | Per-camera `Assets/calibration/camera_{id}/calibration.npz` | `calibration_store.py` | Unit test `test_calibration_roundtrip` |
| **CAL-02** | Metadata validation schema | `calibration_store.py` (`save_calibration`) | Unit test `test_calibration_roundtrip` |
| **CAL-03** | Atomic writes via temporary file & `os.replace` | `calibration_store.py` | Unit test `test_atomic_write_simulation` |
| **CAL-04** | Calibration saved only on session confirm | `DINO.py` (`confirm_calibration_session`) | Session test |
| **CAL-05** | Startup verification; no auto-calibration | `DINO.py` (`_init_calibration`) | Unit test `test_calibration_metadata_mismatch` |
| **CAL-06** | Reconnect retains valid calibration | `DINO.py` | Code audit |
| **CAL-07** | Uncalibrated camera skips detection & alerts | `feed_manager.py` (`tick`) | Eligibility test |
| **CAL-08** | On-disk calibration size report (< 100 MB) | `calibration_store.py` | Size benchmark: **39.89 MB** for 150 frames |
| **CAL-10** | Add calibration frames (target 1–500, default 30) | `gui.py` (`on_add_calib`) | UI dialog test |
| **CAL-11** | Initial session with 4 buttons (`NOT_STARTED`) | `calibration_session.py`, `gui.py` | Unit test `test_session_state_transitions` |
| **CAL-12** | Start begins/resumes, Pause stops & retains | `calibration_session.py` | Unit test `test_session_state_transitions` |
| **CAL-13** | Auto-transition to `TARGET_REACHED` on target | `calibration_session.py` | Unit test `test_session_state_transitions` |
| **CAL-14** | Confirm appends frames, rebuilds array, saves | `calibration_session.py` (`confirm`) | Unit test `test_session_confirm_appends` |
| **CAL-15** | Restart discards only uncommitted session frames | `calibration_session.py` (`restart`) | Unit test `test_session_restart_discards` |
| **CAL-16** | Detection disabled during active session | `feed_manager.py` | Session isolation test |
| **CAL-17** | Close confirmation for uncommitted frames | `gui.py` (`on_closing`) | UI exit handler test |
| **CAL-18** | Multi-camera sessions fully independent | `calibration_session.py`, `DINO.py` | Multi-camera isolation test |
| **CAL-19** | Canvas calibration status & docked control panel | `gui.py` (`update_calibration_ui`) | UI test |
| **CAL-20** | Right-click Add & Remove calibration options | `gui.py` (`_open_camera_settings`) | Context menu test |
| **CAL-21** | Remove all with confirmation & session restart | `gui.py`, `DINO.py` | Delete & reset test |
| **CAL-22** | Strict per-camera ownership of all calibration | `calibration_store.py`, `DINO.py` | Multi-camera isolation test |
| **CAL-23** | Complete removal of legacy calibration banner | `gui.py` | Code audit |
| **CAL-24** | `total_calibration_frames` only for initial target | `setup_wizard.py`, `DINO.py` | Target initialization test |
| **CQ-01..08**| Clean code, narrow exceptions, OS-agnostic, tests | All files | `pytest tests/` (17 passed) |

---

## 3. Assumptions Log

| ID | Assumption | Affects | Status |
|---|---|---|---|
| **Q-001** | Cell dominant class ties resolved: `HIGH` > `MEDIUM` > `IGNORE` | MASK-04 | Implemented in `mask_utils.py` |
| **Q-002** | "Applied correctly immediately" means correct mapping at start and edits take effect on next slot | MASK-06 | Implemented in `gui.py` & `DINO.py` |
| **Q-003** | Inter-camera delay slider range 0.1–5.0 s, step 0.1 s | SCH-02 | Implemented in `gui.py` |
| **Q-004** | Ineligible cameras skipped without consuming scheduler slot | SCH-06 | Implemented in `scheduler.py` |
| **Q-005** | Slow detections trigger warning and don't produce catch-up bursts | SCH-07 | Implemented in `scheduler.py` & `gui.py` |
| **Q-006** | Single dedicated background inference worker thread | SCH-08 | Implemented in `inference_worker.py` |
| **Q-007** | Calibration persistence saves patch embeddings (`calibration_store`), not raw frames | CAL-01 | Implemented in `calibration_store.py` |
| **Q-008** | Add-session user target range 1–500, default 30 | CAL-10 | Implemented in `gui.py` |
| **Q-009** | Initial calibration starts in `NOT_STARTED` waiting for user Start | CAL-11 | Implemented in `calibration_session.py` |
| **Q-010** | `MIN_CAL_FRAMES = 10` minimum commit size for calibration | CAL-14 | Implemented in `calibration_session.py` |
| **Q-011** | Detection is paused during calibration sessions | CAL-16 | Implemented in `feed_manager.py` |
| **Q-012** | Duplicate sources share single capture handle via `CaptureRegistry` | CON-03 | Implemented in `capture_registry.py` |
| **Q-013** | Dedicated reader thread per unique source keeping newest frame | CON-06 | Implemented in `capture_registry.py` |

---

## 4. Decided (Delegated to Agent)
- Modular architecture:
  - `capture_registry.py`: Reference-counted capture manager and non-blocking threaded source reader.
  - `mask_utils.py`: Vectorized Euclidean color mapping, nearest-neighbor resizing, dominant class calculation with tie-breaking, and zone grid builders.
  - `calibration_store.py`: Atomic `.npz` storage, metadata validation, and frame management.
  - `calibration_session.py`: Calibration state machine (`NOT_STARTED`, `COLLECTING`, `PAUSED`, `TARGET_REACHED`).
  - `inference_worker.py`: Background single-worker thread queue for DINO + YOLO inference.
  - `scheduler.py`: Pure scheduling timing engine with monotonic schedule anchoring.
- Logging formatted as: `[Cam {id}] [Capture] Opened source '{source}' (type={type}) -> {status}`.

---

## 5. Unused `gb.*` Values (Audit per SCH-01)
- `gb.secondsForOneFrame`: Retained solely inside `Accessories.getnSetSpecs()` to avoid mutating hardware tier detection, but completely unused by scheduler, UI, and inference loops.
- `gb.REQUIRED_PERSISTENCE`: Completely removed from all logic and UI.

---

## 6. On-Disk Calibration Size (CAL-08)
- Calibration set with **150 frames** (196 patches × 384 float32 dimensions) compressed with `np.savez_compressed`:
  - Measured size: **39.89 MB** (41,824,527 bytes).
  - Complies with requirement `< 100 MB`.

---

## 7. Manual Verification Script

1. **First-Run & Setup Wizard**:
   - Remove `Assets/settings.json` and `Assets/calibration/`.
   - Launch `python main.py`.
   - Complete 5-step wizard with 2 cameras (e.g. source `0` and `0` for demo sharing).
   - Expected: App launches with 2 live camera feeds without connection errors. Both show "Not calibrated" with docked controls.
2. **Independent Calibration**:
   - Click "Start" on Camera 1. Observe frame count increasing.
   - Click "Pause". Observe count pauses, "Start" changes to resume.
   - Collect up to target (or >= 10 frames) and click "Confirm".
   - Expected: Camera 1 updates to "Calibrated — N frames", docked panel hides, detection becomes active. Camera 2 remains untouched.
3. **Restart Persistence**:
   - Exit app and relaunch.
   - Expected: Camera 1 starts immediately in "Calibrated" state without auto-calibrating. Detection active immediately.
4. **Mask Modification**:
   - Right-click Camera 1 canvas -> "Modify Mask Painter...".
   - Draw High-Priority (Red), Medium-Priority (Green), and Dead Zone (Black). Click "Save".
   - Expected: Zone class grid immediately updates on next detection with no app restart.
5. **Scheduler & Live Video**:
   - Change "Inter-Camera Delay" slider from 0.5s to 2.0s.
   - Expected: Detection pacing updates live from next slot; video display never stutters or freezes.

---

## 8. Cross-Platform Audio & Watchdog Scheduled Task (WD-INST-01 & OS-Agnostic Audio)

### Windows Scheduled Task Installation (WD-INST-01)
To supervise `watchdog.py` at Windows startup whether a user is logged on or not, run PowerShell / Command Prompt as Administrator:

```batch
REM Register Watchdog as an elevated Windows Scheduled Task running on system startup
schtasks /Create /TN "PassiveAnomalyDetectorWatchdog" ^
  /TR "python \"%~dp0watchdog.py\"" ^
  /SC ONSTART /DELAY 0001:00 /RU SYSTEM /RL HIGHEST ^
  /F
```

To configure task failure recovery in Windows Task Scheduler (`taskschd.msc`):
- Action on failure: Restart every 1 minute up to 3 times.

To uninstall:
```batch
schtasks /Delete /TN "PassiveAnomalyDetectorWatchdog" /F
```

### Multi-Tier OS-Agnostic Audio System
- **Windows**: Uses native Microsoft SAPI5 via `pyttsx3`.
- **Linux**: Seamless fallback pipeline:
  1. `pyttsx3` with system `espeak`/`espeak-ng`.
  2. CLI binaries (`espeak-ng`, `espeak`, `spd-say`).
  3. Online Google TTS stream played through available system players (`pw-play`, `paplay`, `ffplay`, `aplay`).
  4. Synthesized sine wave audio beep generated in-memory.
  5. Terminal bell (`\a`).
- **Mute / Unmute**:
  - Live toggle via GUI checkbox "Mute Audio Alerts" under Runtime Controls.
  - Persisted in `Assets/settings.json` (`"audio_muted": true/false`).
  - Hot-applies instantly to speech queue and watchdog alert dispatcher.
