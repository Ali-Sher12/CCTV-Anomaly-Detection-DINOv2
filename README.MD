# Passive Anomaly Detector PLUS +

A lightweight, hybrid CCTV monitoring tool designed for gates, entryways, driveways, and monitored perimeters. The system combines vision transformer feature modeling (DINOv2 and v3) with object detection (YOLO) and text-to-speech audio feedback to identify unexpected changes in your camera feed.

It does not rely solely on traditional object classifiers. Instead, it builds a visual baseline of what your camera view normally looks like and flags any patch of the frame that deviates from baseline normality.

---

## Key Features

- **Vision Transformer Anomaly Detection**: Uses DINOv2 to extract dense patch embeddings and detect structural, spatial, or lighting anomalies.
- **Hybrid DINO + YOLO Detection**: Combines patch anomaly analysis with YOLO object recognition for enhanced situational awareness, or switch to pure DINO mode.
- **Retro Windows UI**: Built with a classic Windows desktop interface featuring live camera preview, real-time control sliders, and an embedded terminal log.
- **Interactive Mask Editor**: Built-in Paint-like editor to paint priority and exclusion zones directly over your camera feed.
- **Audio Voice Alerts**: Real-time voice announcements of anomaly severity states (Normal, Alert, Critical).
- **Email & Local Logging**: Sends instant email notifications with snapshot attachments during alerts, with automatic offline fallback logging to the `logs` directory.
- **Persistent Settings**: Saves all runtime controls, thresholds, stream configurations, and email credentials to `Assets/settings.json` across sessions.
- **Startup Loading Screen**: Shows a clean initialization window while heavy neural network models, audio engines, and camera devices load into memory.

---

## How It Works

1. **Grid Patch Extraction**: The camera feed is partitioned into a grid of patches. DINOv2 converts each patch into high-dimensional feature embeddings.
2. **Calibration Phase**: During startup or when requested, the system records baseline embeddings across a configurable number of frames.
3. **Anomaly Scoring**: Incoming frames are compared patch-by-patch against the calibration baseline using Euclidean distance.
4. **Severity Tiers**:
   - **NORMAL**: Scene matches baseline conditions.
   - **ALERT**: Noticeable visual change detected.
   - **CRITICAL**: Significant anomaly detected, particularly in high-priority zones.
5. **Persistence Filtering**: Requires anomalous frames to persist across consecutive readings before confirming an alert, suppressing transient camera noise.
6. **Online Self-Updating Baseline**: When enabled, clean frames gently update baseline statistics so gradual ambient lighting shifts do not cause false alarms.

---

## Zone Mask System

You can control which parts of the frame matter using the built-in Zone Painter:

- **Green (Medium Priority)**: Standard surveillance area evaluated under standard alert thresholds.
- **Red (High Priority)**: Critical areas (e.g. gates, doors) evaluated under stricter thresholds.
- **Black (Dead Zone / Ignored)**: Ignored areas (e.g. trees swaying in the wind, roads, timestamps).

Click **Modify Mask...** in the application to launch the editor. Use Brush, Bucket Fill, Circle, or Rectangle tools, then save and click **Apply & Restart** to reload.

---

## Installation & Setup

### Prerequisites
- Windows OS
- Python 3.14 (or Python 3.10+)
- Webcam or RTSP / HTTP IP Camera Stream

### Option 1: Automatic Setup via Batch File
Run the included installation script as Administrator:
```bat
install_dependencies.bat
```

To remove all installed dependencies later:
```bat
delete_dependencies.bat
```

### Option 2: Manual Installation via pip
```bash
pip install torch torchvision transformers numpy opencv-python pillow psutil ultralytics pyttsx3 scipy huggingface_hub safetensors tqdm PyYAML certifi filelock fsspec packaging regex idna typing_extensions
```

---

## Running the Application

Start the application with:
```bash
python main.py
```

### Keyboard Shortcuts:
- **Esc**: Safely closes the application and saves all current settings.

---

## Controls and Configuration

### Runtime Controls (Apply Immediately)
- **Frame Interval (s)**: Processing delay between analyzed frames.
- **Display Delay (ms)**: GUI refresh interval.
- **Persistence Filter**: Number of consecutive anomalous frames required before triggering an alert.
- **Alert Tolerance**: Number of anomalous patches permitted outside high-priority zones before triggering an alert.
- **Report Cooldown (s)**: Minimum duration between consecutive email notifications and logs.
- **Auto-Update Calibration**: Enables adaptive background learning from clean frames.
- **Thresholds (Alert / Critical)**: Sensitivity cutoffs for standard and high-priority zones.

### Settings Requiring Restart (Staged Controls)
- **DINO Only (no Hybrid)**: Disables YOLO model processing for lightweight DINO-only execution.
- **Video Stream & URL**: Toggle between local USB webcam (0) and remote IP camera streams (e.g. `http://IP:PORT/video`).
- **Calibration Frames**: Number of frames recorded during calibration baseline capture.
- **Email Settings**: Dedicated configuration window for Sender Email, App Password, and Receiver Email.
- **Apply & Restart Button**: Commits all staged settings and re-initializes the camera and model pipeline.

---

## Project Structure

```
.
├── Assets/                 # UI assets, mask.png, logo.png, and settings.json
├── Models/                 # Pretrained DINOv2 and YOLO weights
├── logs/                   # Local anomaly text logs and saved alert images
├── Accessories.py          # System specs, timing helpers, and hardware checks
├── DINO.py                 # DINOv2 feature extractor and anomaly math pipeline
├── Globals.py              # Application globals and configuration defaults
├── YOLO.py                 # YOLO detection and overlay pipeline
├── audio.py                # Text-to-speech audio alert engine
├── gui.py                  # Main retro Tkinter graphical interface
├── loading_screen.py       # Startup splash loading window
├── mask_editor.py          # Dedicated paint-style mask editor
├── settings_store.py       # JSON configuration persistence layer
├── main.py                 # Application entry point
├── install_dependencies.bat# Automated dependency installation script
├── delete_dependencies.bat # Automated dependency removal script
└── README.MD               # Documentation
```