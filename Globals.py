"""Globals.py — Global runtime state and default configurations."""

# --- Runtime Configuration ---
delay = 3
allowed_error = 1
anomaly_report_wait = 15
auto_update_calibration = False
audio_muted = False

TIER_THRESHOLDS = {
    "ALERT": 45.0,
    "CRITICAL": 50.0,
}

TIER_THRESHOLDS_HIGH_PRIORITY = {
    "ALERT": 35.0,
    "CRITICAL": 40.0,
}

# --- Settings & Models ---
totalCalibrationFrames = 50
DINO_ONLY = False
useGPU = False
DINO_MODEL_VERSION = "facebook/dinov2-small"
YOLO_MODEL_VERSION = "Models/YOLO/yolo26s.pt"

EMAIL_SENDER = ""
EMAIL_PASSWORD = ""
EMAIL_RECEIVER = ""
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
LOG_DIR = "logs"

# --- Visual Colors (BGR for OpenCV) ---
TIER_COLORS = {
    "ALERT": (0, 165, 255),    # Orange
    "CRITICAL": (0, 0, 255),   # Red
}

severity_order = ["NORMAL", "ALERT", "CRITICAL"]

# Reference colors for mask processing (BGR)
medium_priority_region = (0, 255, 0)  # Green
high_priority_color = (0, 0, 255)     # Red
dead_zone_color = (0, 0, 0)           # Black

gui = None

