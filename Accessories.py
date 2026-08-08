import time
import torch
import cv2
import Globals as gb
import numpy as np
import socket
import smtplib
import threading
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from transformers import AutoImageProcessor, AutoModel

_start = time.perf_counter()
_lastFrameTimeSeconds = 0

def getTimeSeconds():
    return time.perf_counter()-_start

def getTimeMiliseconds():
    return (time.perf_counter()-_start)*1000

def printSeconds():
    print(getTimeSeconds()," seconds passed.")

def printMiliseconds():
    print(getTimeMiliseconds()," miliseconds passed.")

def FrameEligiblebyTime(secondsForOneFrame):
    global _lastFrameTimeSeconds
    time_cache = getTimeSeconds()
    if time_cache-_lastFrameTimeSeconds>secondsForOneFrame:
        _lastFrameTimeSeconds = time_cache
        return True
    else:
        return False


#############   Model Setup    #############
def Model_Setup():
    gb.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small",cache_dir="Models",local_files_only=True)
    gb.model = AutoModel.from_pretrained("facebook/dinov2-small",cache_dir="Models",local_files_only=True)
    gb.model.eval()

    _input_size = gb.processor.crop_size["height"]
    _patch_size = gb.model.config.patch_size
    gb.grid_h = gb.grid_w = _input_size // _patch_size


def get_patch_embeddings(frame):
    inputs = gb.processor(images=frame, return_tensors="pt",do_resize=False, do_center_crop=False)
    with torch.no_grad():
        outputs = gb.model(**inputs, interpolate_pos_encoding=True)
    patch_embeddings = outputs.last_hidden_state[0, 1:, :]
    return patch_embeddings.numpy()


def finalize_calibration(calibration_store):
    return np.stack(calibration_store, axis=0)  # shape: (num_calib_frames, num_patches, embedding_dim)


def compute_patch_scores(new_embeddings, calibration_array):
    diffs = calibration_array - new_embeddings[np.newaxis, :, :]
    distances = np.linalg.norm(diffs, axis=2)
    patch_scores = np.min(distances, axis=0)
    nearest_slot_per_patch = np.argmin(distances, axis=0)
    return patch_scores, nearest_slot_per_patch


def compute_tier(score_grid, zone_grid_high_priority):
    tier_grid = np.full(score_grid.shape, "NORMAL", dtype=object)

    is_high_priority = zone_grid_high_priority == 1
    is_normal_zone = ~is_high_priority

    tier_grid[is_normal_zone & (score_grid >= gb.TIER_THRESHOLDS["ALERT"])] = "ALERT"
    tier_grid[is_normal_zone & (score_grid >= gb.TIER_THRESHOLDS["CRITICAL"])] = "CRITICAL"

    tier_grid[is_high_priority & (score_grid >= gb.TIER_THRESHOLDS_HIGH_PRIORITY["ALERT"])] = "ALERT"
    tier_grid[is_high_priority & (score_grid >= gb.TIER_THRESHOLDS_HIGH_PRIORITY["CRITICAL"])] = "CRITICAL"

    return tier_grid


def apply_persistence_filter(tier_grid, counters, required):
    is_anomalous = tier_grid != "NORMAL"
    counters = np.where(is_anomalous, counters + 1, 0)
    confirmed_tier_grid = np.where(counters >= required, tier_grid, "NORMAL")
    return confirmed_tier_grid, counters


def is_frame_eligible(raw_tier_grid, zone_grid_high_priority, allowed_error):
    is_high_priority = (zone_grid_high_priority == 1).flatten()
    is_medium_priority = ~is_high_priority
    flat_tier = raw_tier_grid.flatten()

    high_priority_anomalous = np.any(is_high_priority & (flat_tier != "NORMAL"))
    if high_priority_anomalous:
        return False

    medium_priority_critical = np.any(is_medium_priority & (flat_tier == "CRITICAL"))
    if medium_priority_critical:
        return False

    medium_priority_alert_count = np.sum(is_medium_priority & (flat_tier == "ALERT"))
    return medium_priority_alert_count < allowed_error


def self_fix_calibration(calibration_array, new_embeddings, nearest_slot_per_patch, frame_eligible):
    if not frame_eligible:
        return
    patch_indices = np.arange(new_embeddings.shape[0])
    calibration_array[nearest_slot_per_patch, patch_indices, :] = new_embeddings


def draw_overlay(frame, current_highlight, alpha=0.45):
    if current_highlight is None:
        return frame.copy()

    tier_grid, overall_tier = current_highlight
    grid_h, grid_w = tier_grid.shape
    frame_h, frame_w = frame.shape[:2]

    color_grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    highlight_mask = np.zeros((grid_h, grid_w), dtype=np.uint8)

    for tier, color in gb.TIER_COLORS.items():
        matches = tier_grid == tier
        color_grid[matches] = color
        highlight_mask[matches] = 1

    color_full = cv2.resize(color_grid, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
    mask_full = cv2.resize(highlight_mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)

    blended_full = cv2.addWeighted(frame, 1 - alpha, color_full, alpha, 0)

    result = frame.copy()
    result[mask_full == 1] = blended_full[mask_full == 1]

    label_color = gb.TIER_COLORS.get(overall_tier, (255, 255, 255))
    cv2.putText(result, overall_tier, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)

    return result


#############   Anomaly Reporting    #############

def is_internet_available(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except OSError:
        gb.gui.log("Email send failed. No internet.")        
        return False


def send_email(subject, body, frame):
    try:
        msg = MIMEMultipart()
        msg["From"] = gb.EMAIL_SENDER
        msg["To"] = gb.EMAIL_RECEIVER
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        success, encoded_image = cv2.imencode(".jpg", frame)
        if success:
            image_attachment = MIMEImage(encoded_image.tobytes(), name="anomaly.jpg")
            msg.attach(image_attachment)

        with smtplib.SMTP(gb.SMTP_SERVER, gb.SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(gb.EMAIL_SENDER, gb.EMAIL_PASSWORD)
            server.send_message(msg)
        gb.gui.log("Anomaly successfully reported at : " + gb.EMAIL_RECEIVER)
        return True
    except Exception as e:
        gb.gui.log("Email send failed.")
        return False


def log_anomaly_locally(subject, body, frame, timestamp):
    os.makedirs(gb.LOG_DIR, exist_ok=True)

    log_path = os.path.join(gb.LOG_DIR, "anomaly_log.txt")
    with open(log_path, "a") as f:
        f.write(f"[{timestamp}] {subject} - {body}\n")

    image_path = os.path.join(gb.LOG_DIR, f"anomaly_{timestamp}.jpg")
    cv2.imwrite(image_path, frame)


def report_anomaly(overall_tier, frame, timestamp):
    subject = f"Anomaly Detected: {overall_tier}"
    body = f"An anomaly of tier '{overall_tier}' was detected at {timestamp}."

    if is_internet_available():
        send_email(subject, body, frame)
        log_anomaly_locally(subject, body, frame, timestamp)
    else:
        log_anomaly_locally(subject, body, frame, timestamp)


def _dispatch_report(overall_tier, highlighted_frame):
    frame_copy = highlighted_frame.copy()
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    thread = threading.Thread(
        target=report_anomaly,
        args=(overall_tier, frame_copy, timestamp),
        daemon=True
    )
    thread.start()


def handle_anomaly_reporting(overall_tier, highlighted_frame):
    now = getTimeSeconds()

    if overall_tier == "NORMAL":
        gb.reporting_active = False
        return

    if not gb.reporting_active:
        _dispatch_report(overall_tier, highlighted_frame)
        gb.reporting_active = True
        gb.last_report_time = now
    else:
        if now - gb.last_report_time >= gb.anomaly_report_wait:
            _dispatch_report(overall_tier, highlighted_frame)
            gb.last_report_time = now
