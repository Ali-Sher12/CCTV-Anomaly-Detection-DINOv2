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
    inputs = gb.processor(images=frame, return_tensors="pt")
    with torch.no_grad():
        outputs = gb.model(**inputs, interpolate_pos_encoding=True)
    patch_embeddings = outputs.last_hidden_state[0, 1:, :]
    return patch_embeddings.numpy()


#############   NN Mode Calibration    #############

def finalize_calibration(calibration_store):
    return np.stack(calibration_store, axis=0)  # shape: (num_calib_frames, num_patches, embedding_dim)


def compute_patch_scores(new_embeddings, calibration_array):
    diffs = calibration_array - new_embeddings[np.newaxis, :, :]
    distances = np.linalg.norm(diffs, axis=2)
    patch_scores = np.min(distances, axis=0)
    nearest_slot_per_patch = np.argmin(distances, axis=0)
    return patch_scores, nearest_slot_per_patch


def self_fix_calibration(calibration_array, new_embeddings, nearest_slot_per_patch, frame_eligible):
    if not frame_eligible:
        return
    patch_indices = np.arange(new_embeddings.shape[0])
    calibration_array[nearest_slot_per_patch, patch_indices, :] = new_embeddings


#############   Mahalanobis Mode Calibration    #############

def compute_patch_scores_mahalanobis(new_embeddings, calibration_mean, calibration_precision):
    """
    new_embeddings:       (num_patches, embedding_dim)
    calibration_mean:     (num_patches, embedding_dim)
    calibration_precision:(num_patches, embedding_dim, embedding_dim) -- inverse covariance per patch
    Returns patch_scores: (num_patches,) -- Mahalanobis distance per patch
    """
    diffs = new_embeddings - calibration_mean  # (num_patches, embedding_dim)

    # For each patch p, compute diffs[p] @ precision[p] @ diffs[p] (a scalar per patch)
    temp = np.einsum('pi,pij->pj', diffs, calibration_precision)
    squared_distances = np.einsum('pj,pj->p', temp, diffs)

    # Guard against tiny negative values from floating point rounding before sqrt
    squared_distances = np.maximum(squared_distances, 0)
    return np.sqrt(squared_distances)


def finalize_calibration_mahalanobis(calibration_store):
    """
    calibration_store: list of length totalCalibrationFrames, each shape (num_patches, embedding_dim)
    Returns: (mean, precision)
        mean:      (num_patches, embedding_dim)
        precision: (num_patches, embedding_dim, embedding_dim) -- inverse of the regularized covariance
    """
    stacked = np.stack(calibration_store, axis=0)  # (num_frames, num_patches, embedding_dim)
    num_frames, num_patches, embedding_dim = stacked.shape

    mean = np.mean(stacked, axis=0)  # (num_patches, embedding_dim)

    precision = np.zeros((num_patches, embedding_dim, embedding_dim), dtype=np.float64)
    identity = np.eye(embedding_dim)

    print(f"[Mahalanobis Calibration] Computing covariance + inverse for {num_patches} patches "
          f"({embedding_dim} dims each, {num_frames} samples) -- this may take a little while...")

    for p in range(num_patches):
        patch_samples = stacked[:, p, :]  # (num_frames, embedding_dim)
        covariance = np.cov(patch_samples, rowvar=False)  # (embedding_dim, embedding_dim)
        covariance_regularized = covariance + gb.MAHALANOBIS_EPSILON * identity
        precision[p] = np.linalg.inv(covariance_regularized)

    # Diagnostic: score the calibration frames against their own mean/precision so you
    # can see what a "typical normal" score looks like, and tune TIER_THRESHOLDS_MAHALANOBIS
    # in Globals.py accordingly.
    all_scores = np.stack([
        compute_patch_scores_mahalanobis(stacked[i], mean, precision) for i in range(num_frames)
    ])
    print(f"[Mahalanobis Calibration] typical normal score -- "
          f"mean: {all_scores.mean():.2f}, std: {all_scores.std():.2f}, max: {all_scores.max():.2f}")
    print("Use these numbers as a starting point to tune TIER_THRESHOLDS_MAHALANOBIS in Globals.py")

    return mean, precision


def self_fix_calibration_mahalanobis(calibration_mean, new_embeddings, frame_eligible, alpha):
    """
    Updates the running mean only, in place, via an exponential moving average.
    The covariance/precision matrices stay fixed after initial calibration --
    updating them online would require re-inverting a large matrix per patch every
    accepted frame, which is too expensive for real-time video.
    """
    if not frame_eligible:
        return
    calibration_mean += alpha * (new_embeddings - calibration_mean)


#############   Shared Tiering / Persistence / Eligibility    #############

def compute_tier(score_grid, zone_grid_high_priority, thresholds, thresholds_high_priority):
    tier_grid = np.full(score_grid.shape, "NORMAL", dtype=object)

    is_high_priority = zone_grid_high_priority == 1
    is_normal_zone = ~is_high_priority

    tier_grid[is_normal_zone & (score_grid >= thresholds["ALERT"])] = "ALERT"
    tier_grid[is_normal_zone & (score_grid >= thresholds["CRITICAL"])] = "CRITICAL"

    tier_grid[is_high_priority & (score_grid >= thresholds_high_priority["ALERT"])] = "ALERT"
    tier_grid[is_high_priority & (score_grid >= thresholds_high_priority["CRITICAL"])] = "CRITICAL"

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
    return medium_priority_alert_count <= allowed_error


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
    """Quick connectivity check. Tries to open a socket to Google's DNS.
    Fails fast (within `timeout` seconds) instead of letting smtplib hang."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except OSError:
        return False


def send_email(subject, body, frame):
    """Builds and sends an email with the highlighted frame attached as a jpg.
    Returns True on success, False on any failure (so the caller can fall back to logging)."""
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
        return True
    except Exception as e:
        print("Email send failed:", e)
        return False


def log_anomaly_locally(subject, body, frame, timestamp):
    """Fallback used when there's no internet. Appends a line to a text log
    and saves the highlighted frame as a jpg, both inside gb.LOG_DIR."""
    os.makedirs(gb.LOG_DIR, exist_ok=True)

    log_path = os.path.join(gb.LOG_DIR, "anomaly_log.txt")
    with open(log_path, "a") as f:
        f.write(f"[{timestamp}] {subject} - {body}\n")

    image_path = os.path.join(gb.LOG_DIR, f"anomaly_{timestamp}.jpg")
    cv2.imwrite(image_path, frame)


def report_anomaly(overall_tier, frame, timestamp):
    """Runs on a background thread. Decides email vs local log, and executes it.
    This function itself is blocking, but since it runs in its own thread,
    the main video loop never waits on it."""
    subject = f"Anomaly Detected: {overall_tier}"
    body = f"An anomaly of tier '{overall_tier}' was detected at {timestamp}."

    if is_internet_available():
        success = send_email(subject, body, frame)
        if not success:
            log_anomaly_locally(subject, body, frame, timestamp)
    else:
        log_anomaly_locally(subject, body, frame, timestamp)


def _dispatch_report(overall_tier, highlighted_frame):
    """Takes a snapshot copy of the frame and launches report_anomaly on a
    daemon thread, so the main loop can continue immediately."""
    frame_copy = highlighted_frame.copy()
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    thread = threading.Thread(
        target=report_anomaly,
        args=(overall_tier, frame_copy, timestamp),
        daemon=True
    )
    thread.start()


def handle_anomaly_reporting(overall_tier, highlighted_frame):
    """State machine:
    - No anomaly -> reset the reporting state (timer reset).
    - Anomaly, not currently reporting -> send immediately, start the cooldown.
    - Anomaly, already reporting -> send again only once the wait has elapsed.
    This runs independently of the persistence filter; it only looks at the
    final overall_tier for the frame."""
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
