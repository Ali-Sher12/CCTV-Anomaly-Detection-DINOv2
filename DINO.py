import time
import torch
import cv2
import Globals as gb
import Accessories as Acc
import numpy as np
import socket
import smtplib
import threading
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from transformers import AutoImageProcessor, AutoModel

class DINO_MODEL:

    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small",cache_dir="Models",local_files_only=True)
        self.model = AutoModel.from_pretrained("facebook/dinov2-small",cache_dir="Models",local_files_only=True)
        self.model.eval()
        self._input_size = self.processor.crop_size["height"]
        self._patch_size = self.model.config.patch_size
        self.grid_h = self.grid_w = self._input_size // self._patch_size
        self.frame = None
        self.persistence_counters = None
        self.zone_grid_high_priority = None
        self.new_mask = None
        self.mask_image = None
        self.is_active = None
        self.is_high_priority = None
        self.is_watched = None
        self.high_priority_mask = None
        self.highlighted_frame = None
        self.overall_tier = "NORMAL"

    def getFrame(self,f):
        self.frame = f.copy()

    def _get_patch_embeddings(self,final_frame):
        inputs = self.processor(images=final_frame, return_tensors="pt",do_resize=False, do_center_crop=False)
        with torch.no_grad():
            outputs = self.model(**inputs, interpolate_pos_encoding=True)
        patch_embeddings = outputs.last_hidden_state[0, 1:, :]
        return patch_embeddings.numpy()

    def _finalize_calibration(self):
        return np.stack(gb.calibration_store, axis=0)  # shape: (num_calib_frames, num_patches, embedding_dim)


    def compute_patch_scores(self,new_embeddings, calibration_array):
        diffs = calibration_array - new_embeddings[np.newaxis, :, :]
        distances = np.linalg.norm(diffs, axis=2)
        patch_scores = np.min(distances, axis=0)
        nearest_slot_per_patch = np.argmin(distances, axis=0)
        return patch_scores, nearest_slot_per_patch


    def compute_tier(self,score_grid, zone_grid_high_priority):
        tier_grid = np.full(score_grid.shape, "NORMAL", dtype=object)
        is_high_priority = zone_grid_high_priority == 1
        is_normal_zone = ~is_high_priority
        tier_grid[is_normal_zone & (score_grid >= gb.TIER_THRESHOLDS["ALERT"])] = "ALERT"
        tier_grid[is_normal_zone & (score_grid >= gb.TIER_THRESHOLDS["CRITICAL"])] = "CRITICAL"
        tier_grid[is_high_priority & (score_grid >= gb.TIER_THRESHOLDS_HIGH_PRIORITY["ALERT"])] = "ALERT"
        tier_grid[is_high_priority & (score_grid >= gb.TIER_THRESHOLDS_HIGH_PRIORITY["CRITICAL"])] = "CRITICAL"
        return tier_grid

    def apply_persistence_filter(self,tier_grid, counters, required):
        is_anomalous = tier_grid != "NORMAL"
        counters = np.where(is_anomalous, counters + 1, 0)
        confirmed_tier_grid = np.where(counters >= required, tier_grid, "NORMAL")
        return confirmed_tier_grid, counters

    def is_frame_eligible(self,raw_tier_grid, zone_grid_high_priority, allowed_error):
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


    def self_fix_calibration(self,calibration_array, new_embeddings, nearest_slot_per_patch, frame_eligible):
        if not frame_eligible:
            return
        patch_indices = np.arange(new_embeddings.shape[0])
        calibration_array[nearest_slot_per_patch, patch_indices, :] = new_embeddings


    def draw_overlay(self):
        alpha=0.45
        if gb.current_highlight is None:
            return self.frame.copy()

        tier_grid, overall_tier = gb.current_highlight
        self.grid_h, self.grid_w = tier_grid.shape
        frame_h, frame_w = self.frame.shape[:2]

        color_grid = np.zeros((self.grid_h, self.grid_w, 3), dtype=np.uint8)
        highlight_mask = np.zeros((self.grid_h, self.grid_w), dtype=np.uint8)

        for tier, color in gb.TIER_COLORS.items():
            matches = tier_grid == tier
            color_grid[matches] = color
            highlight_mask[matches] = 1

        color_full = cv2.resize(color_grid, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
        mask_full = cv2.resize(highlight_mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)

        blended_full = cv2.addWeighted(self.frame, 1 - alpha, color_full, alpha, 0)

        result = self.frame.copy()
        result[mask_full == 1] = blended_full[mask_full == 1]

        label_color = gb.TIER_COLORS.get(overall_tier, (255, 255, 255))
        if overall_tier == "CRITICAL":
            overall_tier = "ANOMALY"
        cv2.putText(result, overall_tier, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)

        return result


    #############   Anomaly Reporting    #############

    def is_internet_available(self,host="8.8.8.8", port=53, timeout=3):
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            return True
        except OSError:
            gb.gui.log("Email send failed. No internet.")        
            return False


    def send_email(self,subject, body):
        try:
            msg = MIMEMultipart()
            msg["From"] = gb.EMAIL_SENDER
            msg["To"] = gb.EMAIL_RECEIVER
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            success, encoded_image = cv2.imencode(".jpg", self.highlighted_frame)
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


    def log_anomaly_locally(self,subject, body, timestamp):
        os.makedirs(gb.LOG_DIR, exist_ok=True)

        log_path = os.path.join(gb.LOG_DIR, "anomaly_log.txt")
        with open(log_path, "a") as f:
            f.write(f"[{timestamp}] {subject} - {body}\n")

        image_path = os.path.join(gb.LOG_DIR, f"anomaly_{timestamp}.jpg")
        cv2.imwrite(image_path, self.highlighted_frame)


    def report_anomaly(self, timestamp):
        subject = f"Anomaly Detected: {self.overall_tier}"
        body = f"An anomaly of tier '{self.overall_tier}' was detected at {timestamp}."

        if self.is_internet_available():
            self.send_email(subject, body)
            self.log_anomaly_locally(subject, body, timestamp)
        else:
            self.log_anomaly_locally(subject, body, timestamp)


    def _dispatch_report(self):
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        thread = threading.Thread(
            target=self.report_anomaly,
            args=(timestamp,),
            daemon=True
        )
        thread.start()


    def handle_anomaly_reporting(self):
        now = Acc.getTimeSeconds()

        if self.overall_tier == "NORMAL":
            gb.reporting_active = False
            return

        if not gb.reporting_active:
            self._dispatch_report()
            gb.reporting_active = True
            gb.last_report_time = now
        else:
            if now - gb.last_report_time >= gb.anomaly_report_wait:
                self._dispatch_report()
                gb.last_report_time = now

    def camera_init(self):
        #############    Mask Setup    #############
        self.persistence_counters = np.zeros((self.grid_h, self.grid_w), dtype=int)
        self.mask_image = cv2.imread("Assets/mask.png")
        self.is_active = np.all(self.mask_image == gb.medium_priority_region, axis=2)
        self.is_high_priority = np.all(self.mask_image == gb.high_priority_color, axis=2)
        self.is_watched = self.is_active | self.is_high_priority    

        mask = self.is_watched.astype(np.float32)
        self.new_mask = cv2.resize(mask, (gb.frameShape[0], gb.frameShape[1]), interpolation=cv2.INTER_NEAREST)

        self.high_priority_mask = self.is_high_priority.astype(np.float32)
        self.zone_grid_high_priority = cv2.resize(self.high_priority_mask,(self.grid_w, self.grid_h),interpolation=cv2.INTER_NEAREST)
        ############################################

    def DINO_computation_loop(self):    
        Ignored_zone_done_mask = self.frame.copy()
        Ignored_zone_done_mask[self.new_mask == 0] = 0
        final_frame = cv2.cvtColor(Ignored_zone_done_mask, cv2.COLOR_BGR2RGB)
        final_frame = cv2.resize(final_frame,(224,224),interpolation=cv2.INTER_AREA)

        if Acc.FrameEligiblebyTime(gb.secondsForOneFrame):
            embeddings = self._get_patch_embeddings(final_frame)
            if gb.initialCalibration:
                gb.calibration_store.append(embeddings)
                gb.currentCalibrationFramesHeld += 1
                if gb.currentCalibrationFramesHeld < gb.totalCalibrationFrames:
                    gb.initialCalibration = True
                    gb.gui.log(f"Calibration: {gb.currentCalibrationFramesHeld+1} / {gb.totalCalibrationFrames}")
                else:
                    gb.initialCalibration = False
                    gb.calibration_array = self._finalize_calibration()
                    gb.gui.log("Calibration complete.")
                return self.frame,self.overall_tier

            else:
                patch_scores, nearest_slot_per_patch = self.compute_patch_scores(embeddings, gb.calibration_array)
                score_grid = patch_scores.reshape(self.grid_h, self.grid_w)

                raw_tier_grid = self.compute_tier(score_grid, self.zone_grid_high_priority)
                tier_grid, self.persistence_counters = self.apply_persistence_filter(raw_tier_grid, self.persistence_counters, gb.REQUIRED_PERSISTENCE)
                self.overall_tier = max(set(tier_grid.flatten()), key=gb.severity_order.index)

                gb.current_highlight = (tier_grid, self.overall_tier)

                frame_eligible = self.is_frame_eligible(raw_tier_grid, self.zone_grid_high_priority, gb.allowed_error)
                if gb.auto_update_calibration:
                    self.self_fix_calibration(gb.calibration_array, embeddings, nearest_slot_per_patch, frame_eligible)

                ##### Anomaly Reporting #####
                self.highlighted_frame = self.draw_overlay()
                self.handle_anomaly_reporting()
                return self.highlighted_frame,self.overall_tier
                ##############################
        return self.draw_overlay(),self.overall_tier