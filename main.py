import cv2
import numpy as np
import tkinter as tk
import Accessories as Acc
import Globals as gb
import pygame
from gui import AnomalyDetectionGUI

def main():

    Acc.Model_Setup()
    pygame.mixer.init()
    aud_alert = pygame.mixer.Sound("Assets/Audio/alert.wav")    
    aud_critical = pygame.mixer.Sound("Assets/Audio/critical.wav")
    channel_alert = aud_alert.play(loops=-1)
    channel_alert.stop()
    channel_crit = aud_critical.play(loops=-1)
    channel_crit.stop()
    
    persistence_counters = None
    cap = None
    zone_grid_high_priority = None
    new_mask = None

    ############# Tkinter Setup #############
    root = tk.Tk()
    gb.gui = AnomalyDetectionGUI(root)
    #########################################

    def setup_camera():
        nonlocal cap, persistence_counters, zone_grid_high_priority, new_mask

        # Release previous capture if re-opening
        if cap is not None:
            cap.release()

        ############# Video Capture Setup #############
        if gb.doVideoStream:
            gb.url = "http://10.13.12.117:8080/video"
        else:
            gb.url = 0

        cap = cv2.VideoCapture(gb.url)
        ###############################################

        gb.frameShape[0] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        gb.frameShape[1] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        persistence_counters = np.zeros((gb.grid_h, gb.grid_w), dtype=int)

        #############    Mask Setup    #############
        mask_image = cv2.imread("Assets/mask.png")
        is_active = np.all(mask_image == gb.medium_priority_region, axis=2)
        is_high_priority = np.all(mask_image == gb.high_priority_color, axis=2)
        is_watched = is_active | is_high_priority    

        mask = is_watched.astype(np.float32)
        new_mask = cv2.resize(mask, (gb.frameShape[0], gb.frameShape[1]), interpolation=cv2.INTER_NEAREST)

        high_priority_mask = is_high_priority.astype(np.float32)
        zone_grid_high_priority = cv2.resize(high_priority_mask,(gb.grid_w, gb.grid_h),interpolation=cv2.INTER_NEAREST)
        ############################################

        gb.gui.log("Camera initialized. Starting calibration...")

    setup_camera()

    def update_loop():
        nonlocal persistence_counters,channel_alert,channel_crit

        if not gb.gui.is_running():
            return

        # Check if GUI requested a restart (Apply & Restart was clicked)
        if gb.gui.restart_requested:
            gb.gui.clear_restart_flag()
            setup_camera()
            root.after(50, update_loop)
            return

        if cap is None or not cap.isOpened():
            gb.gui.log("Camera not available. Retrying...")
            root.after(1000, update_loop)
            return

        frameRead, frame = cap.read()
        if not frameRead:
            if gb.doVideoStream:
                root.after(10, update_loop)
                return
            else:
                gb.gui.log("End of video stream.")
                return

        ##########     frame processing     ##########
        frame = cv2.flip(frame, 1)
        Ignored_zone_done_mask = frame.copy()
        Ignored_zone_done_mask[new_mask == 0] = 0
        final_frame = cv2.cvtColor(Ignored_zone_done_mask, cv2.COLOR_BGR2RGB)
        final_frame = cv2.resize(final_frame,(224,224),interpolation=cv2.INTER_AREA)

        if Acc.FrameEligiblebyTime(gb.secondsForOneFrame):
            embeddings = Acc.get_patch_embeddings(final_frame)
            if gb.initialCalibration:
                gb.calibration_store.append(embeddings)
                gb.currentCalibrationFramesHeld += 1
                if gb.currentCalibrationFramesHeld < gb.totalCalibrationFrames:
                    gb.initialCalibration = True
                    gb.gui.log(f"Calibration: {gb.currentCalibrationFramesHeld+1} / {gb.totalCalibrationFrames}")
                else:
                    gb.initialCalibration = False
                    gb.calibration_array = Acc.finalize_calibration(gb.calibration_store)
                    gb.gui.log("Calibration complete.")

            else:
                patch_scores, nearest_slot_per_patch = Acc.compute_patch_scores(embeddings, gb.calibration_array)
                score_grid = patch_scores.reshape(gb.grid_h, gb.grid_w)

                raw_tier_grid = Acc.compute_tier(score_grid, zone_grid_high_priority)
                tier_grid, persistence_counters = Acc.apply_persistence_filter(raw_tier_grid, persistence_counters, gb.REQUIRED_PERSISTENCE)
                overall_tier = max(set(tier_grid.flatten()), key=gb.severity_order.index)

                gb.current_highlight = (tier_grid, overall_tier)

                frame_eligible = Acc.is_frame_eligible(raw_tier_grid, zone_grid_high_priority, gb.allowed_error)
                if gb.auto_update_calibration:
                    Acc.self_fix_calibration(gb.calibration_array, embeddings, nearest_slot_per_patch, frame_eligible)

                if overall_tier == "ALERT":
                    if channel_crit != None and channel_crit.get_busy():
                        channel_crit.stop()
                    if not (channel_alert != None and channel_alert.get_busy()):
                        channel_alert = aud_alert.play(loops=-1)

                elif overall_tier == "CRITICAL" and not channel_crit.get_busy():                    
                    if channel_alert != None and channel_alert.get_busy():
                        channel_alert.stop()
                    if not (channel_crit != None and channel_crit.get_busy()):
                        channel_crit = aud_critical.play(loops=-1)
                else:
                    if channel_alert != None and channel_alert.get_busy():
                        channel_alert.stop()
                    if channel_crit != None and channel_crit.get_busy():
                        channel_crit.stop()                       

                ##### Anomaly Reporting #####
                highlighted_frame = Acc.draw_overlay(frame, gb.current_highlight)
                Acc.handle_anomaly_reporting(overall_tier, highlighted_frame)
                ##############################

                gb.gui.update_status(overall_tier, overall_tier)

        processed_frame = Acc.draw_overlay(frame, gb.current_highlight)
        ##############################################
        gb.gui.update_frame(processed_frame)

        # Schedule the next frame; delay controls responsiveness
        root.after(gb.delay, update_loop)

    # Start the update loop
    root.after(100, update_loop)

    # Bind ESC to close
    root.bind("<Escape>", lambda e: gb.gui.on_closing())

    root.mainloop()

    if cap is not None:
        cap.release()

if __name__ == "__main__":
    main()
