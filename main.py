import cv2
import numpy as np
import Accessories as Acc
import Globals as gb

if __name__ == "__main__":

    Acc.Model_Setup()

    persistence_counters = np.zeros((gb.grid_h, gb.grid_w), dtype=int)

    ############# Video Capture Setup #############
    if gb.doVideoStream:
        gb.url = "http://192.168.18.98:8080/video"

    cap = cv2.VideoCapture(gb.url)
    ###############################################

    gb.frameShape[0] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    gb.frameShape[1] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

    # Pick which threshold set applies once, rather than branching every frame
    if gb.NN_MODE:
        active_thresholds = gb.TIER_THRESHOLDS
        active_thresholds_high_priority = gb.TIER_THRESHOLDS_HIGH_PRIORITY
    else:
        active_thresholds = gb.TIER_THRESHOLDS_MAHALANOBIS
        active_thresholds_high_priority = gb.TIER_THRESHOLDS_HIGH_PRIORITY_MAHALANOBIS

    while True:
#        Acc.printSeconds()
        frameRead, frame = cap.read()
        if not frameRead:
            if gb.doVideoStream:
                continue
            else:
                break

        ##########     frame processing     ##########
        frame = cv2.flip(frame, 1)
        Ignored_zone_done_mask = frame.copy()
        Ignored_zone_done_mask[new_mask == 0] = 0
        final_frame = cv2.cvtColor(Ignored_zone_done_mask, cv2.COLOR_BGR2RGB)

        if Acc.FrameEligiblebyTime(gb.secondsForOneFrame):
            embeddings = Acc.get_patch_embeddings(final_frame)
            if gb.initialCalibration:
                gb.calibration_store.append(embeddings)
                gb.calibration_frames.append(final_frame)
                gb.currentCalibrationFramesHeld += 1
                if gb.currentCalibrationFramesHeld < gb.totalCalibrationFrames:
                    gb.initialCalibration = True
                    print("Calibration in Progress : ",gb.currentCalibrationFramesHeld+1," / ",gb.totalCalibrationFrames,end="\r")
                else:
                    gb.initialCalibration = False
                    print("Calibration Status : Complete............ALL OK")
                    if gb.NN_MODE:
                        gb.calibration_array = Acc.finalize_calibration(gb.calibration_store)
                    else:
                        gb.calibration_mean, gb.calibration_precision = Acc.finalize_calibration_mahalanobis(gb.calibration_store)

            else:
                if gb.NN_MODE:
                    patch_scores, nearest_slot_per_patch = Acc.compute_patch_scores(embeddings, gb.calibration_array)
                else:
                    patch_scores = Acc.compute_patch_scores_mahalanobis(embeddings, gb.calibration_mean, gb.calibration_precision)

                score_grid = patch_scores.reshape(gb.grid_h, gb.grid_w)

                raw_tier_grid = Acc.compute_tier(score_grid, zone_grid_high_priority, active_thresholds, active_thresholds_high_priority)
                tier_grid, persistence_counters = Acc.apply_persistence_filter(raw_tier_grid, persistence_counters, gb.REQUIRED_PERSISTENCE)
                overall_tier = max(set(tier_grid.flatten()), key=gb.severity_order.index)

                gb.current_highlight = (tier_grid, overall_tier)

                frame_eligible = Acc.is_frame_eligible(raw_tier_grid, zone_grid_high_priority, gb.allowed_error)

                if gb.NN_MODE:
                    Acc.self_fix_calibration(gb.calibration_array, embeddings, nearest_slot_per_patch, frame_eligible)
                else:
                    Acc.self_fix_calibration_mahalanobis(gb.calibration_mean, embeddings, frame_eligible, gb.MAHALANOBIS_ALPHA)

                ##### Anomaly Reporting #####
                highlighted_frame = Acc.draw_overlay(frame, gb.current_highlight)
                Acc.handle_anomaly_reporting(overall_tier, highlighted_frame)
                ##############################

        processed_frame = Acc.draw_overlay(frame, gb.current_highlight)

        ##############################################

        cv2.imshow("Processed Stream", processed_frame)
        if cv2.waitKey(gb.delay) == 27:
            break

cap.release()
cv2.destroyAllWindows()
