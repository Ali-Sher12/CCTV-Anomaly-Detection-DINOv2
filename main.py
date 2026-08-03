import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModel
import torch
import Accessories as Acc

##### Tier thresholds & colors #####
TIER_THRESHOLDS = {
    "ALERT":    59,
    "CRITICAL": 60,
}
TIER_COLORS = {
    "ALERT":    (0, 165, 255),   # orange
    "CRITICAL": (0, 0, 255),     # red
}
severity_order = ["NORMAL", "ALERT", "CRITICAL"]
####################################


def get_patch_embeddings(frame):
    inputs = processor(images=frame, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, interpolate_pos_encoding=True)
    patch_embeddings = outputs.last_hidden_state[0, 1:, :]
    return patch_embeddings.numpy()


def compute_patch_scores(new_embeddings, calibration_store):
    reference = np.stack(calibration_store, axis=0)
    diffs = reference - new_embeddings[np.newaxis, :, :]
    distances = np.linalg.norm(diffs, axis=2)
    patch_scores = np.min(distances, axis=0)
    return patch_scores


def compute_tier(score_grid):
    tier_grid = np.full(score_grid.shape, "NORMAL", dtype=object)
    tier_grid[score_grid >= TIER_THRESHOLDS["ALERT"]] = "ALERT"
    tier_grid[score_grid >= TIER_THRESHOLDS["CRITICAL"]] = "CRITICAL"
    return tier_grid


def apply_persistence_filter(tier_grid, counters, required):
    is_anomalous = tier_grid != "NORMAL"
    counters = np.where(is_anomalous, counters + 1, 0)
    confirmed_tier_grid = np.where(counters >= required, tier_grid, "NORMAL")
    return confirmed_tier_grid, counters


def draw_overlay(frame, current_highlight, alpha=0.45):
    if current_highlight is None:
        return frame.copy()

    tier_grid, overall_tier = current_highlight
    grid_h, grid_w = tier_grid.shape
    frame_h, frame_w = frame.shape[:2]

    color_grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    highlight_mask = np.zeros((grid_h, grid_w), dtype=np.uint8)

    for tier, color in TIER_COLORS.items():
        matches = tier_grid == tier
        color_grid[matches] = color
        highlight_mask[matches] = 1

    color_full = cv2.resize(color_grid, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
    mask_full = cv2.resize(highlight_mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)

    blended_full = cv2.addWeighted(frame, 1 - alpha, color_full, alpha, 0)

    result = frame.copy()
    result[mask_full == 1] = blended_full[mask_full == 1]

    label_color = TIER_COLORS.get(overall_tier, (255, 255, 255))
    cv2.putText(result, overall_tier, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)

    return result


#########   Globals    #########
totalCalibrationFrames = 150
secondsForOneFrame = 0.5
delay = 1
doVideoStream = False
url = 0
NN_MODE = True
REPLACE_HIGHEST_SCORE = True
################################

##### Don't change these #####
embeddings = None
REQUIRED_PERSISTENCE = 1
persistence_counters = None
current_highlight = None
calibration_store = []
foreground_color = (0, 255, 0)
dead_zone_color = (0, 0, 0)
frameShape = [-1, -1]
calibration_frames = []
frame_scores = []
##############################

##### Single use variables #####
initialCalibration = True
currentCalibrationFramesHeld = 0
################################

#############   Model Setup    #############
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
#processor.crop_size = {"height": 518, "width": 518}
#processor.size = {"shortest_edge": 518}
model = AutoModel.from_pretrained("facebook/dinov2-small")
model.eval()

_input_size = processor.crop_size["height"]
_patch_size = model.config.patch_size
grid_h = grid_w = _input_size // _patch_size
############################################

if __name__ == "__main__":

    if not NN_MODE:
        calibration_store = None
        calibration_store = {}

    persistence_counters = np.zeros((grid_h, grid_w), dtype=int)

    ############# Video Capture Setup #############
    if doVideoStream:
        url = "http://192.168.18.98:8080/video"

    cap = cv2.VideoCapture(url)
    ###############################################

    frameShape[0] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameShape[1] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #############    Mask Setup    #############
    mask_image = cv2.imread("Assets/mask.png")
    is_active = np.all(mask_image == foreground_color, axis=2)
    mask = is_active.astype(np.float32)
    new_mask = cv2.resize(mask, (frameShape[0], frameShape[1]), interpolation=cv2.INTER_NEAREST)
    ############################################

    while True:
#        Acc.printSeconds()
        frameRead, frame = cap.read()
        if not frameRead:
            if doVideoStream:
                continue
            else:
                break

        ##########     frame processing     ##########
        frame = cv2.flip(frame, 1)
        Ignored_zone_done_mask = frame.copy()
        Ignored_zone_done_mask[new_mask == 0] = 0
        final_frame = cv2.cvtColor(Ignored_zone_done_mask, cv2.COLOR_BGR2RGB)

        if Acc.FrameEligiblebyTime(secondsForOneFrame):
            embeddings = get_patch_embeddings(final_frame)
            if initialCalibration:
                calibration_store.append(embeddings)
                calibration_frames.append(final_frame)
                currentCalibrationFramesHeld += 1
                initialCalibration = True if currentCalibrationFramesHeld < totalCalibrationFrames else False

            else:
                patch_scores = compute_patch_scores(embeddings, calibration_store)
                score_grid = patch_scores.reshape(grid_h, grid_w)

                tier_grid = compute_tier(score_grid)
                tier_grid, persistence_counters = apply_persistence_filter(tier_grid, persistence_counters, REQUIRED_PERSISTENCE)
                overall_tier = max(set(tier_grid.flatten()), key=severity_order.index)

                current_highlight = (tier_grid, overall_tier)
#                print(f"score range: min={score_grid.min():.4f}, max={score_grid.max():.4f}")
        processed_frame = draw_overlay(frame, current_highlight)

        ##############################################

        cv2.imshow("Processed Stream", processed_frame)
        if cv2.waitKey(delay) == 27:
            break

cap.release()
cv2.destroyAllWindows()