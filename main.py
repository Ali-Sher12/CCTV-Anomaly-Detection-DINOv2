import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModel
import torch
import Accessories as Acc

def get_patch_embeddings(frame):
    inputs = processor(images=frame, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    patch_embeddings = outputs.last_hidden_state[0, 1:, :]  # drop CLS token
    return patch_embeddings.numpy()


def compute_patch_scores(new_embeddings, calibration_store):
    """
    For each patch position, computes the distance from the new frame's
    embedding at that position to the closest calibration embedding at
    that same position (nearest-neighbor distance).
    """
    reference = np.stack(calibration_store, axis=0)          # (N, num_patches, dim)
    diffs = reference - new_embeddings[np.newaxis, :, :]      # (N, num_patches, dim)
    distances = np.linalg.norm(diffs, axis=2)                 # (N, num_patches)
    patch_scores = np.min(distances, axis=0)                  # (num_patches,)
    return patch_scores


##### Tier thresholds (tune these once you see real score ranges) #####
TIER_THRESHOLDS = {
    "LOG":      15,
    "ALERT":    40,
    "CRITICAL": 60,
}
##########################################################################

def compute_tier(score_grid, thresholds=TIER_THRESHOLDS):
    """
    Converts a per-patch score grid into tier labels.
    Returns: tier_grid - (grid_h, grid_w) array of strings
    """
    tier_grid = np.full(score_grid.shape, "NORMAL", dtype=object)
    tier_grid[score_grid >= thresholds["LOG"]] = "LOG"
    tier_grid[score_grid >= thresholds["ALERT"]] = "ALERT"
    tier_grid[score_grid >= thresholds["CRITICAL"]] = "CRITICAL"
    return tier_grid


def apply_persistence_filter(tier_grid, counters, required):
    """
    Suppresses tiers that haven't been anomalous for enough consecutive
    cycles yet.
    """
    is_anomalous = tier_grid != "NORMAL"
    counters = np.where(is_anomalous, counters + 1, 0)
    confirmed_tier_grid = np.where(counters >= required, tier_grid, "NORMAL")
    return confirmed_tier_grid, counters


TIER_COLORS = {
    "LOG":      (0, 255, 255),   # yellow (BGR)
    "ALERT":    (0, 165, 255),   # orange
    "CRITICAL": (0, 0, 255),     # red
}

def draw_overlay(frame, current_highlight, alpha=0.45):
    """
    Draws the tier overlay on top of a frame, based on the last
    computed inference result. Persists until a new result replaces it.
    """
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
totalCalibrationFrames = 10
secondsForOneFrame = 1
delay = 13
doVideoStream = True
ArduinoMode = False
url = "Assets/sample.mp4"
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
                # Perform inference. Replacement logic (score-close-frame
                # swap) deliberately deferred — not implemented yet.
                patch_scores = compute_patch_scores(embeddings, calibration_store)
                score_grid = patch_scores.reshape(grid_h, grid_w)

                tier_grid = compute_tier(score_grid)
                tier_grid, persistence_counters = apply_persistence_filter(
                    tier_grid, persistence_counters, REQUIRED_PERSISTENCE
                )

                severity_order = ["NORMAL", "LOG", "ALERT", "CRITICAL"]
                overall_tier = max(set(tier_grid.flatten()), key=severity_order.index)

                current_highlight = (tier_grid, overall_tier)
                print(f"score range: min={score_grid.min():.4f}, max={score_grid.max():.4f}")
        # Runs every frame regardless of eligibility: draws the last
        # computed result, or a clean frame if calibration isn't done yet.
        processed_frame = draw_overlay(frame, current_highlight)
        ##############################################

        cv2.imshow("Processed Stream", processed_frame)
        if cv2.waitKey(delay) == 27:
            break

cap.release()
cv2.destroyAllWindows()