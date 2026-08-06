#########   Globals    #########
totalCalibrationFrames = 20
secondsForOneFrame = 0.5
delay = 1
doVideoStream = False
url = 0
NN_MODE = True
REQUIRED_PERSISTENCE = 1
allowed_error = 5
################################

##### Tier thresholds & colors ##### - You can change/tune/tinker these as well
TIER_THRESHOLDS = {
    "ALERT":    59,
    "CRITICAL": 60,
}
TIER_THRESHOLDS_HIGH_PRIORITY = {
    "ALERT":    45,
    "CRITICAL": 50,
}
TIER_COLORS = {
    "ALERT":    (0, 165, 255),   # orange
    "CRITICAL": (0, 0, 255),     # red
}
severity_order = ["NORMAL", "ALERT", "CRITICAL"]
####################################

##### Don't change these #####
embeddings = None
persistence_counters = None
current_highlight = None
calibration_store = []
medium_priority_region = (0, 255, 0)
high_priority_color = (0, 0, 255)
dead_zone_color = (0, 0, 0)
frameShape = [-1, -1]
calibration_frames = []
frame_scores = []
calibration_array = None
##############################

##### Single use variables #####
initialCalibration = True
currentCalibrationFramesHeld = 0
################################

##### Model and Processor #####
processor = None
model = None
grid_h = 0
grid_w = 0
###############################
