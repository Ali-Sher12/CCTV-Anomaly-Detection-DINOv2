#########   Globals    #########
totalCalibrationFrames = 800
secondsForOneFrame = 0.5
delay = 1
doVideoStream = False
url = 0
NN_MODE = False
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

# Mahalanobis distances live on a different scale than Euclidean ones — a "normal"
# score tends to sit near sqrt(embedding_dim) (~19-20 for a ~384-dim embedding).
# These are ROUGH starting points only. After your first calibration run, check the
# printed "typical normal score" diagnostic and adjust these to match your real data.
TIER_THRESHOLDS_MAHALANOBIS = {
    "ALERT":    25,
    "CRITICAL": 32,
}
TIER_THRESHOLDS_HIGH_PRIORITY_MAHALANOBIS = {
    "ALERT":    20,
    "CRITICAL": 26,
}

TIER_COLORS = {
    "ALERT":    (0, 165, 255),   # orange
    "CRITICAL": (0, 0, 255),     # red
}
severity_order = ["NORMAL", "ALERT", "CRITICAL"]
####################################

##### Mahalanobis-specific config #####
MAHALANOBIS_EPSILON = 1e-3   # regularization added to covariance diagonal before inversion
MAHALANOBIS_ALPHA = 0.03     # self-fix mean update rate (slow-to-moderate)
########################################

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
calibration_array = None          # used in NN_MODE
calibration_mean = None           # used in Mahalanobis mode, shape: (num_patches, embedding_dim)
calibration_precision = None      # used in Mahalanobis mode, shape: (num_patches, embedding_dim, embedding_dim)
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

##### Anomaly Reporting Config ##### - Fill in your email details below
anomaly_report_wait = 30          # seconds to wait before re-sending an alert for an ongoing anomaly

EMAIL_SENDER = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"      # use an app password, not your real password
EMAIL_RECEIVER = "receiver_email@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

LOG_DIR = "logs"                  # folder where offline anomaly logs/images get saved
#####################################

##### Anomaly Reporting State ##### - Don't change these
reporting_active = False
last_report_time = 0
####################################
