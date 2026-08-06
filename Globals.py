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

##### Anomaly Reporting Config ##### - Fill in your email details below
anomaly_report_wait = 15          # seconds to wait before re-sending an alert for an ongoing anomaly

EMAIL_SENDER = "woejack42699@gmail.com"
EMAIL_PASSWORD = "rtir pwts riht sdjt"      # use an app password, not your real password
EMAIL_RECEIVER = "muhammad.ali.sher.official@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

LOG_DIR = "logs"                  # folder where offline anomaly logs/images get saved
#####################################

##### Anomaly Reporting State ##### - Don't change these
reporting_active = False
last_report_time = 0
####################################
