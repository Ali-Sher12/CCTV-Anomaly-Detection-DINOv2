#########   Globals- can be toggled at runtime    #########
secondsForOneFrame = 0.2
delay = 3
REQUIRED_PERSISTENCE = 1
allowed_error = 1
anomaly_report_wait = 15
TIER_THRESHOLDS = {
    "ALERT":    45,
    "CRITICAL": 50,
}
TIER_THRESHOLDS_HIGH_PRIORITY = {
    "ALERT":    35,
    "CRITICAL": 40,
}
auto_update_calibration = False
####################################

##### Globals - thse can be toggled but effects take place on restart. (load/writing to file) #####
totalCalibrationFrames = 50
doVideoStream = False
url = 0
DINO_ONLY = True #Hybrid if false

EMAIL_SENDER = "woejack42699@gmail.com"
EMAIL_PASSWORD = "rtir pwts riht sdjt"      # use an app password, not your real password
EMAIL_RECEIVER = "muhammad.ali.sher.official@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
LOG_DIR = "logs"                  # folder where offline anomaly logs/images get saved
####################################

##### Don't change these #####
TIER_COLORS = {
    "ALERT":    (0, 165, 255),   # orange
    "CRITICAL": (0, 0, 255),     # red
}
severity_order = ["NORMAL", "ALERT", "CRITICAL"]
embeddings = None

current_highlight = None
calibration_store = []
medium_priority_region = (0, 255, 0)
high_priority_color = (0, 0, 255)
dead_zone_color = (0, 0, 0)
frameShape = [-1, -1]
calibration_frames = []
frame_scores = []
calibration_array = None
reporting_active = False
last_report_time = 0

##############################

##### Single use variables - dont change these #####
initialCalibration = True
currentCalibrationFramesHeld = 0
################################
gui = None

