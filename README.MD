This project is currently in its infancy, more features will be added over time.

## CCTV Anomaly Detector

A small CCTV tool that watches your gate or driveway and tells you when something looks different from normal. It does not know what a "car" or a "person" is.
It just learns what your camera view usually looks like, and flags any part of the frame that suddenly looks new.

## How it works

The camera feed is broken into a grid of small patches. A vision model called DINOv2 turns each patch into a set of numbers (an embedding).
During the first few seconds after you start the app, it collects these numbers for many frames and treats that as "normal."
After that, every new frame gets compared patch by patch against the normal set. If a patch looks very different, it gets marked as an anomaly.
Depending on how different it is, and whether it falls inside a zone you marked as high priority, it gets a tier:

- **NORMAL** – nothing unusual
- **ALERT** – something changed, worth a look
- **CRITICAL** – a strong change, likely something is there

When the tier goes to ALERT or CRITICAL, the app plays a sound, highlights the area on screen, and can send you an email with a photo.
If there is no internet, it saves the alert and photo to a local log folder instead (logging is done regardless of internet avilability).

The system also slowly updates its idea of "normal" using frames that look clean, so small lighting changes over the day do not keep triggering false alerts.

## Requirements

pip install opencv-python numpy Pillow torch transformers pygame-ce
```
Use python 3.14.

```

Run.bat opens the control window and starts the camera. The first thing it does is calibrate, meaning it watches your empty driveway or gate for a number of frames to learn what "normal" looks like.
Do not walk in front of the camera during this part. You will see a "Calibration: x / y" counter in the window while this happens.
Once calibration is done, the app starts checking every new frame for anomalies.

Press **Esc** at any time to close the app.

## Using the mask editor

Not every part of the camera view matters equally. Maybe your gate area matters a lot, but a tree branch that sways in the wind should not count. That is what the mask is for.

Click **"Modify Mask..."** in the GUI to open the paint tool. You get three colors:

- **Green** – medium priority zone, watched normally
- **Red** – high priority zone, any anomaly here is treated seriously
- **Black** – dead zone, completely ignored

Use the brush, fill bucket, circle, or rectangle tool to paint over your camera view. Save when done, then click **"Apply & Restart"** in the main window to load the new mask. This also restarts the camera and re-runs calibration.

## Adjusting settings while it runs

Most sliders in the GUI apply right away, no restart needed:

- **Frame Interval** – how often (in seconds) a frame actually gets analyzed
- **Display Delay** – how fast the video preview refreshes
- **Persistence Filter** – how many frames in a row must be anomalous before it counts as a real alert (helps avoid one weird frame causing a false alarm)
- **Alert Tolerance** – how many odd patches outside high priority zones are allowed before triggering an alert
- **Report Cooldown** – minimum time between repeated alert emails/logs
- **Auto-Update Calibration** – lets the system slowly adjust its idea of normal using clean frames
- **Alert / Critical thresholds** – how different a patch needs to be to count as ALERT or CRITICAL, with separate (usually stricter) values for high priority zones

Settings under **"Settings (Require Restart)"**, like switching to a video stream or changing the number of calibration frames, only apply once you press **Apply & Restart**.

## A few tips

- Calibrate with the scene empty and in normal lighting. If you calibrate at night and then run it in daylight, you will get a lot of false alerts.
- If you get too many false alerts, try raising the thresholds a bit or increasing the Persistence Filter.
- The high priority zone (red) is meant for the actual gate or entry point. Keep the rest of the frame as medium priority (green) or dead zone (black) if it is not useful, like the sky or a busy street in the background.
- Local logs and saved anomaly photos live in the `logs` folder, so check there if email alerts are not reaching you.