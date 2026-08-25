import tkinter as tk
import time
import os
import Globals as gb
from PIL import Image, ImageTk
from mask_editor import MaskEditor
from settings_store import load_settings, save_settings
import cv2


# ─── Load persisted settings into Globals on import ──────────────
_saved = load_settings() or {}
_SAVED_STREAM_URL = _saved.get("stream_url", "http://192.168.18.98:8080/video")


class AnomalyDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Passive Anomaly Detector PLUS +")
        self.root.configure(bg='#c0c0c0')
        self.root.resizable(False, False)

        self._running = True
        self._restart_requested = False
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Staged restart values — GUI-only until "Apply & Restart" is clicked
        self._staged_video_stream = gb.doVideoStream
        self._staged_calib_frames = gb.totalCalibrationFrames
        self._staged_url = _SAVED_STREAM_URL
        self._staged_dino_only = gb.DINO_ONLY
        self._mask_modified = False

        # Fonts
        self.font_classic = ('MS Sans Serif', 8)
        self.font_title = ('MS Sans Serif', 10, 'bold')
        self.font_status = ('Fixedsys', 10)
        self.font_restart_hint = ('MS Sans Serif', 8, 'bold')

        # Build UI
        self._build_ui()
        self._init_controls()

        # Set window icon if logo exists
        self._set_app_icon()

    def _set_app_icon(self):
        """Set the window icon from Assets/logo.png."""
        logo_path = "Assets/logo.png"
        if os.path.exists(logo_path):
            try:
                self._icon_img = ImageTk.PhotoImage(file=logo_path)
                self.root.iconphoto(False, self._icon_img)
            except Exception:
                pass

    # ─── UI Construction ──────────────────────────────────────────

    def _build_ui(self):
        # Top banner (navy blue accent) with optional icon
        banner_frame = tk.Frame(self.root, bg='#000080')
        banner_frame.pack(fill=tk.X, pady=(0, 5))

        logo_path = "Assets/logo.png"
        self._banner_icon = None
        if os.path.exists(logo_path):
            try:
                b_img = Image.open(logo_path)
                b_img.thumbnail((24, 24), Image.Resampling.LANCZOS)
                self._banner_icon = ImageTk.PhotoImage(b_img)
                tk.Label(banner_frame, image=self._banner_icon, bg='#000080').pack(side=tk.LEFT, padx=(8, 2), pady=2)
            except Exception:
                pass

        banner = tk.Label(banner_frame, text="Passive Anomaly Detector PLUS +",
                          bg='#000080', fg='white', font=self.font_title, anchor='w')
        banner.pack(side=tk.LEFT, padx=(2, 10), pady=3)

        # Main layout
        main_frame = tk.Frame(self.root, bg='#c0c0c0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left Panel — Camera Feed + Terminal
        left_panel = tk.Frame(main_frame, bg='#c0c0c0')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        feed_frame = tk.Frame(left_panel, bg='#c0c0c0', relief=tk.SUNKEN, bd=2)
        feed_frame.pack(fill=tk.X, padx=0, pady=(0, 4))

        self.canvas = tk.Canvas(feed_frame, width=640, height=480, bg='black', highlightthickness=0)
        self.canvas.pack(padx=5, pady=5)

        # Terminal log area below the camera feed
        term_frame = tk.LabelFrame(left_panel, text="Log", bg='#c0c0c0',
                                   font=self.font_classic, relief=tk.GROOVE, bd=2)
        term_frame.pack(fill=tk.BOTH, expand=True)

        self.terminal = tk.Text(term_frame, height=6, bg='black', fg='#00ff00',
                                font=('Courier New', 9), state=tk.DISABLED,
                                wrap=tk.WORD, bd=0, highlightthickness=0)
        term_scroll = tk.Scrollbar(term_frame, command=self.terminal.yview)
        self.terminal.config(yscrollcommand=term_scroll.set)
        term_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.terminal.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Right Panel — Controls
        right_panel = tk.Frame(main_frame, bg='#c0c0c0')
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))

        self._build_status_group(right_panel)
        self._build_runtime_group(right_panel)
        self._build_threshold_group(right_panel)
        self._build_restart_group(right_panel)

        # Status Bar
        self.statusbar = tk.Label(self.root, text="\u25b8 Ready", bd=1, relief=tk.SUNKEN,
                                  anchor='w', bg='#c0c0c0', font=self.font_status)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.photo = None

    def _build_status_group(self, parent):
        grp = tk.LabelFrame(parent, text="System Status", bg='#c0c0c0',
                            font=self.font_classic, relief=tk.GROOVE, bd=2)
        grp.pack(fill=tk.X, pady=(0, 8), ipadx=5, ipady=3)

        self.lbl_status = tk.Label(grp, text="Status: NORMAL", bg='#c0c0c0',
                                   font=self.font_title, fg='green')
        self.lbl_status.pack(anchor='w')

        self.lbl_calibration = tk.Label(grp, text="Calibration: Pending", bg='#c0c0c0',
                                        font=self.font_classic)
        self.lbl_calibration.pack(anchor='w')

    def _build_runtime_group(self, parent):
        grp = tk.LabelFrame(parent, text="Runtime Controls", bg='#c0c0c0',
                            font=self.font_classic, relief=tk.GROOVE, bd=2)
        grp.pack(fill=tk.X, pady=(0, 8), ipadx=5, ipady=3)

        self.scale_interval = self._create_slider(grp, "Frame Interval (s)", 0.1, 5.0, 0.1,
                                                  self._on_interval)
        self.scale_delay = self._create_slider(grp, "Display Delay (ms)", 1, 100, 1,
                                               self._on_delay)
        self.scale_persistence = self._create_slider(grp, "Persistence Filter", 1, 10, 1,
                                                     self._on_persistence)
        self.scale_error = self._create_slider(grp, "Alert Tolerance", 1, 20, 1,
                                               self._on_error)
        self.scale_cooldown = self._create_slider(grp, "Report Cooldown (s)", 5, 120, 1,
                                                  self._on_cooldown)

        self.var_auto_update = tk.BooleanVar(value=gb.auto_update_calibration)
        self.chk_auto_update = tk.Checkbutton(grp, text="Auto-Update Calibration",
                                              variable=self.var_auto_update,
                                              bg='#c0c0c0', font=self.font_classic,
                                              command=self._on_auto_update)
        self.chk_auto_update.pack(anchor='w', pady=(4, 0))

    def _build_threshold_group(self, parent):
        grp = tk.LabelFrame(parent, text="Thresholds", bg='#c0c0c0',
                            font=self.font_classic, relief=tk.GROOVE, bd=2)
        grp.pack(fill=tk.X, pady=(0, 8), ipadx=5, ipady=3)

        self.scale_alert = self._create_slider(grp, "Alert", 0, 200, 1, self._on_alert)
        self.scale_crit = self._create_slider(grp, "Critical", 0, 200, 1, self._on_crit)

        tk.Label(grp, text="\u2014 High-Priority Zones \u2014", bg='#c0c0c0',
                 font=self.font_classic).pack(pady=(6, 2))

        self.scale_hp_alert = self._create_slider(grp, "HP Alert", 0, 200, 1, self._on_hp_alert)
        self.scale_hp_crit = self._create_slider(grp, "HP Critical", 0, 200, 1, self._on_hp_crit)

    def _build_restart_group(self, parent):
        grp = tk.LabelFrame(parent, text="Settings (Require Restart)", bg='#c0c0c0',
                            font=self.font_classic, relief=tk.GROOVE, bd=2)
        grp.pack(fill=tk.X, ipadx=5, ipady=3)

        # DINO Only checkbox
        self.var_dino_only = tk.BooleanVar()
        self.chk_dino_only = tk.Checkbutton(grp, text="DINO Only (no Hybrid)",
                                            variable=self.var_dino_only,
                                            bg='#c0c0c0', font=self.font_classic,
                                            command=self._on_dino_only_staged)
        self.chk_dino_only.pack(anchor='w')

        # Video Stream checkbox
        self.var_video = tk.BooleanVar()
        self.chk_video = tk.Checkbutton(grp, text="Video Stream", variable=self.var_video,
                                        bg='#c0c0c0', font=self.font_classic,
                                        command=self._on_video_staged)
        self.chk_video.pack(anchor='w')

        # URL entry — only visible when Video Stream is checked
        self.url_row = tk.Frame(grp, bg='#c0c0c0')
        # not packed yet — shown/hidden by _on_video_staged
        tk.Label(self.url_row, text="URL:", bg='#c0c0c0',
                 font=self.font_classic).pack(side=tk.LEFT)
        self.entry_url = tk.Entry(self.url_row, width=22, font=self.font_classic)
        self.entry_url.pack(side=tk.LEFT, padx=3)
        self.entry_url.insert(0, self._staged_url)
        self.entry_url.bind("<KeyRelease>", self._on_url_staged)

        # Calibration frames entry
        calib_row = tk.Frame(grp, bg='#c0c0c0')
        calib_row.pack(fill=tk.X, pady=3)
        tk.Label(calib_row, text="Calibration Frames:", bg='#c0c0c0',
                 font=self.font_classic).pack(side=tk.LEFT)
        self.entry_calib = tk.Entry(calib_row, width=5, font=self.font_classic)
        self.entry_calib.pack(side=tk.LEFT, padx=5)
        self.entry_calib.bind("<KeyRelease>", self._on_calib_frames_staged)

        # Email Settings button — opens a separate window
        tk.Button(grp, text="Email Settings...", command=self._open_email_settings,
                  relief=tk.RAISED, bd=2, bg='#c0c0c0',
                  font=self.font_classic).pack(pady=3, fill=tk.X)

        # Modify Mask button
        tk.Button(grp, text="Modify Mask...", command=self._open_mask_editor,
                  relief=tk.RAISED, bd=2, bg='#c0c0c0',
                  font=self.font_classic).pack(pady=3, fill=tk.X)

        # "Restart Required" label — hidden until a staged value diverges
        self.lbl_restart_hint = tk.Label(grp, text="Restart required", bg='#c0c0c0',
                                         fg='#006400', font=self.font_restart_hint)
        # not packed yet

        # Apply & Restart button
        self.btn_apply_restart = tk.Button(grp, text="Apply && Restart",
                                           command=self._apply_and_restart,
                                           relief=tk.RAISED, bd=2, bg='#c0c0c0',
                                           font=self.font_classic, state=tk.DISABLED)
        self.btn_apply_restart.pack(pady=5, fill=tk.X)

    # ─── Widget helpers ───────────────────────────────────────────

    def _create_slider(self, parent, label_text, from_, to, resolution, command):
        frame = tk.Frame(parent, bg='#c0c0c0')
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text=label_text, bg='#c0c0c0', font=self.font_classic,
                 width=18, anchor='w').pack(side=tk.LEFT)
        scale = tk.Scale(frame, from_=from_, to=to, resolution=resolution,
                         orient=tk.HORIZONTAL, bg='#c0c0c0', font=self.font_classic,
                         command=command, length=120)
        scale.pack(side=tk.RIGHT)
        return scale

    # ─── Initialise controls from current globals ─────────────────

    def _init_controls(self):
        # Runtime sliders
        self.scale_interval.set(gb.secondsForOneFrame)
        self.scale_delay.set(gb.delay)
        self.scale_persistence.set(gb.REQUIRED_PERSISTENCE)
        self.scale_error.set(gb.allowed_error)
        self.scale_cooldown.set(gb.anomaly_report_wait)

        # Threshold sliders
        self.scale_alert.set(gb.TIER_THRESHOLDS["ALERT"])
        self.scale_crit.set(gb.TIER_THRESHOLDS["CRITICAL"])
        self.scale_hp_alert.set(gb.TIER_THRESHOLDS_HIGH_PRIORITY["ALERT"])
        self.scale_hp_crit.set(gb.TIER_THRESHOLDS_HIGH_PRIORITY["CRITICAL"])

        # Restart-required controls — set to current *active* values
        self._staged_video_stream = gb.doVideoStream
        self._staged_calib_frames = gb.totalCalibrationFrames
        self._staged_dino_only = gb.DINO_ONLY
        self._mask_modified = False

        self.var_video.set(gb.doVideoStream)
        self.var_dino_only.set(gb.DINO_ONLY)
        self._toggle_url_visibility()
        self.entry_calib.delete(0, tk.END)
        self.entry_calib.insert(0, str(gb.totalCalibrationFrames))

    # ─── Runtime slider callbacks (write immediately) ─────────────

    def _on_interval(self, val):
        gb.secondsForOneFrame = float(val)

    def _on_delay(self, val):
        gb.delay = int(float(val))

    def _on_persistence(self, val):
        gb.REQUIRED_PERSISTENCE = int(float(val))

    def _on_error(self, val):
        gb.allowed_error = int(float(val))

    def _on_cooldown(self, val):
        gb.anomaly_report_wait = int(float(val))

    def _on_auto_update(self):
        gb.auto_update_calibration = self.var_auto_update.get()

    # ─── Threshold slider callbacks (write immediately) ───────────

    def _on_alert(self, val):
        gb.TIER_THRESHOLDS["ALERT"] = float(val)

    def _on_crit(self, val):
        gb.TIER_THRESHOLDS["CRITICAL"] = float(val)

    def _on_hp_alert(self, val):
        gb.TIER_THRESHOLDS_HIGH_PRIORITY["ALERT"] = float(val)

    def _on_hp_crit(self, val):
        gb.TIER_THRESHOLDS_HIGH_PRIORITY["CRITICAL"] = float(val)

    # ─── Restart-required staged callbacks (GUI-only) ─────────────

    def _on_dino_only_staged(self):
        """Toggle is GUI-only. Does NOT touch gb.DINO_ONLY."""
        self._staged_dino_only = self.var_dino_only.get()
        self._check_restart_needed()

    def _on_video_staged(self):
        """Toggle is GUI-only. Does NOT touch gb.doVideoStream."""
        self._staged_video_stream = self.var_video.get()
        self._toggle_url_visibility()
        self._check_restart_needed()

    def _toggle_url_visibility(self):
        """Show URL entry when video stream is checked, hide otherwise."""
        if self.var_video.get():
            self.url_row.pack(fill=tk.X, pady=2, after=self.chk_video)
        else:
            self.url_row.pack_forget()

    def _on_url_staged(self, event=None):
        """Entry is GUI-only. Does NOT touch gb.url."""
        self._staged_url = self.entry_url.get().strip()
        self._check_restart_needed()

    def _on_calib_frames_staged(self, event=None):
        """Entry is GUI-only. Does NOT touch gb.totalCalibrationFrames."""
        try:
            val = int(self.entry_calib.get())
            if val > 0:
                self._staged_calib_frames = val
        except ValueError:
            pass
        self._check_restart_needed()

    def _open_mask_editor(self):
        """Opens the paint-like mask editor as a modal window."""
        MaskEditor(self.root, mask_path="Assets/mask.png",
                   on_save_callback=self._on_mask_saved)

    def _on_mask_saved(self):
        """Called when the mask editor saves successfully."""
        self._mask_modified = True
        self._check_restart_needed()
        self.log("Mask saved. Click 'Apply & Restart' to load new mask.")

    # ─── Email Settings Window ────────────────────────────────────

    def _open_email_settings(self):
        """Opens a separate Toplevel window to edit email sender, receiver, and password."""
        win = tk.Toplevel(self.root)
        win.title("Email Settings")
        win.configure(bg='#c0c0c0')
        win.resizable(False, False)
        win.grab_set()  # modal

        font = self.font_classic
        font_bold = self.font_restart_hint

        # Banner
        tk.Label(win, text="Email Configuration", bg='#000080', fg='white',
                 font=font_bold, anchor='w', padx=8).pack(fill=tk.X)

        body = tk.LabelFrame(win, text="SMTP Settings", bg='#c0c0c0',
                             font=font, relief=tk.GROOVE, bd=2)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Sender
        row1 = tk.Frame(body, bg='#c0c0c0')
        row1.pack(fill=tk.X, pady=3, padx=5)
        tk.Label(row1, text="Sender Email:", bg='#c0c0c0', font=font, width=15,
                 anchor='w').pack(side=tk.LEFT)
        entry_sender = tk.Entry(row1, width=30, font=font)
        entry_sender.pack(side=tk.LEFT, padx=3)
        entry_sender.insert(0, gb.EMAIL_SENDER)

        # Password
        row2 = tk.Frame(body, bg='#c0c0c0')
        row2.pack(fill=tk.X, pady=3, padx=5)
        tk.Label(row2, text="App Password:", bg='#c0c0c0', font=font, width=15,
                 anchor='w').pack(side=tk.LEFT)
        entry_password = tk.Entry(row2, width=30, font=font, show="*")
        entry_password.pack(side=tk.LEFT, padx=3)
        entry_password.insert(0, gb.EMAIL_PASSWORD)

        # Receiver
        row3 = tk.Frame(body, bg='#c0c0c0')
        row3.pack(fill=tk.X, pady=3, padx=5)
        tk.Label(row3, text="Receiver Email:", bg='#c0c0c0', font=font, width=15,
                 anchor='w').pack(side=tk.LEFT)
        entry_receiver = tk.Entry(row3, width=30, font=font)
        entry_receiver.pack(side=tk.LEFT, padx=3)
        entry_receiver.insert(0, gb.EMAIL_RECEIVER)

        # Note
        tk.Label(body, text="Changes are saved immediately.",
                 bg='#c0c0c0', fg='#006400', font=font_bold).pack(pady=(5, 0))

        # Status label for feedback
        lbl_email_status = tk.Label(body, text="", bg='#c0c0c0', font=font)
        lbl_email_status.pack(pady=2)

        # Buttons
        btn_frame = tk.Frame(body, bg='#c0c0c0')
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        def save_email():
            gb.EMAIL_SENDER = entry_sender.get().strip()
            gb.EMAIL_PASSWORD = entry_password.get().strip()
            gb.EMAIL_RECEIVER = entry_receiver.get().strip()
            save_settings(stream_url=self._staged_url)
            lbl_email_status.config(text="Saved.", fg='#006400')
            self.log(f"Email settings updated.")
            win.after(800, win.destroy)

        tk.Button(btn_frame, text="Save", command=save_email,
                  relief=tk.RAISED, bd=2, bg='#c0c0c0', font=font,
                  width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=win.destroy,
                  relief=tk.RAISED, bd=2, bg='#c0c0c0', font=font,
                  width=10).pack(side=tk.LEFT, padx=5)

    # ─── Restart logic ────────────────────────────────────────────

    def _check_restart_needed(self):
        """Show/hide the 'Restart required' hint and enable/disable the button."""
        needs_restart = (
            self._staged_video_stream != gb.doVideoStream or
            self._staged_calib_frames != gb.totalCalibrationFrames or
            self._staged_dino_only != gb.DINO_ONLY or
            self._mask_modified
        )
        if needs_restart:
            self.lbl_restart_hint.pack(pady=(5, 0))
            self.btn_apply_restart.config(state=tk.NORMAL)
        else:
            self.lbl_restart_hint.pack_forget()
            self.btn_apply_restart.config(state=tk.DISABLED)

    def _apply_and_restart(self):
        """Commit staged values to globals, then trigger a full recalibration.
        main.py's update_loop detects self._restart_requested and re-runs setup."""
        # Commit staged values
        gb.doVideoStream = self._staged_video_stream
        gb.totalCalibrationFrames = self._staged_calib_frames
        gb.DINO_ONLY = self._staged_dino_only
        if gb.doVideoStream and self._staged_url:
            gb.url = self._staged_url

        # Reset calibration state
        gb.initialCalibration = True
        gb.currentCalibrationFramesHeld = 0
        gb.calibration_store = []
        gb.calibration_frames = []
        gb.calibration_array = None
        gb.current_highlight = None

        # Save all settings to file
        save_settings(stream_url=self._staged_url)

        # Signal main.py to re-open camera / reload mask
        self._restart_requested = True
        self._mask_modified = False

        # Hide the hint — staged values now match active
        self.lbl_restart_hint.pack_forget()
        self.btn_apply_restart.config(state=tk.DISABLED)

        self.lbl_calibration.config(text="Calibration: Restarting...")
        self.log("Settings applied. Restarting pipeline...")

    @property
    def restart_requested(self):
        """main.py polls this once per frame."""
        return self._restart_requested

    def clear_restart_flag(self):
        self._restart_requested = False

    # ─── External API (called by main.py) ─────────────────────────

    def update_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = cv2.resize(rgb_frame, (640, 480))

        image = Image.fromarray(rgb_frame)
        self.photo = ImageTk.PhotoImage(image=image)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        if gb.initialCalibration:
            self.lbl_calibration.config(
                text=f"Calibration: {gb.currentCalibrationFramesHeld}/{gb.totalCalibrationFrames}")
        else:
            self.lbl_calibration.config(text="Calibration: Complete")

    def update_status(self, status_text):
        self.lbl_status.config(text=f"Status: {status_text}")
        color_map = {"NORMAL": "green", "ALERT": "orange", "CRITICAL": "red"}
        self.lbl_status.config(fg=color_map.get(status_text, "green"))

    def log(self, message):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {message}\n"
        self.terminal.config(state=tk.NORMAL)
        self.terminal.insert(tk.END, line)
        self.terminal.see(tk.END)
        self.terminal.config(state=tk.DISABLED)
        self.statusbar.config(text=f"\u25b8 {message}")

    def is_running(self):
        return self._running

    def on_closing(self):
        # Save all current settings before exit
        save_settings(stream_url=self._staged_url)
        self._running = False
        self.root.destroy()
