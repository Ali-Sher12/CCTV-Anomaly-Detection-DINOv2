"""gui.py — Multi-camera Tkinter UI.

Features:
- N camera canvases in a 2-column grid.
- Docked non-modal calibration panels per camera with 4 buttons (Start, Pause, Confirm, Restart).
- Live-editable "Inter-Camera Delay (s)" slider saved on release.
- Right-click per-camera settings with threshold overrides, mask editor, and calibration actions.
- Complete removal of persistence UI and legacy banners.
"""

import os
import sys
import time
import shutil
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np

import Globals as gb
from mask_editor import MaskEditor
from settings_store import save_settings
from calibration_session import SessionState, CalibrationSession
from capture_registry import CaptureRegistry

CANVAS_W = 480
CANVAS_H = 360
COLS = 2


class AnomalyDetectionGUI:

    def __init__(self, root, cameras: list | None = None, settings_data: dict | None = None):
        self.root = root
        self.root.title("Passive Anomaly Detector PLUS +")
        self.root.configure(bg="#c0c0c0")
        self.root.resizable(True, True)
        self.root.minsize(900, 600)

        self._running = True
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self._settings_data = settings_data or {}
        self._cameras_cfg = cameras or []
        self._feeds = {}
        self._feed_manager = None

        # Per-camera UI widgets
        self._canvases: dict[int, tk.Canvas] = {}
        self._photos: dict[int, ImageTk.PhotoImage] = {}
        self._image_ids: dict[int, int] = {}
        self._status_labels: dict[int, tk.Label] = {}
        self._calib_labels: dict[int, tk.Label] = {}
        self._calib_panels: dict[int, tk.Frame] = {}
        self._calib_progress: dict[int, ttk.Progressbar] = {}
        self._calib_btns: dict[int, dict[str, tk.Button]] = {}
        self._calib_count_labels: dict[int, tk.Label] = {}

        # Fonts
        self.font_classic = ("MS Sans Serif", 8)
        self.font_bold = ("MS Sans Serif", 8, "bold")
        self.font_title = ("MS Sans Serif", 9, "bold")
        self.font_status = ("Fixedsys", 10)

        self._build_ui()
        self._init_controls()
        self._set_app_icon()

    def set_feed_manager(self, feed_manager):
        self._feed_manager = feed_manager

    def set_feeds(self, feeds):
        self._feeds = {feed.id: feed for feed in feeds}

    def _get_feed(self, camera_id: int):
        return self._feeds.get(camera_id)

    def _set_app_icon(self):
        logo_path = os.path.join("Assets", "logo.png")
        if os.path.exists(logo_path):
            try:
                self._icon_img = ImageTk.PhotoImage(file=logo_path)
                self.root.iconphoto(False, self._icon_img)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # UI Building
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Top banner
        banner_frame = tk.Frame(self.root, bg="#000080")
        banner_frame.pack(fill=tk.X, pady=(0, 5))

        logo_path = os.path.join("Assets", "logo.png")
        self._banner_icon = None
        if os.path.exists(logo_path):
            try:
                b_img = Image.open(logo_path)
                b_img.thumbnail((24, 24), Image.Resampling.LANCZOS)
                self._banner_icon = ImageTk.PhotoImage(b_img)
                tk.Label(banner_frame, image=self._banner_icon,
                         bg="#000080").pack(side=tk.LEFT, padx=(8, 2), pady=2)
            except Exception:
                pass

        tk.Label(banner_frame, text="Passive Anomaly Detector PLUS +",
                 bg="#000080", fg="white", font=self.font_title,
                 anchor="w").pack(side=tk.LEFT, padx=(2, 10), pady=3)

        # Main layout
        main_frame = tk.Frame(self.root, bg="#c0c0c0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel (cameras)
        left_panel = tk.Frame(main_frame, bg="#c0c0c0")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self._build_camera_grid(left_panel)

        # Right panel (controls)
        right_panel = tk.Frame(main_frame, bg="#c0c0c0")
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))

        self._build_runtime_group(right_panel)
        self._build_threshold_group(right_panel)
        self._build_system_group(right_panel)

        # Bottom status bar
        self.statusbar = tk.Label(self.root, text="▸ Ready", bd=1, relief=tk.SUNKEN,
                                  anchor="w", bg="#c0c0c0", font=self.font_status)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

    def _build_camera_grid(self, parent):
        # Scrollable container for camera grid
        scroll_canvas = tk.Canvas(parent, bg="#c0c0c0", highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=scroll_canvas.yview)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        grid_frame = tk.Frame(scroll_canvas, bg="#c0c0c0")
        grid_window = scroll_canvas.create_window((0, 0), window=grid_frame, anchor="nw")

        def _on_frame_configure(event):
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

        def _on_canvas_configure(event):
            # Keep inner frame matched with viewport width
            if event.width > 10:
                scroll_canvas.itemconfig(grid_window, width=event.width)

        grid_frame.bind("<Configure>", _on_frame_configure)
        scroll_canvas.bind("<Configure>", _on_canvas_configure)

        # Cross-platform mousewheel scrolling
        def _on_mousewheel(event):
            if event.num == 4:
                scroll_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                scroll_canvas.yview_scroll(1, "units")
            elif event.delta:
                scroll_canvas.yview_scroll(-1 * int(event.delta / 120), "units")

        def _bind_mousewheel(widget):
            widget.bind("<MouseWheel>", _on_mousewheel)
            widget.bind("<Button-4>", _on_mousewheel)
            widget.bind("<Button-5>", _on_mousewheel)
            for child in widget.winfo_children():
                _bind_mousewheel(child)

        _bind_mousewheel(scroll_canvas)
        _bind_mousewheel(grid_frame)

        grid_frame.grid_columnconfigure(0, weight=1, uniform="cam_col")
        grid_frame.grid_columnconfigure(1, weight=1, uniform="cam_col")

        cfgs = self._cameras_cfg if self._cameras_cfg else [{"id": 1, "name": "Camera 1"}]

        for i, cam_cfg in enumerate(cfgs):
            cam_id = cam_cfg["id"]
            cam_name = cam_cfg.get("name", f"Camera {cam_id}")
            row, col = divmod(i, COLS)

            cell = tk.Frame(grid_frame, bg="#c0c0c0", relief=tk.SUNKEN, bd=2)
            cell.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
            _bind_mousewheel(cell)

            # Camera header
            tk.Label(cell, text=f"Camera {cam_id} — {cam_name}",
                     bg="#c0c0c0", font=self.font_title).pack(anchor="w", padx=4, pady=(2, 0))

            # Canvas
            canvas = tk.Canvas(cell, width=CANVAS_W, height=CANVAS_H,
                               bg="black", highlightthickness=0)
            canvas.pack(padx=4, pady=4)

            # Context menu bindings
            for b in ("<Button-3>", "<Button-2>", "<Control-Button-1>"):
                canvas.bind(b, lambda e, cid=cam_id: self._open_camera_settings(cid))

            # Status label
            lbl_status = tk.Label(cell, text="Status: NORMAL", bg="#c0c0c0",
                                  font=self.font_title, fg="green")
            lbl_status.pack(anchor="w", padx=4)

            # Calibration status line (CAL-19)
            lbl_calib = tk.Label(cell, text="Calibration: Initialising...", bg="#c0c0c0",
                                 font=self.font_classic)
            lbl_calib.pack(anchor="w", padx=4, pady=(0, 2))

            # Docked calibration control panel (CAL-19)
            panel = tk.Frame(cell, bg="#dcdcdc", relief=tk.GROOVE, bd=1)
            panel.pack(fill=tk.X, padx=4, pady=(0, 4))

            top_row = tk.Frame(panel, bg="#dcdcdc")
            top_row.pack(fill=tk.X, padx=4, pady=2)

            prog = ttk.Progressbar(top_row, orient="horizontal", length=200, mode="determinate")
            prog.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

            lbl_count = tk.Label(top_row, text="0/100", bg="#dcdcdc", font=self.font_classic)
            lbl_count.pack(side=tk.RIGHT)

            btn_row = tk.Frame(panel, bg="#dcdcdc")
            btn_row.pack(fill=tk.X, padx=4, pady=(0, 2))

            btns = {}
            btns["start"] = tk.Button(btn_row, text="Start", font=self.font_classic,
                                      command=lambda cid=cam_id: self._on_start_calib(cid),
                                      bg="#00aa00", fg="white", relief=tk.RAISED, bd=1, width=7)
            btns["start"].pack(side=tk.LEFT, padx=2)

            btns["pause"] = tk.Button(btn_row, text="Pause", font=self.font_classic,
                                      command=lambda cid=cam_id: self._on_pause_calib(cid),
                                      bg="#c0c0c0", relief=tk.RAISED, bd=1, width=7)
            btns["pause"].pack(side=tk.LEFT, padx=2)

            btns["confirm"] = tk.Button(btn_row, text="Confirm", font=self.font_classic,
                                        command=lambda cid=cam_id: self._on_confirm_calib(cid),
                                        bg="#0000aa", fg="white", relief=tk.RAISED, bd=1, width=8)
            btns["confirm"].pack(side=tk.LEFT, padx=2)

            btns["restart"] = tk.Button(btn_row, text="Restart", font=self.font_classic,
                                        command=lambda cid=cam_id: self._on_restart_calib(cid),
                                        bg="#aa0000", fg="white", relief=tk.RAISED, bd=1, width=7)
            btns["restart"].pack(side=tk.LEFT, padx=2)

            self._canvases[cam_id] = canvas
            self._status_labels[cam_id] = lbl_status
            self._calib_labels[cam_id] = lbl_calib
            self._calib_panels[cam_id] = panel
            self._calib_progress[cam_id] = prog
            self._calib_btns[cam_id] = btns
            self._calib_count_labels[cam_id] = lbl_count
            self._photos[cam_id] = None

            _bind_mousewheel(cell)

    # ------------------------------------------------------------------
    # Right Panel (Runtime & Thresholds)
    # ------------------------------------------------------------------

    def _build_runtime_group(self, parent):
        grp = tk.LabelFrame(parent, text="Runtime Controls", bg="#c0c0c0",
                            font=self.font_classic, relief=tk.GROOVE, bd=2)
        grp.pack(fill=tk.X, pady=(0, 8), ipadx=5, ipady=3)

        # Inter-camera delay slider (SCH-02, SCH-03)
        delay_frame = tk.Frame(grp, bg="#c0c0c0")
        delay_frame.pack(fill=tk.X, pady=2)
        tk.Label(delay_frame, text="Inter-Camera Delay (s)", bg="#c0c0c0",
                 font=self.font_classic, width=20, anchor="w").pack(side=tk.LEFT)
        self.scale_inter_delay = tk.Scale(delay_frame, from_=0.1, to=5.0, resolution=0.1,
                                          orient=tk.HORIZONTAL, bg="#c0c0c0",
                                          font=self.font_classic, command=self._on_inter_delay_change,
                                          length=120)
        self.scale_inter_delay.pack(side=tk.RIGHT)
        self.scale_inter_delay.bind("<ButtonRelease-1>", self._on_inter_delay_release)

        # Display delay slider
        self.scale_delay = self._create_slider(grp, "Display Delay (ms)", 1, 100, 1, self._on_delay)

        # Auto-update checkbutton
        self.var_auto_update = tk.BooleanVar(value=gb.auto_update_calibration)
        tk.Checkbutton(grp, text="Auto-Update Calibration",
                       variable=self.var_auto_update, bg="#c0c0c0",
                       font=self.font_classic,
                       command=self._on_auto_update).pack(anchor="w", pady=(4, 0))

        # Mute audio checkbutton
        self.var_mute_audio = tk.BooleanVar(value=gb.audio_muted)
        tk.Checkbutton(grp, text="Mute Audio Alerts",
                       variable=self.var_mute_audio, bg="#c0c0c0",
                       font=self.font_classic,
                       command=self._on_mute_audio).pack(anchor="w", pady=(2, 0))

    def _build_threshold_group(self, parent):
        grp = tk.LabelFrame(parent, text="Global Thresholds", bg="#c0c0c0",
                            font=self.font_classic, relief=tk.GROOVE, bd=2)
        grp.pack(fill=tk.X, pady=(0, 8), ipadx=5, ipady=3)

        tk.Label(grp, text="(Right-click a feed for overrides)",
                 bg="#c0c0c0", font=self.font_classic, fg="#555555").pack(anchor="w", pady=(0, 4))

        self.scale_alert = self._create_slider(grp, "Alert", 0, 200, 1, self._on_alert)
        self.scale_crit = self._create_slider(grp, "Critical", 0, 200, 1, self._on_crit)
        tk.Label(grp, text="— High-Priority Zones —", bg="#c0c0c0",
                 font=self.font_classic).pack(pady=(6, 2))
        self.scale_hp_alert = self._create_slider(grp, "HP Alert", 0, 200, 1, self._on_hp_alert)
        self.scale_hp_crit = self._create_slider(grp, "HP Critical", 0, 200, 1, self._on_hp_crit)

    def _build_system_group(self, parent):
        grp = tk.LabelFrame(parent, text="System", bg="#c0c0c0",
                            font=self.font_classic, relief=tk.GROOVE, bd=2)
        grp.pack(fill=tk.X, pady=(0, 8), ipadx=5, ipady=4)

        btn_reset = tk.Button(grp, text="Reset to Initial Setup...",
                              command=self._on_reset_clicked,
                              relief=tk.RAISED, bd=2, bg="#b33939", fg="white",
                              activebackground="#8b0000", activeforeground="white",
                              font=self.font_bold)
        btn_reset.pack(fill=tk.X, padx=3, pady=2)

    def _create_slider(self, parent, label_text, from_, to, resolution, command):
        frame = tk.Frame(parent, bg="#c0c0c0")
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text=label_text, bg="#c0c0c0",
                 font=self.font_classic, width=18, anchor="w").pack(side=tk.LEFT)
        scale = tk.Scale(frame, from_=from_, to=to, resolution=resolution,
                         orient=tk.HORIZONTAL, bg="#c0c0c0",
                         font=self.font_classic, command=command, length=120)
        scale.pack(side=tk.RIGHT)
        return scale

    def _init_controls(self):
        init_delay = self._settings_data.get("inter_camera_delay", 0.5)
        self.scale_inter_delay.set(init_delay)
        self.scale_delay.set(gb.delay)
        self.scale_alert.set(gb.TIER_THRESHOLDS["ALERT"])
        self.scale_crit.set(gb.TIER_THRESHOLDS["CRITICAL"])
        self.scale_hp_alert.set(gb.TIER_THRESHOLDS_HIGH_PRIORITY["ALERT"])
        self.scale_hp_crit.set(gb.TIER_THRESHOLDS_HIGH_PRIORITY["CRITICAL"])

    # ------------------------------------------------------------------
    # Slider Callbacks
    # ------------------------------------------------------------------

    def _on_inter_delay_change(self, val):
        delay_f = float(val)
        if self._feed_manager:
            self._feed_manager.set_inter_camera_delay(delay_f)

    def _on_inter_delay_release(self, event):
        delay_f = float(self.scale_inter_delay.get())
        self._settings_data["inter_camera_delay"] = delay_f
        save_settings(self._settings_data)
        self.log(f"Inter-camera delay saved: {delay_f:.1f}s")

    def _on_delay(self, val):
        gb.delay = int(float(val))

    def _on_auto_update(self):
        gb.auto_update_calibration = self.var_auto_update.get()

    def _on_mute_audio(self):
        muted = self.var_mute_audio.get()
        gb.audio_muted = muted
        self._settings_data["audio_muted"] = muted
        save_settings(self._settings_data)
        self.log(f"Audio alerts {'MUTED' if muted else 'UNMUTED'}.")

    def _on_alert(self, val):
        val = float(val)
        gb.TIER_THRESHOLDS["ALERT"] = val
        if "default_thresholds" in self._settings_data:
            self._settings_data["default_thresholds"]["ALERT"] = val

    def _on_crit(self, val):
        val = float(val)
        gb.TIER_THRESHOLDS["CRITICAL"] = val
        if "default_thresholds" in self._settings_data:
            self._settings_data["default_thresholds"]["CRITICAL"] = val

    def _on_hp_alert(self, val):
        val = float(val)
        gb.TIER_THRESHOLDS_HIGH_PRIORITY["ALERT"] = val
        if "default_thresholds" in self._settings_data:
            self._settings_data["default_thresholds"]["HIGH_PRIORITY_ALERT"] = val

    def _on_hp_crit(self, val):
        val = float(val)
        gb.TIER_THRESHOLDS_HIGH_PRIORITY["CRITICAL"] = val
        if "default_thresholds" in self._settings_data:
            self._settings_data["default_thresholds"]["HIGH_PRIORITY_CRITICAL"] = val

    # ------------------------------------------------------------------
    # Calibration Session Actions (CAL-10 to CAL-24)
    # ------------------------------------------------------------------

    def _on_start_calib(self, camera_id: int):
        feed = self._get_feed(camera_id)
        if feed:
            feed.start_calibration_session()
            self.log(f"[Cam {camera_id}] Calibration collection started.")

    def _on_pause_calib(self, camera_id: int):
        feed = self._get_feed(camera_id)
        if feed:
            feed.pause_calibration_session()
            self.log(f"[Cam {camera_id}] Calibration collection paused.")

    def _on_confirm_calib(self, camera_id: int):
        feed = self._get_feed(camera_id)
        if feed:
            ok, err = feed.confirm_calibration_session()
            if not ok:
                messagebox.showerror("Calibration Error", err, parent=self.root)

    def _on_restart_calib(self, camera_id: int):
        feed = self._get_feed(camera_id)
        if feed:
            feed.restart_calibration_session()
            self.log(f"[Cam {camera_id}] Calibration session restarted (uncommitted frames discarded).")

    # ------------------------------------------------------------------
    # Calibration UI Updater (called every tick)
    # ------------------------------------------------------------------

    def update_calibration_ui(self, camera_id: int, feed):
        calib_lbl = self._calib_labels.get(camera_id)
        panel = self._calib_panels.get(camera_id)
        prog = self._calib_progress.get(camera_id)
        count_lbl = self._calib_count_labels.get(camera_id)
        btns = self._calib_btns.get(camera_id)

        if not calib_lbl or not panel or not btns:
            return

        session = feed.calibration_session
        if session is not None:
            # Active session
            panel.pack(fill=tk.X, padx=4, pady=(0, 4))
            target = session.target_frames
            collected = session.collected_count
            state_str = session.state.value

            calib_lbl.config(text=f"Calibrating {collected}/{target} — {state_str}")
            prog.config(maximum=target, value=collected)
            count_lbl.config(text=f"{collected}/{target}")

            # Button states
            btns["start"].config(state=tk.NORMAL if session.can_start else tk.DISABLED)
            btns["pause"].config(state=tk.NORMAL if session.can_pause else tk.DISABLED)
            btns["confirm"].config(state=tk.NORMAL if session.can_confirm else tk.DISABLED)
            btns["restart"].config(state=tk.NORMAL if session.can_restart else tk.DISABLED)
        else:
            # Not in session -> show status line and hide docked panel
            calib_lbl.config(text=feed.calibration_status)
            panel.pack_forget()

    # ------------------------------------------------------------------
    # Right-Click Per-Camera Settings Dialog
    # ------------------------------------------------------------------

    def _open_camera_settings(self, camera_id: int):
        feed = self._get_feed(camera_id)
        if feed is None:
            self.log(f"[Cam {camera_id}] Feed not ready.")
            return

        win = tk.Toplevel(self.root)
        win.title(f"Camera {camera_id} — {feed.name} Settings")
        win.configure(bg="#c0c0c0")
        win.resizable(False, False)
        win.transient(self.root)

        font = self.font_classic
        font_bold = self.font_title

        tk.Label(win, text=f"Camera {camera_id} — {feed.name}",
                 bg="#000080", fg="white", font=font_bold,
                 padx=8).pack(fill=tk.X)

        body = tk.Frame(win, bg="#c0c0c0")
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 1. Per-Camera Controls
        ctrl_frame = tk.LabelFrame(body, text="Camera Settings", bg="#c0c0c0",
                                   font=font, relief=tk.GROOVE, bd=2)
        ctrl_frame.pack(fill=tk.X, pady=(0, 8), padx=2, ipady=3)

        def make_cam_slider(parent, label, from_, to, res, getter, setter):
            r = tk.Frame(parent, bg="#c0c0c0"); r.pack(fill=tk.X, pady=2)
            tk.Label(r, text=label, bg="#c0c0c0", font=font, width=22, anchor="w").pack(side=tk.LEFT)
            s = tk.Scale(r, from_=from_, to=to, resolution=res, orient=tk.HORIZONTAL,
                         bg="#c0c0c0", font=font, command=lambda v: setter(v), length=130)
            s.set(getter())
            s.pack(side=tk.RIGHT)
            return s

        make_cam_slider(ctrl_frame, "Alert Tolerance", 1, 20, 1,
                        lambda: feed.allowed_error,
                        lambda v: setattr(feed, "allowed_error", int(float(v))))
        make_cam_slider(ctrl_frame, "Report Cooldown (s)", 5, 120, 1,
                        lambda: feed.anomaly_report_wait,
                        lambda v: setattr(feed, "anomaly_report_wait", int(float(v))))

        # 2. Threshold Overrides
        thresh_frame = tk.LabelFrame(body, text="Threshold Overrides", bg="#c0c0c0",
                                     font=font, relief=tk.GROOVE, bd=2)
        thresh_frame.pack(fill=tk.X, pady=(0, 8), padx=2, ipady=3)

        var_override = tk.BooleanVar(value=bool(feed.thresholds))
        sliders_subframe = tk.Frame(thresh_frame, bg="#c0c0c0")

        def on_toggle_override():
            if var_override.get():
                if feed.thresholds is None:
                    feed.thresholds = {
                        "ALERT": gb.TIER_THRESHOLDS["ALERT"],
                        "CRITICAL": gb.TIER_THRESHOLDS["CRITICAL"],
                        "HIGH_PRIORITY_ALERT": gb.TIER_THRESHOLDS_HIGH_PRIORITY["ALERT"],
                        "HIGH_PRIORITY_CRITICAL": gb.TIER_THRESHOLDS_HIGH_PRIORITY["CRITICAL"],
                    }
                sliders_subframe.pack(fill=tk.X, padx=4, pady=2)
                self.log(f"[Cam {camera_id}] Custom thresholds enabled.")
            else:
                feed.thresholds = None
                sliders_subframe.pack_forget()
                self.log(f"[Cam {camera_id}] Reverted to global thresholds.")

        tk.Checkbutton(thresh_frame, text="Enable per-camera threshold overrides",
                       variable=var_override, bg="#c0c0c0", font=font,
                       command=on_toggle_override).pack(anchor="w", padx=4, pady=2)

        if var_override.get():
            sliders_subframe.pack(fill=tk.X, padx=4, pady=2)

        def make_thresh_slider(label, key, hp=False):
            r = tk.Frame(sliders_subframe, bg="#c0c0c0"); r.pack(fill=tk.X, pady=2)
            tk.Label(r, text=label, bg="#c0c0c0", font=font, width=18, anchor="w").pack(side=tk.LEFT)
            initial = (feed.thresholds or {}).get(
                key,
                gb.TIER_THRESHOLDS_HIGH_PRIORITY[key.replace("HIGH_PRIORITY_", "")] if hp else gb.TIER_THRESHOLDS[key]
            )
            def on_val(v, k=key):
                if feed.thresholds is not None:
                    feed.thresholds[k] = float(v)
            s = tk.Scale(r, from_=0, to=200, resolution=0.5, orient=tk.HORIZONTAL,
                         bg="#c0c0c0", font=font, command=on_val, length=130)
            s.set(initial)
            s.pack(side=tk.RIGHT)

        make_thresh_slider("Alert", "ALERT")
        make_thresh_slider("Critical", "CRITICAL")
        make_thresh_slider("HP Alert", "HIGH_PRIORITY_ALERT", hp=True)
        make_thresh_slider("HP Critical", "HIGH_PRIORITY_CRITICAL", hp=True)

        # 3. Zone Mask Modification
        mask_frame = tk.LabelFrame(body, text="Zone Mask", bg="#c0c0c0",
                                   font=font, relief=tk.GROOVE, bd=2)
        mask_frame.pack(fill=tk.X, pady=(0, 8), padx=2, ipady=3)

        def open_mask():
            def on_mask_saved():
                feed.update_mask()
                self.log(f"[Cam {camera_id}] Mask updated and active.")
            MaskEditor(win, mask_path=feed.mask_path, on_save_callback=on_mask_saved)

        tk.Button(mask_frame, text="Modify Mask Painter...", command=open_mask,
                  relief=tk.RAISED, bd=2, bg="#c0c0c0", font=font).pack(fill=tk.X, padx=6, pady=2)

        # 4. Calibration Management (CAL-20, CAL-21)
        cal_sec = tk.LabelFrame(body, text="Calibration", bg="#c0c0c0",
                                font=font, relief=tk.GROOVE, bd=2)
        cal_sec.pack(fill=tk.X, pady=(0, 8), padx=2, ipady=3)

        def on_add_calib():
            target_str = simpledialog.askstring(
                "Add Calibration Frames",
                f"Enter number of frames to add for Camera {camera_id} (1–500):",
                initialvalue="30",
                parent=win
            )
            if not target_str:
                return
            try:
                target = int(target_str.strip())
                if target < 1 or target > 500:
                    raise ValueError()
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter an integer between 1 and 500.", parent=win)
                return

            feed.start_calibration_session(target_frames=target)
            self.log(f"[Cam {camera_id}] Started Add Calibration Session (target: {target} frames).")
            win.destroy()

        def on_remove_calib():
            n = len(feed.calibration_store)
            confirm = messagebox.askyesno(
                "Delete Calibration",
                f"Permanently delete all {n} calibration frames for Camera {camera_id} ({feed.name})?\n\n"
                "Detection stops on this camera until you calibrate again.",
                parent=win,
                icon="warning"
            )
            if not confirm:
                return

            ok, err = feed.remove_all_calibration_and_recalibrate()
            if not ok:
                messagebox.showerror("Delete Failed", err, parent=win)
            else:
                win.destroy()

        tk.Button(cal_sec, text="Add calibration frames...", command=on_add_calib,
                  relief=tk.RAISED, bd=2, bg="#c0c0c0", font=font).pack(fill=tk.X, padx=6, pady=2)
        tk.Button(cal_sec, text="Remove all calibration frames and recalibrate...", command=on_remove_calib,
                  relief=tk.RAISED, bd=2, bg="#c0c0c0", fg="#880000", font=font).pack(fill=tk.X, padx=6, pady=2)

        # Close button
        tk.Button(win, text="Close", command=win.destroy,
                  relief=tk.RAISED, bd=2, bg="#c0c0c0", font=font, width=12).pack(pady=6)

        win.update_idletasks()
        try:
            win.grab_set()
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    # Frame & Status Updates (Main Thread)
    # ------------------------------------------------------------------

    def update_camera_frame(self, camera_id: int, frame: np.ndarray):
        canvas = self._canvases.get(camera_id)
        if canvas is None or frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_LINEAR)
        img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=img)
        self._photos[camera_id] = photo

        img_id = self._image_ids.get(camera_id)
        if img_id is None:
            self._image_ids[camera_id] = canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        else:
            canvas.itemconfigure(img_id, image=photo)

    def update_camera_status(self, camera_id: int, status_text: str):
        lbl = self._status_labels.get(camera_id)
        if lbl is None:
            return
        lbl.config(text=f"Status: {status_text}")
        color_map = {
            "NORMAL": "green",
            "ALERT": "orange",
            "CRITICAL": "red",
            "OFFLINE": "#888888"
        }
        lbl.config(fg=color_map.get(status_text, "green"))

    def log(self, message: str):
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {message}")
        self.statusbar.config(text=f"▸ {message}")

    def is_running(self) -> bool:
        return self._running

    def _on_reset_clicked(self):
        # --- First confirmation ---
        confirmed1 = messagebox.askyesno(
            "Reset Application",
            "This will permanently delete ALL saved data:\n"
            "  • Camera configuration (settings.json)\n"
            "  • All calibration data\n"
            "  • All zone masks\n"
            "  • Heartbeat and watchdog logs\n\n"
            "The application will restart from the Setup Wizard.\n\n"
            "Are you sure you want to continue?",
            parent=self.root,
            icon="warning"
        )
        if not confirmed1:
            return

        # --- Second confirmation ---
        confirmed2 = messagebox.askyesno(
            "Confirm Full Reset",
            "This is your final confirmation.\n\n"
            "ALL data will be wiped and CANNOT be recovered.\n\n"
            "Proceed with full reset?",
            parent=self.root,
            icon="warning"
        )
        if not confirmed2:
            return

        base_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(base_dir, "Assets")

        # Paths to wipe
        wipe_files = [
            os.path.join(assets_dir, "settings.json"),
            os.path.join(assets_dir, "heartbeat.txt"),
            os.path.join(assets_dir, "app.pid"),
            os.path.join(assets_dir, "watchdog_restart_log.json"),
            os.path.join(assets_dir, "watchdog_log.txt"),
        ]
        wipe_dirs = [
            os.path.join(assets_dir, "calibration"),
            os.path.join(assets_dir, "masks"),
        ]

        for path in wipe_files:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"[Reset] Deleted {path}")
                except OSError as e:
                    print(f"[Reset] Error deleting {path}: {e}")

        for path in wipe_dirs:
            if os.path.exists(path):
                try:
                    shutil.rmtree(path)
                    print(f"[Reset] Deleted directory {path}")
                except OSError as e:
                    print(f"[Reset] Error deleting directory {path}: {e}")

        # Release all video capture handles BEFORE exec so the new process
        # inherits no open /dev/video* file descriptors (fixes source=0
        # "cannot open" on re-launch after reset).
        try:
            CaptureRegistry().release_all()
        except Exception as e:
            print(f"[Reset] Warning: release_all failed: {e}")

        self._running = False
        self.root.destroy()
        os.execv(sys.executable, [sys.executable] + sys.argv)

    def on_closing(self):
        # Check if any camera has uncommitted calibration session frames (CAL-17)
        uncommitted_cams = []
        for feed in self._feeds.values():
            if feed.calibration_session and feed.calibration_session.collected_count > 0:
                uncommitted_cams.append((feed.id, feed.calibration_session.collected_count))

        if uncommitted_cams:
            details = ", ".join([f"Cam {cid}: {cnt} frames" for cid, cnt in uncommitted_cams])
            ans = messagebox.askyesno(
                "Uncommitted Calibration Frames",
                f"There are uncommitted calibration frames ({details}).\n\nDiscard these uncommitted frames and exit?",
                parent=self.root
            )
            if not ans:
                return

        save_settings(self._settings_data)
        self._running = False
        self.root.destroy()

