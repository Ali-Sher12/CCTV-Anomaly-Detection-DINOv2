"""setup_wizard.py — First-run 5-step Setup Wizard.

Shown by main.py when load_settings() returns None.
On Finish: writes settings.json with setup_complete=True and generates
per-camera all-green mask PNGs in Assets/masks/.
"""

import os
import threading
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

import Globals as gb
import Accessories as Acc
from settings_store import save_settings, default_settings
from capture_registry import normalize_source


class SetupWizard(tk.Toplevel):
    """Modal wizard. Call wait_window() after construction to block until done."""

    STEPS = 5

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Passive Anomaly Detector — First-Run Setup")
        self.resizable(False, False)
        self.configure(bg="#c0c0c0")
        self.transient(parent)
        try:
            self.grab_set()
        except tk.TclError:
            pass
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._data = default_settings()
        self._step = 0

        self.font = ("MS Sans Serif", 9)
        self.font_bold = ("MS Sans Serif", 9, "bold")
        self.font_title = ("MS Sans Serif", 11, "bold")

        self._cameras: list[dict] = []
        self._cam_id_counter = 1

        # Threshold variables (Step 4)
        self._var_alert = tk.DoubleVar(value=45.0)
        self._var_crit = tk.DoubleVar(value=50.0)
        self._var_hp_alert = tk.DoubleVar(value=35.0)
        self._var_hp_crit = tk.DoubleVar(value=40.0)

        # Step-5 variables
        self._var_calib = tk.IntVar(value=100)
        self._var_delay = tk.DoubleVar(value=0.5)
        self._var_dino_only = tk.BooleanVar(value=False)

        self._build_chrome()
        self._show_step(0)
        self._center()

    def _build_chrome(self):
        banner = tk.Frame(self, bg="#000080")
        banner.pack(fill=tk.X)
        tk.Label(banner, text="Passive Anomaly Detector — Setup Wizard",
                 bg="#000080", fg="white", font=self.font_title,
                 anchor="w", padx=10).pack(side=tk.LEFT, pady=5)

        self._lbl_step = tk.Label(self, text="", bg="#c0c0c0", font=self.font_bold)
        self._lbl_step.pack(anchor="e", padx=10, pady=(4, 0))

        self._content = tk.Frame(self, bg="#c0c0c0")
        self._content.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        ttk.Separator(self, orient="horizontal").pack(fill=tk.X, padx=5)

        nav = tk.Frame(self, bg="#c0c0c0")
        nav.pack(fill=tk.X, padx=10, pady=6)
        self._btn_back = tk.Button(nav, text="◀ Back", command=self._go_back,
                                   relief=tk.RAISED, bd=2, bg="#c0c0c0",
                                   font=self.font, width=10)
        self._btn_back.pack(side=tk.LEFT)
        self._btn_next = tk.Button(nav, text="Next ▶", command=self._go_next,
                                   relief=tk.RAISED, bd=2, bg="#c0c0c0",
                                   font=self.font, width=10)
        self._btn_next.pack(side=tk.RIGHT)

    def _show_step(self, step: int):
        self._step = step
        for w in self._content.winfo_children():
            w.destroy()

        self._lbl_step.config(text=f"Step {step + 1} of {self.STEPS}")
        self._btn_back.config(state=tk.NORMAL if step > 0 else tk.DISABLED)
        is_last = step == self.STEPS - 1
        self._btn_next.config(text="Finish" if is_last else "Next ▶")

        builders = [
            self._build_step1_device,
            self._build_step2_cameras,
            self._build_step3_email,
            self._build_step4_thresholds,
            self._build_step5_summary,
        ]
        builders[step]()

    def _go_next(self):
        if not self._validate_step(self._step):
            return
        self._collect_step(self._step)
        if self._step < self.STEPS - 1:
            self._show_step(self._step + 1)
        else:
            self._finish()

    def _go_back(self):
        if self._step > 0:
            self._show_step(self._step - 1)

    # ------------------------------------------------------------------
    # Step 1 — Device Scan
    # ------------------------------------------------------------------

    def _build_step1_device(self):
        f = self._content
        tk.Label(f, text="Step 1 — Device Scan", bg="#c0c0c0",
                 font=self.font_bold).pack(anchor="w", pady=(0, 6))
        tk.Label(f, text="Hardware specs determine which AI models will be used.",
                 bg="#c0c0c0", font=self.font).pack(anchor="w")

        Acc.getnSetSpecs()

        txt = tk.Text(f, height=10, width=60, bg="black", fg="#00ff00",
                      font=("Courier New", 9), state=tk.DISABLED,
                      relief=tk.SUNKEN, bd=2)
        txt.pack(pady=8, fill=tk.X)
        txt.config(state=tk.NORMAL)
        txt.insert(tk.END, Acc.specLog1 or "(no spec info)")
        txt.config(state=tk.DISABLED)

        if "Tier: HIGH" in (Acc.specLog1 or ""):
            self._data["device_tier"] = "HIGH"
        elif "Tier: LOW" in (Acc.specLog1 or ""):
            self._data["device_tier"] = "LOW"
        else:
            self._data["device_tier"] = "MID"

        self._data["dino_model_version"] = gb.DINO_MODEL_VERSION
        self._data["yolo_model_version"] = gb.YOLO_MODEL_VERSION

    # ------------------------------------------------------------------
    # Step 2 — Cameras
    # ------------------------------------------------------------------

    def _build_step2_cameras(self):
        f = self._content
        tk.Label(f, text="Step 2 — Cameras", bg="#c0c0c0",
                 font=self.font_bold).pack(anchor="w", pady=(0, 6))
        tk.Label(f, text="Add at least one camera. Source: integer index (0) or stream URL.",
                 bg="#c0c0c0", font=self.font).pack(anchor="w")

        list_frame = tk.Frame(f, bg="#c0c0c0", relief=tk.SUNKEN, bd=2)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=6)

        canvas = tk.Canvas(list_frame, bg="#c0c0c0", highlightthickness=0, height=220)
        sb = tk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._cam_inner = tk.Frame(canvas, bg="#c0c0c0")
        self._cam_window = canvas.create_window((0, 0), window=self._cam_inner, anchor="nw")
        self._cam_inner.bind("<Configure>",
                             lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        self._cam_canvas = canvas

        for cam in self._cameras:
            self._add_camera_row(cam)

        btn_row = tk.Frame(f, bg="#c0c0c0")
        btn_row.pack(anchor="w")
        tk.Button(btn_row, text="+ Add Camera", command=self._add_camera,
                  relief=tk.RAISED, bd=2, bg="#c0c0c0",
                  font=self.font).pack(side=tk.LEFT, padx=(0, 10))

        if not self._cameras:
            self._add_camera()

    def _add_camera(self):
        cam_id = self._cam_id_counter
        self._cam_id_counter += 1
        cam = {
            "id": cam_id,
            "name_var": tk.StringVar(value=f"Camera {cam_id}"),
            "source_var": tk.StringVar(value=str(cam_id - 1)),
            "status_var": tk.StringVar(value=""),
        }
        self._cameras.append(cam)
        self._add_camera_row(cam)
        self._cam_canvas.update_idletasks()
        self._cam_canvas.configure(scrollregion=self._cam_canvas.bbox("all"))

    def _add_camera_row(self, cam: dict):
        row = tk.Frame(self._cam_inner, bg="#c0c0c0", pady=3)
        row.pack(fill=tk.X, padx=4)

        tk.Label(row, text=f"#{cam['id']}", bg="#c0c0c0", font=self.font, width=3).pack(side=tk.LEFT)
        tk.Label(row, text="Name:", bg="#c0c0c0", font=self.font).pack(side=tk.LEFT)
        tk.Entry(row, textvariable=cam["name_var"], width=14, font=self.font).pack(side=tk.LEFT, padx=3)
        tk.Label(row, text="Source:", bg="#c0c0c0", font=self.font).pack(side=tk.LEFT)
        tk.Entry(row, textvariable=cam["source_var"], width=22, font=self.font).pack(side=tk.LEFT, padx=3)

        lbl_status = tk.Label(row, textvariable=cam["status_var"], bg="#c0c0c0", font=self.font, width=3)
        lbl_status.pack(side=tk.LEFT, padx=3)

        def test_this(c=cam):
            c["status_var"].set("⏳")
            raw = c["source_var"].get().strip()

            def _test_worker():
                cap = None
                ok = False
                try:
                    src = normalize_source(raw)
                    cap = cv2.VideoCapture(src)
                    ok = cap.isOpened()
                    if ok:
                        ret, frame = cap.read()
                        ok = ret and frame is not None
                except Exception:
                    ok = False
                finally:
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass
                c["status_var"].set("✅" if ok else "❌")

            threading.Thread(target=_test_worker, daemon=True).start()

        def remove_this(c=cam):
            if len(self._cameras) <= 1:
                messagebox.showwarning("Remove Camera", "At least one camera is required.", parent=self)
                return
            self._cameras.remove(c)
            row.destroy()

        tk.Button(row, text="Test", command=test_this, relief=tk.RAISED, bd=2, bg="#c0c0c0",
                  font=self.font, width=5).pack(side=tk.LEFT, padx=2)
        tk.Button(row, text="Remove", command=remove_this, relief=tk.RAISED, bd=2, bg="#c0c0c0",
                  font=self.font, width=7).pack(side=tk.LEFT)

    # ------------------------------------------------------------------
    # Step 3 — Email
    # ------------------------------------------------------------------

    def _build_step3_email(self):
        f = self._content
        tk.Label(f, text="Step 3 — Email Notifications", bg="#c0c0c0",
                 font=self.font_bold).pack(anchor="w", pady=(0, 6))
        tk.Label(f, text="Leave blank to disable email alerts.", bg="#c0c0c0", font=self.font).pack(anchor="w")

        body = tk.Frame(f, bg="#c0c0c0")
        body.pack(fill=tk.X, pady=8)

        existing = self._data.get("email", {})

        def row(label, default, show=""):
            r = tk.Frame(body, bg="#c0c0c0"); r.pack(fill=tk.X, pady=3)
            tk.Label(r, text=label, bg="#c0c0c0", font=self.font, width=18, anchor="w").pack(side=tk.LEFT)
            e = tk.Entry(r, width=36, font=self.font, show=show)
            e.insert(0, default)
            e.pack(side=tk.LEFT)
            return e

        self._entry_sender = row("Sender Email:", existing.get("sender", ""))
        self._entry_password = row("App Password:", existing.get("app_password", ""), show="*")
        rcv = existing.get("receivers", [])
        self._entry_receivers = row("Receivers (comma-sep):", ", ".join(rcv) if rcv else "")

    # ------------------------------------------------------------------
    # Step 4 — Thresholds
    # ------------------------------------------------------------------

    def _build_step4_thresholds(self):
        f = self._content
        tk.Label(f, text="Step 4 — Default Thresholds", bg="#c0c0c0",
                 font=self.font_bold).pack(anchor="w", pady=(0, 6))
        tk.Label(f, text="These apply to all cameras unless overridden per-camera.",
                 bg="#c0c0c0", font=self.font).pack(anchor="w")

        dt = self._data.get("default_thresholds", {})
        self._var_alert.set(dt.get("ALERT", 45.0))
        self._var_crit.set(dt.get("CRITICAL", 50.0))
        self._var_hp_alert.set(dt.get("HIGH_PRIORITY_ALERT", 35.0))
        self._var_hp_crit.set(dt.get("HIGH_PRIORITY_CRITICAL", 40.0))

        sliders = tk.Frame(f, bg="#c0c0c0")
        sliders.pack(fill=tk.X, pady=8)

        def sl(parent, label, var, from_=0, to=200):
            r = tk.Frame(parent, bg="#c0c0c0"); r.pack(fill=tk.X, pady=3)
            tk.Label(r, text=label, bg="#c0c0c0", font=self.font, width=24, anchor="w").pack(side=tk.LEFT)
            tk.Scale(r, variable=var, from_=from_, to=to, resolution=0.5,
                     orient=tk.HORIZONTAL, bg="#c0c0c0", length=180, font=self.font).pack(side=tk.RIGHT)

        sl(sliders, "Alert", self._var_alert)
        sl(sliders, "Critical", self._var_crit)
        sl(sliders, "HP Alert", self._var_hp_alert)
        sl(sliders, "HP Critical", self._var_hp_crit)

    # ------------------------------------------------------------------
    # Step 5 — Summary / Finish
    # ------------------------------------------------------------------

    def _build_step5_summary(self):
        f = self._content
        tk.Label(f, text="Step 5 — Suggested Values & Summary", bg="#c0c0c0",
                 font=self.font_bold).pack(anchor="w", pady=(0, 6))

        tier = self._data.get("device_tier", "MID")
        cam_count = len(self._cameras)

        calib_map = {"LOW": 60, "MID": 100, "HIGH": 150}
        delay_map = {"LOW": 1.5, "MID": 0.5, "HIGH": 0.2}
        self._var_calib.set(calib_map.get(tier, 100))
        self._var_delay.set(delay_map.get(tier, 0.5))
        self._var_dino_only.set(gb.DINO_ONLY)

        body = tk.Frame(f, bg="#c0c0c0")
        body.pack(fill=tk.X, pady=6)

        r1 = tk.Frame(body, bg="#c0c0c0"); r1.pack(fill=tk.X, pady=3)
        tk.Label(r1, text="Calibration Frames:", bg="#c0c0c0", font=self.font, width=26, anchor="w").pack(side=tk.LEFT)
        tk.Entry(r1, textvariable=self._var_calib, width=8, font=self.font).pack(side=tk.LEFT)

        r2 = tk.Frame(body, bg="#c0c0c0"); r2.pack(fill=tk.X, pady=3)
        tk.Label(r2, text="Inter-Camera Delay (s):", bg="#c0c0c0", font=self.font, width=26, anchor="w").pack(side=tk.LEFT)
        tk.Entry(r2, textvariable=self._var_delay, width=8, font=self.font).pack(side=tk.LEFT)

        r3 = tk.Frame(body, bg="#c0c0c0"); r3.pack(fill=tk.X, pady=3)
        tk.Checkbutton(r3, text="DINO Only (disable YOLO hybrid)",
                       variable=self._var_dino_only, bg="#c0c0c0", font=self.font).pack(side=tk.LEFT)

        info_lines = [
            f"Device Tier : {tier}",
            f"Cameras     : {cam_count}",
            f"DINO model  : {self._data.get('dino_model_version', '')}",
            f"YOLO model  : {self._data.get('yolo_model_version', '')}",
            "",
            "Click Finish to save settings and start the app.",
        ]
        tk.Label(body, text="\n".join(info_lines), bg="#c0c0c0",
                 font=("Courier New", 9), justify=tk.LEFT,
                 relief=tk.SUNKEN, bd=1, padx=6, pady=4).pack(fill=tk.X, pady=(10, 0))

    def _validate_step(self, step: int) -> bool:
        if step == 1:
            if not self._cameras:
                messagebox.showwarning("Cameras", "Add at least one camera.", parent=self)
                return False
        return True

    def _collect_step(self, step: int):
        if step == 3:
            self._data["default_thresholds"] = {
                "ALERT": float(self._var_alert.get()),
                "CRITICAL": float(self._var_crit.get()),
                "HIGH_PRIORITY_ALERT": float(self._var_hp_alert.get()),
                "HIGH_PRIORITY_CRITICAL": float(self._var_hp_crit.get()),
            }
        elif step == 2:
            receivers_raw = self._entry_receivers.get().strip()
            receivers = [r.strip() for r in receivers_raw.split(",") if r.strip()]
            self._data["email"] = {
                "sender": self._entry_sender.get().strip(),
                "app_password": self._entry_password.get().strip(),
                "receivers": receivers,
            }

    def _finish(self):
        try:
            calib_frames = int(self._var_calib.get())
        except Exception:
            calib_frames = 100
        try:
            inter_delay = float(self._var_delay.get())
        except Exception:
            inter_delay = 0.5

        self._data["total_calibration_frames"] = calib_frames
        self._data["inter_camera_delay"] = inter_delay
        self._data["dino_only"] = bool(self._var_dino_only.get())

        cameras = []
        for cam in self._cameras:
            raw_src = cam["source_var"].get().strip()
            src = normalize_source(raw_src)
            mask_path = os.path.join("Assets", "masks", f"camera_{cam['id']}_mask.png")
            cameras.append({
                "id": cam["id"],
                "name": cam["name_var"].get().strip(),
                "source": src,
                "mask_path": mask_path,
                "thresholds": None,
                "allowed_error": 1,
                "anomaly_report_wait": 10,
            })
        self._data["cameras"] = cameras

        # Generate default masks
        os.makedirs(os.path.join("Assets", "masks"), exist_ok=True)
        for cam_cfg in cameras:
            self._generate_mask(cam_cfg)

        self._data["setup_complete"] = True
        save_settings(self._data)

        self.grab_release()
        self.destroy()

    def _generate_mask(self, cam_cfg: dict):
        src = cam_cfg["source"]
        path = cam_cfg["mask_path"]
        w, h = 640, 480
        try:
            cap = cv2.VideoCapture(src)
            if cap.isOpened():
                fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if fw > 0 and fh > 0:
                    w, h = fw, fh
            cap.release()
        except Exception:
            pass

        mask = np.zeros((h, w, 3), dtype=np.uint8)
        mask[:] = gb.medium_priority_region
        cv2.imwrite(path, mask)

    def _on_close(self):
        if messagebox.askyesno("Cancel Setup", "Cancel setup? App will not start.", parent=self):
            self.grab_release()
            self.destroy()

    def _center(self):
        self.update_idletasks()
        w = self.winfo_width()
        h = self.winfo_height()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")

