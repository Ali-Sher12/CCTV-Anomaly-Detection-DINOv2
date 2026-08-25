import tkinter as tk
import os
from PIL import Image, ImageTk


class LoadingScreen:
    """A standalone loading splash window that stays on screen during initial model/system setup."""

    def __init__(self, root):
        self.root = root
        self.splash = tk.Toplevel(root)
        self.splash.title("Passive Anomaly Detector PLUS +")
        self.splash.configure(bg='#c0c0c0', bd=3, relief=tk.RAISED)
        self.splash.overrideredirect(True)

        w, h = 420, 240
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.splash.geometry(f"{w}x{h}+{x}+{y}")

        # Top banner
        tk.Label(self.splash, text="Passive Anomaly Detector PLUS +",
                 bg='#000080', fg='white', font=('MS Sans Serif', 10, 'bold'),
                 padx=10, pady=4).pack(fill=tk.X)

        content = tk.Frame(self.splash, bg='#c0c0c0')
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Logo
        logo_path = "Assets/logo.png"
        if os.path.exists(logo_path):
            try:
                img = Image.open(logo_path)
                img.thumbnail((120, 80), Image.Resampling.LANCZOS)
                self._photo = ImageTk.PhotoImage(img)
                tk.Label(content, image=self._photo, bg='#c0c0c0').pack(pady=(5, 5))
            except Exception:
                pass

        tk.Label(content, text="Initializing models and camera feed...",
                 bg='#c0c0c0', font=('MS Sans Serif', 9)).pack(pady=5)
        self.status_lbl = tk.Label(content, text="Please wait...",
                                   bg='#c0c0c0', font=('MS Sans Serif', 8, 'italic'),
                                   fg='#444444')
        self.status_lbl.pack()

        self.root.update()

    def set_status(self, text):
        try:
            self.status_lbl.config(text=text)
            self.root.update()
        except Exception:
            pass

    def close(self):
        try:
            self.splash.destroy()
        except Exception:
            pass
