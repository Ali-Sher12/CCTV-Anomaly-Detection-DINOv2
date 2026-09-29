"""audio.py — serialised TTS queue, camera-aware anomaly messages.

report(camera_id, tier) pushes "ANOMALY DETECTED AT CAMERA N" onto a
queue for ALERT or CRITICAL tiers.  A single daemon worker thread drains
the queue one item at a time so concurrent multi-camera alerts never
overlap or get dropped.
"""

import queue
import threading
import shutil
import subprocess
import pyttsx3
import Globals as gb


class AudioEngine:

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._warned_missing = False
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def report(self, camera_id: int, tier: str):
        """Queue a speech alert for ALERT or CRITICAL tiers only."""
        if tier in ("ALERT", "CRITICAL"):
            self._queue.put(f"ANOMALY DETECTED AT CAMERA {camera_id}")

    # ------------------------------------------------------------------
    # Worker (background thread — never touches Tk widgets)
    # ------------------------------------------------------------------

    def _speak_text(self, text: str) -> bool:
        """Attempt to speak using pyttsx3 or CLI speech engines."""
        # 1. Try pyttsx3
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            return True
        except Exception:
            pass

        # 2. Try CLI tools if pyttsx3 fails on Linux
        for cmd in ("espeak-ng", "espeak", "spd-say"):
            if shutil.which(cmd):
                try:
                    subprocess.run([cmd, text], check=True, timeout=5)
                    return True
                except Exception:
                    pass

        return False

    def _run(self):
        while True:
            text = self._queue.get()   # blocks until an item is available
            try:
                spoken = self._speak_text(text)
                if not spoken:
                    # Fallback audible terminal bell so alert is not silent
                    print("\a", end="", flush=True)
                    if not self._warned_missing:
                        self._warned_missing = True
                        msg = ("[Audio] TTS engine unavailable. On Linux, please run: "
                               "'sudo pacman -S espeak-ng' (or 'sudo apt-get install espeak-ng')")
                        print(msg)
                        if gb.gui:
                            gb.gui.log(msg)
            except Exception as e:
                print(f"[Audio Error] {e}")
            finally:
                self._queue.task_done()

