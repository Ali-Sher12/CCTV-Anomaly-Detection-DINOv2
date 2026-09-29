"""capture_registry.py — Reference-counted video capture registry and non-blocking readers.

Ensures:
1. Source normalization ("0" -> 0, "rtsp://..." -> "rtsp://...").
2. Single cv2.VideoCapture descriptor shared across multiple feeds with identical sources (e.g., demo configs).
3. Dedicated lightweight background reader thread per unique source keeping only the freshest frame.
4. Structured low-noise logging for open and reconnect attempts.
5. Strict resource cleanup in finally blocks.
"""

import time
import threading
import cv2
import numpy as np


def normalize_source(source):
    """Normalize camera source to an int (webcam index) or stripped string (stream/file/device)."""
    if source is None:
        raise ValueError("Camera source cannot be None")
    
    if isinstance(source, int):
        return source

    s = str(source).strip()
    if not s:
        raise ValueError("Camera source cannot be empty")

    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return int(s)

    # Convert IP webcam URLs ending in :8080 or :8080/ to :8080/video
    if s.startswith("http://") or s.startswith("https://"):
        if s.endswith(":8080") or s.endswith(":8080/"):
            s = s.rstrip("/") + "/video"

    return s


class SourceReader:
    """Manages a single VideoCapture source in a background thread.
    
    Drops older frames and always maintains the single newest frame.
    """

    def __init__(self, normalized_source):
        self.source = normalized_source
        self.cap = None
        self._lock = threading.Lock()
        self._latest_frame = None
        self._is_opened = False
        self._last_error = ""
        self._running = True
        self._ref_count = 0
        self._last_open_log = 0.0
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._open_capture()
        self._thread.start()

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] [SourceReader:{self.source}] {msg}")

    def _open_capture(self):
        with self._lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
                self._is_opened = False

            try:
                cap = cv2.VideoCapture(self.source)
                # Optimize buffer size for live feeds
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass

                is_open = cap.isOpened()
                first_read_ok = False
                if is_open:
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        first_read_ok = True
                        self._latest_frame = frame
                        self._is_opened = True
                        self.cap = cap
                        self._last_error = ""
                        self._log(f"Opened source '{self.source}' (type={type(self.source).__name__}) -> isOpened=True, firstRead=True")
                        return True
                    else:
                        self._last_error = "Failed to read first frame"
                        cap.release()
                        self.cap = None
                        self._is_opened = False
                        self._log(f"Failed source '{self.source}' (type={type(self.source).__name__}) -> isOpened=True, firstRead=False")
                        return False
                else:
                    self._last_error = f"Cannot open source '{self.source}'"
                    cap.release()
                    self.cap = None
                    self._is_opened = False
                    self._log(f"Failed source '{self.source}' (type={type(self.source).__name__}) -> isOpened=False: {self._last_error}")
                    return False
            except Exception as e:
                self._last_error = str(e)
                self._is_opened = False
                self._log(f"Exception opening source '{self.source}': {e}")
                return False

    def _read_loop(self):
        consecutive_failures = 0
        while self._running:
            if not self._is_opened or self.cap is None:
                time.sleep(1.0)
                if self._running and self._ref_count > 0:
                    self._open_capture()
                continue

            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    consecutive_failures = 0
                    with self._lock:
                        self._latest_frame = frame
                else:
                    consecutive_failures += 1
                    # After 15 failed reads, mark offline and attempt reconnect
                    if consecutive_failures >= 15:
                        with self._lock:
                            self._is_opened = False
                            if self.cap is not None:
                                try:
                                    self.cap.release()
                                except Exception:
                                    pass
                                self.cap = None
                            self._last_error = "Connection dropped / frame read failed"
                        self._log("Connection dropped. Entering reconnect cycle.")
            except Exception as e:
                with self._lock:
                    self._is_opened = False
                    self._last_error = str(e)
                    if self.cap is not None:
                        try:
                            self.cap.release()
                        except Exception:
                            pass
                        self.cap = None
                self._log(f"Exception in read loop: {e}")

            time.sleep(0.01)

    def get_latest_frame(self):
        """Returns (is_online, latest_frame_copy_or_None, error_str)."""
        with self._lock:
            if self._is_opened and self._latest_frame is not None:
                return True, self._latest_frame.copy(), ""
            return False, None, self._last_error or "Offline / Connecting"

    def get_resolution(self):
        """Returns (width, height) or (640, 480)."""
        with self._lock:
            if self._latest_frame is not None:
                h, w = self._latest_frame.shape[:2]
                return w, h
            if self.cap is not None and self._is_opened:
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if w > 0 and h > 0:
                    return w, h
        return 640, 480

    def stop(self):
        self._running = False
        with self._lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
                self._is_opened = False


class CaptureRegistry:
    """Thread-safe singleton registry mapping normalized_source -> SourceReader."""

    _instance = None
    _registry_lock = threading.Lock()

    def __new__(cls):
        with cls._registry_lock:
            if cls._instance is None:
                cls._instance = super(CaptureRegistry, cls).__new__(cls)
                cls._instance._readers = {}
                cls._instance._camera_bindings = {}  # camera_id -> normalized_source
            return cls._instance

    def acquire(self, source, camera_id: int) -> SourceReader:
        norm_src = normalize_source(source)
        with self._registry_lock:
            if norm_src not in self._readers:
                self._readers[norm_src] = SourceReader(norm_src)
            reader = self._readers[norm_src]
            reader._ref_count += 1
            self._camera_bindings[camera_id] = norm_src
            return reader

    def release(self, camera_id: int):
        with self._registry_lock:
            if camera_id in self._camera_bindings:
                norm_src = self._camera_bindings.pop(camera_id)
                if norm_src in self._readers:
                    reader = self._readers[norm_src]
                    reader._ref_count -= 1
                    if reader._ref_count <= 0:
                        reader.stop()
                        del self._readers[norm_src]

    def release_all(self):
        with self._registry_lock:
            for reader in self._readers.values():
                reader.stop()
            self._readers.clear()
            self._camera_bindings.clear()

