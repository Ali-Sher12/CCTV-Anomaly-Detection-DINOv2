"""watchdog.py — Standalone watchdog process and heartbeat monitor.

Monitors Assets/heartbeat.txt and Assets/app.pid.
If the heartbeat goes stale:
1. Speaks an audible alert.
2. Gracefully terminates any hung app process.
3. Automatically restarts the app (subject to a restart-loop guard).
4. Dispatches email notifications if configured.

Strict isolation (WD-01):
- Does NOT import any module from the main application.
- Uses only Python standard library + psutil + pyttsx3.
"""

import os
import sys
import json
import time
import shutil
import smtplib
import subprocess
import datetime
import tempfile
import urllib.request
import urllib.parse
import wave
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


def load_watchdog_config(config_path: str | Path) -> dict:
    """Load and strictly validate watchdog configuration (WD-04)."""
    p = Path(config_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Watchdog config file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Watchdog config must be a JSON object")

    # Required fields
    app_install_path = data.get("app_install_path")
    if not app_install_path:
        raise ValueError("Config validation failed: 'app_install_path' is required")

    app_entrypoint = data.get("app_entrypoint")
    if not app_entrypoint or not isinstance(app_entrypoint, list):
        raise ValueError("Config validation failed: 'app_entrypoint' must be a non-empty list")

    # Defaults
    heartbeat_interval = float(data.get("heartbeat_interval", 5))
    stale_after = float(data.get("stale_after", 60))
    check_interval = float(data.get("check_interval", 20))
    startup_grace = float(data.get("startup_grace", 120))
    restart_grace = float(data.get("restart_grace", 150))
    terminate_timeout = float(data.get("terminate_timeout", 10))
    max_restarts = int(data.get("max_restarts", 3))
    restart_window = float(data.get("restart_window", 600))
    heartbeat_path = data.get("heartbeat_path", "Assets/heartbeat.txt")
    pid_path = data.get("pid_path", "Assets/app.pid")
    email_cfg = data.get("email")

    # Validation: stale_after >= 3 * heartbeat_interval (WD-04)
    if stale_after < 3 * heartbeat_interval:
        raise ValueError(
            f"Config validation failed: 'stale_after' ({stale_after}s) must be at least "
            f"3x 'heartbeat_interval' ({heartbeat_interval}s = {3 * heartbeat_interval}s)"
        )

    return {
        "heartbeat_interval": heartbeat_interval,
        "stale_after": stale_after,
        "check_interval": check_interval,
        "startup_grace": startup_grace,
        "restart_grace": restart_grace,
        "terminate_timeout": terminate_timeout,
        "max_restarts": max_restarts,
        "restart_window": restart_window,
        "app_install_path": str(Path(app_install_path).resolve()),
        "app_entrypoint": app_entrypoint,
        "heartbeat_path": heartbeat_path,
        "pid_path": pid_path,
        "email": email_cfg,
    }


def is_heartbeat_stale(heartbeat_path: Path, stale_after: float, current_time: float | None = None) -> tuple[bool, str]:
    """Check if the heartbeat file is missing or older than stale_after seconds."""
    now = current_time if current_time is not None else time.time()
    if not heartbeat_path.exists():
        return True, "Heartbeat file does not exist"

    try:
        content_ts = None
        try:
            txt = heartbeat_path.read_text(encoding="utf-8").strip()
            if txt:
                content_ts = float(txt)
        except Exception:
            pass

        if content_ts is not None:
            effective_ts = content_ts
        else:
            effective_ts = heartbeat_path.stat().st_mtime

        age = now - effective_ts

        if age > stale_after:
            return True, f"Heartbeat is stale ({age:.1f}s old > {stale_after}s threshold)"
        return False, f"Heartbeat healthy ({age:.1f}s old)"
    except Exception as e:
        return True, f"Error reading heartbeat file: {e}"


def load_restart_log(log_path: Path) -> list[float]:
    """Load list of past restart epoch timestamps from JSON (WD-06, WD-12)."""
    if not log_path.exists():
        return []
    try:
        data = json.loads(log_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [float(x) for x in data if isinstance(x, (int, float))]
    except Exception:
        pass
    return []


def save_restart_log(log_path: Path, restarts: list[float]):
    """Atomically persist restart history (WD-06)."""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = log_path.parent / f"watchdog_restart_log_{os.getpid()}.tmp"
        tmp.write_text(json.dumps(restarts, indent=2), encoding="utf-8")
        os.replace(tmp, log_path)
    except Exception as e:
        print(f"[Watchdog] Warning: failed to save restart log: {e}")


def clean_and_count_restarts(restarts: list[float], restart_window: float, current_time: float) -> list[float]:
    """Filter out restart timestamps outside the rolling window (WD-07)."""
    return [t for t in restarts if (current_time - t) <= restart_window]


def should_trigger_loop_guard(recent_restarts_count: int, max_restarts: int) -> bool:
    """Check if restart count in window meets or exceeds max_restarts."""
    return recent_restarts_count >= max_restarts


def append_watchdog_log(log_path: Path, message: str, current_time: float | None = None):
    """Append structured log message with ISO-8601 timestamp (WD-11)."""
    now_dt = datetime.datetime.fromtimestamp(
        current_time if current_time is not None else time.time(),
        tz=datetime.timezone.utc
    )
    iso_ts = now_dt.isoformat()
    line = f"[{iso_ts}] {message}\n"
    print(line.strip())
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        print(f"[Watchdog] Warning: could not write log file: {e}")


def _play_watchdog_audio(filepath: str) -> bool:
    for player in ("pw-play", "paplay", "ffplay", "aplay"):
        if shutil.which(player):
            cmd = [player, filepath]
            if player == "ffplay":
                cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", filepath]
            try:
                res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8)
                if res.returncode == 0:
                    return True
            except Exception:
                pass
    return False


def _play_watchdog_beep(duration_sec: float = 0.4, freq_hz: float = 880.0) -> bool:
    sr = 22050
    num_samples = int(sr * duration_sec)
    import math
    import struct
    waveform = bytearray()
    for i in range(num_samples):
        val = int(32767.0 * math.sin(2.0 * math.pi * freq_hz * (i / sr)))
        # clamp
        val = max(-32768, min(32767, val))
        waveform.extend(struct.pack("<h", val))

    temp_wav = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_wav = f.name
        with wave.open(temp_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(bytes(waveform))
        return _play_watchdog_audio(temp_wav)
    except Exception:
        return False
    finally:
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception:
                pass


def speak_alert(text: str) -> bool:
    """Speak text using pyttsx3, CLI speech, online TTS, or synthesized audio (WD-05, WD-07)."""
    if pyttsx3 is not None:
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            return True
        except Exception:
            pass

    for cmd in ("espeak-ng", "espeak", "spd-say"):
        if shutil.which(cmd):
            try:
                subprocess.run([cmd, text], check=True, timeout=5)
                return True
            except Exception:
                pass

    # Online TTS fallback
    try:
        url = ("https://translate.google.com/translate_tts?ie=UTF-8&tl=en&client=tw-ob&q="
               + urllib.parse.quote(text))
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = resp.read()
        temp_mp3 = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                temp_mp3 = f.name
                f.write(data)
            if _play_watchdog_audio(temp_mp3):
                return True
        finally:
            if temp_mp3 and os.path.exists(temp_mp3):
                try:
                    os.remove(temp_mp3)
                except Exception:
                    pass
    except Exception:
        pass

    # Synthesized audio beep fallback
    if _play_watchdog_beep():
        return True

    # Audible terminal bell
    print("\a", end="", flush=True)
    return False



def send_watchdog_email(email_cfg: dict | None, subject: str, body: str) -> bool:
    """Send alert email via standard library smtplib (WD-08)."""
    if not email_cfg or not isinstance(email_cfg, dict):
        return False

    sender = email_cfg.get("sender")
    password = email_cfg.get("app_password")
    receivers = email_cfg.get("receivers") or []
    smtp_server = email_cfg.get("smtp_server", "smtp.gmail.com")
    smtp_port = int(email_cfg.get("smtp_port", 587))

    if not sender or not password or not receivers:
        return False

    try:
        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = ", ".join(receivers)
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP(smtp_server, smtp_port, timeout=10)
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receivers, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"[Watchdog] Email send failed: {e}")
        return False


def terminate_process(pid_path: Path, timeout: float = 10.0, expected_entrypoint: list[str] | None = None) -> bool:
    """Read app.pid and terminate existing process gracefully before killing (WD-06)."""
    if not pid_path.exists():
        return False

    try:
        raw = pid_path.read_text(encoding="utf-8").strip()
        if not raw.isdigit():
            return False
        pid = int(raw)
    except Exception:
        return False

    if psutil is None:
        # Fallback if psutil is unavailable
        try:
            os.kill(pid, 15) # SIGTERM
            time.sleep(1.0)
            return True
        except Exception:
            return False

    try:
        if not psutil.pid_exists(pid):
            return False

        proc = psutil.Process(pid)
        # Verify process is likely our app
        try:
            cmdline = " ".join(proc.cmdline())
            if expected_entrypoint:
                entry_match = any(e in cmdline for e in expected_entrypoint if len(e) > 3)
                if not entry_match and "python" not in proc.name().lower():
                    # Mismatched process, skip killing wrong PID
                    return False
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except psutil.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)
        return True
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False
    except Exception as e:
        print(f"[Watchdog] Process termination error on PID {pid}: {e}")
        return False


class WatchdogService:
    """State machine and loop controller for the watchdog service."""

    def __init__(self, config: dict, clock=None, sleeper=None):
        self.cfg = config
        self.clock = clock if clock is not None else time.time
        self.sleeper = sleeper if sleeper is not None else time.sleep

        self.install_dir = Path(config["app_install_path"])
        self.heartbeat_path = self.install_dir / config["heartbeat_path"]
        self.pid_path = self.install_dir / config["pid_path"]
        self.log_path = self.install_dir / "Assets" / "watchdog_log.txt"
        self.restart_log_path = self.install_dir / "Assets" / "watchdog_restart_log.json"

        # State tracking (WD-09)
        self.in_down_episode = False
        self.loop_guard_tripped = False
        self._running = True

    def log(self, msg: str):
        append_watchdog_log(self.log_path, msg, current_time=self.clock())

    def run_single_check(self) -> str:
        """Run one staleness evaluation and reaction cycle. Returns state string."""
        now = self.clock()
        stale, reason = is_heartbeat_stale(self.heartbeat_path, self.cfg["stale_after"], current_time=now)

        if not stale:
            if self.in_down_episode:
                # Recovery transition (WD-10)
                self.in_down_episode = False
                self.loop_guard_tripped = False
                self.log(f"RECOVERY: App heartbeat restored ({reason}).")
                speak_alert("Passive anomaly detector has recovered.")
            return "HEALTHY"

        # Down episode active
        if not self.in_down_episode:
            # First stale detection of episode (WD-05, WD-08, WD-09)
            self.in_down_episode = True
            self.log(f"STALE DETECTED: {reason}. Beginning recovery procedure.")
            speak_alert("Anomaly detector is not responding.")

            # Send first-down email
            if self.cfg.get("email"):
                send_watchdog_email(
                    self.cfg["email"],
                    "[ALERT] Passive Anomaly Detector Is Not Responding",
                    f"Watchdog detected a stale heartbeat at {datetime.datetime.fromtimestamp(now, tz=datetime.timezone.utc).isoformat()}.\n"
                    f"Reason: {reason}\n"
                    f"Attempting automatic application recovery."
                )

        # Evaluate restart-loop guard (WD-07)
        past_restarts = load_restart_log(self.restart_log_path)
        recent_restarts = clean_and_count_restarts(past_restarts, self.cfg["restart_window"], now)

        if should_trigger_loop_guard(len(recent_restarts), self.cfg["max_restarts"]):
            if not self.loop_guard_tripped:
                self.loop_guard_tripped = True
                self.log(f"RESTART-LOOP GUARD TRIPPED: {len(recent_restarts)} restarts in past {self.cfg['restart_window']}s. Halting auto-restarts.")
                speak_alert("Multiple restart attempts have failed. Manual attention is needed.")

                if self.cfg.get("email"):
                    send_watchdog_email(
                        self.cfg["email"],
                        "[CRITICAL] Watchdog Restart-Loop Guard Tripped",
                        f"Passive Anomaly Detector has failed {len(recent_restarts)} restart attempts in {self.cfg['restart_window']}s.\n"
                        f"Automatic restarts are suspended. Manual human attention is required."
                    )
            return "LOOP_GUARD_TRIPPED"

        # Run restart procedure (WD-06)
        self.log("RESTART PROCEDURE: Terminating existing process if active...")
        terminated = terminate_process(
            self.pid_path,
            timeout=self.cfg["terminate_timeout"],
            expected_entrypoint=self.cfg["app_entrypoint"]
        )
        if terminated:
            self.log("RESTART PROCEDURE: Stale process terminated.")

        self.log(f"RESTART PROCEDURE: Launching {self.cfg['app_entrypoint']} in {self.cfg['app_install_path']}...")
        try:
            subprocess.Popen(
                self.cfg["app_entrypoint"],
                cwd=self.cfg["app_install_path"]
            )
            self.log("RESTART PROCEDURE: Fresh instance launched successfully.")
        except Exception as e:
            self.log(f"RESTART PROCEDURE ERROR: Failed to launch process: {e}")
            return "LAUNCH_ERROR"

        # Record restart timestamp
        recent_restarts.append(now)
        save_restart_log(self.restart_log_path, recent_restarts)

        # Wait restart_grace seconds before resuming checks
        self.log(f"RESTART PROCEDURE: Pausing checks for restart grace period ({self.cfg['restart_grace']}s)...")
        self.sleeper(self.cfg["restart_grace"])
        return "RESTARTED"

    def run_forever(self):
        """Main service entry loop (WD-02, WD-03, WD-12)."""
        self.log(f"Watchdog service started. Startup grace period: {self.cfg['startup_grace']}s.")
        self.sleeper(self.cfg["startup_grace"])

        while self._running:
            try:
                self.run_single_check()
            except Exception as e:
                self.log(f"Watchdog loop check error: {e}")

            try:
                self.sleeper(self.cfg["check_interval"])
            except Exception:
                pass


def main():
    config_file = Path(__file__).resolve().parent / "watchdog_config.json"
    try:
        config = load_watchdog_config(config_file)
    except Exception as e:
        print(f"[Watchdog Fatal] Configuration error: {e}")
        sys.exit(1)

    service = WatchdogService(config)
    service.run_forever()


if __name__ == "__main__":
    main()

