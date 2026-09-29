"""audio.py — serialised TTS queue, camera-aware anomaly messages.

report(camera_id, tier) pushes "ANOMALY DETECTED AT CAMERA N" onto a
queue for ALERT or CRITICAL tiers. A single daemon worker thread drains
the queue one item at a time so concurrent multi-camera alerts never
overlap or get dropped.

Supports Windows (SAPI5 via pyttsx3) and Linux (pyttsx3/espeak-ng/spd-say,
online TTS fallback via pw-play/paplay/ffplay/aplay, synthesized WAV beep,
and terminal bell fallback).
Supports dynamic mute/unmute via gb.audio_muted.
"""

import os
import queue
import shutil
import subprocess
import tempfile
import threading
import urllib.parse
import urllib.request
import wave
import numpy as np

import Globals as gb

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


def play_audio_file(filepath: str) -> bool:
    """Cross-platform audio file playback via available system audio players."""
    # Linux players (PipeWire, PulseAudio, FFmpeg, ALSA)
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


def play_synthesized_beep(duration_sec: float = 0.4, freq_hz: float = 880.0) -> bool:
    """Generate and play a pleasant audible beep waveform as offline audio fallback."""
    sr = 22050
    num_samples = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    # Sine wave with smooth fade-in and fade-out to prevent audio pops
    envelope = np.ones(num_samples)
    fade_len = int(sr * 0.02)
    if fade_len > 0 and 2 * fade_len < num_samples:
        envelope[:fade_len] = np.linspace(0, 1, fade_len)
        envelope[-fade_len:] = np.linspace(1, 0, fade_len)
    waveform = (np.sin(2 * np.pi * freq_hz * t) * envelope * 32767).astype(np.int16)

    temp_wav = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_wav = f.name
        with wave.open(temp_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(waveform.tobytes())
        return play_audio_file(temp_wav)
    except Exception:
        return False
    finally:
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception:
                pass


class AudioEngine:

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def report(self, camera_id: int, tier: str):
        """Queue a speech alert for ALERT or CRITICAL tiers only."""
        if getattr(gb, "audio_muted", False):
            return
        if tier in ("ALERT", "CRITICAL"):
            self._queue.put(f"ANOMALY DETECTED AT CAMERA {camera_id}")

    # ------------------------------------------------------------------
    # Worker (background thread — never touches Tk widgets)
    # ------------------------------------------------------------------

    def _speak_text(self, text: str) -> bool:
        """Attempt to speak using pyttsx3, CLI speech engines, online TTS, or synthesized audio."""
        if getattr(gb, "audio_muted", False):
            return True

        # 1. Try pyttsx3 (SAPI5 on Windows, NSSpeech on macOS, espeak on Linux if installed)
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

        # 2. Try CLI speech utilities if pyttsx3 fails or espeak library is missing
        for cmd in ("espeak-ng", "espeak", "spd-say"):
            if shutil.which(cmd):
                try:
                    subprocess.run([cmd, text], check=True, timeout=5)
                    return True
                except Exception:
                    pass

        # 3. Try high-quality online TTS fallback if system has internet connection
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
                if play_audio_file(temp_mp3):
                    return True
            finally:
                if temp_mp3 and os.path.exists(temp_mp3):
                    try:
                        os.remove(temp_mp3)
                    except Exception:
                        pass
        except Exception:
            pass

        # 4. Synthesized audio beep via system sound player (pw-play/paplay/ffplay/aplay)
        if play_synthesized_beep():
            return True

        # 5. Terminal bell fallback
        print("\a", end="", flush=True)
        return False

    def _run(self):
        while True:
            text = self._queue.get()   # blocks until an item is available
            try:
                if not getattr(gb, "audio_muted", False):
                    self._speak_text(text)
            except Exception as e:
                print(f"[Audio Error] {e}")
            finally:
                self._queue.task_done()
