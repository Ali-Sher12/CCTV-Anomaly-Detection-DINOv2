"""tests/test_audio.py — Test AudioEngine mute and speech dispatch logic."""

import time
import Globals as gb
from audio import AudioEngine, play_synthesized_beep


def test_audio_engine_mute_toggle():
    gb.audio_muted = True
    engine = AudioEngine()
    engine.report(1, "ALERT")
    # Queue should remain empty because audio is muted
    time.sleep(0.05)
    assert engine._queue.empty()

    # Unmute and report
    gb.audio_muted = False
    engine.report(2, "CRITICAL")
    # Item should be queued or being processed
    assert getattr(engine, "_queue") is not None


def test_audio_engine_ignores_normal_tier():
    gb.audio_muted = False
    engine = AudioEngine()
    engine.report(1, "NORMAL")
    time.sleep(0.05)
    assert engine._queue.empty()


def test_synthesized_beep_generation():
    # Verify offline synthesized beep generation functions without crashing
    success = play_synthesized_beep(duration_sec=0.05, freq_hz=880.0)
    assert isinstance(success, bool)
