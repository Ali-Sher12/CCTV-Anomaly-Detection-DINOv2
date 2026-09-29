import pytest
import numpy as np
from calibration_session import CalibrationSession, SessionState, MIN_CAL_FRAMES


def test_session_state_transitions():
    sess = CalibrationSession(target_frames=15, is_initial=True)
    assert sess.state == SessionState.NOT_STARTED
    assert sess.can_start
    assert not sess.can_pause
    assert not sess.can_confirm
    assert not sess.can_restart

    # Start
    assert sess.start()
    assert sess.state == SessionState.COLLECTING
    assert not sess.can_start
    assert sess.can_pause
    assert not sess.can_confirm
    assert sess.can_restart

    # Add 5 frames
    dummy_emb = np.zeros((196, 384), dtype=np.float32)
    for _ in range(5):
        sess.add_frame(dummy_emb)

    assert sess.collected_count == 5
    assert sess.state == SessionState.COLLECTING

    # Pause
    assert sess.pause()
    assert sess.state == SessionState.PAUSED
    assert sess.can_start
    assert not sess.can_pause
    # Cannot confirm yet because collected < MIN_CAL_FRAMES (10)
    assert not sess.can_confirm
    assert sess.can_restart

    # Resume
    assert sess.start()
    assert sess.state == SessionState.COLLECTING

    # Add 5 more frames (total 10)
    for _ in range(5):
        sess.add_frame(dummy_emb)
    sess.pause()
    assert sess.collected_count == 10
    # Now >= MIN_CAL_FRAMES, so confirm is enabled in PAUSED
    assert sess.can_confirm

    # Resume and collect up to target (15)
    sess.start()
    for _ in range(5):
        sess.add_frame(dummy_emb)

    assert sess.state == SessionState.TARGET_REACHED
    assert sess.collected_count == 15
    assert not sess.can_start
    assert not sess.can_pause
    assert sess.can_confirm
    assert sess.can_restart


def test_session_confirm_appends():
    sess = CalibrationSession(target_frames=10)
    sess.start()
    dummy_emb = np.zeros((196, 384), dtype=np.float32)
    for _ in range(10):
        sess.add_frame(dummy_emb)

    existing_store = [dummy_emb for _ in range(5)]
    updated_store = sess.confirm(existing_store)
    assert len(updated_store) == 15
    assert sess.collected_count == 0


def test_session_restart_discards():
    sess = CalibrationSession(target_frames=20)
    sess.start()
    dummy_emb = np.zeros((196, 384), dtype=np.float32)
    for _ in range(12):
        sess.add_frame(dummy_emb)

    assert sess.collected_count == 12
    sess.restart()
    assert sess.state == SessionState.NOT_STARTED
    assert sess.collected_count == 0
