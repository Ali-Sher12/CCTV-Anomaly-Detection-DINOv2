"""calibration_session.py — Per-camera calibration session state machine.

Implements CAL-10 to CAL-24:
States:
  NOT_STARTED    : Session created, 0 frames collected.
  COLLECTING     : Taking one frame per slot of this camera.
  PAUSED         : Collection stopped, uncommitted frames kept.
  TARGET_REACHED : Target frame count reached, collection stopped automatically.

Four actions: Start, Pause, Confirm, Restart.
"""

from enum import Enum
import numpy as np

MIN_CAL_FRAMES = 10


class SessionState(Enum):
    NOT_STARTED = "NOT_STARTED"
    COLLECTING = "COLLECTING"
    PAUSED = "PAUSED"
    TARGET_REACHED = "TARGET_REACHED"


class CalibrationSession:
    """Manages an in-progress calibration collection session for one camera."""

    def __init__(self, target_frames: int = 100, is_initial: bool = False):
        self.target_frames = max(1, int(target_frames))
        self.is_initial = is_initial
        self.state = SessionState.NOT_STARTED
        self.session_frames: list[np.ndarray] = []

    @property
    def collected_count(self) -> int:
        return len(self.session_frames)

    @property
    def can_start(self) -> bool:
        return self.state in (SessionState.NOT_STARTED, SessionState.PAUSED)

    @property
    def can_pause(self) -> bool:
        return self.state == SessionState.COLLECTING

    @property
    def can_confirm(self) -> bool:
        if self.state == SessionState.TARGET_REACHED:
            return True
        if self.state == SessionState.PAUSED and len(self.session_frames) >= MIN_CAL_FRAMES:
            return True
        return False

    @property
    def can_restart(self) -> bool:
        return self.state in (SessionState.COLLECTING, SessionState.PAUSED, SessionState.TARGET_REACHED)

    def start(self) -> bool:
        if not self.can_start:
            return False
        self.state = SessionState.COLLECTING
        return True

    def pause(self) -> bool:
        if not self.can_pause:
            return False
        self.state = SessionState.PAUSED
        return True

    def restart(self) -> bool:
        if not self.can_restart:
            return False
        self.session_frames.clear()
        self.state = SessionState.NOT_STARTED
        return True

    def add_frame(self, embedding: np.ndarray) -> bool:
        """Add an embedding frame if collecting. Returns True if target reached."""
        if self.state != SessionState.COLLECTING:
            return False

        self.session_frames.append(embedding)
        if len(self.session_frames) >= self.target_frames:
            self.state = SessionState.TARGET_REACHED
            return True
        return False

    def confirm(self, existing_store: list[np.ndarray]) -> list[np.ndarray]:
        """Appends uncommitted session frames to existing calibration store."""
        if not self.can_confirm:
            raise RuntimeError(f"Cannot confirm session in state {self.state} with {len(self.session_frames)} frames (min {MIN_CAL_FRAMES})")
        
        new_store = list(existing_store) + list(self.session_frames)
        self.session_frames.clear()
        return new_store

