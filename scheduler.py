"""scheduler.py — Staggered multi-camera scheduler with monotonic clock anchoring.

Implements SCH-04, SCH-05, SCH-06, SCH-07:
- Deterministic slot timing: next_slot = prev_slot + delay (anchored, no drift)
- Skips ineligible cameras without consuming time slots
- Prevents burst catch-ups if inference takes longer than delay
- Emits slow detection metrics/warnings
"""

import time


class StaggeredScheduler:
    """Pure scheduling logic engine, clock-injectable for unit testing."""

    def __init__(self, inter_camera_delay: float = 0.5, clock=None):
        self._delay = max(0.1, float(inter_camera_delay))
        self._clock = clock if clock is not None else time.monotonic
        self._current_slot_time = self._clock()
        self._next_slot_time = self._current_slot_time
        self._current_camera_index = 0
        self._inference_in_flight = False
        self._slow_detection_warning = ""

    @property
    def inter_camera_delay(self) -> float:
        return self._delay

    @inter_camera_delay.setter
    def inter_camera_delay(self, val: float):
        self._delay = max(0.1, float(val))

    @property
    def slow_detection_warning(self) -> str:
        return self._slow_detection_warning

    def notify_inference_started(self):
        self._inference_in_flight = True
        self._current_slot_time = self._next_slot_time

    def notify_inference_finished(self, duration: float):
        self._inference_in_flight = False
        now = self._clock()
        if duration > self._delay:
            self._slow_detection_warning = f"Detection slower than delay (measured {duration:.2f}s)"
            # Detection exceeded slot interval: anchor next slot from now to prevent burst
            self._next_slot_time = now + self._delay
        else:
            self._slow_detection_warning = ""
            # Anchored schedule: next slot is exactly previous slot time + delay
            nominal_next = self._current_slot_time + self._delay
            if nominal_next <= now:
                self._next_slot_time = now + self._delay
            else:
                self._next_slot_time = nominal_next

    def get_next_camera_for_slot(self, eligible_flags: list[bool]) -> int | None:
        """Determines if a slot is due and returns the eligible camera index, or None."""
        now = self._clock()
        if self._inference_in_flight:
            return None

        if now < self._next_slot_time:
            return None

        n = len(eligible_flags)
        if n == 0:
            return None

        # Find next eligible camera starting from current index
        for _ in range(n):
            idx = self._current_camera_index
            self._current_camera_index = (self._current_camera_index + 1) % n
            if eligible_flags[idx]:
                return idx

        return None

