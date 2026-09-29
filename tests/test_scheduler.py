import pytest
from scheduler import StaggeredScheduler

class FakeClock:
    def __init__(self, start=0.0):
        self.time = start

    def __call__(self):
        return self.time

    def advance(self, dt):
        self.time += dt


def test_scheduler_round_robin_timing():
    clock = FakeClock(start=0.0)
    scheduler = StaggeredScheduler(inter_camera_delay=0.5, clock=clock)

    # 3 cameras, all eligible
    eligible = [True, True, True]

    # At t=0.0, Cam 0 is due
    idx = scheduler.get_next_camera_for_slot(eligible)
    assert idx == 0
    scheduler.notify_inference_started()
    clock.advance(0.1) # inference takes 0.1s
    scheduler.notify_inference_finished(0.1)

    # At t=0.1, Cam 1 is not due yet (delay=0.5)
    assert scheduler.get_next_camera_for_slot(eligible) is None

    # Advance to t=0.5
    clock.advance(0.4) # total 0.5s
    idx = scheduler.get_next_camera_for_slot(eligible)
    assert idx == 1
    scheduler.notify_inference_started()
    clock.advance(0.1)
    scheduler.notify_inference_finished(0.1)

    # Advance to t=1.0
    clock.advance(0.4)
    idx = scheduler.get_next_camera_for_slot(eligible)
    assert idx == 2
    scheduler.notify_inference_started()
    clock.advance(0.1)
    scheduler.notify_inference_finished(0.1)

    # Advance to t=1.5 -> Cam 0 again
    clock.advance(0.4)
    idx = scheduler.get_next_camera_for_slot(eligible)
    assert idx == 0


def test_skip_ineligible_cameras():
    clock = FakeClock(start=0.0)
    scheduler = StaggeredScheduler(inter_camera_delay=0.5, clock=clock)

    # Cam 0 eligible, Cam 1 offline (False), Cam 2 eligible
    eligible = [True, False, True]

    # Slot 1: Cam 0
    idx = scheduler.get_next_camera_for_slot(eligible)
    assert idx == 0
    scheduler.notify_inference_started()
    scheduler.notify_inference_finished(0.05)

    # Advance to next slot
    clock.advance(0.5)
    # Slot 2: Cam 1 is skipped, Cam 2 takes it immediately
    idx = scheduler.get_next_camera_for_slot(eligible)
    assert idx == 2
    scheduler.notify_inference_started()
    scheduler.notify_inference_finished(0.05)


def test_slow_inference_no_burst():
    clock = FakeClock(start=0.0)
    scheduler = StaggeredScheduler(inter_camera_delay=0.5, clock=clock)
    eligible = [True, True, True]

    # Cam 0 starts at t=0
    idx = scheduler.get_next_camera_for_slot(eligible)
    assert idx == 0
    scheduler.notify_inference_started()

    # Inference takes 1.8s (> 0.5s)
    clock.advance(1.8)
    scheduler.notify_inference_finished(1.8)
    assert "Detection slower than delay" in scheduler.slow_detection_warning

    # Immediately after finishing at t=1.8, scheduler should NOT burst.
    # Next slot should be anchored to t=1.8 + 0.5 = 2.3
    assert scheduler.get_next_camera_for_slot(eligible) is None

    # Advance by 0.5s to t=2.3
    clock.advance(0.5)
    idx = scheduler.get_next_camera_for_slot(eligible)
    assert idx == 1  # Cam 1 takes next slot calmly


def test_scheduler_10s_exact_sequence():
    clock = FakeClock(start=0.0)
    scheduler = StaggeredScheduler(inter_camera_delay=0.5, clock=clock)
    eligible = [True, True, True]

    history = []
    dt = 0.01  # 10ms simulation resolution
    total_ticks = int(10.0 / dt)

    for _ in range(total_ticks):
        cam_idx = scheduler.get_next_camera_for_slot(eligible)
        if cam_idx is not None:
            history.append((round(clock.time, 2), cam_idx))
            scheduler.notify_inference_started()
            # Fast inference finished in current tick
            scheduler.notify_inference_finished(0.02)
        clock.advance(dt)

    assert len(history) == 20  # 10s / 0.5s = 20 detections
    for k, (t, cam) in enumerate(history):
        expected_t = round(k * 0.5, 2)
        expected_cam = k % 3
        assert abs(t - expected_t) < 0.02, f"Step {k}: expected time {expected_t}, got {t}"
        assert cam == expected_cam, f"Step {k}: expected cam {expected_cam}, got {cam}"

