"""Tests for badminton_tracker.detector."""

from __future__ import annotations

import numpy as np
import pytest

from badminton_tracker.detector import BallDetector, VelocitySmoother


def _blank_frame(w: int = 320, h: int = 240) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_detector_returns_no_shots_on_blank_video() -> None:
    """A pitch-black video should produce zero contour detections and zero shots."""
    det = BallDetector(use_background_subtractor=False)
    for i in range(60):
        det.process_frame(_blank_frame(), i)
    contacts = det.detect_shot_contacts()
    assert contacts == []
    # No frame should have a position because the mask is always empty.
    assert all(d.position is None for d in det.detections)


def test_detector_finds_bright_shuttle_centroid() -> None:
    """A single bright spot in a frame should be picked up as a detection."""
    det = BallDetector(use_background_subtractor=False, min_area=1.0)
    frame = _blank_frame()
    # Bright blob at (160, 120).
    import cv2
    cv2.circle(frame, (160, 120), 5, (255, 255, 255), -1)
    out = det.process_frame(frame, 0)
    assert out.position is not None
    px, py = out.position
    # Allow some slack — Kalman is initialised to the first measurement.
    assert abs(px - 160) < 5
    assert abs(py - 120) < 5


def test_shot_contact_detected_on_synthetic_direction_change() -> None:
    """Build a synthetic detection history with one sharp reversal and
    verify the detector flags it."""
    det = BallDetector(use_background_subtractor=False)
    # Inject detections directly. We bypass process_frame to keep this test
    # independent of the OpenCV mask pipeline.
    from badminton_tracker.detector import DetectionFrame

    # Frames 0..9 moving right, frames 10..19 moving left (sharp 180-deg flip).
    for i in range(10):
        det._detections.append(
            DetectionFrame(frame_index=i, position=(float(100 + i * 5), 100.0),
                           velocity=(5.0, 0.0))
        )
    for i in range(10, 20):
        det._detections.append(
            DetectionFrame(frame_index=i, position=(float(145 - (i - 10) * 5), 100.0),
                           velocity=(-5.0, 0.0))
        )

    contacts = det.detect_shot_contacts()
    assert len(contacts) >= 1
    # The reversal happens at frame 10.
    assert any(c.frame_index == 10 for c in contacts)
    # cos angle for a perfect reversal is -1.
    assert all(c.cos_angle <= -0.9 for c in contacts)


def test_velocity_smoother_initializes_with_first_measurement() -> None:
    s = VelocitySmoother()
    pos, vel = s.update((10.0, 20.0))
    assert pos == (10.0, 20.0)
    assert vel == (0.0, 0.0)


def test_refractory_window_prevents_duplicate_contacts() -> None:
    """Two reversals within the refractory window should produce only one contact."""
    from badminton_tracker.detector import DetectionFrame

    det = BallDetector(use_background_subtractor=False, refractory_frames=10)
    # alternating velocity each frame -> many reversals
    for i in range(30):
        v = (5.0, 0.0) if i % 2 == 0 else (-5.0, 0.0)
        det._detections.append(
            DetectionFrame(frame_index=i, position=(100.0 + i, 100.0), velocity=v)
        )
    contacts = det.detect_shot_contacts()
    # Without refractory we would have ~29 contacts; with refractory_frames=10
    # we expect roughly ceil(30/10) = 3.
    assert len(contacts) <= 4
