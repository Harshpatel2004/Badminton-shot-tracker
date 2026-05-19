"""Shuttle detection and shot-contact extraction.

The detector combines HSV color thresholding, background subtraction, and
contour analysis to track the shuttlecock frame by frame. A lightweight
Kalman-style velocity state is maintained for smoothing. Shot contacts are
detected when the velocity vector reverses sharply between consecutive
frames (dot product of unit velocities falls below a threshold).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# Velocity smoother
# ---------------------------------------------------------------------------

class VelocitySmoother:
    """Very small constant-velocity Kalman filter for a 2-D point.

    State: [x, y, vx, vy].  Measurement: [x, y].

    We use this instead of raw frame-to-frame differences because the
    shuttle detection is noisy at low resolution.
    """

    def __init__(self, process_noise: float = 1e-2, meas_noise: float = 1.0) -> None:
        self.kf = cv2.KalmanFilter(4, 2)
        dt = 1.0
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * meas_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self._initialized = False

    def update(self, measurement: Optional[Point]) -> Tuple[Optional[Point], Optional[Point]]:
        """Step the filter. Returns (smoothed_position, velocity)."""
        if measurement is None:
            if not self._initialized:
                return None, None
            pred = self.kf.predict()
            pos = (float(pred[0]), float(pred[1]))
            vel = (float(pred[2]), float(pred[3]))
            return pos, vel

        z = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
        if not self._initialized:
            self.kf.statePost = np.array(
                [[measurement[0]], [measurement[1]], [0.0], [0.0]],
                dtype=np.float32,
            )
            self._initialized = True
            return (float(measurement[0]), float(measurement[1])), (0.0, 0.0)

        self.kf.predict()
        est = self.kf.correct(z)
        pos = (float(est[0]), float(est[1]))
        vel = (float(est[2]), float(est[3]))
        return pos, vel


# ---------------------------------------------------------------------------
# Detection results
# ---------------------------------------------------------------------------

@dataclass
class DetectionFrame:
    frame_index: int
    position: Optional[Point]
    velocity: Optional[Point]


@dataclass
class ShotContact:
    frame_index: int
    position: Point        # image coordinates
    velocity_before: Point
    velocity_after: Point
    cos_angle: float


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

@dataclass
class BallDetector:
    """Detects the shuttlecock and extracts shot contact frames."""

    hsv_lower: Sequence[int] = (0, 0, 200)
    hsv_upper: Sequence[int] = (180, 60, 255)
    min_area: float = 4.0
    max_area: float = 400.0
    direction_cos_threshold: float = -0.3
    min_speed_px: float = 2.0
    refractory_frames: int = 6
    use_background_subtractor: bool = True

    # Internal state — initialised in __post_init__.
    _smoother: VelocitySmoother = field(init=False)
    _bg_sub: Optional[cv2.BackgroundSubtractor] = field(init=False, default=None)
    _detections: List[DetectionFrame] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._smoother = VelocitySmoother()
        if self.use_background_subtractor:
            self._bg_sub = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=25, detectShadows=False
            )
        self._detections = []

    # ------------------------------------------------------------------
    # Per-frame detection
    # ------------------------------------------------------------------

    def _mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Build the binary mask for shuttle-candidate pixels."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array(self.hsv_lower, dtype=np.uint8)
        upper = np.array(self.hsv_upper, dtype=np.uint8)
        color_mask = cv2.inRange(hsv, lower, upper)

        if self._bg_sub is not None:
            motion_mask = self._bg_sub.apply(frame_bgr)
            # Combine color and motion. We keep pixels that are bright AND moving.
            mask = cv2.bitwise_and(color_mask, motion_mask)
        else:
            mask = color_mask

        # Small open then dilate to clean noise.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def _best_contour_centroid(self, mask: np.ndarray) -> Optional[Point]:
        """Return the centroid of the most plausible shuttle blob, if any."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: Optional[Tuple[float, Point]] = None
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            # Score by closeness to expected size — favor small, roundish blobs.
            score = -abs(area - 30.0)
            if best is None or score > best[0]:
                best = (score, (float(cx), float(cy)))
        return best[1] if best else None

    def process_frame(self, frame_bgr: np.ndarray, frame_index: int) -> DetectionFrame:
        """Run detection on a single BGR frame and append to internal history."""
        mask = self._mask(frame_bgr)
        meas = self._best_contour_centroid(mask)
        pos, vel = self._smoother.update(meas)
        det = DetectionFrame(frame_index=frame_index, position=pos, velocity=vel)
        self._detections.append(det)
        return det

    # ------------------------------------------------------------------
    # Shot-contact analysis
    # ------------------------------------------------------------------

    def detect_shot_contacts(self) -> List[ShotContact]:
        """Scan accumulated detections and return frames where the velocity
        vector reverses sharply enough to imply a racket contact.

        The dot product of consecutive unit velocity vectors falls below
        ``direction_cos_threshold`` for a real contact (e.g. cos(150 deg) = -0.87).
        A refractory window prevents the same contact being reported twice.
        """
        contacts: List[ShotContact] = []
        last_contact_frame = -10_000

        dets = [d for d in self._detections if d.position is not None and d.velocity is not None]
        for i in range(1, len(dets)):
            prev = dets[i - 1]
            curr = dets[i]
            v0 = np.array(prev.velocity, dtype=np.float64)
            v1 = np.array(curr.velocity, dtype=np.float64)
            s0 = float(np.linalg.norm(v0))
            s1 = float(np.linalg.norm(v1))
            if s0 < self.min_speed_px or s1 < self.min_speed_px:
                continue
            cos_a = float(np.dot(v0, v1) / (s0 * s1))
            if cos_a <= self.direction_cos_threshold:
                if curr.frame_index - last_contact_frame < self.refractory_frames:
                    continue
                contacts.append(
                    ShotContact(
                        frame_index=curr.frame_index,
                        position=curr.position,  # type: ignore[arg-type]
                        velocity_before=(float(v0[0]), float(v0[1])),
                        velocity_after=(float(v1[0]), float(v1[1])),
                        cos_angle=cos_a,
                    )
                )
                last_contact_frame = curr.frame_index
        return contacts

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def detections(self) -> List[DetectionFrame]:
        return list(self._detections)

    def reset(self) -> None:
        self._smoother = VelocitySmoother()
        if self.use_background_subtractor:
            self._bg_sub = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=25, detectShadows=False
            )
        self._detections = []
