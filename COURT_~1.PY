"""Image <-> court coordinate mapping via a perspective homography."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


@dataclass
class CourtMapper:
    """Maps pixel coordinates to real court coordinates (in metres).

    The input is the four court corners (TL, TR, BR, BL) as observed in the
    video frame, plus the real-world court dimensions. Internally we build a
    3x3 perspective homography with ``cv2.getPerspectiveTransform``.
    """

    image_corners: Sequence[Point]
    court_length_m: float = 13.4   # along the long axis (y)
    court_width_m: float = 6.1     # along the short axis (x)

    def __post_init__(self) -> None:
        if len(self.image_corners) != 4:
            raise ValueError("image_corners must have exactly 4 points")
        src = np.asarray(self.image_corners, dtype=np.float32)
        # Court frame: x in [0, width], y in [0, length].
        # We choose TL = (0, 0), TR = (width, 0), BR = (width, length), BL = (0, length).
        dst = np.array(
            [
                [0.0, 0.0],
                [self.court_width_m, 0.0],
                [self.court_width_m, self.court_length_m],
                [0.0, self.court_length_m],
            ],
            dtype=np.float32,
        )
        self._H = cv2.getPerspectiveTransform(src, dst)
        self._H_inv = np.linalg.inv(self._H)

    # ------------------------------------------------------------------
    # Forward / inverse transforms
    # ------------------------------------------------------------------

    def image_to_court(self, points: Iterable[Point]) -> np.ndarray:
        """Map a list of image points to court metres. Returns Nx2 array."""
        pts = np.asarray(list(points), dtype=np.float32).reshape(-1, 1, 2)
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        out = cv2.perspectiveTransform(pts, self._H)
        return out.reshape(-1, 2)

    def court_to_image(self, points: Iterable[Point]) -> np.ndarray:
        """Inverse map: court metres back to image pixels."""
        pts = np.asarray(list(points), dtype=np.float32).reshape(-1, 1, 2)
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        out = cv2.perspectiveTransform(pts, self._H_inv)
        return out.reshape(-1, 2)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw_court_overlay(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw the court polygon, net line, and service boxes onto ``frame``."""
        out = frame.copy()

        # Outline.
        pts = self.court_to_image(
            [
                (0.0, 0.0),
                (self.court_width_m, 0.0),
                (self.court_width_m, self.court_length_m),
                (0.0, self.court_length_m),
            ]
        ).astype(np.int32)
        cv2.polylines(out, [pts.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=thickness)

        # Net (midline along the length axis).
        net = self.court_to_image(
            [(0.0, self.court_length_m / 2.0), (self.court_width_m, self.court_length_m / 2.0)]
        ).astype(np.int32)
        cv2.line(out, tuple(net[0]), tuple(net[1]), color, thickness)

        # Short service lines (1.98 m from net on each side).
        sv = 1.98
        for offset in (-sv, sv):
            y = self.court_length_m / 2.0 + offset
            line = self.court_to_image([(0.0, y), (self.court_width_m, y)]).astype(np.int32)
            cv2.line(out, tuple(line[0]), tuple(line[1]), color, max(1, thickness - 1))

        # Centre line between the two service boxes on each half.
        cx = self.court_width_m / 2.0
        center = self.court_to_image(
            [(cx, self.court_length_m / 2.0 - sv), (cx, self.court_length_m / 2.0 + sv)]
        ).astype(np.int32)
        cv2.line(out, tuple(center[0]), tuple(center[1]), color, max(1, thickness - 1))

        return out

    # ------------------------------------------------------------------
    # Zones
    # ------------------------------------------------------------------

    def zone_for_point(self, x_m: float, y_m: float) -> str:
        """Return one of 6 zone labels for a court point.

        Long axis split into front/mid/back (3), short axis into left/right (2).
        """
        # x runs across width (0..width). y runs along length (0..length).
        # "Front" = closest to the net (the middle band of length).
        third = self.court_length_m / 3.0
        if y_m < third:
            band = "back-far"     # far from camera side
        elif y_m < 2 * third:
            band = "mid"
        else:
            band = "back-near"    # near camera side

        # But the resume calls it front/mid/back x left/right. Re-label so
        # that the side closest to y=0 is "back" and farther is "front" from
        # the camera's POV. We expose a simpler 3-band naming below.
        if y_m < third:
            band = "back"
        elif y_m < 2 * third:
            band = "mid"
        else:
            band = "front"

        side = "left" if x_m < self.court_width_m / 2.0 else "right"
        return f"{band}-{side}"

    @staticmethod
    def zone_names() -> List[str]:
        """The 6 canonical zone labels."""
        return [
            "back-left", "back-right",
            "mid-left", "mid-right",
            "front-left", "front-right",
        ]
