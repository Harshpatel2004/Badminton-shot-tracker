"""Annotated video output: draws skeletons, shuttle trail, and detected shots."""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .court_mapper import CourtMapper
from .detector import DetectionFrame, ShotContact
from .pose_tracker import FrameTracking, PoseTracker


Point = Tuple[float, float]


@dataclass
class VisualizerOptions:
    draw_skeleton: bool = True
    draw_trail: bool = True
    trail_length: int = 20
    draw_court: bool = True
    contact_marker_radius: int = 14
    contact_marker_persist_frames: int = 30


class VideoVisualizer:
    """Writes an annotated MP4 (or AVI) of the analysed match."""

    def __init__(
        self,
        out_path: str,
        frame_size: Tuple[int, int],
        fps: float,
        court_mapper: Optional[CourtMapper] = None,
        options: Optional[VisualizerOptions] = None,
    ) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        self.out_path = out_path
        self.frame_size = frame_size
        self.fps = float(fps)
        self.court_mapper = court_mapper
        self.options = options or VisualizerOptions()

        # Pick a codec by file extension.
        ext = os.path.splitext(out_path)[1].lower()
        if ext == ".avi":
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self.writer = cv2.VideoWriter(out_path, fourcc, self.fps, frame_size)
        if not self.writer.isOpened():
            raise IOError(f"Failed to open video writer for {out_path}")

        self._trail: Deque[Point] = deque(maxlen=self.options.trail_length)
        # frame_index -> ShotContact, with each rendered for N frames.
        self._recent_contacts: List[Tuple[int, ShotContact]] = []

    # ------------------------------------------------------------------

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()

    def __enter__(self) -> "VideoVisualizer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------

    def _draw_skeleton(self, frame: np.ndarray, tracking: FrameTracking) -> None:
        connections = PoseTracker.pose_connections()
        colors = [(255, 128, 0), (0, 200, 255)]  # one color per player
        for player in tracking.players:
            color = colors[player.player_id % len(colors)]
            lm = player.landmarks
            for a, b in connections:
                if a >= len(lm) or b >= len(lm):
                    continue
                if player.visibility[a] < 0.3 or player.visibility[b] < 0.3:
                    continue
                pa = (int(lm[a][0]), int(lm[a][1]))
                pb = (int(lm[b][0]), int(lm[b][1]))
                cv2.line(frame, pa, pb, color, 2)
            for (x, y), v in zip(lm, player.visibility):
                if v < 0.3:
                    continue
                cv2.circle(frame, (int(x), int(y)), 3, color, -1)
            x, y, w, h = player.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
            cv2.putText(frame, f"P{player.player_id + 1}", (x, max(20, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _draw_trail(self, frame: np.ndarray) -> None:
        if len(self._trail) < 2:
            return
        pts = list(self._trail)
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            color = (0, int(255 * alpha), int(255 * (1 - alpha)))
            cv2.line(
                frame,
                (int(pts[i - 1][0]), int(pts[i - 1][1])),
                (int(pts[i][0]), int(pts[i][1])),
                color,
                2,
            )
        cv2.circle(frame, (int(pts[-1][0]), int(pts[-1][1])), 5, (0, 255, 255), -1)

    def _draw_contacts(self, frame: np.ndarray, frame_index: int) -> None:
        keep: List[Tuple[int, ShotContact]] = []
        for first_seen, contact in self._recent_contacts:
            age = frame_index - first_seen
            if age > self.options.contact_marker_persist_frames:
                continue
            keep.append((first_seen, contact))
            alpha = 1.0 - age / max(1, self.options.contact_marker_persist_frames)
            r = int(self.options.contact_marker_radius * (1.0 + 0.4 * (1.0 - alpha)))
            color = (0, 0, int(255 * alpha) + 80)
            cv2.circle(frame, (int(contact.position[0]), int(contact.position[1])),
                       r, color, 3)
            cv2.putText(frame, "SHOT",
                        (int(contact.position[0]) + r, int(contact.position[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        self._recent_contacts = keep

    # ------------------------------------------------------------------

    def write_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        detection: Optional[DetectionFrame] = None,
        tracking: Optional[FrameTracking] = None,
        contacts_this_frame: Optional[Sequence[ShotContact]] = None,
    ) -> None:
        out = frame_bgr.copy()
        if self.options.draw_court and self.court_mapper is not None:
            try:
                out = self.court_mapper.draw_court_overlay(out)
            except Exception:
                pass

        if detection is not None and detection.position is not None:
            self._trail.append(detection.position)

        if self.options.draw_trail:
            self._draw_trail(out)

        if tracking is not None and self.options.draw_skeleton:
            self._draw_skeleton(out, tracking)

        if contacts_this_frame:
            for c in contacts_this_frame:
                self._recent_contacts.append((frame_index, c))
        self._draw_contacts(out, frame_index)

        cv2.putText(out, f"Frame {frame_index}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Make sure the frame is the expected size.
        if (out.shape[1], out.shape[0]) != self.frame_size:
            out = cv2.resize(out, self.frame_size)
        self.writer.write(out)
