"""Player pose and face tracking using MediaPipe.

The :class:`PoseTracker` runs MediaPipe Pose on each frame and returns the
player's keypoints. For two-player scenarios we run a light face detection
to split players by their face bounding box and assign the closest pose to
each face. MediaPipe Pose ships single-person, so we approximate by
splitting the frame in two halves around the net and running the model
twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import mediapipe as mp  # type: ignore
    _MP_OK = True
except Exception:  # pragma: no cover — handled at runtime
    mp = None  # type: ignore
    _MP_OK = False


Point = Tuple[float, float]


@dataclass
class PlayerPose:
    player_id: int
    landmarks: List[Point]    # 33 landmarks in image pixels (x, y)
    visibility: List[float]   # per-landmark visibility 0..1
    bbox: Tuple[int, int, int, int]   # x, y, w, h around the player


@dataclass
class FrameTracking:
    frame_index: int
    players: List[PlayerPose] = field(default_factory=list)


class PoseTracker:
    """Wraps MediaPipe Pose + Face Detection for 1-2 player tracking."""

    def __init__(
        self,
        max_players: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        split_horizontal: bool = True,
    ) -> None:
        if not _MP_OK:
            raise RuntimeError(
                "mediapipe is not installed — install it via `pip install mediapipe`."
            )
        self.max_players = max(1, min(2, max_players))
        self.split_horizontal = split_horizontal
        self._pose_models = [
            mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            for _ in range(self.max_players)
        ]
        self._face = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=min_detection_confidence
        )

    # ------------------------------------------------------------------

    def close(self) -> None:
        for p in self._pose_models:
            try:
                p.close()
            except Exception:
                pass
        try:
            self._face.close()
        except Exception:
            pass

    def __enter__(self) -> "PoseTracker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------

    def _run_pose(self, sub_bgr: np.ndarray, offset: Tuple[int, int], model_idx: int) -> Optional[PlayerPose]:
        import cv2  # local to avoid hard dep at import time

        rgb = cv2.cvtColor(sub_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose_models[model_idx].process(rgb)
        if not result.pose_landmarks:
            return None

        h, w = sub_bgr.shape[:2]
        ox, oy = offset
        landmarks: List[Point] = []
        vis: List[float] = []
        xs: List[float] = []
        ys: List[float] = []
        for lm in result.pose_landmarks.landmark:
            x = lm.x * w + ox
            y = lm.y * h + oy
            landmarks.append((float(x), float(y)))
            vis.append(float(lm.visibility))
            xs.append(x)
            ys.append(y)
        if not xs:
            return None
        x0, y0 = int(min(xs)), int(min(ys))
        x1, y1 = int(max(xs)), int(max(ys))
        return PlayerPose(
            player_id=model_idx,
            landmarks=landmarks,
            visibility=vis,
            bbox=(x0, y0, max(1, x1 - x0), max(1, y1 - y0)),
        )

    def process_frame(self, frame_bgr: np.ndarray, frame_index: int) -> FrameTracking:
        h, w = frame_bgr.shape[:2]
        players: List[PlayerPose] = []

        if self.max_players == 1:
            p = self._run_pose(frame_bgr, (0, 0), 0)
            if p is not None:
                players.append(p)
            return FrameTracking(frame_index=frame_index, players=players)

        # Split frame in half along the long axis (assumes camera placed at
        # one end of the court, net horizontal across the middle).
        if self.split_horizontal:
            mid = h // 2
            top = frame_bgr[0:mid, :, :]
            bot = frame_bgr[mid:h, :, :]
            p_top = self._run_pose(top, (0, 0), 0)
            p_bot = self._run_pose(bot, (0, mid), 1)
            if p_top is not None:
                p_top.player_id = 0
                players.append(p_top)
            if p_bot is not None:
                p_bot.player_id = 1
                players.append(p_bot)
        else:
            mid = w // 2
            left = frame_bgr[:, 0:mid, :]
            right = frame_bgr[:, mid:w, :]
            p_left = self._run_pose(left, (0, 0), 0)
            p_right = self._run_pose(right, (mid, 0), 1)
            if p_left is not None:
                p_left.player_id = 0
                players.append(p_left)
            if p_right is not None:
                p_right.player_id = 1
                players.append(p_right)

        return FrameTracking(frame_index=frame_index, players=players)

    # ------------------------------------------------------------------

    @staticmethod
    def pose_connections() -> List[Tuple[int, int]]:
        """Return the standard MediaPipe Pose skeleton connections."""
        if not _MP_OK:
            return []
        return list(mp.solutions.pose.POSE_CONNECTIONS)
