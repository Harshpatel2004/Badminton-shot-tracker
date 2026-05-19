"""Self-contained demo for badminton-shot-tracker.

Synthesises a short fake "match" video: a moving white circle on a darker
background that changes direction periodically (each direction change
simulates a racket contact). The full pipeline is then run on the synthetic
video and the heatmap PNG plus shots CSV are written to ``demo_output/``.

Run with::

    python run_demo.py

No real footage required — this proves the project is wired together
end-to-end.
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import cv2
import numpy as np

from badminton_tracker.court_mapper import CourtMapper
from badminton_tracker.detector import BallDetector
from badminton_tracker.heatmap import (
    build_heatmap,
    save_heatmap_png,
    save_zone_chart_png,
    zone_breakdown,
)
from badminton_tracker.utils import (
    ShotRecord,
    ensure_dir,
    frame_to_time,
    save_results_csv,
)


WIDTH, HEIGHT = 640, 480
FPS = 30
TOTAL_FRAMES = 240   # 8 seconds


def _synth_video(path: str) -> List[Tuple[int, Tuple[int, int]]]:
    """Generate the demo video. Returns the ground-truth contact frames."""
    ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        # Fall back to AVI/MJPG if mp4v isn't available.
        alt = os.path.splitext(path)[0] + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(alt, fourcc, FPS, (WIDTH, HEIGHT))
        path = alt

    # Court-ish background.
    bg = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)
    cv2.rectangle(bg, (60, 60), (WIDTH - 60, HEIGHT - 60), (50, 80, 50), -1)
    cv2.rectangle(bg, (60, 60), (WIDTH - 60, HEIGHT - 60), (255, 255, 255), 2)
    # Net line.
    cv2.line(bg, (60, HEIGHT // 2), (WIDTH - 60, HEIGHT // 2), (255, 255, 255), 1)

    # Trajectory: bouncing back and forth between two players, with sharp
    # direction reversals at frames 40, 90, 140, 190.
    contacts: List[Tuple[int, Tuple[int, int]]] = []
    pos = np.array([120.0, 120.0])
    vel = np.array([5.5, 4.0])

    direction_changes = {
        40:  np.array([-5.5, 4.0]),
        90:  np.array([5.5, -4.0]),
        140: np.array([-5.5, -4.0]),
        190: np.array([5.5, 4.0]),
    }

    rng = np.random.default_rng(42)

    for i in range(TOTAL_FRAMES):
        if i in direction_changes:
            vel = direction_changes[i].copy()
            contacts.append((i, (int(pos[0]), int(pos[1]))))

        # Tiny noise so detection is non-trivial.
        pos = pos + vel + rng.normal(0, 0.2, size=2)

        # Reflect off frame edges so the shuttle stays in view.
        if pos[0] < 80 or pos[0] > WIDTH - 80:
            vel[0] *= -1
            pos[0] = np.clip(pos[0], 80, WIDTH - 80)
        if pos[1] < 80 or pos[1] > HEIGHT - 80:
            vel[1] *= -1
            pos[1] = np.clip(pos[1], 80, HEIGHT - 80)

        frame = bg.copy()
        # Shuttle: bright white circle.
        cv2.circle(frame, (int(pos[0]), int(pos[1])), 6, (255, 255, 255), -1)
        # Faint glow so the bg subtractor has something to chew on.
        cv2.circle(frame, (int(pos[0]), int(pos[1])), 9, (220, 220, 220), 1)
        writer.write(frame)

    writer.release()
    print(f"[demo] Wrote synthetic video to {path}")
    print(f"[demo] Injected ground-truth contacts at frames: {[c[0] for c in contacts]}")
    return contacts


def main() -> int:
    out_dir = ensure_dir("demo_output")
    video_path = os.path.join(out_dir, "synthetic_match.mp4")
    gt = _synth_video(video_path)
    if not os.path.isfile(video_path):
        # mp4v unsupported on some systems -> fallback wrote .avi
        video_path = os.path.splitext(video_path)[0] + ".avi"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[error] Could not open synthetic video {video_path}", file=sys.stderr)
        return 1

    # Court corners chosen to match the rectangle we drew in _synth_video.
    mapper = CourtMapper(
        image_corners=[(60, 60), (WIDTH - 60, 60), (WIDTH - 60, HEIGHT - 60), (60, HEIGHT - 60)],
        court_width_m=6.1,
        court_length_m=13.4,
    )
    detector = BallDetector(
        hsv_lower=(0, 0, 200),
        hsv_upper=(180, 60, 255),
        min_area=4.0,
        max_area=400.0,
        direction_cos_threshold=-0.2,
        min_speed_px=1.5,
        refractory_frames=8,
        use_background_subtractor=True,
    )

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detector.process_frame(frame, idx)
        idx += 1
    cap.release()

    contacts = detector.detect_shot_contacts()
    print(f"[demo] Detected {len(contacts)} contacts at frames "
          f"{[c.frame_index for c in contacts]}")

    img_pts = [c.position for c in contacts]
    court_pts = mapper.image_to_court(img_pts) if img_pts else np.empty((0, 2))

    records: List[ShotRecord] = []
    for c, (cx, cy) in zip(contacts, court_pts):
        records.append(
            ShotRecord(
                frame_index=c.frame_index,
                time_s=frame_to_time(c.frame_index, FPS),
                image_x=c.position[0],
                image_y=c.position[1],
                court_x_m=float(cx),
                court_y_m=float(cy),
                zone=mapper.zone_for_point(float(cx), float(cy)),
            )
        )
    csv_path = os.path.join(out_dir, "shots.csv")
    save_results_csv(csv_path, records)
    print(f"[demo] Wrote {csv_path}")

    heatmap = build_heatmap(
        court_points=[(p[0], p[1]) for p in court_pts],
        court_width_m=mapper.court_width_m,
        court_length_m=mapper.court_length_m,
    )
    hm_path = os.path.join(out_dir, "heatmap.png")
    save_heatmap_png(heatmap, hm_path,
                     court_width_m=mapper.court_width_m,
                     court_length_m=mapper.court_length_m)
    print(f"[demo] Wrote {hm_path}")

    bd = zone_breakdown(
        [(p[0], p[1]) for p in court_pts],
        court_width_m=mapper.court_width_m,
        court_length_m=mapper.court_length_m,
    )
    zone_path = os.path.join(out_dir, "zones.png")
    save_zone_chart_png(bd, zone_path)
    print(f"[demo] Wrote {zone_path}")
    print(f"[demo] Zone counts: {bd.counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
