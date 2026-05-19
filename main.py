"""CLI entry point for badminton-shot-tracker.

Example
-------
    python -m badminton_tracker.main \
        --video examples/match.mp4 \
        --output output/ \
        --config configs/default.yaml \
        --show
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import cv2
import numpy as np

from .court_mapper import CourtMapper
from .detector import BallDetector, ShotContact
from .heatmap import (
    build_heatmap,
    save_heatmap_png,
    save_zone_chart_png,
    zone_breakdown,
)
from .utils import (
    ShotRecord,
    ensure_dir,
    frame_to_time,
    get_court_corners,
    load_config,
    save_results_csv,
)
from .visualizer import VideoVisualizer, VisualizerOptions


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="badminton-shot-tracker",
        description="Detect shuttle shot-contact points and build a court heatmap.",
    )
    p.add_argument("--video", required=True, help="Path to the input match video.")
    p.add_argument("--output", default="output", help="Directory for outputs.")
    p.add_argument("--config", default="configs/default.yaml", help="YAML config.")
    p.add_argument("--show", action="store_true", help="Show live preview window.")
    p.add_argument("--no-video", action="store_true", help="Skip annotated video output.")
    p.add_argument("--no-pose", action="store_true",
                   help="Skip MediaPipe pose tracking (useful for headless tests).")
    p.add_argument("--max-frames", type=int, default=0,
                   help="If > 0, stop after this many frames (useful for quick runs).")
    return p


def run(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    if not os.path.isfile(args.video):
        print(f"[error] Video not found: {args.video}", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    out_dir = ensure_dir(args.output)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[error] Could not open video {args.video}", file=sys.stderr)
        return 3

    fps = cap.get(cv2.CAP_PROP_FPS) or float(cfg.get("fps", 30))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Court / detector setup
    court_corners = get_court_corners(cfg)
    court_size = cfg.get("court_size_m", {})
    mapper = CourtMapper(
        image_corners=court_corners,
        court_length_m=float(court_size.get("length", 13.4)),
        court_width_m=float(court_size.get("width", 6.1)),
    )

    det_cfg = cfg.get("detection", {})
    shuttle_hsv = cfg.get("shuttle_hsv", {})
    shuttle_area = cfg.get("shuttle_area", {})
    detector = BallDetector(
        hsv_lower=shuttle_hsv.get("lower", [0, 0, 200]),
        hsv_upper=shuttle_hsv.get("upper", [180, 60, 255]),
        min_area=float(shuttle_area.get("min", 4)),
        max_area=float(shuttle_area.get("max", 400)),
        direction_cos_threshold=float(det_cfg.get("direction_cos_threshold", -0.3)),
        min_speed_px=float(det_cfg.get("min_speed_px", 2.0)),
        refractory_frames=int(det_cfg.get("refractory_frames", 6)),
    )

    # Optional pose tracker.
    pose = None
    if not args.no_pose:
        try:
            from .pose_tracker import PoseTracker

            pose = PoseTracker(max_players=2)
        except Exception as e:
            print(f"[warn] Pose tracking disabled: {e}", file=sys.stderr)
            pose = None

    # Annotated video writer.
    writer = None
    if not args.no_video:
        out_video = os.path.join(out_dir, "annotated.mp4")
        try:
            writer = VideoVisualizer(
                out_path=out_video,
                frame_size=(width, height),
                fps=fps,
                court_mapper=mapper,
                options=VisualizerOptions(
                    draw_skeleton=bool(cfg.get("output", {}).get("draw_skeleton", True)),
                    draw_trail=bool(cfg.get("output", {}).get("draw_trail", True)),
                    trail_length=int(cfg.get("output", {}).get("trail_length", 20)),
                ),
            )
        except Exception as e:
            print(f"[warn] Video writer disabled: {e}", file=sys.stderr)
            writer = None

    # Two-pass-ish loop: per frame we detect, optionally pose-track, and
    # buffer for shot detection at the end. We keep the contacts list growing
    # so the visualizer can render them as soon as they're found.
    frame_idx = 0
    contacts_by_frame: dict = {}
    last_contacts_seen = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            det = detector.process_frame(frame, frame_idx)

            tracking = None
            if pose is not None:
                try:
                    tracking = pose.process_frame(frame, frame_idx)
                except Exception:
                    tracking = None

            # Re-scan for new contacts so we can draw the marker immediately.
            new_contacts = detector.detect_shot_contacts()
            recent = new_contacts[last_contacts_seen:]
            last_contacts_seen = len(new_contacts)
            for c in recent:
                contacts_by_frame.setdefault(c.frame_index, []).append(c)

            if writer is not None:
                writer.write_frame(
                    frame_bgr=frame,
                    frame_index=frame_idx,
                    detection=det,
                    tracking=tracking,
                    contacts_this_frame=contacts_by_frame.get(frame_idx, []),
                )

            if args.show:
                preview = frame.copy()
                if det.position is not None:
                    cv2.circle(preview, (int(det.position[0]), int(det.position[1])),
                               6, (0, 255, 255), 2)
                cv2.imshow("badminton-shot-tracker", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.close()
        if pose is not None:
            pose.close()
        if args.show:
            cv2.destroyAllWindows()

    contacts = detector.detect_shot_contacts()
    print(f"[info] Frames processed: {frame_idx}")
    print(f"[info] Shot contacts detected: {len(contacts)}")

    # Map to court frame.
    img_pts = [c.position for c in contacts]
    court_pts = mapper.image_to_court(img_pts) if img_pts else np.empty((0, 2))

    # Save CSV.
    records: List[ShotRecord] = []
    for c, (cx, cy) in zip(contacts, court_pts):
        records.append(
            ShotRecord(
                frame_index=c.frame_index,
                time_s=frame_to_time(c.frame_index, fps),
                image_x=c.position[0],
                image_y=c.position[1],
                court_x_m=float(cx),
                court_y_m=float(cy),
                zone=mapper.zone_for_point(float(cx), float(cy)),
            )
        )
    csv_path = os.path.join(out_dir, "shots.csv")
    save_results_csv(csv_path, records)
    print(f"[info] Wrote {csv_path}")

    # Heatmap + zone chart.
    hm_cfg = cfg.get("heatmap", {})
    heatmap = build_heatmap(
        court_points=[(p[0], p[1]) for p in court_pts],
        court_width_m=mapper.court_width_m,
        court_length_m=mapper.court_length_m,
        grid_x=int(hm_cfg.get("grid_x", 61)),
        grid_y=int(hm_cfg.get("grid_y", 134)),
        sigma=float(hm_cfg.get("gaussian_sigma", 2.0)),
    )
    hm_path = os.path.join(out_dir, "heatmap.png")
    save_heatmap_png(
        heatmap,
        hm_path,
        court_width_m=mapper.court_width_m,
        court_length_m=mapper.court_length_m,
    )
    print(f"[info] Wrote {hm_path}")

    bd = zone_breakdown(
        [(p[0], p[1]) for p in court_pts],
        court_width_m=mapper.court_width_m,
        court_length_m=mapper.court_length_m,
    )
    zone_path = os.path.join(out_dir, "zones.png")
    save_zone_chart_png(bd, zone_path)
    print(f"[info] Wrote {zone_path}")
    print(f"[info] Zone counts: {bd.counts}")
    return 0


def main() -> None:  # pragma: no cover
    raise SystemExit(run())


if __name__ == "__main__":  # pragma: no cover
    main()
