"""Utility helpers: config loading, CSV export, fps math."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config from ``path`` and return a plain dict.

    Raises FileNotFoundError if the file does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping, got {type(cfg).__name__}")
    return cfg


def get_court_corners(cfg: Dict[str, Any]) -> List[Tuple[float, float]]:
    """Return the four court corners in TL, TR, BR, BL order as floats."""
    cc = cfg.get("court_corners", {})
    keys = ("top_left", "top_right", "bottom_right", "bottom_left")
    pts: List[Tuple[float, float]] = []
    for k in keys:
        if k not in cc:
            raise KeyError(f"Missing court corner '{k}' in config")
        x, y = cc[k]
        pts.append((float(x), float(y)))
    return pts


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

@dataclass
class ShotRecord:
    frame_index: int
    time_s: float
    image_x: float
    image_y: float
    court_x_m: float
    court_y_m: float
    zone: str


def save_results_csv(path: str, records: Iterable[ShotRecord]) -> None:
    """Write shot records to a CSV file (creates parent dir if needed)."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fields = [
        "frame_index",
        "time_s",
        "image_x",
        "image_y",
        "court_x_m",
        "court_y_m",
        "zone",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "frame_index": r.frame_index,
                    "time_s": round(r.time_s, 4),
                    "image_x": round(r.image_x, 2),
                    "image_y": round(r.image_y, 2),
                    "court_x_m": round(r.court_x_m, 3),
                    "court_y_m": round(r.court_y_m, 3),
                    "zone": r.zone,
                }
            )


# ---------------------------------------------------------------------------
# Time / fps helpers
# ---------------------------------------------------------------------------

def frame_to_time(frame_index: int, fps: float) -> float:
    """Convert a frame index to seconds."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    return float(frame_index) / float(fps)


def time_to_frame(time_s: float, fps: float) -> int:
    """Convert seconds to a frame index (rounded down)."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    return int(time_s * fps)


def ensure_dir(path: str) -> str:
    """Create directory ``path`` if missing and return it."""
    os.makedirs(path, exist_ok=True)
    return path


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` to the inclusive range [lo, hi]."""
    return max(lo, min(hi, value))
