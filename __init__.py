"""badminton_tracker — real-time shot detection and court heatmap toolkit.

Public modules:
    detector       : shuttle detection and shot-contact extraction
    pose_tracker   : MediaPipe-based player pose tracking
    court_mapper   : image -> court (metres) homography
    heatmap        : 2-D heatmap and zone breakdown
    visualizer     : annotated video output
    utils          : config loading, CSV export, fps helpers
"""

__version__ = "0.1.0"
__author__ = "Harsh Patel"

from . import (
    detector,
    pose_tracker,
    court_mapper,
    heatmap,
    visualizer,
    utils,
)

__all__ = [
    "detector",
    "pose_tracker",
    "court_mapper",
    "heatmap",
    "visualizer",
    "utils",
]
