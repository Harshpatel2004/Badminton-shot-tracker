"""Heatmap and zone-breakdown helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

# Use a non-interactive backend so this works on headless CI / servers.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:
    from scipy.ndimage import gaussian_filter  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# 2-D heatmap
# ---------------------------------------------------------------------------

def build_heatmap(
    court_points: Sequence[Point],
    court_width_m: float = 6.1,
    court_length_m: float = 13.4,
    grid_x: int = 61,
    grid_y: int = 134,
    sigma: float = 2.0,
) -> np.ndarray:
    """Build a 2-D density heatmap from a list of court-frame points.

    Parameters
    ----------
    court_points : iterable of (x_m, y_m) pairs in court metres.
    court_width_m, court_length_m : court extent.
    grid_x, grid_y : output grid resolution.
    sigma : Gaussian blur sigma in grid cells.

    Returns the heatmap as a (grid_y, grid_x) numpy array, scaled 0..1.
    """
    if grid_x <= 0 or grid_y <= 0:
        raise ValueError("grid_x and grid_y must be positive")

    pts = np.asarray(list(court_points), dtype=np.float64)
    if pts.size == 0:
        return np.zeros((grid_y, grid_x), dtype=np.float32)

    # Clip to court bounds so out-of-range points don't blow up the histogram.
    xs = np.clip(pts[:, 0], 0, court_width_m)
    ys = np.clip(pts[:, 1], 0, court_length_m)

    hist, _, _ = np.histogram2d(
        ys, xs,
        bins=[grid_y, grid_x],
        range=[[0, court_length_m], [0, court_width_m]],
    )
    hist = hist.astype(np.float32)
    if _HAS_SCIPY:
        hist = gaussian_filter(hist, sigma=sigma)
    else:
        # Fallback: cheap separable Gaussian via cv2 if scipy missing.
        try:
            import cv2  # type: ignore

            k = max(3, int(2 * round(sigma) + 1) | 1)
            hist = cv2.GaussianBlur(hist, (k, k), sigma)
        except Exception:
            pass

    m = hist.max()
    if m > 0:
        hist = hist / m
    return hist


def save_heatmap_png(
    heatmap: np.ndarray,
    out_path: str,
    title: str = "Badminton Shot Heatmap",
    court_width_m: float = 6.1,
    court_length_m: float = 13.4,
    cmap: str = "hot",
) -> str:
    """Render the heatmap with court overlay and save to PNG."""
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 8), dpi=140)
    ax.imshow(
        heatmap,
        origin="lower",
        extent=[0, court_width_m, 0, court_length_m],
        cmap=cmap,
        aspect="equal",
        interpolation="bilinear",
    )

    # Court outline.
    ax.plot(
        [0, court_width_m, court_width_m, 0, 0],
        [0, 0, court_length_m, court_length_m, 0],
        color="white",
        linewidth=1.5,
    )
    # Net.
    ax.plot([0, court_width_m], [court_length_m / 2, court_length_m / 2],
            color="cyan", linewidth=1.5, linestyle="--")
    # Service lines.
    for off in (-1.98, 1.98):
        y = court_length_m / 2 + off
        ax.plot([0, court_width_m], [y, y], color="white", linewidth=0.8, alpha=0.7)
    # Centre line.
    ax.plot([court_width_m / 2, court_width_m / 2],
            [court_length_m / 2 - 1.98, court_length_m / 2 + 1.98],
            color="white", linewidth=0.8, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("Court width (m)")
    ax.set_ylabel("Court length (m)")
    ax.set_xlim(0, court_width_m)
    ax.set_ylim(0, court_length_m)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Zone breakdown
# ---------------------------------------------------------------------------

@dataclass
class ZoneBreakdown:
    counts: Dict[str, int]
    total: int

    def percentages(self) -> Dict[str, float]:
        if self.total == 0:
            return {z: 0.0 for z in self.counts}
        return {z: 100.0 * c / self.total for z, c in self.counts.items()}


def zone_breakdown(
    court_points: Iterable[Point],
    court_width_m: float = 6.1,
    court_length_m: float = 13.4,
) -> ZoneBreakdown:
    """Count points per zone in a 3x2 split (front/mid/back x left/right)."""
    zones = [
        "back-left", "back-right",
        "mid-left", "mid-right",
        "front-left", "front-right",
    ]
    counts: Dict[str, int] = {z: 0 for z in zones}
    total = 0
    third = court_length_m / 3.0
    for x, y in court_points:
        if not (0.0 <= x <= court_width_m and 0.0 <= y <= court_length_m):
            continue
        if y < third:
            band = "back"
        elif y < 2 * third:
            band = "mid"
        else:
            band = "front"
        side = "left" if x < court_width_m / 2.0 else "right"
        counts[f"{band}-{side}"] += 1
        total += 1
    return ZoneBreakdown(counts=counts, total=total)


def save_zone_chart_png(breakdown: ZoneBreakdown, out_path: str, title: str = "Shot count by zone") -> str:
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    zones = list(breakdown.counts.keys())
    counts = [breakdown.counts[z] for z in zones]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    bars = ax.bar(zones, counts, color="#2a9d8f")
    ax.set_title(title)
    ax.set_ylabel("Shots")
    ax.set_xlabel("Zone")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), str(c),
                ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path
