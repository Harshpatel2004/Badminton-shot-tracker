"""Tests for badminton_tracker.court_mapper."""

from __future__ import annotations

import numpy as np
import pytest

from badminton_tracker.court_mapper import CourtMapper


# A simple axis-aligned "image rectangle" of court corners we can reason
# about analytically. The image corners and court corners line up so the
# homography is effectively an affine scaling.
IMAGE_CORNERS = [(0.0, 0.0), (100.0, 0.0), (100.0, 200.0), (0.0, 200.0)]


def _mk() -> CourtMapper:
    return CourtMapper(image_corners=IMAGE_CORNERS, court_width_m=6.1, court_length_m=13.4)


def test_corners_map_to_court_corners() -> None:
    m = _mk()
    out = m.image_to_court(IMAGE_CORNERS)
    expected = np.array([
        [0.0, 0.0],
        [6.1, 0.0],
        [6.1, 13.4],
        [0.0, 13.4],
    ])
    assert np.allclose(out, expected, atol=1e-3)


def test_center_pixel_maps_to_court_center() -> None:
    m = _mk()
    out = m.image_to_court([(50.0, 100.0)])
    assert out.shape == (1, 2)
    cx, cy = out[0]
    assert abs(cx - 3.05) < 1e-3
    assert abs(cy - 6.7) < 1e-3


def test_round_trip_image_court_image() -> None:
    m = _mk()
    pts = [(25.0, 50.0), (75.0, 175.0), (10.0, 10.0)]
    court = m.image_to_court(pts)
    back = m.court_to_image([(c[0], c[1]) for c in court])
    assert np.allclose(np.asarray(pts), back, atol=1e-3)


def test_zone_assignment() -> None:
    m = _mk()
    # back-left: x in [0, 3.05], y in [0, 4.466...]
    assert m.zone_for_point(1.0, 1.0) == "back-left"
    assert m.zone_for_point(5.0, 1.0) == "back-right"
    # mid-band: y in [4.466, 8.933]
    assert m.zone_for_point(1.0, 6.7) == "mid-left"
    assert m.zone_for_point(5.0, 6.7) == "mid-right"
    # front-band: y in [8.933, 13.4]
    assert m.zone_for_point(1.0, 12.0) == "front-left"
    assert m.zone_for_point(5.0, 12.0) == "front-right"


def test_invalid_corner_count_raises() -> None:
    with pytest.raises(ValueError):
        CourtMapper(image_corners=[(0, 0), (1, 0), (1, 1)])


def test_draw_court_overlay_returns_same_shape() -> None:
    m = _mk()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    out = m.draw_court_overlay(frame)
    assert out.shape == frame.shape
    # Drawing should change at least some pixels.
    assert (out != frame).any()
