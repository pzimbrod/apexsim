"""Unit tests for synthetic track-layout builders."""

from __future__ import annotations

import unittest

import numpy as np

from apexsim.track import (
    build_circular_track,
    build_figure_eight_track,
    build_straight_track,
)
from apexsim.utils.exceptions import TrackDataError


class TrackLayoutBuilderTests(unittest.TestCase):
    """Validate geometry properties of synthetic track-layout helpers."""

    def test_straight_layout_has_zero_curvature_and_expected_length(self) -> None:
        """Build straight layout with near-zero curvature over the full length."""
        track = build_straight_track(length=1_000.0, sample_count=501)

        self.assertAlmostEqual(track.length, 1_000.0, delta=1e-9)
        self.assertLess(float(np.max(np.abs(track.curvature))), 1e-12)
        self.assertLess(float(np.max(np.abs(track.grade))), 1e-12)
        self.assertLess(float(np.max(np.abs(track.banking))), 1e-12)

    def test_circle_layout_has_constant_curvature_away_from_seam(self) -> None:
        """Build circular layout with nearly constant interior curvature."""
        radius = 50.0
        track = build_circular_track(radius=radius, sample_count=720)

        interior = track.curvature[20:-20]
        self.assertAlmostEqual(float(np.mean(interior)), 1.0 / radius, delta=3e-3)
        self.assertLess(float(np.std(interior)), 2e-3)
        self.assertAlmostEqual(track.length, 2.0 * np.pi * radius, delta=0.5)

    def test_circle_layout_can_flip_curvature_sign_for_clockwise_direction(self) -> None:
        """Build clockwise circle layout with negative signed interior curvature."""
        track = build_circular_track(radius=50.0, sample_count=720, clockwise=True)

        interior = track.curvature[20:-20]
        self.assertLess(float(np.mean(interior)), -1e-3)

    def test_figure_eight_changes_curvature_sign_and_crosses_center(self) -> None:
        """Build figure-eight layout with positive and negative curvature lobes."""
        track = build_figure_eight_track(lobe_radius=80.0, sample_count=1_200)

        self.assertLess(float(np.min(track.curvature)), -1e-3)
        self.assertGreater(float(np.max(track.curvature)), 1e-3)

        near_center = np.hypot(track.x, track.y) < 1e-6
        self.assertGreaterEqual(int(np.count_nonzero(near_center)), 3)

    def test_layout_builders_reject_invalid_parameters(self) -> None:
        """Reject non-physical geometric parameters for layout generation."""
        with self.assertRaises(TrackDataError):
            build_straight_track(length=0.0)
        with self.assertRaises(TrackDataError):
            build_circular_track(radius=-10.0)
        with self.assertRaises(TrackDataError):
            build_figure_eight_track(lobe_radius=0.0)
        with self.assertRaises(TrackDataError):
            build_circular_track(sample_count=3)


if __name__ == "__main__":
    unittest.main()
