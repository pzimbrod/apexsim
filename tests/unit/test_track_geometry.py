"""Unit tests for track geometry calculations."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lap_time_sim.track.geometry import build_track_data
from lap_time_sim.track.io import load_track_csv
from lap_time_sim.utils.exceptions import TrackDataError


class TrackGeometryTests(unittest.TestCase):
    """Track parser and geometry checks."""

    def test_curvature_of_circle_is_nearly_constant(self) -> None:
        """Approximate constant curvature for a discretized circle."""
        radius = 120.0
        t = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        elevation = np.zeros_like(x)
        banking = np.zeros_like(x)

        track = build_track_data(x, y, elevation, banking)
        curvature = track.curvature_1pm[10:-10]
        self.assertAlmostEqual(float(np.mean(curvature)), 1.0 / radius, delta=2e-3)
        self.assertGreater(track.length_m, 700.0)

    def test_loader_rejects_missing_columns(self) -> None:
        """Reject CSV inputs missing required track columns."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_track.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["x", "y", "elevation"])
                writer.writerow([0.0, 0.0, 0.0])
                writer.writerow([1.0, 0.0, 0.0])
                writer.writerow([2.0, 0.0, 0.0])
                writer.writerow([3.0, 0.0, 0.0])

            with self.assertRaises(TrackDataError):
                load_track_csv(path)

    def test_loader_rejects_missing_file(self) -> None:
        """Reject non-existing track files."""
        with self.assertRaises(TrackDataError):
            load_track_csv("does_not_exist.csv")

    def test_loader_rejects_file_without_header(self) -> None:
        """Reject CSV files that do not include a header row."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "headerless.csv"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(TrackDataError):
                load_track_csv(path)

    def test_loader_rejects_too_few_rows(self) -> None:
        """Reject CSV inputs with fewer than four data points."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "too_short.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["x", "y", "elevation", "banking"])
                writer.writerow([0.0, 0.0, 0.0, 0.0])
                writer.writerow([1.0, 0.0, 0.0, 0.0])
                writer.writerow([2.0, 0.0, 0.0, 0.0])

            with self.assertRaises(TrackDataError):
                load_track_csv(path)


if __name__ == "__main__":
    unittest.main()
