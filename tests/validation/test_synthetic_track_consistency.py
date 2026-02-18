"""Physical-consistency checks on synthetic benchmark tracks."""

from __future__ import annotations

import unittest

import numpy as np

from apexsim.simulation import build_simulation_config, simulate_lap
from apexsim.track import (
    build_circular_track,
    build_figure_eight_track,
    build_straight_track,
)
from apexsim.vehicle import PointMassPhysics, build_point_mass_model
from tests.helpers import sample_vehicle_parameters

INTERIOR_TRIM = 30
SIGN_CHANGE_ZERO_TOLERANCE = 1e-4


class SyntheticTrackPhysicalConsistencyTests(unittest.TestCase):
    """Validate core physical invariants on canonical synthetic layouts."""

    def setUp(self) -> None:
        """Create a shared model and solver config for scenario validations."""
        vehicle = sample_vehicle_parameters()
        self.model = build_point_mass_model(vehicle=vehicle, physics=PointMassPhysics())
        self.config = build_simulation_config(max_speed=115.0)

    def test_straight_track_has_no_lateral_dynamics(self) -> None:
        """Keep lateral acceleration and curvature at zero on a straight track."""
        track = build_straight_track(length=1_000.0, sample_count=501)
        result = simulate_lap(track=track, model=self.model, config=self.config)

        self.assertLess(float(np.max(np.abs(track.curvature))), 1e-12)
        self.assertLess(float(np.max(np.abs(result.lateral_accel))), 1e-12)

    def test_circle_track_reaches_quasi_steady_cornering_state(self) -> None:
        """Hold near-constant speed and near-zero longitudinal accel in the corner."""
        track = build_circular_track(radius=50.0, sample_count=720)
        result = simulate_lap(track=track, model=self.model, config=self.config)

        interior = slice(INTERIOR_TRIM, -INTERIOR_TRIM)
        speed = result.speed[interior]
        longitudinal_accel = result.longitudinal_accel[interior]
        lateral_accel = result.lateral_accel[interior]

        self.assertLess(float(np.std(speed)), 0.2)
        self.assertLess(float(np.mean(np.abs(longitudinal_accel))), 0.25)
        self.assertGreater(float(np.min(lateral_accel)), 0.0)

    def test_figure_eight_track_contains_turn_transition_dynamics(self) -> None:
        """Show sign changes in lateral dynamics and accel/brake transitions."""
        track = build_figure_eight_track(lobe_radius=80.0, sample_count=1_200)
        result = simulate_lap(track=track, model=self.model, config=self.config)

        lateral_accel = result.lateral_accel
        longitudinal_accel = result.longitudinal_accel

        self.assertLess(float(np.min(lateral_accel)), -1.0)
        self.assertGreater(float(np.max(lateral_accel)), 1.0)
        self.assertLess(float(np.min(longitudinal_accel)), -1.0)
        self.assertGreater(float(np.max(longitudinal_accel)), 1.0)

        sign = np.sign(
            np.where(np.abs(lateral_accel) < SIGN_CHANGE_ZERO_TOLERANCE, 0.0, lateral_accel)
        )
        non_zero_sign = sign[sign != 0.0]
        sign_changes = int(np.count_nonzero(np.diff(non_zero_sign)))
        self.assertGreaterEqual(sign_changes, 1)


if __name__ == "__main__":
    unittest.main()
