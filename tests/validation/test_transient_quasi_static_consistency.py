"""Validation tests for transient/quasi-static consistency on simple tracks."""

from __future__ import annotations

import unittest

import numpy as np

from apexsim.simulation import (
    TransientConfig,
    TransientNumericsConfig,
    TransientRuntimeConfig,
    build_simulation_config,
    simulate_lap,
)
from apexsim.track import build_straight_track
from apexsim.vehicle import PointMassPhysics, build_point_mass_model
from tests.helpers import sample_vehicle_parameters


class TransientQuasiStaticConsistencyTests(unittest.TestCase):
    """Check that transient point-mass results match quasi-static baselines."""

    def test_point_mass_transient_matches_quasi_static_on_straight_track(self) -> None:
        """Keep point-mass transient and quasi-static outputs aligned on a straight."""
        track = build_straight_track(length=700.0, sample_count=161)
        model = build_point_mass_model(
            vehicle=sample_vehicle_parameters(),
            physics=PointMassPhysics(
                max_drive_accel=7.5,
                max_brake_accel=16.0,
                friction_coefficient=1.8,
            ),
        )

        # A non-zero initial speed avoids standing-start singular behavior and
        # keeps the comparison focused on solver consistency.
        quasi_config = build_simulation_config(
            max_speed=80.0,
            initial_speed=20.0,
            compute_backend="numpy",
            solver_mode="quasi_static",
        )
        transient_config = build_simulation_config(
            max_speed=80.0,
            initial_speed=20.0,
            compute_backend="numpy",
            solver_mode="transient_oc",
            transient=TransientConfig(
                numerics=TransientNumericsConfig(
                    max_iterations=20,
                    control_interval=8,
                    control_smoothness_weight=0.0,
                ),
                runtime=TransientRuntimeConfig(verbosity=0),
            ),
        )

        quasi_result = simulate_lap(track=track, model=model, config=quasi_config)
        transient_result = simulate_lap(track=track, model=model, config=transient_config)

        self.assertEqual(transient_result.solver_mode, "transient_oc")
        self.assertEqual(quasi_result.solver_mode, "quasi_static")
        self.assertLess(float(np.max(np.abs(track.curvature))), 1e-12)

        lap_time_rel_error = abs(transient_result.lap_time - quasi_result.lap_time) / max(
            quasi_result.lap_time,
            1e-9,
        )
        speed_linf_error = float(np.max(np.abs(transient_result.speed - quasi_result.speed)))
        energy_rel_error = abs(transient_result.energy - quasi_result.energy) / max(
            quasi_result.energy,
            1e-9,
        )

        self.assertLess(lap_time_rel_error, 0.01)
        self.assertLess(speed_linf_error, 0.5)
        self.assertLess(energy_rel_error, 0.01)
        self.assertLess(float(np.max(np.abs(transient_result.lateral_accel))), 1e-9)
        self.assertLess(float(np.max(np.abs(quasi_result.lateral_accel))), 1e-9)

    def test_point_mass_transient_with_pid_scheduling_stays_close_on_straight(self) -> None:
        """Bound transient-vs-quasi-static mismatch when PID scheduling is enabled."""
        track = build_straight_track(length=700.0, sample_count=161)
        model = build_point_mass_model(
            vehicle=sample_vehicle_parameters(),
            physics=PointMassPhysics(
                max_drive_accel=7.5,
                max_brake_accel=16.0,
                friction_coefficient=1.8,
            ),
        )
        quasi_config = build_simulation_config(
            max_speed=80.0,
            initial_speed=20.0,
            compute_backend="numpy",
            solver_mode="quasi_static",
        )
        transient_config = build_simulation_config(
            max_speed=80.0,
            initial_speed=20.0,
            compute_backend="numpy",
            solver_mode="transient_oc",
            transient=TransientConfig(
                numerics=TransientNumericsConfig(
                    max_iterations=20,
                    control_interval=8,
                    control_smoothness_weight=0.0,
                    pid_gain_scheduling_mode="physics_informed",
                ),
                runtime=TransientRuntimeConfig(verbosity=0),
            ),
        )

        quasi_result = simulate_lap(track=track, model=model, config=quasi_config)
        transient_result = simulate_lap(track=track, model=model, config=transient_config)
        lap_time_rel_error = abs(transient_result.lap_time - quasi_result.lap_time) / max(
            quasi_result.lap_time,
            1e-9,
        )
        speed_linf_error = float(np.max(np.abs(transient_result.speed - quasi_result.speed)))
        energy_rel_error = abs(transient_result.energy - quasi_result.energy) / max(
            quasi_result.energy,
            1e-9,
        )

        self.assertLess(lap_time_rel_error, 0.02)
        self.assertLess(speed_linf_error, 1.0)
        self.assertLess(energy_rel_error, 0.02)


if __name__ == "__main__":
    unittest.main()
