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
from apexsim.tire import default_axle_tire_parameters
from apexsim.track import build_circular_track, build_straight_track
from apexsim.vehicle import (
    PointMassPhysics,
    SingleTrackPhysics,
    build_point_mass_model,
    build_single_track_model,
)
from tests.helpers import sample_vehicle_parameters


class TransientQuasiStaticConsistencyTests(unittest.TestCase):
    """Cross-check transient and quasi-static solver outputs on simple tracks."""

    def _point_mass_model(self) -> object:
        """Return validation-configured point-mass model instance.

        Returns:
            Point-mass model instance used in consistency tests.
        """
        return build_point_mass_model(
            vehicle=sample_vehicle_parameters(),
            physics=PointMassPhysics(
                max_drive_accel=7.5,
                max_brake_accel=16.0,
                friction_coefficient=1.8,
            ),
        )

    def _single_track_model(self) -> object:
        """Return low-speed-stable single-track model for PID consistency checks.

        Returns:
            Single-track model instance configured for stable PID validation.
        """
        return build_single_track_model(
            vehicle=sample_vehicle_parameters(),
            tires=default_axle_tire_parameters(),
            physics=SingleTrackPhysics(
                max_drive_accel=7.5,
                max_brake_accel=16.0,
                peak_slip_angle=0.12,
                max_steer_angle=0.55,
                # Lower steer-rate limit stabilizes PID-vs-quasi-static agreement.
                max_steer_rate=2.0,
            ),
        )

    def _quasi_config(self, *, max_speed: float, initial_speed: float) -> object:
        """Build quasi-static config used in consistency comparisons.

        Args:
            max_speed: Runtime speed cap [m/s].
            initial_speed: Initial speed at the first track sample [m/s].

        Returns:
            Validated quasi-static simulation config.
        """
        return build_simulation_config(
            max_speed=max_speed,
            initial_speed=initial_speed,
            compute_backend="numpy",
            solver_mode="quasi_static",
        )

    def _transient_pid_config(self, *, max_speed: float, initial_speed: float) -> object:
        """Build transient PID config used in consistency comparisons.

        Args:
            max_speed: Runtime speed cap [m/s].
            initial_speed: Initial speed at the first track sample [m/s].

        Returns:
            Validated transient PID simulation config.
        """
        return build_simulation_config(
            max_speed=max_speed,
            initial_speed=initial_speed,
            compute_backend="numpy",
            solver_mode="transient_oc",
            transient=TransientConfig(
                numerics=TransientNumericsConfig(
                    control_smoothness_weight=0.0,
                    # Avoid artificial clipping dominating straight/circle checks.
                    max_time_step=1.0,
                ),
                runtime=TransientRuntimeConfig(
                    driver_model="pid",
                    verbosity=0,
                ),
            ),
        )

    @staticmethod
    def _relative_error(value: float, reference: float) -> float:
        """Return relative error with finite denominator floor.

        Args:
            value: Candidate scalar value.
            reference: Reference scalar value.

        Returns:
            Relative absolute error with a denominator floor.
        """
        return abs(value - reference) / max(abs(reference), 1e-9)

    def test_point_mass_transient_matches_quasi_static_on_straight_track(self) -> None:
        """Validate point-mass longitudinal consistency on a straight track."""
        track = build_straight_track(length=700.0, sample_count=161)
        model = self._point_mass_model()
        quasi_result = simulate_lap(
            track=track,
            model=model,
            config=self._quasi_config(max_speed=14.0, initial_speed=12.0),
        )
        transient_result = simulate_lap(
            track=track,
            model=model,
            config=self._transient_pid_config(max_speed=14.0, initial_speed=12.0),
        )

        self.assertLess(float(np.max(np.abs(track.curvature))), 1e-12)
        self.assertLess(float(np.max(np.abs(quasi_result.lateral_accel))), 1e-9)
        self.assertLess(float(np.max(np.abs(transient_result.lateral_accel))), 1e-9)

        lap_time_rel_error = self._relative_error(transient_result.lap_time, quasi_result.lap_time)
        speed_linf_error = float(np.max(np.abs(transient_result.speed - quasi_result.speed)))
        energy_rel_error = self._relative_error(transient_result.energy, quasi_result.energy)

        self.assertLess(lap_time_rel_error, 0.005)
        self.assertLess(speed_linf_error, 0.05)
        self.assertLess(energy_rel_error, 0.005)

    def test_point_mass_transient_matches_quasi_static_on_circular_track(self) -> None:
        """Validate point-mass lateral consistency on a circular track."""
        track = build_circular_track(radius=100.0, sample_count=241)
        model = self._point_mass_model()
        quasi_result = simulate_lap(
            track=track,
            model=model,
            config=self._quasi_config(max_speed=14.0, initial_speed=12.0),
        )
        transient_result = simulate_lap(
            track=track,
            model=model,
            config=self._transient_pid_config(max_speed=14.0, initial_speed=12.0),
        )

        self.assertGreater(float(np.max(np.abs(quasi_result.lateral_accel))), 0.1)
        self.assertGreater(float(np.max(np.abs(transient_result.lateral_accel))), 0.1)

        lap_time_rel_error = self._relative_error(transient_result.lap_time, quasi_result.lap_time)
        speed_linf_error = float(np.max(np.abs(transient_result.speed - quasi_result.speed)))
        energy_rel_error = self._relative_error(transient_result.energy, quasi_result.energy)

        self.assertLess(lap_time_rel_error, 0.005)
        self.assertLess(speed_linf_error, 0.2)
        self.assertLess(energy_rel_error, 0.01)

    def test_single_track_transient_matches_quasi_static_on_straight_track(self) -> None:
        """Validate single-track longitudinal consistency on a straight track."""
        track = build_straight_track(length=700.0, sample_count=161)
        model = self._single_track_model()
        quasi_result = simulate_lap(
            track=track,
            model=model,
            config=self._quasi_config(max_speed=14.0, initial_speed=12.0),
        )
        transient_result = simulate_lap(
            track=track,
            model=model,
            config=self._transient_pid_config(max_speed=14.0, initial_speed=12.0),
        )

        self.assertLess(float(np.max(np.abs(track.curvature))), 1e-12)
        self.assertLess(float(np.max(np.abs(quasi_result.lateral_accel))), 1e-9)
        self.assertLess(float(np.max(np.abs(transient_result.lateral_accel))), 1e-9)

        lap_time_rel_error = self._relative_error(transient_result.lap_time, quasi_result.lap_time)
        speed_linf_error = float(np.max(np.abs(transient_result.speed - quasi_result.speed)))
        speed_vx_linf_error = float(np.max(np.abs(transient_result.speed - transient_result.vx)))
        sideslip_ratio = float(
            np.max(
                np.abs(transient_result.vy)
                / np.maximum(np.abs(transient_result.vx), 1e-9)
            )
        )
        self.assertTrue(np.isfinite(transient_result.energy))
        self.assertTrue(np.isfinite(quasi_result.energy))

        # PID-transient single-track checks are intentionally bounded, not strict.
        self.assertLess(lap_time_rel_error, 0.05)
        self.assertLess(speed_linf_error, 1.0)
        self.assertLess(speed_vx_linf_error, 1e-12)
        self.assertLessEqual(sideslip_ratio, 0.35 + 1e-9)

    def test_single_track_transient_matches_quasi_static_on_circular_track(self) -> None:
        """Validate single-track lateral consistency on a circular track."""
        track = build_circular_track(radius=100.0, sample_count=241)
        model = self._single_track_model()
        quasi_result = simulate_lap(
            track=track,
            model=model,
            config=self._quasi_config(max_speed=14.0, initial_speed=12.0),
        )
        transient_result = simulate_lap(
            track=track,
            model=model,
            config=self._transient_pid_config(max_speed=14.0, initial_speed=12.0),
        )

        quasi_lateral_peak = float(np.max(np.abs(quasi_result.lateral_accel)))
        transient_lateral_peak = float(np.max(np.abs(transient_result.lateral_accel)))
        self.assertGreater(quasi_lateral_peak, 0.1)
        self.assertGreater(transient_lateral_peak, 0.1)

        lap_time_rel_error = self._relative_error(transient_result.lap_time, quasi_result.lap_time)
        speed_linf_error = float(np.max(np.abs(transient_result.speed - quasi_result.speed)))
        speed_vx_linf_error = float(np.max(np.abs(transient_result.speed - transient_result.vx)))
        sideslip_ratio = float(
            np.max(
                np.abs(transient_result.vy)
                / np.maximum(np.abs(transient_result.vx), 1e-9)
            )
        )
        lateral_peak_rel_error = self._relative_error(
            transient_lateral_peak,
            quasi_lateral_peak,
        )
        self.assertTrue(np.isfinite(transient_result.energy))
        self.assertTrue(np.isfinite(quasi_result.energy))

        # Single-track transient PID is compared with bounded model-appropriate tolerances.
        self.assertLess(lap_time_rel_error, 0.08)
        self.assertLess(speed_linf_error, 1.0)
        self.assertLess(lateral_peak_rel_error, 0.08)
        self.assertLess(speed_vx_linf_error, 1e-12)
        self.assertLessEqual(sideslip_ratio, 0.35 + 1e-9)


if __name__ == "__main__":
    unittest.main()
