"""Validation tests for physical sensitivity invariants on synthetic tracks."""

from __future__ import annotations

import unittest
from collections.abc import Callable
from dataclasses import replace

import numpy as np

from apexsim.analysis.sensitivity import (
    SensitivityNumerics,
    SensitivityParameter,
    SensitivityRuntime,
    compute_sensitivities,
)
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
    SingleTrackNumerics,
    SingleTrackPhysics,
    build_point_mass_model,
    build_single_track_model,
)
from tests.helpers import sample_vehicle_parameters


class SensitivityPhysicalPlausibilityTests(unittest.TestCase):
    """Validate physically expected sensitivity limits in E2E scenarios."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create shared synthetic tracks, vehicle data, and FD settings."""
        cls.vehicle = sample_vehicle_parameters()
        cls.low_drag_vehicle = replace(cls.vehicle, drag_coefficient=0.01)
        cls.tires = default_axle_tire_parameters()
        cls.circle_track = build_circular_track(radius=40.0, sample_count=401)
        cls.straight_track = build_straight_track(length=1_200.0, sample_count=301)
        cls.circle_config = build_simulation_config(max_speed=115.0, compute_backend="numpy")
        cls.straight_config = build_simulation_config(
            max_speed=90.0,
            initial_speed=0.0,
            compute_backend="numpy",
        )
        cls.fd_runtime = SensitivityRuntime(method="finite_difference")
        cls.fd_numerics = SensitivityNumerics(
            finite_difference_scheme="central",
            finite_difference_relative_step=1e-3,
            finite_difference_absolute_step=1e-5,
        )

    def _fd_sensitivity(
        self,
        objective: Callable[[dict[str, float]], float],
        parameters: list[SensitivityParameter],
    ) -> dict[str, float]:
        """Compute finite-difference sensitivities for objective and parameters.

        Args:
            objective: Scalar objective callable evaluated by the sensitivity API.
            parameters: Sensitivity-parameter definitions used for perturbations.

        Returns:
            Mapping of parameter name to finite-difference sensitivity.
        """
        result = compute_sensitivities(
            objective=objective,
            parameters=parameters,
            runtime=self.fd_runtime,
            numerics=self.fd_numerics,
        )
        self.assertEqual(result.method, "finite_difference")
        return result.sensitivities

    def test_circle_drive_sensitivity_is_near_zero_when_grip_limited_point_mass(self) -> None:
        """Keep circle lap-time and max|ay| sensitivity near zero vs drive limit."""

        def lap_time_objective(parameters: dict[str, float]) -> float:
            model = build_point_mass_model(
                vehicle=self.low_drag_vehicle,
                physics=PointMassPhysics(
                    max_drive_accel=parameters["max_drive_accel"],
                    max_brake_accel=16.0,
                    friction_coefficient=1.7,
                ),
            )
            return simulate_lap(
                track=self.circle_track,
                model=model,
                config=self.circle_config,
            ).lap_time

        def max_abs_lateral_accel_objective(parameters: dict[str, float]) -> float:
            model = build_point_mass_model(
                vehicle=self.low_drag_vehicle,
                physics=PointMassPhysics(
                    max_drive_accel=parameters["max_drive_accel"],
                    max_brake_accel=16.0,
                    friction_coefficient=1.7,
                ),
            )
            lap = simulate_lap(track=self.circle_track, model=model, config=self.circle_config)
            return float(np.max(np.abs(lap.lateral_accel)))

        low_drive_lap = lap_time_objective({"max_drive_accel": 4.0})
        high_drive_lap = lap_time_objective({"max_drive_accel": 12.0})
        self.assertAlmostEqual(high_drive_lap, low_drive_lap, delta=1e-3)

        parameters = [SensitivityParameter(name="max_drive_accel", value=8.0, lower_bound=0.5)]
        lap_sensitivities = self._fd_sensitivity(lap_time_objective, parameters)
        ay_sensitivities = self._fd_sensitivity(max_abs_lateral_accel_objective, parameters)

        self.assertAlmostEqual(lap_sensitivities["max_drive_accel"], 0.0, delta=1e-3)
        self.assertAlmostEqual(ay_sensitivities["max_drive_accel"], 0.0, delta=1e-12)

    def test_circle_drive_sensitivity_is_near_zero_when_grip_limited_single_track(self) -> None:
        """Keep circle lap-time and max|ay| sensitivity near zero vs drive limit."""

        def lap_time_objective(parameters: dict[str, float]) -> float:
            model = build_single_track_model(
                vehicle=self.low_drag_vehicle,
                tires=self.tires,
                physics=SingleTrackPhysics(
                    max_drive_accel=parameters["max_drive_accel"],
                    max_brake_accel=16.0,
                    peak_slip_angle=0.12,
                ),
                numerics=SingleTrackNumerics(),
            )
            return simulate_lap(
                track=self.circle_track,
                model=model,
                config=self.circle_config,
            ).lap_time

        def max_abs_lateral_accel_objective(parameters: dict[str, float]) -> float:
            model = build_single_track_model(
                vehicle=self.low_drag_vehicle,
                tires=self.tires,
                physics=SingleTrackPhysics(
                    max_drive_accel=parameters["max_drive_accel"],
                    max_brake_accel=16.0,
                    peak_slip_angle=0.12,
                ),
                numerics=SingleTrackNumerics(),
            )
            lap = simulate_lap(track=self.circle_track, model=model, config=self.circle_config)
            return float(np.max(np.abs(lap.lateral_accel)))

        low_drive_lap = lap_time_objective({"max_drive_accel": 4.0})
        high_drive_lap = lap_time_objective({"max_drive_accel": 12.0})
        self.assertAlmostEqual(high_drive_lap, low_drive_lap, delta=1e-3)

        parameters = [SensitivityParameter(name="max_drive_accel", value=8.0, lower_bound=0.5)]
        lap_sensitivities = self._fd_sensitivity(lap_time_objective, parameters)
        ay_sensitivities = self._fd_sensitivity(max_abs_lateral_accel_objective, parameters)

        self.assertAlmostEqual(lap_sensitivities["max_drive_accel"], 0.0, delta=1e-3)
        self.assertAlmostEqual(ay_sensitivities["max_drive_accel"], 0.0, delta=1e-10)

    def test_circle_yaw_moment_sensitivity_matches_model_structure(self) -> None:
        """Keep point-mass yaw responses zero and single-track responses non-zero."""

        def point_mass_max_abs_yaw(parameters: dict[str, float]) -> float:
            model = build_point_mass_model(
                vehicle=self.vehicle,
                physics=PointMassPhysics(
                    max_drive_accel=parameters["max_drive_accel"],
                    max_brake_accel=parameters["max_brake_accel"],
                    friction_coefficient=parameters["friction_coefficient"],
                ),
            )
            lap = simulate_lap(track=self.circle_track, model=model, config=self.circle_config)
            return float(np.max(np.abs(lap.yaw_moment)))

        point_mass_parameters = [
            SensitivityParameter(name="max_drive_accel", value=8.0, lower_bound=0.5),
            SensitivityParameter(name="max_brake_accel", value=16.0, lower_bound=0.5),
            SensitivityParameter(name="friction_coefficient", value=1.7, lower_bound=0.2),
        ]
        point_mass_baseline = point_mass_max_abs_yaw(
            {
                "max_drive_accel": 8.0,
                "max_brake_accel": 16.0,
                "friction_coefficient": 1.7,
            }
        )
        point_mass_sensitivities = self._fd_sensitivity(
            point_mass_max_abs_yaw,
            point_mass_parameters,
        )

        self.assertEqual(point_mass_baseline, 0.0)
        for sensitivity in point_mass_sensitivities.values():
            self.assertAlmostEqual(sensitivity, 0.0, delta=1e-12)

        def single_track_max_abs_yaw(parameters: dict[str, float]) -> float:
            model = build_single_track_model(
                vehicle=self.vehicle,
                tires=self.tires,
                physics=SingleTrackPhysics(
                    max_drive_accel=parameters["max_drive_accel"],
                    max_brake_accel=16.0,
                    peak_slip_angle=parameters["peak_slip_angle"],
                ),
                numerics=SingleTrackNumerics(),
            )
            lap = simulate_lap(track=self.circle_track, model=model, config=self.circle_config)
            return float(np.max(np.abs(lap.yaw_moment)))

        single_track_parameters = [
            SensitivityParameter(name="max_drive_accel", value=8.0, lower_bound=0.5),
            SensitivityParameter(name="peak_slip_angle", value=0.12, lower_bound=0.02),
        ]
        single_track_baseline = single_track_max_abs_yaw(
            {
                "max_drive_accel": 8.0,
                "peak_slip_angle": 0.12,
            }
        )
        single_track_sensitivities = self._fd_sensitivity(
            single_track_max_abs_yaw,
            single_track_parameters,
        )

        self.assertGreater(single_track_baseline, 1e-6)
        self.assertTrue(any(abs(value) > 1e-3 for value in single_track_sensitivities.values()))

    def test_straight_max_lateral_accel_sensitivity_is_zero(self) -> None:
        """Keep max|ay| and its sensitivities near zero on straight tracks."""

        def point_mass_max_abs_lateral_accel(parameters: dict[str, float]) -> float:
            model = build_point_mass_model(
                vehicle=self.vehicle,
                physics=PointMassPhysics(
                    max_drive_accel=parameters["max_drive_accel"],
                    max_brake_accel=16.0,
                    friction_coefficient=parameters["friction_coefficient"],
                ),
            )
            lap = simulate_lap(track=self.straight_track, model=model, config=self.straight_config)
            return float(np.max(np.abs(lap.lateral_accel)))

        point_mass_parameters = [
            SensitivityParameter(name="max_drive_accel", value=8.0, lower_bound=0.5),
            SensitivityParameter(name="friction_coefficient", value=1.7, lower_bound=0.2),
        ]
        point_mass_sensitivities = self._fd_sensitivity(
            point_mass_max_abs_lateral_accel,
            point_mass_parameters,
        )
        self.assertEqual(
            point_mass_max_abs_lateral_accel(
                {"max_drive_accel": 8.0, "friction_coefficient": 1.7}
            ),
            0.0,
        )
        for sensitivity in point_mass_sensitivities.values():
            self.assertAlmostEqual(sensitivity, 0.0, delta=1e-12)

        def single_track_max_abs_lateral_accel(parameters: dict[str, float]) -> float:
            model = build_single_track_model(
                vehicle=self.vehicle,
                tires=self.tires,
                physics=SingleTrackPhysics(
                    max_drive_accel=parameters["max_drive_accel"],
                    max_brake_accel=16.0,
                    peak_slip_angle=parameters["peak_slip_angle"],
                ),
                numerics=SingleTrackNumerics(),
            )
            lap = simulate_lap(track=self.straight_track, model=model, config=self.straight_config)
            return float(np.max(np.abs(lap.lateral_accel)))

        single_track_parameters = [
            SensitivityParameter(name="max_drive_accel", value=8.0, lower_bound=0.5),
            SensitivityParameter(name="peak_slip_angle", value=0.12, lower_bound=0.02),
        ]
        single_track_sensitivities = self._fd_sensitivity(
            single_track_max_abs_lateral_accel,
            single_track_parameters,
        )
        self.assertEqual(
            single_track_max_abs_lateral_accel({"max_drive_accel": 8.0, "peak_slip_angle": 0.12}),
            0.0,
        )
        for sensitivity in single_track_sensitivities.values():
            self.assertAlmostEqual(sensitivity, 0.0, delta=1e-12)

    def test_straight_lap_time_mass_and_drag_share_positive_sign_with_reference_mass(self) -> None:
        """Keep mass/drag lap-time sensitivities positive with force-equivalent scaling."""
        track = build_straight_track(length=700.0, sample_count=161)
        config = build_simulation_config(
            max_speed=90.0,
            initial_speed=0.0,
            compute_backend="numpy",
            solver_mode="transient_oc",
            transient=TransientConfig(
                numerics=TransientNumericsConfig(max_time_step=1.0),
                runtime=TransientRuntimeConfig(driver_model="pid", verbosity=0),
            ),
        )
        reference_mass = self.vehicle.mass

        def lap_time_objective(parameters: dict[str, float]) -> float:
            model = build_single_track_model(
                vehicle=replace(
                    self.vehicle,
                    mass=parameters["mass"],
                    drag_coefficient=parameters["drag_coefficient"],
                ),
                tires=self.tires,
                physics=SingleTrackPhysics(
                    max_drive_accel=8.0,
                    max_brake_accel=16.0,
                    reference_mass=reference_mass,
                    peak_slip_angle=0.12,
                    max_steer_rate=1.0,
                ),
                numerics=SingleTrackNumerics(),
            )
            return simulate_lap(track=track, model=model, config=config).lap_time

        sensitivities = self._fd_sensitivity(
            lap_time_objective,
            [
                SensitivityParameter(name="mass", value=self.vehicle.mass, lower_bound=100.0),
                SensitivityParameter(
                    name="drag_coefficient",
                    value=self.vehicle.drag_coefficient,
                    lower_bound=0.05,
                ),
            ],
        )
        self.assertGreater(sensitivities["mass"], 0.0)
        self.assertGreater(sensitivities["drag_coefficient"], 0.0)

    def test_straight_traction_limited_case_is_insensitive_to_drive_limit(self) -> None:
        """Keep lap-time/max(ax) insensitive to drive limit when tire grip bottlenecks."""
        reference_speed = 10.0
        low_mu = 0.25
        drive_base = 8.0

        model_base = build_point_mass_model(
            vehicle=self.vehicle,
            physics=PointMassPhysics(
                max_drive_accel=drive_base,
                max_brake_accel=16.0,
                friction_coefficient=low_mu,
            ),
        )
        model_high_drive = build_point_mass_model(
            vehicle=self.vehicle,
            physics=PointMassPhysics(
                max_drive_accel=12.0,
                max_brake_accel=16.0,
                friction_coefficient=low_mu,
            ),
        )
        base_accel = model_base.max_longitudinal_accel(
            speed=reference_speed,
            lateral_accel_required=0.0,
            grade=0.0,
            banking=0.0,
        )
        high_drive_accel = model_high_drive.max_longitudinal_accel(
            speed=reference_speed,
            lateral_accel_required=0.0,
            grade=0.0,
            banking=0.0,
        )
        self.assertAlmostEqual(base_accel, high_drive_accel, delta=1e-12)

        def lap_time_objective(parameters: dict[str, float]) -> float:
            model = build_point_mass_model(
                vehicle=self.vehicle,
                physics=PointMassPhysics(
                    max_drive_accel=parameters["max_drive_accel"],
                    max_brake_accel=16.0,
                    friction_coefficient=low_mu,
                ),
            )
            return simulate_lap(
                track=self.straight_track,
                model=model,
                config=self.straight_config,
            ).lap_time

        def max_longitudinal_accel_objective(parameters: dict[str, float]) -> float:
            model = build_point_mass_model(
                vehicle=self.vehicle,
                physics=PointMassPhysics(
                    max_drive_accel=parameters["max_drive_accel"],
                    max_brake_accel=16.0,
                    friction_coefficient=low_mu,
                ),
            )
            lap = simulate_lap(track=self.straight_track, model=model, config=self.straight_config)
            return float(np.max(lap.longitudinal_accel))

        parameters = [
            SensitivityParameter(
                name="max_drive_accel",
                value=drive_base,
                lower_bound=0.5,
            ),
        ]
        lap_sensitivities = self._fd_sensitivity(lap_time_objective, parameters)
        ax_sensitivities = self._fd_sensitivity(max_longitudinal_accel_objective, parameters)

        self.assertAlmostEqual(lap_sensitivities["max_drive_accel"], 0.0, delta=1e-10)
        self.assertAlmostEqual(ax_sensitivities["max_drive_accel"], 0.0, delta=1e-10)

    def test_straight_power_limited_case_is_insensitive_to_friction(self) -> None:
        """Keep lap-time/max(ax) insensitive to friction when drive limit bottlenecks."""
        reference_speed = 10.0
        power_limited_drive = 2.0

        high_mu_model = build_point_mass_model(
            vehicle=self.vehicle,
            physics=PointMassPhysics(
                max_drive_accel=power_limited_drive,
                max_brake_accel=16.0,
                friction_coefficient=1.7,
            ),
        )
        low_mu_model = build_point_mass_model(
            vehicle=self.vehicle,
            physics=PointMassPhysics(
                max_drive_accel=power_limited_drive,
                max_brake_accel=16.0,
                friction_coefficient=1.2,
            ),
        )
        high_mu_accel = high_mu_model.max_longitudinal_accel(
            speed=reference_speed,
            lateral_accel_required=0.0,
            grade=0.0,
            banking=0.0,
        )
        low_mu_accel = low_mu_model.max_longitudinal_accel(
            speed=reference_speed,
            lateral_accel_required=0.0,
            grade=0.0,
            banking=0.0,
        )
        self.assertAlmostEqual(high_mu_accel, low_mu_accel, delta=1e-12)

        def lap_time_objective(parameters: dict[str, float]) -> float:
            model = build_point_mass_model(
                vehicle=self.vehicle,
                physics=PointMassPhysics(
                    max_drive_accel=power_limited_drive,
                    max_brake_accel=16.0,
                    friction_coefficient=parameters["friction_coefficient"],
                ),
            )
            return simulate_lap(
                track=self.straight_track,
                model=model,
                config=self.straight_config,
            ).lap_time

        def max_longitudinal_accel_objective(parameters: dict[str, float]) -> float:
            model = build_point_mass_model(
                vehicle=self.vehicle,
                physics=PointMassPhysics(
                    max_drive_accel=power_limited_drive,
                    max_brake_accel=16.0,
                    friction_coefficient=parameters["friction_coefficient"],
                ),
            )
            lap = simulate_lap(track=self.straight_track, model=model, config=self.straight_config)
            return float(np.max(lap.longitudinal_accel))

        parameters = [
            SensitivityParameter(
                name="friction_coefficient",
                value=1.7,
                lower_bound=0.2,
            ),
        ]
        lap_sensitivities = self._fd_sensitivity(lap_time_objective, parameters)
        ax_sensitivities = self._fd_sensitivity(max_longitudinal_accel_objective, parameters)

        self.assertAlmostEqual(lap_sensitivities["friction_coefficient"], 0.0, delta=1e-10)
        self.assertAlmostEqual(ax_sensitivities["friction_coefficient"], 0.0, delta=1e-10)


if __name__ == "__main__":
    unittest.main()
