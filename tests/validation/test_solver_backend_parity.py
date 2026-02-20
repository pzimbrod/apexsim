"""Cross-backend parity checks for quasi-static and transient solver paths."""

from __future__ import annotations

import importlib.util
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

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
NUMBA_AVAILABLE = importlib.util.find_spec("numba") is not None
SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None
TORCHDIFFEQ_AVAILABLE = importlib.util.find_spec("torchdiffeq") is not None


def _relative_error(value: float, reference: float) -> float:
    """Return stable relative error with finite denominator floor.

    Args:
        value: Candidate value.
        reference: Reference value.

    Returns:
        Relative absolute error with safe denominator floor.
    """
    return abs(value - reference) / max(abs(reference), 1e-9)


@unittest.skipUnless(TORCH_AVAILABLE, "torch backend not available")
class TorchBackendParityTests(unittest.TestCase):
    """Validate NumPy/Torch parity for solver core algorithmics."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build shared models/tracks and backend-specific configs."""
        vehicle = sample_vehicle_parameters()
        cls.point_mass = build_point_mass_model(
            vehicle=vehicle,
            physics=PointMassPhysics(
                max_drive_accel=8.0,
                max_brake_accel=16.0,
                friction_coefficient=1.65,
            ),
        )
        cls.single_track = build_single_track_model(
            vehicle=vehicle,
            tires=default_axle_tire_parameters(),
            physics=SingleTrackPhysics(
                max_drive_accel=8.0,
                max_brake_accel=16.0,
                peak_slip_angle=0.12,
                max_steer_angle=0.55,
                max_steer_rate=2.2,
            ),
        )
        cls.straight_track = build_straight_track(length=700.0, sample_count=161)
        cls.circle_track = build_circular_track(radius=100.0, sample_count=241)
        cls.numpy_qs = build_simulation_config(
            compute_backend="numpy",
            max_speed=34.0,
            initial_speed=0.0,
            solver_mode="quasi_static",
        )
        cls.torch_qs = build_simulation_config(
            compute_backend="torch",
            max_speed=34.0,
            initial_speed=0.0,
            solver_mode="quasi_static",
        )
        transient = TransientConfig(
            numerics=TransientNumericsConfig(
                integration_method="rk4",
                max_iterations=60,
            ),
            runtime=TransientRuntimeConfig(
                driver_model="pid",
                verbosity=0,
            ),
        )
        cls.numpy_transient = build_simulation_config(
            compute_backend="numpy",
            max_speed=34.0,
            initial_speed=0.0,
            solver_mode="transient_oc",
            transient=transient,
        )
        cls.torch_transient = build_simulation_config(
            compute_backend="torch",
            max_speed=34.0,
            initial_speed=0.0,
            solver_mode="transient_oc",
            transient=transient,
        )

    def test_quasi_static_point_mass_numpy_torch_parity_on_straight(self) -> None:
        """Keep point-mass quasi-static profile aligned across NumPy and Torch."""
        numpy_result = simulate_lap(
            track=self.straight_track,
            model=self.point_mass,
            config=self.numpy_qs,
        )
        torch_result = simulate_lap(
            track=self.straight_track,
            model=self.point_mass,
            config=self.torch_qs,
        )
        self.assertLess(_relative_error(torch_result.lap_time, numpy_result.lap_time), 5e-6)
        self.assertLess(
            float(np.max(np.abs(torch_result.speed - numpy_result.speed))),
            1e-6,
        )

    def test_quasi_static_single_track_numpy_torch_parity_on_circle(self) -> None:
        """Keep single-track quasi-static profile aligned across NumPy and Torch."""
        numpy_result = simulate_lap(
            track=self.circle_track,
            model=self.single_track,
            config=self.numpy_qs,
        )
        torch_result = simulate_lap(
            track=self.circle_track,
            model=self.single_track,
            config=self.torch_qs,
        )
        self.assertLess(_relative_error(torch_result.lap_time, numpy_result.lap_time), 5e-6)
        self.assertLess(
            float(np.max(np.abs(torch_result.speed - numpy_result.speed))),
            1e-6,
        )

    def test_transient_pid_point_mass_numpy_torch_parity_on_straight(self) -> None:
        """Keep transient PID point-mass traces aligned across NumPy and Torch."""
        numpy_result = simulate_lap(
            track=self.straight_track,
            model=self.point_mass,
            config=self.numpy_transient,
        )
        torch_result = simulate_lap(
            track=self.straight_track,
            model=self.point_mass,
            config=self.torch_transient,
        )
        self.assertLess(_relative_error(torch_result.lap_time, numpy_result.lap_time), 2e-3)
        self.assertLess(
            float(np.max(np.abs(torch_result.speed - numpy_result.speed))),
            2e-2,
        )
        self.assertLess(
            float(
                np.max(
                    np.abs(
                        torch_result.longitudinal_accel
                        - numpy_result.longitudinal_accel
                    )
                )
            ),
            2e-2,
        )

    def test_transient_pid_single_track_numpy_torch_parity_on_circle(self) -> None:
        """Keep transient PID single-track traces aligned across NumPy and Torch."""
        numpy_result = simulate_lap(
            track=self.circle_track,
            model=self.single_track,
            config=self.numpy_transient,
        )
        torch_result = simulate_lap(
            track=self.circle_track,
            model=self.single_track,
            config=self.torch_transient,
        )
        self.assertLess(_relative_error(torch_result.lap_time, numpy_result.lap_time), 4e-2)
        self.assertLess(
            float(np.max(np.abs(torch_result.speed - numpy_result.speed))),
            8e-1,
        )
        self.assertLess(
            float(
                np.max(
                    np.abs(
                        np.asarray(torch_result.lateral_accel, dtype=float)
                        - np.asarray(numpy_result.lateral_accel, dtype=float)
                    )
                )
            ),
            2.5,
        )

    @unittest.skipUnless(
        TORCHDIFFEQ_AVAILABLE and SCIPY_AVAILABLE,
        "torchdiffeq/scipy not available",
    )
    def test_transient_optimal_control_point_mass_numpy_torch_parity_on_straight(self) -> None:
        """Keep transient OC point-mass output aligned between NumPy and Torch."""
        track = build_straight_track(length=700.0, sample_count=81)
        transient_oc = TransientConfig(
            numerics=TransientNumericsConfig(
                integration_method="rk4",
                max_iterations=5,
                control_interval=1,
                tolerance=1e-2,
                control_smoothness_weight=0.0,
            ),
            runtime=TransientRuntimeConfig(
                driver_model="optimal_control",
                verbosity=0,
            ),
        )
        numpy_config = build_simulation_config(
            compute_backend="numpy",
            max_speed=14.0,
            initial_speed=12.0,
            solver_mode="transient_oc",
            transient=transient_oc,
        )
        torch_config = build_simulation_config(
            compute_backend="torch",
            max_speed=14.0,
            initial_speed=12.0,
            solver_mode="transient_oc",
            transient=transient_oc,
        )
        numpy_result = simulate_lap(track=track, model=self.point_mass, config=numpy_config)
        torch_result = simulate_lap(track=track, model=self.point_mass, config=torch_config)
        self.assertLess(_relative_error(torch_result.lap_time, numpy_result.lap_time), 5e-3)
        self.assertLess(
            float(np.max(np.abs(torch_result.speed - numpy_result.speed))),
            0.2,
        )


@unittest.skipUnless(NUMBA_AVAILABLE, "numba backend not available")
class NumbaBackendParityTests(unittest.TestCase):
    """Validate Numba quasi-static parity against NumPy reference."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build point-mass model and compact straight track."""
        cls.model = build_point_mass_model(
            vehicle=sample_vehicle_parameters(),
            physics=PointMassPhysics(
                max_drive_accel=8.0,
                max_brake_accel=16.0,
                friction_coefficient=1.65,
            ),
        )
        cls.track = build_straight_track(length=700.0, sample_count=161)
        cls.numpy_config = build_simulation_config(
            compute_backend="numpy",
            max_speed=34.0,
            initial_speed=0.0,
            solver_mode="quasi_static",
        )
        cls.numba_config = build_simulation_config(
            compute_backend="numba",
            max_speed=34.0,
            initial_speed=0.0,
            solver_mode="quasi_static",
        )

    def test_quasi_static_point_mass_numba_matches_numpy(self) -> None:
        """Keep numba point-mass quasi-static output aligned with numpy."""
        numpy_result = simulate_lap(track=self.track, model=self.model, config=self.numpy_config)
        numba_result = simulate_lap(track=self.track, model=self.model, config=self.numba_config)
        self.assertLess(_relative_error(numba_result.lap_time, numpy_result.lap_time), 5e-6)
        self.assertLess(
            float(np.max(np.abs(numba_result.speed - numpy_result.speed))),
            1e-6,
        )

    @unittest.skipUnless(SCIPY_AVAILABLE, "scipy not available")
    def test_transient_optimal_control_point_mass_numba_matches_numpy(self) -> None:
        """Keep transient OC point-mass output aligned between NumPy and Numba."""
        track = build_straight_track(length=700.0, sample_count=81)
        transient_oc = TransientConfig(
            numerics=TransientNumericsConfig(
                integration_method="rk4",
                max_iterations=5,
                control_interval=1,
                tolerance=1e-2,
                control_smoothness_weight=0.0,
            ),
            runtime=TransientRuntimeConfig(
                driver_model="optimal_control",
                verbosity=0,
            ),
        )
        numpy_config = build_simulation_config(
            compute_backend="numpy",
            max_speed=14.0,
            initial_speed=12.0,
            solver_mode="transient_oc",
            transient=transient_oc,
        )
        numba_config = build_simulation_config(
            compute_backend="numba",
            max_speed=14.0,
            initial_speed=12.0,
            solver_mode="transient_oc",
            transient=transient_oc,
        )
        numpy_result = simulate_lap(track=track, model=self.model, config=numpy_config)
        numba_result = simulate_lap(track=track, model=self.model, config=numba_config)
        self.assertLess(_relative_error(numba_result.lap_time, numpy_result.lap_time), 5e-3)
        self.assertLess(
            float(np.max(np.abs(numba_result.speed - numpy_result.speed))),
            1e-6,
        )
