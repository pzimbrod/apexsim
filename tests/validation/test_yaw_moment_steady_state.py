"""Validation tests for steady-state yaw-moment behavior on circular tracks."""

from __future__ import annotations

import importlib.util
import unittest
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
from apexsim.track import build_circular_track
from apexsim.utils.exceptions import ConfigurationError
from apexsim.vehicle import (
    PointMassPhysics,
    SingleTrackPhysics,
    build_point_mass_model,
    build_single_track_model,
)
from tests.helpers import sample_vehicle_parameters

NUMBA_AVAILABLE = importlib.util.find_spec("numba") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None


class YawMomentSteadyStateValidationTests(unittest.TestCase):
    """Validate steady-state yaw-moment residual behavior on circular tracks."""

    BACKENDS = ("numpy", "numba", "torch")
    SOLVER_MODES = ("quasi_static", "transient_oc")
    MODELS = ("point_mass", "single_track")

    @classmethod
    def setUpClass(cls) -> None:
        """Create shared fixtures for yaw-residual validation scenarios."""
        cls.vehicle = sample_vehicle_parameters()
        cls.tires = default_axle_tire_parameters()
        cls.circle_track = build_circular_track(radius=100.0, sample_count=81)
        cls.circle_track_sensitivity = build_circular_track(radius=100.0, sample_count=61)
        cls.fd_runtime = SensitivityRuntime(method="finite_difference")
        cls.fd_numerics = SensitivityNumerics(
            finite_difference_scheme="forward",
            finite_difference_relative_step=1e-3,
            finite_difference_absolute_step=1e-5,
        )

    @staticmethod
    def _backend_available(backend: str) -> bool:
        """Return whether a backend dependency is available in this environment.

        Args:
            backend: Backend identifier.

        Returns:
            ``True`` when the backend can be executed in this environment.
        """
        if backend == "numba":
            return NUMBA_AVAILABLE
        if backend == "torch":
            return TORCH_AVAILABLE
        return True

    def _build_model(
        self,
        *,
        model_name: str,
        vehicle: object | None = None,
        max_steer_angle: float = 0.55,
        max_steer_rate: float = 2.0,
    ) -> object:
        """Build one validation model instance.

        Args:
            model_name: Model identifier.
            vehicle: Optional vehicle parameters override.
            max_steer_angle: Single-track steering-angle limit [rad].
            max_steer_rate: Single-track steering-rate limit [rad/s].

        Returns:
            Configured point-mass or single-track model instance.
        """
        resolved_vehicle = self.vehicle if vehicle is None else vehicle
        if model_name == "point_mass":
            return build_point_mass_model(
                vehicle=resolved_vehicle,
                physics=PointMassPhysics(
                    max_drive_accel=7.5,
                    max_brake_accel=16.0,
                    friction_coefficient=1.8,
                ),
            )

        return build_single_track_model(
            vehicle=resolved_vehicle,
            tires=self.tires,
            physics=SingleTrackPhysics(
                max_drive_accel=7.5,
                max_brake_accel=16.0,
                peak_slip_angle=0.12,
                max_steer_angle=max_steer_angle,
                max_steer_rate=max_steer_rate,
            ),
        )

    @staticmethod
    def _steady_state_mask(longitudinal_accel: np.ndarray) -> np.ndarray:
        """Build steady-state mask on circle tracks via interior+|ax| filtering.

        Args:
            longitudinal_accel: Longitudinal-acceleration trace [m/s^2].

        Returns:
            Boolean mask selecting steady-state samples.
        """
        accel = np.asarray(longitudinal_accel, dtype=float)
        n = int(accel.size)
        trim = max(8, int(0.25 * n))
        interior = np.zeros(n, dtype=bool)
        if n <= 2 * trim:
            interior[:] = True
        else:
            interior[trim : n - trim] = True

        accel_mask = np.abs(accel) <= 0.3
        combined = interior & accel_mask
        minimum_points = max(3, n // 10)
        if int(np.count_nonzero(combined)) < minimum_points:
            return interior
        return combined

    def _steady_state_eta(
        self,
        *,
        model: object,
        yaw_moment: np.ndarray,
        lateral_accel: np.ndarray,
        longitudinal_accel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute normalized steady-state yaw-residual trace.

        Args:
            model: Vehicle model holding physical scales.
            yaw_moment: Yaw-moment trace [N*m].
            lateral_accel: Lateral-acceleration trace [m/s^2].
            longitudinal_accel: Longitudinal-acceleration trace [m/s^2].

        Returns:
            Tuple ``(eta, mask)`` with normalized yaw residual and steady-state mask.
        """
        mass = float(model.vehicle.mass)  # type: ignore[attr-defined]
        wheelbase = float(model.vehicle.wheelbase)  # type: ignore[attr-defined]
        scale = np.maximum(mass * np.abs(lateral_accel) * wheelbase, 1.0)
        eta = np.abs(yaw_moment) / scale
        mask = self._steady_state_mask(longitudinal_accel)
        return eta, mask

    def _steady_state_eta_stats(self, *, model: object, lap: object) -> tuple[float, float]:
        """Compute steady-state mean/max normalized yaw residual for one lap run.

        Args:
            model: Vehicle model used for the lap simulation.
            lap: Lap result instance.

        Returns:
            Tuple ``(mean_eta, max_eta)`` over the steady-state mask.
        """
        eta, mask = self._steady_state_eta(
            model=model,
            yaw_moment=np.asarray(lap.yaw_moment, dtype=float),
            lateral_accel=np.asarray(lap.lateral_accel, dtype=float),
            longitudinal_accel=np.asarray(lap.longitudinal_accel, dtype=float),
        )
        return float(np.mean(eta[mask])), float(np.max(eta[mask]))

    def _build_config(
        self,
        *,
        backend: str,
        solver_mode: str,
        driver_model: str = "pid",
        max_time_step: float = 1.0,
        max_iterations: int = 30,
        control_interval: int = 12,
        steer_kp: float | None = None,
        steer_ki: float | None = None,
        steer_kd: float | None = None,
        steer_vy_damping: float | None = None,
        steer_integral_limit: float | None = None,
    ) -> object:
        """Build one simulation configuration used in validation matrix runs.

        Args:
            backend: Compute backend identifier.
            solver_mode: Solver-mode identifier.
            driver_model: Transient driver model identifier.
            max_time_step: Upper integration step bound [s].
            max_iterations: Maximum transient optimizer iterations.
            control_interval: Control mesh interval in samples.
            steer_kp: Optional transient PID steering proportional gain.
            steer_ki: Optional transient PID steering integral gain.
            steer_kd: Optional transient PID steering derivative gain.
            steer_vy_damping: Optional transient lateral-speed damping gain.
            steer_integral_limit: Optional transient PID integral clamp.

        Returns:
            Validated simulation config instance.
        """
        if solver_mode == "quasi_static":
            return build_simulation_config(
                max_speed=14.0,
                initial_speed=12.0,
                compute_backend=backend,
                solver_mode=solver_mode,
            )

        transient_numerics_kwargs: dict[str, float | int] = {
            "max_time_step": max_time_step,
            "control_smoothness_weight": 0.0,
            "max_iterations": max_iterations,
            "control_interval": control_interval,
        }
        if steer_kp is not None:
            transient_numerics_kwargs["pid_steer_kp"] = steer_kp
        if steer_ki is not None:
            transient_numerics_kwargs["pid_steer_ki"] = steer_ki
        if steer_kd is not None:
            transient_numerics_kwargs["pid_steer_kd"] = steer_kd
        if steer_vy_damping is not None:
            transient_numerics_kwargs["pid_steer_vy_damping"] = steer_vy_damping
        if steer_integral_limit is not None:
            transient_numerics_kwargs["pid_steer_integral_limit"] = steer_integral_limit

        return build_simulation_config(
            max_speed=14.0,
            initial_speed=12.0,
            compute_backend=backend,
            solver_mode=solver_mode,
            transient=TransientConfig(
                numerics=TransientNumericsConfig(**transient_numerics_kwargs),
                runtime=TransientRuntimeConfig(driver_model=driver_model, verbosity=0),
            ),
        )

    def test_circle_steady_state_yaw_residual_matrix(self) -> None:
        """Validate steady-state yaw-residual levels across model/backend/mode matrix."""
        for backend in self.BACKENDS:
            if not self._backend_available(backend):
                continue
            for solver_mode in self.SOLVER_MODES:
                for model_name in self.MODELS:
                    with self.subTest(
                        backend=backend,
                        solver_mode=solver_mode,
                        model=model_name,
                    ):
                        model = self._build_model(model_name=model_name)
                        config = self._build_config(backend=backend, solver_mode=solver_mode)
                        lap = simulate_lap(track=self.circle_track, model=model, config=config)
                        mean_eta, max_eta = self._steady_state_eta_stats(model=model, lap=lap)

                        if model_name == "single_track" and solver_mode == "transient_oc":
                            self.assertLessEqual(mean_eta, 5e-3)
                            self.assertLessEqual(max_eta, 3e-2)
                        else:
                            self.assertLessEqual(max_eta, 1e-10)

    def test_circle_steady_state_yaw_residual_sensitivity_matrix(self) -> None:
        """Keep steady-state yaw-residual sensitivities bounded across full matrix."""
        parameter_names = ("mass", "cg_height", "yaw_inertia", "drag_coefficient")
        lower_bounds = {
            "mass": 100.0,
            "cg_height": 0.05,
            "yaw_inertia": 10.0,
            "drag_coefficient": 0.01,
        }

        for backend in self.BACKENDS:
            if not self._backend_available(backend):
                continue
            for solver_mode in self.SOLVER_MODES:
                for model_name in self.MODELS:
                    with self.subTest(
                        backend=backend,
                        solver_mode=solver_mode,
                        model=model_name,
                    ):
                        baseline_vehicle = self.vehicle
                        config = self._build_config(backend=backend, solver_mode=solver_mode)
                        cache: dict[tuple[float, ...], float] = {}

                        def objective(
                            parameter_values: dict[str, float],
                            *,
                            _cache: dict[tuple[float, ...], float] = cache,
                            _baseline_vehicle: object = baseline_vehicle,
                            _model_name: str = model_name,
                            _config: object = config,
                        ) -> float:
                            key = tuple(float(parameter_values[name]) for name in parameter_names)
                            cached = _cache.get(key)
                            if cached is not None:
                                return cached

                            vehicle = replace(
                                _baseline_vehicle,
                                **{
                                    name: float(parameter_values[name])
                                    for name in parameter_names
                                },
                            )
                            model = self._build_model(model_name=_model_name, vehicle=vehicle)
                            lap = simulate_lap(
                                track=self.circle_track_sensitivity,
                                model=model,
                                config=_config,
                            )
                            mean_eta, _ = self._steady_state_eta_stats(model=model, lap=lap)
                            _cache[key] = mean_eta
                            return mean_eta

                        result = compute_sensitivities(
                            objective=objective,
                            parameters=[
                                SensitivityParameter(
                                    name=name,
                                    value=float(getattr(baseline_vehicle, name)),
                                    lower_bound=lower_bounds[name],
                                )
                                for name in parameter_names
                            ],
                            runtime=self.fd_runtime,
                            numerics=self.fd_numerics,
                        )
                        self.assertEqual(result.method, "finite_difference")

                        if model_name == "single_track" and solver_mode == "transient_oc":
                            bound = 1e-3
                        else:
                            bound = 1e-8
                        for sensitivity in result.sensitivities.values():
                            self.assertLessEqual(abs(sensitivity), bound)

    def test_circle_pid_driver_aggressive_gains_raise_yaw_residual(self) -> None:
        """Use yaw residual as instability indicator for unrealistic PID steering gains."""
        baseline_model = self._build_model(model_name="single_track")
        baseline_config = self._build_config(backend="numpy", solver_mode="transient_oc")
        baseline_lap = simulate_lap(
            track=self.circle_track,
            model=baseline_model,
            config=baseline_config,
        )
        baseline_mean_eta, _baseline_max_eta = self._steady_state_eta_stats(
            model=baseline_model,
            lap=baseline_lap,
        )

        aggressive_model = self._build_model(
            model_name="single_track",
            max_steer_angle=1.2,
            max_steer_rate=40.0,
        )
        aggressive_config = self._build_config(
            backend="numpy",
            solver_mode="transient_oc",
            steer_kp=80.0,
            steer_ki=20.0,
            steer_kd=0.0,
            steer_vy_damping=-2.0,
            steer_integral_limit=10.0,
        )
        aggressive_lap = simulate_lap(
            track=self.circle_track,
            model=aggressive_model,
            config=aggressive_config,
        )
        aggressive_mean_eta, _aggressive_max_eta = self._steady_state_eta_stats(
            model=aggressive_model,
            lap=aggressive_lap,
        )

        self.assertLessEqual(baseline_mean_eta, 5e-3)
        self.assertGreaterEqual(aggressive_mean_eta, 5e-2)
        self.assertGreaterEqual(
            aggressive_mean_eta,
            20.0 * max(baseline_mean_eta, 1e-6),
        )

    @unittest.skipUnless(SCIPY_AVAILABLE, "SciPy not installed")
    def test_circle_optimal_control_low_iteration_case_exposes_nonsteady_yaw_residual(self) -> None:
        """Fail fast when low-iteration optimal-control setup cannot converge."""
        track = build_circular_track(radius=100.0, sample_count=41)
        model = self._build_model(model_name="single_track")
        config = self._build_config(
            backend="numpy",
            solver_mode="transient_oc",
            driver_model="optimal_control",
            max_time_step=1.0,
            max_iterations=2,
            control_interval=6,
        )
        with self.assertRaises(ConfigurationError):
            simulate_lap(track=track, model=model, config=config)


if __name__ == "__main__":
    unittest.main()
