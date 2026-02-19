"""Unit tests for transient optimal-control solver modules."""

from __future__ import annotations

import builtins
import importlib.util
import io
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

from apexsim.analysis import (
    SensitivityNumerics,
    SensitivityRuntime,
    SensitivityStudyParameter,
    run_lap_sensitivity_study,
)
from apexsim.simulation import (
    PidSpeedSchedule,
    TransientConfig,
    TransientNumericsConfig,
    TransientPidGainSchedulingConfig,
    TransientRuntimeConfig,
    build_simulation_config,
    simulate_lap,
)
from apexsim.simulation.model_api import ModelDiagnostics
from apexsim.simulation.profile import SpeedProfileResult
from apexsim.simulation.runner import _compute_diagnostics
from apexsim.simulation.transient_numba import solve_transient_lap_numba
from apexsim.simulation.transient_numpy import (
    _build_control_mesh_positions,
    _decode_point_mass_controls,
    _decode_single_track_controls,
    _expand_mesh_controls,
    _require_scipy_optimize,
    _resolve_pid_gain_scheduling,
    _resolve_schedule_speed_nodes,
    build_physics_informed_pid_gain_scheduling,
    solve_transient_lap_numpy,
)
from apexsim.simulation.transient_torch import (
    _build_control_node_count,
    solve_transient_lap_torch,
)
from apexsim.tire import default_axle_tire_parameters
from apexsim.track import build_circular_track, build_straight_track
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


class _FakeScipyOptimize:
    """Minimal SciPy optimize stand-in used for offline transient tests."""

    @staticmethod
    def minimize(
        objective: Any,
        x0: np.ndarray,
        method: str,
        options: dict[str, Any],
        callback: Any | None = None,
    ) -> Any:
        """Return deterministic one-shot optimization result.

        Args:
            objective: Objective callable.
            x0: Initial decision vector.
            method: Optimizer method identifier.
            options: Optimizer options mapping.
            callback: Optional callback invoked once with ``x0``.

        Returns:
            Namespace with ``x``, ``fun``, and ``success`` fields.
        """
        del method, options
        x0_array = np.asarray(x0, dtype=float)
        if callback is not None:
            callback(x0_array)
        return SimpleNamespace(
            x=x0_array,
            fun=float(objective(x0_array)),
            success=True,
            nit=1,
        )


def _patch_transient_dependency_specs() -> Any:
    """Patch dependency discovery for transient solver extras.

    Returns:
        Mock patcher context object.
    """
    real_find_spec = importlib.util.find_spec
    return patch(
        "apexsim.simulation.config.importlib.util.find_spec",
        side_effect=lambda name: (
            object() if name in {"scipy", "torchdiffeq"} else real_find_spec(name)
        ),
    )


def _custom_pid_schedule() -> TransientPidGainSchedulingConfig:
    """Return compact custom PID schedule used by transient tests.

    Returns:
        Valid scheduling config with longitudinal and steering schedules.
    """
    nodes = (0.0, 20.0, 60.0)
    return TransientPidGainSchedulingConfig(
        longitudinal_kp=PidSpeedSchedule(nodes, (0.8, 0.75, 0.7)),
        longitudinal_ki=PidSpeedSchedule(nodes, (0.02, 0.015, 0.01)),
        longitudinal_kd=PidSpeedSchedule(nodes, (0.06, 0.055, 0.05)),
        steer_kp=PidSpeedSchedule(nodes, (1.8, 1.4, 0.9)),
        steer_ki=PidSpeedSchedule(nodes, (0.08, 0.05, 0.03)),
        steer_kd=PidSpeedSchedule(nodes, (0.16, 0.12, 0.09)),
        steer_vy_damping=PidSpeedSchedule(nodes, (0.2, 0.27, 0.35)),
    )


class TransientSolverTests(unittest.TestCase):
    """Validate transient OC solver behavior and dispatch integration."""

    def test_solve_transient_lap_numpy_point_mass_runs(self) -> None:
        """Solve point-mass transient profile on numpy backend."""
        track = build_straight_track(length=240.0, sample_count=61)
        model = build_point_mass_model(
            vehicle=sample_vehicle_parameters(),
            physics=PointMassPhysics(),
        )
        with (
            _patch_transient_dependency_specs(),
            patch(
                "apexsim.simulation.transient_numpy._require_scipy_optimize",
                return_value=_FakeScipyOptimize(),
            ),
        ):
            config = build_simulation_config(
                compute_backend="numpy",
                solver_mode="transient_oc",
                max_speed=70.0,
                initial_speed=15.0,
                transient=TransientConfig(
                    runtime=TransientRuntimeConfig(driver_model="optimal_control")
                ),
            )
            profile = solve_transient_lap_numpy(track=track, model=model, config=config)

        self.assertEqual(profile.speed.shape, track.arc_length.shape)
        self.assertGreater(profile.lap_time, 0.0)
        self.assertEqual(profile.steer_cmd.shape, track.arc_length.shape)
        self.assertTrue(np.allclose(profile.vy, 0.0))

    def test_solve_transient_lap_numpy_single_track_runs(self) -> None:
        """Solve single-track transient profile on numpy backend."""
        track = build_circular_track(radius=55.0, sample_count=181)
        model = build_single_track_model(
            vehicle=sample_vehicle_parameters(),
            tires=default_axle_tire_parameters(),
            physics=SingleTrackPhysics(),
        )
        with (
            _patch_transient_dependency_specs(),
            patch(
                "apexsim.simulation.transient_numpy._require_scipy_optimize",
                return_value=_FakeScipyOptimize(),
            ),
        ):
            config = build_simulation_config(
                compute_backend="numpy",
                solver_mode="transient_oc",
                max_speed=70.0,
                initial_speed=20.0,
                transient=TransientConfig(
                    runtime=TransientRuntimeConfig(driver_model="optimal_control")
                ),
            )
            profile = solve_transient_lap_numpy(track=track, model=model, config=config)

        self.assertEqual(profile.speed.shape, track.arc_length.shape)
        self.assertGreater(profile.lap_time, 0.0)
        self.assertGreater(float(np.max(np.abs(profile.yaw_rate))), 0.0)
        self.assertGreater(float(np.max(np.abs(profile.steer_cmd))), 0.0)

    def test_solve_transient_lap_numpy_single_track_runs_with_pid_driver(self) -> None:
        """Solve single-track transient profile using default PID driver mode."""
        track = build_circular_track(radius=55.0, sample_count=181)
        model = build_single_track_model(
            vehicle=sample_vehicle_parameters(),
            tires=default_axle_tire_parameters(),
            physics=SingleTrackPhysics(),
        )
        with _patch_transient_dependency_specs():
            config = build_simulation_config(
                compute_backend="numpy",
                solver_mode="transient_oc",
                max_speed=70.0,
                initial_speed=20.0,
            )
            profile = solve_transient_lap_numpy(track=track, model=model, config=config)

        self.assertEqual(profile.speed.shape, track.arc_length.shape)
        self.assertGreater(profile.lap_time, 0.0)
        self.assertGreater(float(np.max(np.abs(profile.yaw_rate))), 0.0)
        self.assertGreater(float(np.max(np.abs(profile.steer_cmd))), 0.0)

    def test_physics_informed_pid_scheduling_builder_is_deterministic(self) -> None:
        """Build deterministic physics-informed schedules with expected trends."""
        model = build_single_track_model(
            vehicle=sample_vehicle_parameters(),
            tires=default_axle_tire_parameters(),
            physics=SingleTrackPhysics(),
        )
        numerics = TransientNumericsConfig()
        schedule_a = build_physics_informed_pid_gain_scheduling(
            model=model,
            numerics=numerics,
            max_speed=80.0,
        )
        schedule_b = build_physics_informed_pid_gain_scheduling(
            model=model,
            numerics=numerics,
            max_speed=80.0,
        )
        self.assertEqual(schedule_a, schedule_b)
        self.assertIsNotNone(schedule_a.steer_kp)
        self.assertIsNotNone(schedule_a.steer_vy_damping)
        self.assertIsNotNone(schedule_a.longitudinal_kp)
        if (
            schedule_a.steer_kp is None
            or schedule_a.steer_vy_damping is None
            or schedule_a.longitudinal_kp is None
        ):
            self.fail("Expected fully defined physics-informed schedule")
        steer_kp_values = np.asarray(schedule_a.steer_kp.values, dtype=float)
        steer_vy_values = np.asarray(schedule_a.steer_vy_damping.values, dtype=float)
        self.assertTrue(np.all(np.diff(steer_kp_values) <= 1e-12))
        self.assertTrue(np.all(np.diff(steer_vy_values) >= -1e-12))
        self.assertEqual(
            schedule_a.longitudinal_kp.speed_nodes_mps,
            schedule_a.steer_kp.speed_nodes_mps,
        )

    def test_schedule_speed_node_resolution_covers_explicit_and_zero_speed_cases(self) -> None:
        """Resolve schedule nodes for explicit grids and degenerate speed caps."""
        explicit = _resolve_schedule_speed_nodes(
            max_speed=80.0,
            speed_nodes_mps=(0.0, 8.0, 16.0),
        )
        self.assertEqual(explicit, (0.0, 8.0, 16.0))
        zero_cap = _resolve_schedule_speed_nodes(max_speed=0.0)
        self.assertEqual(zero_cap, (0.0, 1.0))

    def test_torch_control_node_count_handles_single_sample(self) -> None:
        """Cover single-sample control node sizing branch in torch helper."""
        self.assertEqual(_build_control_node_count(sample_count=1, control_interval=8), 1)

    def test_resolve_pid_gain_scheduling_mode_switches(self) -> None:
        """Resolve off/physics/custom scheduling branches from runtime numerics."""
        model = build_point_mass_model(vehicle=sample_vehicle_parameters())
        off_config = build_simulation_config(
            compute_backend="numpy",
            solver_mode="transient_oc",
            max_speed=60.0,
            transient=TransientConfig(
                numerics=TransientNumericsConfig(pid_gain_scheduling_mode="off"),
            ),
        )
        self.assertIsNone(
            _resolve_pid_gain_scheduling(model=model, config=off_config),
        )

        physics_config = build_simulation_config(
            compute_backend="numpy",
            solver_mode="transient_oc",
            max_speed=60.0,
            transient=TransientConfig(
                numerics=TransientNumericsConfig(
                    pid_gain_scheduling_mode="physics_informed",
                ),
            ),
        )
        resolved_physics = _resolve_pid_gain_scheduling(
            model=model,
            config=physics_config,
        )
        self.assertIsNotNone(resolved_physics)

        custom_schedule = _custom_pid_schedule()
        custom_config = build_simulation_config(
            compute_backend="numpy",
            solver_mode="transient_oc",
            max_speed=60.0,
            transient=TransientConfig(
                numerics=TransientNumericsConfig(
                    pid_gain_scheduling_mode="custom",
                    pid_gain_scheduling=custom_schedule,
                ),
            ),
        )
        resolved_custom = _resolve_pid_gain_scheduling(model=model, config=custom_config)
        self.assertEqual(resolved_custom, custom_schedule)

    def test_require_scipy_optimize_success_path(self) -> None:
        """Load scipy.optimize when dependency is available."""
        optimize_module = _require_scipy_optimize()
        self.assertTrue(hasattr(optimize_module, "minimize"))

    def test_pid_off_mode_ignores_custom_schedule_object(self) -> None:
        """Keep legacy PID behavior when scheduling mode is explicitly off."""
        track = build_straight_track(length=200.0, sample_count=51)
        model = build_point_mass_model(vehicle=sample_vehicle_parameters())
        baseline_config = build_simulation_config(
            compute_backend="numpy",
            solver_mode="transient_oc",
            max_speed=60.0,
            initial_speed=10.0,
            transient=TransientConfig(
                numerics=TransientNumericsConfig(pid_gain_scheduling_mode="off"),
            ),
        )
        off_with_schedule = build_simulation_config(
            compute_backend="numpy",
            solver_mode="transient_oc",
            max_speed=60.0,
            initial_speed=10.0,
            transient=TransientConfig(
                numerics=TransientNumericsConfig(
                    pid_gain_scheduling_mode="off",
                    pid_gain_scheduling=_custom_pid_schedule(),
                ),
            ),
        )
        profile_baseline = solve_transient_lap_numpy(
            track=track,
            model=model,
            config=baseline_config,
        )
        profile_off_with_schedule = solve_transient_lap_numpy(
            track=track,
            model=model,
            config=off_with_schedule,
        )
        self.assertTrue(np.allclose(profile_baseline.speed, profile_off_with_schedule.speed))
        self.assertAlmostEqual(
            profile_baseline.lap_time,
            profile_off_with_schedule.lap_time,
            places=10,
        )

    def test_single_track_pid_custom_scheduling_runs(self) -> None:
        """Run single-track PID solver with explicit custom schedule tables."""
        track = build_circular_track(radius=70.0, sample_count=121)
        model = build_single_track_model(
            vehicle=sample_vehicle_parameters(),
            tires=default_axle_tire_parameters(),
            physics=SingleTrackPhysics(),
        )
        config = build_simulation_config(
            compute_backend="numpy",
            solver_mode="transient_oc",
            max_speed=75.0,
            initial_speed=18.0,
            transient=TransientConfig(
                numerics=TransientNumericsConfig(
                    pid_gain_scheduling_mode="custom",
                    pid_gain_scheduling=_custom_pid_schedule(),
                ),
            ),
        )
        profile = solve_transient_lap_numpy(track=track, model=model, config=config)
        self.assertTrue(np.isfinite(profile.lap_time))
        self.assertTrue(np.all(np.isfinite(profile.speed)))
        self.assertGreater(float(np.max(np.abs(profile.steer_cmd))), 0.0)

    def test_point_mass_pid_physics_informed_scheduling_runs(self) -> None:
        """Run point-mass PID path with physics-informed gain scheduling."""
        track = build_straight_track(length=220.0, sample_count=61)
        model = build_point_mass_model(vehicle=sample_vehicle_parameters())
        config = build_simulation_config(
            compute_backend="numpy",
            solver_mode="transient_oc",
            max_speed=60.0,
            initial_speed=0.0,
            transient=TransientConfig(
                numerics=TransientNumericsConfig(
                    pid_gain_scheduling_mode="physics_informed",
                ),
            ),
        )
        profile = solve_transient_lap_numpy(track=track, model=model, config=config)
        self.assertGreater(profile.lap_time, 0.0)
        self.assertTrue(np.all(np.isfinite(profile.longitudinal_accel)))

    def test_numba_transient_solver_rejects_wrong_backend(self) -> None:
        """Reject numba transient solver calls for non-numba runtime configs."""
        track = build_straight_track(length=100.0, sample_count=31)
        model = build_point_mass_model(vehicle=sample_vehicle_parameters())
        config = build_simulation_config(max_speed=60.0, compute_backend="numpy")
        with self.assertRaises(ConfigurationError):
            solve_transient_lap_numba(track=track, model=model, config=config)

    @unittest.skipUnless(NUMBA_AVAILABLE, "Numba not installed")
    def test_numba_transient_solver_delegates_to_numpy_core(self) -> None:
        """Run numba transient wrapper with fake SciPy optimization backend."""
        track = build_straight_track(length=180.0, sample_count=41)
        model = build_point_mass_model(vehicle=sample_vehicle_parameters())
        with (
            _patch_transient_dependency_specs(),
            patch(
                "apexsim.simulation.transient_numpy._require_scipy_optimize",
                return_value=_FakeScipyOptimize(),
            ),
        ):
            config = build_simulation_config(
                compute_backend="numba",
                solver_mode="transient_oc",
                max_speed=60.0,
                initial_speed=12.0,
                transient=TransientConfig(
                    runtime=TransientRuntimeConfig(driver_model="optimal_control")
                ),
            )
            profile = solve_transient_lap_numba(track=track, model=model, config=config)
        self.assertGreater(profile.lap_time, 0.0)

    def test_require_scipy_optimize_raises_clear_error_when_missing(self) -> None:
        """Raise configuration error when SciPy optimize import is unavailable."""
        original_import = builtins.__import__

        def fake_import(
            name: str,
            globals_dict: object | None = None,
            locals_dict: object | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name == "scipy.optimize":
                raise ModuleNotFoundError("No module named 'scipy.optimize'")
            return original_import(name, globals_dict, locals_dict, fromlist, level)

        with (
            patch("builtins.__import__", side_effect=fake_import),
            self.assertRaises(ConfigurationError),
        ):
            _require_scipy_optimize()

    def test_control_mesh_helpers_cover_single_sample_and_mismatch_branches(self) -> None:
        """Validate helper branches for control mesh construction/expansion."""
        mesh_single = _build_control_mesh_positions(sample_count=1, control_interval=8)
        self.assertTrue(np.allclose(mesh_single, np.array([0.0], dtype=float)))

        expanded = _expand_mesh_controls(
            node_values=np.array([2.0, 3.0], dtype=float),
            sample_count=1,
            mesh_positions=np.array([0.0, 1.0], dtype=float),
        )
        self.assertTrue(np.allclose(expanded, np.array([2.0], dtype=float)))

        point_mass_model = build_point_mass_model(vehicle=sample_vehicle_parameters())
        with self.assertRaises(ConfigurationError):
            _decode_point_mass_controls(
                model=point_mass_model,
                raw_ax=np.array([0.1, 0.2], dtype=float),
                sample_count=5,
                mesh_positions=None,
            )

        single_track_model = build_single_track_model(
            vehicle=sample_vehicle_parameters(),
            tires=default_axle_tire_parameters(),
            physics=SingleTrackPhysics(),
        )
        with self.assertRaises(ConfigurationError):
            _decode_single_track_controls(
                single_track_model,
                np.array([0.1, -0.2, 0.05, -0.1], dtype=float),
                sample_count=10,
                mesh_positions=None,
            )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_solve_transient_lap_torch_runs_with_fake_odeint(self) -> None:
        """Solve torch transient profile with deterministic fake ode integrator."""
        import torch

        track = build_circular_track(radius=45.0, sample_count=121)
        model = build_single_track_model(
            vehicle=sample_vehicle_parameters(),
            tires=default_axle_tire_parameters(),
            physics=SingleTrackPhysics(),
        )

        def fake_odeint(rhs: Any, y0: Any, t: Any, method: str = "rk4") -> Any:
            """Integrate one step with euler/rk4 style update.

            Args:
                rhs: Derivative callable.
                y0: Initial state tensor.
                t: Integration time support tensor ``[0, dt]``.
                method: Integration method.

            Returns:
                Two-point trajectory tensor stack.
            """
            dt = t[1] - t[0]
            if method == "euler":
                y1 = y0 + dt * rhs(t[0], y0)
            else:
                k1 = rhs(t[0], y0)
                k2 = rhs(t[0] + 0.5 * dt, y0 + 0.5 * dt * k1)
                k3 = rhs(t[0] + 0.5 * dt, y0 + 0.5 * dt * k2)
                k4 = rhs(t[1], y0 + dt * k3)
                y1 = y0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            return torch.stack((y0, y1), dim=0)

        with (
            _patch_transient_dependency_specs(),
            patch(
                "apexsim.simulation.transient_torch._require_torchdiffeq",
                return_value=SimpleNamespace(odeint=fake_odeint),
            ),
            patch("sys.stderr", new=io.StringIO()) as progress_stream,
        ):
            config = build_simulation_config(
                compute_backend="torch",
                solver_mode="transient_oc",
                max_speed=70.0,
                initial_speed=18.0,
                transient=TransientConfig(
                    numerics=TransientNumericsConfig(
                        integration_method="rk4",
                        max_iterations=2,
                        optimizer_lbfgs_max_iter=2,
                        optimizer_adam_steps=2,
                    ),
                    runtime=TransientRuntimeConfig(
                        driver_model="optimal_control",
                        deterministic_seed=0,
                        verbosity=2,
                    ),
                ),
            )
            profile = solve_transient_lap_torch(track=track, model=model, config=config)

        progress_output = progress_stream.getvalue()
        self.assertEqual(tuple(profile.speed.shape), tuple(track.arc_length.shape))
        self.assertTrue(torch.isfinite(profile.lap_time))
        self.assertGreater(float(profile.lap_time.detach().cpu().item()), 0.0)
        self.assertIn("Transient OC (torch)", progress_output)
        self.assertIn("Transient track (torch, single_track", progress_output)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_solve_transient_lap_torch_point_mass_runs_and_converts_to_numpy(self) -> None:
        """Solve torch transient point-mass profile and convert to NumPy result."""
        import torch

        track = build_straight_track(length=220.0, sample_count=61)
        model = build_point_mass_model(
            vehicle=sample_vehicle_parameters(),
            physics=PointMassPhysics(),
        )

        def fake_odeint(rhs: Any, y0: Any, t: Any, method: str = "rk4") -> Any:
            """Return a two-point trajectory for compatibility with solver wiring."""
            del rhs, method
            return torch.stack((y0, y0), dim=0)

        with (
            _patch_transient_dependency_specs(),
            patch(
                "apexsim.simulation.transient_torch._require_torchdiffeq",
                return_value=SimpleNamespace(odeint=fake_odeint),
            ),
        ):
            config = build_simulation_config(
                compute_backend="torch",
                solver_mode="transient_oc",
                max_speed=65.0,
                initial_speed=5.0,
                transient=TransientConfig(
                    numerics=TransientNumericsConfig(
                        max_iterations=2,
                        optimizer_lbfgs_max_iter=2,
                        optimizer_adam_steps=2,
                    ),
                    runtime=TransientRuntimeConfig(
                        driver_model="optimal_control",
                        deterministic_seed=1,
                    ),
                ),
            )
            profile = solve_transient_lap_torch(track=track, model=model, config=config)
            profile_np = profile.to_numpy()

        self.assertEqual(tuple(profile.speed.shape), tuple(track.arc_length.shape))
        self.assertTrue(torch.isfinite(profile.lap_time))
        self.assertEqual(profile_np.speed.shape, track.arc_length.shape)
        self.assertTrue(np.allclose(profile_np.vy, 0.0))
        self.assertTrue(np.allclose(profile_np.yaw_rate, 0.0))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_solve_transient_lap_torch_pid_default_skips_torchdiffeq(self) -> None:
        """Use transient PID mode on torch backend without fallback dependencies."""
        track = build_straight_track(length=180.0, sample_count=41)
        model = build_point_mass_model(vehicle=sample_vehicle_parameters())
        with (
            _patch_transient_dependency_specs(),
            patch(
                "apexsim.simulation.transient_torch._require_torchdiffeq",
                side_effect=AssertionError("torchdiffeq should not be required in pid mode"),
            ),
            patch(
                "apexsim.simulation.transient_numpy.solve_transient_lap_numpy",
                side_effect=AssertionError("numpy fallback should not be used in torch pid mode"),
            ),
        ):
            config = build_simulation_config(
                compute_backend="torch",
                solver_mode="transient_oc",
                max_speed=60.0,
                initial_speed=10.0,
            )
            profile = solve_transient_lap_torch(track=track, model=model, config=config)
            profile_np = profile.to_numpy()
        self.assertGreater(profile_np.lap_time, 0.0)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_solve_transient_lap_torch_pid_supports_custom_gain_scheduling(self) -> None:
        """Run torch PID path with explicit custom speed-dependent gain tables."""
        track = build_circular_track(radius=70.0, sample_count=121)
        model = build_single_track_model(
            vehicle=sample_vehicle_parameters(),
            tires=default_axle_tire_parameters(),
            physics=SingleTrackPhysics(),
        )
        with (
            _patch_transient_dependency_specs(),
            patch(
                "apexsim.simulation.transient_torch._require_torchdiffeq",
                side_effect=AssertionError("torchdiffeq should not be required in pid mode"),
            ),
        ):
            config = build_simulation_config(
                compute_backend="torch",
                solver_mode="transient_oc",
                max_speed=75.0,
                initial_speed=18.0,
                torch_device="cpu",
                torch_compile=False,
                transient=TransientConfig(
                    numerics=TransientNumericsConfig(
                        pid_gain_scheduling_mode="custom",
                        pid_gain_scheduling=_custom_pid_schedule(),
                    ),
                ),
            )
            profile = solve_transient_lap_torch(track=track, model=model, config=config)
        profile_np = profile.to_numpy()
        self.assertTrue(np.isfinite(profile_np.lap_time))
        self.assertTrue(np.all(np.isfinite(profile_np.speed)))
        self.assertGreater(float(np.max(np.abs(profile_np.steer_cmd))), 0.0)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_solve_transient_lap_torch_pid_supports_physics_informed_gain_scheduling(self) -> None:
        """Run torch PID path with deterministic physics-informed scheduling."""
        track = build_straight_track(length=220.0, sample_count=61)
        model = build_point_mass_model(vehicle=sample_vehicle_parameters())
        with (
            _patch_transient_dependency_specs(),
            patch(
                "apexsim.simulation.transient_torch._require_torchdiffeq",
                side_effect=AssertionError("torchdiffeq should not be required in pid mode"),
            ),
        ):
            config = build_simulation_config(
                compute_backend="torch",
                solver_mode="transient_oc",
                max_speed=60.0,
                initial_speed=0.0,
                torch_device="cpu",
                torch_compile=False,
                transient=TransientConfig(
                    numerics=TransientNumericsConfig(
                        pid_gain_scheduling_mode="physics_informed",
                    ),
                ),
            )
            profile = solve_transient_lap_torch(track=track, model=model, config=config)
        profile_np = profile.to_numpy()
        self.assertGreater(profile_np.lap_time, 0.0)
        self.assertTrue(np.all(np.isfinite(profile_np.longitudinal_accel)))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_transient_pid_autodiff_sensitivity_runs_for_single_track_parameters(self) -> None:
        """Compute transient PID AD sensitivities for key single-track parameters."""
        track = build_straight_track(length=180.0, sample_count=41)
        model = build_single_track_model(
            vehicle=sample_vehicle_parameters(),
            tires=default_axle_tire_parameters(),
            physics=SingleTrackPhysics(),
        )
        with _patch_transient_dependency_specs():
            config = build_simulation_config(
                compute_backend="torch",
                solver_mode="transient_oc",
                max_speed=60.0,
                initial_speed=8.0,
                transient=TransientConfig(
                    numerics=TransientNumericsConfig(max_time_step=1.0),
                    runtime=TransientRuntimeConfig(driver_model="pid", verbosity=0),
                ),
                torch_device="cpu",
                torch_compile=False,
            )
            study = run_lap_sensitivity_study(
                track=track,
                model=model,
                simulation_config=config,
                parameters=[
                    SensitivityStudyParameter(name="mass", target="vehicle.mass"),
                    SensitivityStudyParameter(name="cg_height", target="vehicle.cg_height"),
                    SensitivityStudyParameter(name="yaw_inertia", target="vehicle.yaw_inertia"),
                    SensitivityStudyParameter(
                        name="drag_coefficient",
                        target="vehicle.drag_coefficient",
                    ),
                ],
                objectives=("lap_time_s", "energy_kwh"),
            )
        self.assertEqual(study.sensitivity_results["lap_time_s"].method, "autodiff")
        self.assertEqual(study.sensitivity_results["energy_kwh"].method, "autodiff")
        for objective in ("lap_time_s", "energy_kwh"):
            result = study.sensitivity_results[objective]
            for name in ("mass", "cg_height", "yaw_inertia", "drag_coefficient"):
                self.assertTrue(np.isfinite(result.sensitivities[name]))
        lap_time_sensitivities = study.sensitivity_results["lap_time_s"].sensitivities
        self.assertLess(abs(lap_time_sensitivities["mass"]), 1.0)
        self.assertLess(abs(lap_time_sensitivities["drag_coefficient"]), 5.0)
        self.assertLess(abs(lap_time_sensitivities["yaw_inertia"]), 0.1)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_transient_pid_autodiff_matches_fd_for_single_track_on_straight(self) -> None:
        """Keep transient single-track AD sensitivities aligned with FD on straight laps."""
        track = build_straight_track(length=180.0, sample_count=41)
        model = build_single_track_model(
            vehicle=sample_vehicle_parameters(),
            tires=default_axle_tire_parameters(),
            physics=SingleTrackPhysics(),
        )
        with _patch_transient_dependency_specs():
            config = build_simulation_config(
                compute_backend="torch",
                solver_mode="transient_oc",
                max_speed=60.0,
                initial_speed=8.0,
                transient=TransientConfig(
                    numerics=TransientNumericsConfig(max_time_step=1.0),
                    runtime=TransientRuntimeConfig(driver_model="pid", verbosity=0),
                ),
                torch_device="cpu",
                torch_compile=False,
            )
            parameters = [
                SensitivityStudyParameter(name="mass", target="vehicle.mass"),
                SensitivityStudyParameter(
                    name="drag_coefficient",
                    target="vehicle.drag_coefficient",
                ),
            ]
            ad = run_lap_sensitivity_study(
                track=track,
                model=model,
                simulation_config=config,
                parameters=parameters,
                objectives=("lap_time_s",),
                runtime=SensitivityRuntime(method="autodiff"),
            )
            fd = run_lap_sensitivity_study(
                track=track,
                model=model,
                simulation_config=config,
                parameters=parameters,
                objectives=("lap_time_s",),
                runtime=SensitivityRuntime(method="finite_difference"),
                numerics=SensitivityNumerics(finite_difference_relative_step=1e-4),
            )

        ad_values = ad.sensitivity_results["lap_time_s"].sensitivities
        fd_values = fd.sensitivity_results["lap_time_s"].sensitivities
        for name in ("mass", "drag_coefficient"):
            tolerance = max(1e-6, 5e-3 * abs(fd_values[name]))
            self.assertAlmostEqual(ad_values[name], fd_values[name], delta=tolerance)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_transient_pid_autodiff_sensitivity_runs_for_point_mass_parameters(self) -> None:
        """Compute transient PID AD sensitivities for key point-mass parameters."""
        track = build_straight_track(length=180.0, sample_count=41)
        model = build_point_mass_model(vehicle=sample_vehicle_parameters())
        with _patch_transient_dependency_specs():
            config = build_simulation_config(
                compute_backend="torch",
                solver_mode="transient_oc",
                max_speed=60.0,
                initial_speed=8.0,
                transient=TransientConfig(
                    numerics=TransientNumericsConfig(max_time_step=1.0),
                    runtime=TransientRuntimeConfig(driver_model="pid", verbosity=0),
                ),
                torch_device="cpu",
                torch_compile=False,
            )
            study = run_lap_sensitivity_study(
                track=track,
                model=model,
                simulation_config=config,
                parameters=[
                    SensitivityStudyParameter(name="mass", target="vehicle.mass"),
                    SensitivityStudyParameter(name="cg_height", target="vehicle.cg_height"),
                    SensitivityStudyParameter(name="yaw_inertia", target="vehicle.yaw_inertia"),
                    SensitivityStudyParameter(
                        name="drag_coefficient",
                        target="vehicle.drag_coefficient",
                    ),
                ],
                objectives=("lap_time_s", "energy_kwh"),
            )
        self.assertEqual(study.sensitivity_results["lap_time_s"].method, "autodiff")
        self.assertEqual(study.sensitivity_results["energy_kwh"].method, "autodiff")
        for objective in ("lap_time_s", "energy_kwh"):
            result = study.sensitivity_results[objective]
            for name in ("mass", "cg_height", "yaw_inertia", "drag_coefficient"):
                self.assertTrue(np.isfinite(result.sensitivities[name]))

    def test_simulate_lap_transient_populates_extended_fields(self) -> None:
        """Return transient result with populated state/control traces."""
        track = build_straight_track(length=220.0, sample_count=51)
        model = build_point_mass_model(vehicle=sample_vehicle_parameters())
        with (
            _patch_transient_dependency_specs(),
            patch(
                "apexsim.simulation.transient_numpy._require_scipy_optimize",
                return_value=_FakeScipyOptimize(),
            ),
        ):
            config = build_simulation_config(
                compute_backend="numpy",
                solver_mode="transient_oc",
                max_speed=65.0,
                initial_speed=10.0,
                transient=TransientConfig(
                    runtime=TransientRuntimeConfig(driver_model="optimal_control")
                ),
            )
            result = simulate_lap(track=track, model=model, config=config)

        self.assertEqual(result.solver_mode, "transient_oc")
        self.assertIsNotNone(result.time)
        self.assertIsNotNone(result.vx)
        self.assertIsNotNone(result.vy)
        self.assertIsNotNone(result.yaw_rate)
        self.assertIsNotNone(result.steer_cmd)
        self.assertIsNotNone(result.ax_cmd)

    def test_numpy_transient_solver_emits_progress_for_verbosity_two(self) -> None:
        """Emit optimization and track progress lines when verbosity >= 2."""
        track = build_straight_track(length=200.0, sample_count=41)
        model = build_point_mass_model(vehicle=sample_vehicle_parameters())
        with (
            _patch_transient_dependency_specs(),
            patch(
                "apexsim.simulation.transient_numpy._require_scipy_optimize",
                return_value=_FakeScipyOptimize(),
            ),
            patch("sys.stderr", new=io.StringIO()) as progress_stream,
        ):
            config = build_simulation_config(
                compute_backend="numpy",
                solver_mode="transient_oc",
                max_speed=60.0,
                initial_speed=10.0,
                transient=TransientConfig(
                    numerics=TransientNumericsConfig(max_iterations=3),
                    runtime=TransientRuntimeConfig(
                        driver_model="optimal_control",
                        verbosity=2,
                    ),
                ),
            )
            solve_transient_lap_numpy(track=track, model=model, config=config)

        progress_output = progress_stream.getvalue()
        self.assertIn("Transient OC (numpy)", progress_output)
        self.assertIn("Transient track (numpy, point_mass", progress_output)

    @unittest.skipUnless(NUMBA_AVAILABLE, "Numba not installed")
    def test_numba_transient_solver_emits_progress_for_verbosity_two(self) -> None:
        """Emit optimization and track progress lines for numba transient runs."""
        track = build_straight_track(length=200.0, sample_count=41)
        model = build_point_mass_model(vehicle=sample_vehicle_parameters())
        with (
            _patch_transient_dependency_specs(),
            patch(
                "apexsim.simulation.transient_numpy._require_scipy_optimize",
                return_value=_FakeScipyOptimize(),
            ),
            patch("sys.stderr", new=io.StringIO()) as progress_stream,
        ):
            config = build_simulation_config(
                compute_backend="numba",
                solver_mode="transient_oc",
                max_speed=60.0,
                initial_speed=10.0,
                transient=TransientConfig(
                    numerics=TransientNumericsConfig(max_iterations=3),
                    runtime=TransientRuntimeConfig(
                        driver_model="optimal_control",
                        verbosity=2,
                    ),
                ),
            )
            solve_transient_lap_numba(track=track, model=model, config=config)

        progress_output = progress_stream.getvalue()
        self.assertIn("Transient OC (numba)", progress_output)
        self.assertIn("Transient track (numba, point_mass", progress_output)

    def test_numpy_transient_pid_default_does_not_require_scipy_optimizer(self) -> None:
        """Use PID driver by default without invoking SciPy optimizer plumbing."""
        track = build_straight_track(length=180.0, sample_count=41)
        model = build_point_mass_model(vehicle=sample_vehicle_parameters())
        with (
            _patch_transient_dependency_specs(),
            patch(
                "apexsim.simulation.transient_numpy._require_scipy_optimize",
                side_effect=AssertionError("optimizer path should not be used in pid mode"),
            ),
        ):
            config = build_simulation_config(
                compute_backend="numpy",
                solver_mode="transient_oc",
                max_speed=60.0,
                initial_speed=10.0,
            )
            profile = solve_transient_lap_numpy(track=track, model=model, config=config)
        self.assertGreater(profile.lap_time, 0.0)

    def test_compute_diagnostics_rejects_batch_outputs_with_wrong_length(self) -> None:
        """Reject diagnostics_batch outputs that do not contain four signals."""
        track = build_straight_track(length=120.0, sample_count=21)
        n = track.arc_length.size
        profile = SpeedProfileResult(
            speed=np.full(n, 20.0, dtype=float),
            longitudinal_accel=np.zeros(n, dtype=float),
            lateral_accel=np.zeros(n, dtype=float),
            lateral_envelope_iterations=0,
            lap_time=1.0,
        )

        class _BadLengthModel:
            def diagnostics_batch(
                self,
                *,
                speed: np.ndarray,
                longitudinal_accel: np.ndarray,
                lateral_accel: np.ndarray,
                curvature: np.ndarray,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                del speed, longitudinal_accel, lateral_accel, curvature
                zeros = np.zeros(n, dtype=float)
                return zeros, zeros, zeros

        with self.assertRaises(ConfigurationError):
            _compute_diagnostics(track=track, model=_BadLengthModel(), profile=profile)

    def test_compute_diagnostics_rejects_batch_outputs_with_wrong_shape(self) -> None:
        """Reject diagnostics_batch outputs when any signal shape mismatches profile."""
        track = build_straight_track(length=120.0, sample_count=21)
        n = track.arc_length.size
        profile = SpeedProfileResult(
            speed=np.full(n, 20.0, dtype=float),
            longitudinal_accel=np.zeros(n, dtype=float),
            lateral_accel=np.zeros(n, dtype=float),
            lateral_envelope_iterations=0,
            lap_time=1.0,
        )

        class _BadShapeModel:
            def diagnostics_batch(
                self,
                *,
                speed: np.ndarray,
                longitudinal_accel: np.ndarray,
                lateral_accel: np.ndarray,
                curvature: np.ndarray,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                del speed, longitudinal_accel, lateral_accel, curvature
                return (
                    np.zeros(n + 1, dtype=float),
                    np.zeros(n, dtype=float),
                    np.zeros(n, dtype=float),
                    np.zeros(n, dtype=float),
                )

        with self.assertRaises(ConfigurationError):
            _compute_diagnostics(track=track, model=_BadShapeModel(), profile=profile)

    def test_compute_diagnostics_falls_back_to_scalar_api(self) -> None:
        """Compute diagnostics by scalar loop when batch API is absent."""
        track = build_straight_track(length=90.0, sample_count=13)
        n = track.arc_length.size
        profile = SpeedProfileResult(
            speed=np.linspace(10.0, 25.0, n, dtype=float),
            longitudinal_accel=np.linspace(0.0, 1.0, n, dtype=float),
            lateral_accel=np.zeros(n, dtype=float),
            lateral_envelope_iterations=0,
            lap_time=1.0,
        )

        class _ScalarOnlyModel:
            def diagnostics(
                self,
                *,
                speed: float,
                longitudinal_accel: float,
                lateral_accel: float,
                curvature: float,
            ) -> ModelDiagnostics:
                del lateral_accel
                return ModelDiagnostics(
                    yaw_moment=curvature * speed,
                    front_axle_load=1000.0 + longitudinal_accel,
                    rear_axle_load=900.0 - longitudinal_accel,
                    power=1200.0 + speed,
                )

        yaw_moment, front_axle_load, rear_axle_load, power = _compute_diagnostics(
            track=track,
            model=_ScalarOnlyModel(),
            profile=profile,
        )
        self.assertEqual(yaw_moment.shape, (n,))
        self.assertEqual(front_axle_load.shape, (n,))
        self.assertEqual(rear_axle_load.shape, (n,))
        self.assertEqual(power.shape, (n,))
        self.assertGreater(float(np.max(power)), float(np.min(power)))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_sensitivity_uses_transient_torch_objective_path(self) -> None:
        """Route lap-study objective evaluation through transient torch solver."""
        import torch

        track = build_straight_track(length=160.0, sample_count=41)
        model = build_point_mass_model(vehicle=sample_vehicle_parameters())
        with _patch_transient_dependency_specs():
            config = build_simulation_config(
                compute_backend="torch",
                solver_mode="transient_oc",
                max_speed=60.0,
                initial_speed=10.0,
            )

        class _DummyTransientProfile:
            """Minimal profile object compatible with sensitivity objectives."""

            def __init__(self, n: int) -> None:
                speed = torch.full((n,), 20.0, dtype=torch.float64)
                self.speed = speed
                self.longitudinal_accel = torch.zeros_like(speed)
                self.lap_time = torch.as_tensor(12.0, dtype=torch.float64)

        with (
            _patch_transient_dependency_specs(),
            patch(
                "apexsim.simulation.transient_torch.solve_transient_lap_torch",
                return_value=_DummyTransientProfile(track.arc_length.size),
            ),
        ):
            study = run_lap_sensitivity_study(
                track=track,
                model=model,
                simulation_config=config,
                parameters=[SensitivityStudyParameter(name="mass", target="vehicle.mass")],
                runtime=SensitivityRuntime(method="finite_difference"),
            )
        self.assertIn("lap_time_s", study.sensitivity_results)
        self.assertIn("energy_kwh", study.sensitivity_results)
