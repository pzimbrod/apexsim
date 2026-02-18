"""Tests for integrator and simulation config validation."""

from __future__ import annotations

import builtins
import importlib.util
import unittest
import warnings
from unittest.mock import patch

import numpy as np

import apexsim.simulation as simulation_module
from apexsim.simulation.config import (
    NumericsConfig,
    RuntimeConfig,
    SimulationConfig,
    build_simulation_config,
)
from apexsim.simulation.envelope import lateral_speed_limit
from apexsim.simulation.integrator import rk4_step
from apexsim.simulation.transient_common import (
    TransientConfig,
    TransientNumericsConfig,
    TransientRuntimeConfig,
)
from apexsim.utils.exceptions import ConfigurationError

NUMBA_AVAILABLE = importlib.util.find_spec("numba") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class SimulationHelpersTests(unittest.TestCase):
    """Unit tests for numerical helper modules."""

    def test_rk4_step_matches_linear_solution(self) -> None:
        """Match RK4 integration output against an exponential reference."""

        # dy/dt = -2y has exact solution y(t+dt) = y * exp(-2dt).
        def rhs(_: float, state: np.ndarray) -> np.ndarray:
            return -2.0 * state

        initial = np.array([1.0], dtype=float)
        dt = 0.1
        stepped = rk4_step(rhs, 0.0, initial, dt)
        expected = np.exp(-2.0 * dt)
        self.assertAlmostEqual(float(stepped[0]), float(expected), delta=1e-5)

    def test_lateral_speed_limit_handles_straight_and_curved_segments(self) -> None:
        """Return expected envelope speed limits for straight and curved path samples."""
        straight_limit = lateral_speed_limit(curvature=0.0, lateral_accel_limit=8.0, max_speed=50.0)
        self.assertEqual(straight_limit, 50.0)

        curved_limit = lateral_speed_limit(curvature=0.02, lateral_accel_limit=8.0, max_speed=50.0)
        self.assertAlmostEqual(curved_limit, np.sqrt(8.0 / 0.02), places=12)

        saturated_limit = lateral_speed_limit(
            curvature=0.005,
            lateral_accel_limit=8.0,
            max_speed=20.0,
        )
        self.assertEqual(saturated_limit, 20.0)

    def test_simulation_config_validation_raises_for_invalid_values(self) -> None:
        """Raise configuration errors for invalid simulation settings."""
        with self.assertRaises(ConfigurationError):
            SimulationConfig(
                runtime=RuntimeConfig(max_speed=115.0, enable_transient_refinement=False),
                numerics=NumericsConfig(
                    min_speed=0.0,
                    lateral_envelope_max_iterations=20,
                    lateral_envelope_convergence_tolerance=0.1,
                    transient_step=0.01,
                ),
            ).validate()

    def test_build_simulation_config_uses_stable_default_numerics(self) -> None:
        """Build configuration with default numerics when omitted."""
        config = build_simulation_config(max_speed=115.0)
        self.assertEqual(config.numerics, NumericsConfig())
        self.assertFalse(config.runtime.enable_transient_refinement)
        self.assertIsNone(config.runtime.initial_speed)

        with self.assertRaises(ConfigurationError):
            SimulationConfig(
                runtime=RuntimeConfig(max_speed=10.0, enable_transient_refinement=False),
                numerics=NumericsConfig(
                    min_speed=20.0,
                    lateral_envelope_max_iterations=20,
                    lateral_envelope_convergence_tolerance=0.1,
                    transient_step=0.01,
                ),
            ).validate()
        with self.assertRaises(ConfigurationError):
            SimulationConfig(
                runtime=RuntimeConfig(max_speed=115.0, enable_transient_refinement=False),
                numerics=NumericsConfig(
                    min_speed=8.0,
                    lateral_envelope_max_iterations=20,
                    lateral_envelope_convergence_tolerance=0.1,
                    transient_step=0.0,
                ),
            ).validate()
        with self.assertRaises(ConfigurationError):
            SimulationConfig(
                runtime=RuntimeConfig(max_speed=115.0, enable_transient_refinement=False),
                numerics=NumericsConfig(
                    min_speed=8.0,
                    lateral_envelope_max_iterations=0,
                    lateral_envelope_convergence_tolerance=0.1,
                    transient_step=0.01,
                ),
            ).validate()
        with self.assertRaises(ConfigurationError):
            SimulationConfig(
                runtime=RuntimeConfig(max_speed=115.0, enable_transient_refinement=False),
                numerics=NumericsConfig(
                    min_speed=8.0,
                    lateral_envelope_max_iterations=20,
                    lateral_envelope_convergence_tolerance=0.0,
                    transient_step=0.01,
                ),
            ).validate()

    def test_runtime_config_rejects_unknown_compute_backend(self) -> None:
        """Reject runtime configs with unsupported compute backend identifiers."""
        with self.assertRaises(ConfigurationError):
            SimulationConfig(
                runtime=RuntimeConfig(
                    max_speed=115.0,
                    enable_transient_refinement=False,
                    compute_backend="does_not_exist",
                ),
                numerics=NumericsConfig(),
            ).validate()

    def test_runtime_config_rejects_initial_speed_outside_bounds(self) -> None:
        """Reject initial-speed selections outside [0, max_speed]."""
        with self.assertRaises(ConfigurationError):
            build_simulation_config(max_speed=115.0, initial_speed=-1.0)
        with self.assertRaises(ConfigurationError):
            build_simulation_config(max_speed=115.0, initial_speed=120.0)

    def test_build_simulation_config_supports_torch_backend(self) -> None:
        """Build a validated configuration that selects the torch backend."""
        config = build_simulation_config(max_speed=115.0, compute_backend="torch")
        self.assertEqual(config.runtime.compute_backend, "torch")
        self.assertEqual(config.runtime.torch_device, "cpu")
        self.assertFalse(config.runtime.torch_compile)

    def test_runtime_config_rejects_torch_compile_for_torch_backend(self) -> None:
        """Reject torch_compile to keep torch backend solves AD-compatible."""
        with self.assertRaises(ConfigurationError):
            build_simulation_config(
                max_speed=115.0,
                compute_backend="torch",
                torch_compile=True,
            )

    def test_build_simulation_config_supports_initial_speed(self) -> None:
        """Build a validated configuration with explicit initial speed."""
        config = build_simulation_config(max_speed=115.0, initial_speed=20.0)
        self.assertEqual(config.runtime.initial_speed, 20.0)

    def test_build_simulation_config_supports_standing_start(self) -> None:
        """Allow standing starts for straight-line launch scenarios."""
        config = build_simulation_config(max_speed=115.0, initial_speed=0.0)
        self.assertEqual(config.runtime.initial_speed, 0.0)

    def test_runtime_config_rejects_unknown_solver_mode(self) -> None:
        """Reject runtime configs with unsupported solver_mode identifiers."""
        with self.assertRaises(ConfigurationError):
            SimulationConfig(
                runtime=RuntimeConfig(
                    max_speed=115.0,
                    solver_mode="unknown",
                ),
                numerics=NumericsConfig(),
            ).validate()

    def test_transient_config_validation_rejects_invalid_inputs(self) -> None:
        """Reject transient numerical/runtime settings outside valid bounds."""
        with self.assertRaises(ConfigurationError):
            TransientNumericsConfig(integration_method="unknown").validate()
        with self.assertRaises(ConfigurationError):
            TransientNumericsConfig(max_iterations=0).validate()
        with self.assertRaises(ConfigurationError):
            TransientNumericsConfig(tolerance=0.0).validate()
        with self.assertRaises(ConfigurationError):
            TransientNumericsConfig(min_time_step=0.0).validate()
        with self.assertRaises(ConfigurationError):
            TransientNumericsConfig(
                min_time_step=0.1,
                max_time_step=0.05,
            ).validate()
        with self.assertRaises(ConfigurationError):
            TransientNumericsConfig(lateral_constraint_weight=-1.0).validate()
        with self.assertRaises(ConfigurationError):
            TransientNumericsConfig(tracking_weight=-1.0).validate()
        with self.assertRaises(ConfigurationError):
            TransientNumericsConfig(control_smoothness_weight=-1.0).validate()
        with self.assertRaises(ConfigurationError):
            TransientNumericsConfig(control_interval=0).validate()
        with self.assertRaises(ConfigurationError):
            TransientNumericsConfig(optimizer_learning_rate=0.0).validate()
        with self.assertRaises(ConfigurationError):
            TransientNumericsConfig(optimizer_lbfgs_max_iter=0).validate()
        with self.assertRaises(ConfigurationError):
            TransientNumericsConfig(optimizer_adam_steps=0).validate()
        with self.assertRaises(ConfigurationError):
            TransientNumericsConfig(
                pid_longitudinal_integral_limit=-1.0
            ).validate()
        with self.assertRaises(ConfigurationError):
            TransientNumericsConfig(pid_steer_integral_limit=-1.0).validate()
        with self.assertRaises(ConfigurationError):
            TransientRuntimeConfig(ode_backend_policy="bad").validate()
        with self.assertRaises(ConfigurationError):
            TransientRuntimeConfig(optimizer_backend_policy="bad").validate()
        with self.assertRaises(ConfigurationError):
            TransientRuntimeConfig(driver_model="bad").validate()
        with self.assertRaises(ConfigurationError):
            TransientRuntimeConfig(deterministic_seed=-1).validate()
        with self.assertRaises(ConfigurationError):
            TransientRuntimeConfig(verbosity=-1).validate()

    def test_build_simulation_config_maps_deprecated_transient_flag(self) -> None:
        """Map deprecated transient flag to transient solver mode with warning."""
        real_find_spec = importlib.util.find_spec
        with (
            patch(
                "apexsim.simulation.config.importlib.util.find_spec",
                side_effect=lambda name: (
                    object() if name in {"scipy"} else real_find_spec(name)
                ),
            ),
            warnings.catch_warnings(record=True) as captured,
        ):
            warnings.simplefilter("always")
            config = build_simulation_config(
                max_speed=115.0,
                compute_backend="numpy",
                enable_transient_refinement=True,
            )
        self.assertEqual(config.runtime.solver_mode, "transient_oc")
        self.assertTrue(any(item.category is DeprecationWarning for item in captured))

    def test_build_simulation_config_rejects_conflicting_transient_flags(self) -> None:
        """Reject deprecated transient flag when solver mode is explicitly quasi-static."""
        with self.assertRaises(ConfigurationError):
            build_simulation_config(
                max_speed=115.0,
                solver_mode="quasi_static",
                enable_transient_refinement=True,
            )

    def test_transient_optimal_control_requires_backend_specific_dependencies(self) -> None:
        """Require SciPy or torchdiffeq when transient optimal-control mode is selected."""
        real_find_spec = importlib.util.find_spec
        with patch(
            "apexsim.simulation.config.importlib.util.find_spec",
            side_effect=lambda name: (
                None if name == "scipy" else real_find_spec(name)
            ),
        ), self.assertRaises(ConfigurationError):
            build_simulation_config(
                max_speed=115.0,
                compute_backend="numpy",
                solver_mode="transient_oc",
                transient=TransientConfig(
                    runtime=TransientRuntimeConfig(driver_model="optimal_control")
                ),
            )

        with patch(
            "apexsim.simulation.config.importlib.util.find_spec",
            side_effect=lambda name: (
                None if name == "torchdiffeq" else real_find_spec(name)
            ),
        ), self.assertRaises(ConfigurationError):
            build_simulation_config(
                max_speed=115.0,
                compute_backend="torch",
                solver_mode="transient_oc",
                transient=TransientConfig(
                    runtime=TransientRuntimeConfig(driver_model="optimal_control")
                ),
            )

    def test_transient_pid_mode_skips_optimal_control_dependency_checks(self) -> None:
        """Allow transient PID mode without SciPy/torchdiffeq optional dependencies."""
        real_find_spec = importlib.util.find_spec
        with patch(
            "apexsim.simulation.config.importlib.util.find_spec",
            side_effect=lambda name: (
                None if name in {"scipy", "torchdiffeq"} else real_find_spec(name)
            ),
        ):
            config = build_simulation_config(
                max_speed=115.0,
                compute_backend="numpy",
                solver_mode="transient_oc",
            )
        self.assertEqual(config.transient.runtime.driver_model, "pid")

    def test_simulation_config_accepts_transient_custom_config(self) -> None:
        """Accept fully specified transient config in quasi-static runtime mode."""
        config = build_simulation_config(
            max_speed=115.0,
            compute_backend="numpy",
            transient=TransientConfig(
                numerics=TransientNumericsConfig(
                    integration_method="euler",
                    max_iterations=3,
                    tolerance=1e-3,
                ),
                runtime=TransientRuntimeConfig(
                    ode_backend_policy="auto",
                    optimizer_backend_policy="auto",
                    deterministic_seed=0,
                    verbosity=0,
                ),
            ),
        )
        self.assertEqual(config.transient.numerics.integration_method, "euler")

    @unittest.skipUnless(NUMBA_AVAILABLE, "Numba not installed")
    def test_build_simulation_config_supports_numba_backend(self) -> None:
        """Build a validated configuration that selects the numba backend."""
        config = build_simulation_config(max_speed=115.0, compute_backend="numba")
        self.assertEqual(config.runtime.compute_backend, "numba")
        self.assertEqual(config.runtime.torch_device, "cpu")
        self.assertFalse(config.runtime.torch_compile)

    def test_runtime_config_rejects_non_boolean_torch_compile(self) -> None:
        """Reject runtime configs where torch_compile is not a boolean."""
        with self.assertRaises(ConfigurationError):
            SimulationConfig(
                runtime=RuntimeConfig(
                    max_speed=115.0,
                    enable_transient_refinement=False,
                    compute_backend="torch",
                    torch_compile="yes",  # type: ignore[arg-type]
                ),
                numerics=NumericsConfig(),
            ).validate()

    def test_runtime_config_rejects_empty_torch_device(self) -> None:
        """Reject runtime configs where torch_device is an empty string."""
        with self.assertRaises(ConfigurationError):
            SimulationConfig(
                runtime=RuntimeConfig(
                    max_speed=115.0,
                    enable_transient_refinement=False,
                    compute_backend="numpy",
                    torch_device="",
                ),
                numerics=NumericsConfig(),
            ).validate()

    def test_runtime_config_rejects_non_cpu_device_for_cpu_backends(self) -> None:
        """Reject non-CPU torch_device selection for numpy/numba backends."""
        with self.assertRaises(ConfigurationError):
            SimulationConfig(
                runtime=RuntimeConfig(
                    max_speed=115.0,
                    enable_transient_refinement=False,
                    compute_backend="numpy",
                    torch_device="cuda:0",
                ),
                numerics=NumericsConfig(),
            ).validate()

    def test_runtime_config_rejects_torch_compile_for_numpy_backend(self) -> None:
        """Reject torch_compile flag outside of the torch backend."""
        with self.assertRaises(ConfigurationError):
            SimulationConfig(
                runtime=RuntimeConfig(
                    max_speed=115.0,
                    enable_transient_refinement=False,
                    compute_backend="numpy",
                    torch_compile=True,
                ),
                numerics=NumericsConfig(),
            ).validate()

    def test_runtime_config_raises_when_numba_is_missing(self) -> None:
        """Raise configuration error when numba backend is requested without numba."""
        original_import = builtins.__import__

        def _fake_import(
            name: str,
            globals_dict: object | None = None,
            locals_dict: object | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name == "numba":
                raise ModuleNotFoundError("No module named 'numba'")
            return original_import(name, globals_dict, locals_dict, fromlist, level)

        with patch("builtins.__import__", side_effect=_fake_import), self.assertRaises(
            ConfigurationError
        ):
            SimulationConfig(
                runtime=RuntimeConfig(
                    max_speed=115.0,
                    enable_transient_refinement=False,
                    compute_backend="numba",
                ),
                numerics=NumericsConfig(),
            ).validate()

    def test_runtime_config_raises_when_torch_is_missing(self) -> None:
        """Raise configuration error when torch backend is requested without torch."""
        original_import = builtins.__import__

        def _fake_import(
            name: str,
            globals_dict: object | None = None,
            locals_dict: object | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name == "torch":
                raise ModuleNotFoundError("No module named 'torch'")
            return original_import(name, globals_dict, locals_dict, fromlist, level)

        with patch("builtins.__import__", side_effect=_fake_import), self.assertRaises(
            ConfigurationError
        ):
            SimulationConfig(
                runtime=RuntimeConfig(
                    max_speed=115.0,
                    enable_transient_refinement=False,
                    compute_backend="torch",
                ),
                numerics=NumericsConfig(),
            ).validate()

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_runtime_config_rejects_cuda_request_without_cuda_device(self) -> None:
        """Reject CUDA device requests when CUDA runtime is unavailable."""
        import torch

        if torch.cuda.is_available():
            self.skipTest("CUDA is available on this host")

        with self.assertRaises(ConfigurationError):
            SimulationConfig(
                runtime=RuntimeConfig(
                    max_speed=115.0,
                    enable_transient_refinement=False,
                    compute_backend="torch",
                    torch_device="cuda:0",
                ),
                numerics=NumericsConfig(),
            ).validate()

    def test_simulation_module_rejects_unknown_public_symbol(self) -> None:
        """Raise AttributeError for unknown symbols requested via simulation module."""
        with self.assertRaises(AttributeError):
            simulation_module.__getattr__("does_not_exist")


if __name__ == "__main__":
    unittest.main()
