"""Tests for backend acceleration helpers and fallback paths."""

from __future__ import annotations

import builtins
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from apexsim.simulation import numba_profile, torch_profile
from apexsim.simulation.config import build_simulation_config
from apexsim.simulation.numba_profile import (
    _compiled_numba_kernel,
    _compiled_single_track_numba_kernel,
    _point_mass_speed_profile_kernel,
    _single_track_speed_profile_kernel,
    solve_speed_profile_numba,
)
from apexsim.track.io import load_track_csv
from apexsim.utils.exceptions import ConfigurationError

NUMBA_AVAILABLE = importlib.util.find_spec("numba") is not None


class _DummyTorchModule:
    """Dummy torch module used to force compile fallback path in tests."""

    @staticmethod
    def compile(*_: object, **__: object) -> object:
        """Raise compile failure to trigger fallback path.

        Args:
            *_: Positional arguments passed to ``torch.compile``.
            **__: Keyword arguments passed to ``torch.compile``.

        Returns:
            No return value because this stub always raises.

        Raises:
            RuntimeError: Always raised to force fallback behavior.
        """
        raise RuntimeError("compile failed")


class _InvalidNumbaModel:
    """Invalid numba-model stub returning malformed kernel parameter tuple."""

    def validate(self) -> None:
        """Provide no-op validation for protocol compatibility."""

    def numba_speed_profile_parameters(self) -> tuple[float]:
        """Return malformed parameter tuple for negative-path testing.

        Returns:
            One-element tuple instead of the required six scalar parameters.
        """
        return (1.0,)


class _MissingNumbaApiModel:
    """Invalid model stub lacking numba backend API method."""

    def validate(self) -> None:
        """Provide no-op validation for protocol compatibility."""


class BackendAccelerationTests(unittest.TestCase):
    """Validate fallback behavior of backend acceleration helpers."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load shared track fixture for backend helper tests."""
        root = Path(__file__).resolve().parents[2]
        cls.track = load_track_csv(root / "data" / "spa_francorchamps.csv")

    def tearDown(self) -> None:
        """Reset backend caches so tests remain isolated."""
        torch_profile._COMPILED_SOLVER_CACHE.clear()
        numba_profile._COMPILED_NUMBA_KERNEL = None
        numba_profile._COMPILED_SINGLE_TRACK_NUMBA_KERNEL = None

    @staticmethod
    def _kernel_inputs() -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        int,
        float,
    ]:
        """Build deterministic inputs for direct kernel-level testing.

        Returns:
            Tuple of positional arguments expected by
            ``_point_mass_speed_profile_kernel``.
        """
        arc_length = np.array([0.0, 50.0, 120.0, 200.0, 280.0], dtype=np.float64)
        curvature = np.array([0.0, 0.012, 0.010, 0.0, -0.011], dtype=np.float64)
        grade = np.array([0.0, 0.008, -0.006, 0.002, 0.0], dtype=np.float64)
        banking = np.array([0.0, 0.04, -0.02, 0.0, 0.01], dtype=np.float64)
        return (
            arc_length,
            curvature,
            grade,
            banking,
            798.0,
            0.5 * 1.225 * 3.2 * 1.5,
            0.5 * 1.225 * 0.9 * 1.5,
            1.7,
            8.0,
            16.0,
            115.0,
            115.0,
            8.0,
            20,
            0.1,
        )

    @staticmethod
    def _single_track_kernel_inputs() -> tuple[object, ...]:
        """Build deterministic inputs for direct single_track-kernel testing.

        Returns:
            Tuple of positional arguments expected by
            ``_single_track_speed_profile_kernel``.
        """
        arc_length = np.array([0.0, 50.0, 120.0, 200.0, 280.0], dtype=np.float64)
        curvature = np.array([0.0, 0.012, 0.010, 0.0, -0.011], dtype=np.float64)
        grade = np.array([0.0, 0.008, -0.006, 0.002, 0.0], dtype=np.float64)
        banking = np.array([0.0, 0.04, -0.02, 0.0, 0.01], dtype=np.float64)
        return (
            arc_length,
            curvature,
            grade,
            banking,
            798.0,
            0.5 * 1.225 * 3.2 * 1.5,
            0.5 * 1.225 * 0.9 * 1.5,
            0.5,
            0.5,
            0.31,
            3.60,
            1.60,
            1.55,
            0.53,
            8.0,
            16.0,
            0.12,
            0.5,
            12,
            0.05,
            9.5,
            1.35,
            1.85,
            0.97,
            3500.0,
            -0.08,
            0.4,
            9.0,
            1.32,
            1.80,
            0.98,
            3600.0,
            -0.07,
            0.4,
            115.0,
            115.0,
            8.0,
            20,
            0.1,
        )

    def test_python_numba_kernel_executes_and_respects_bounds(self) -> None:
        """Execute kernel in pure Python mode and verify physical invariants."""
        (
            arc_length,
            curvature,
            grade,
            banking,
            mass,
            downforce_scale,
            drag_scale,
            friction_coefficient,
            max_drive_accel,
            max_brake_accel,
            max_speed,
            initial_speed,
            min_speed,
            lateral_iterations_limit,
            lateral_convergence_tolerance,
        ) = self._kernel_inputs()

        speed, longitudinal_accel, lateral_accel, lateral_iterations, lap_time = (
            _point_mass_speed_profile_kernel(
                arc_length,
                curvature,
                grade,
                banking,
                mass,
                downforce_scale,
                drag_scale,
                friction_coefficient,
                max_drive_accel,
                max_brake_accel,
                max_speed,
                initial_speed,
                min_speed,
                lateral_iterations_limit,
                lateral_convergence_tolerance,
            )
        )

        self.assertEqual(speed.shape, curvature.shape)
        self.assertEqual(longitudinal_accel.shape, curvature.shape)
        self.assertEqual(lateral_accel.shape, curvature.shape)
        self.assertGreaterEqual(lateral_iterations, 1)
        self.assertLessEqual(lateral_iterations, lateral_iterations_limit)

        self.assertTrue(np.all(np.isfinite(speed)))
        self.assertTrue(np.all(np.isfinite(longitudinal_accel)))
        self.assertTrue(np.all(np.isfinite(lateral_accel)))
        self.assertTrue(np.all(speed >= min_speed - 1e-12))
        self.assertTrue(np.all(speed <= max_speed + 1e-12))
        np.testing.assert_allclose(lateral_accel, speed * speed * curvature, rtol=0.0, atol=1e-12)
        self.assertGreater(lap_time, 0.0)

        if speed.size > 1:
            self.assertAlmostEqual(float(longitudinal_accel[-1]), float(longitudinal_accel[-2]))

    @unittest.skipUnless(NUMBA_AVAILABLE, "Numba not installed")
    def test_compiled_numba_kernel_matches_python_kernel(self) -> None:
        """Match compiled numba kernel output against Python reference kernel."""
        kernel = _compiled_numba_kernel()
        args = self._kernel_inputs()

        python_result = _point_mass_speed_profile_kernel(*args)
        compiled_result = kernel(*args)

        np.testing.assert_allclose(compiled_result[0], python_result[0], rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(compiled_result[1], python_result[1], rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(compiled_result[2], python_result[2], rtol=1e-11, atol=1e-11)
        self.assertEqual(int(compiled_result[3]), int(python_result[3]))
        self.assertAlmostEqual(float(compiled_result[4]), float(python_result[4]), places=11)

    def test_python_single_track_numba_kernel_executes_and_respects_bounds(self) -> None:
        """Execute single_track kernel in Python mode and verify physical invariants."""
        args = self._single_track_kernel_inputs()
        arc_length = args[0]
        curvature = args[1]
        max_speed = args[-5]
        min_speed = args[-3]
        lateral_iterations_limit = args[-2]

        speed, longitudinal_accel, lateral_accel, lateral_iterations, lap_time = (
            _single_track_speed_profile_kernel(*args)
        )

        self.assertEqual(speed.shape, arc_length.shape)
        self.assertEqual(longitudinal_accel.shape, arc_length.shape)
        self.assertEqual(lateral_accel.shape, arc_length.shape)
        self.assertGreaterEqual(lateral_iterations, 1)
        self.assertLessEqual(lateral_iterations, lateral_iterations_limit)

        self.assertTrue(np.all(np.isfinite(speed)))
        self.assertTrue(np.all(np.isfinite(longitudinal_accel)))
        self.assertTrue(np.all(np.isfinite(lateral_accel)))
        self.assertTrue(np.all(speed >= min_speed - 1e-12))
        self.assertTrue(np.all(speed <= max_speed + 1e-12))
        np.testing.assert_allclose(lateral_accel, speed * speed * curvature, rtol=0.0, atol=1e-12)
        self.assertGreater(lap_time, 0.0)

        if speed.size > 1:
            self.assertAlmostEqual(float(longitudinal_accel[-1]), float(longitudinal_accel[-2]))

    @unittest.skipUnless(NUMBA_AVAILABLE, "Numba not installed")
    def test_compiled_single_track_numba_kernel_matches_python_kernel(self) -> None:
        """Match compiled single_track kernel output against Python reference kernel."""
        kernel = _compiled_single_track_numba_kernel()
        self.assertIs(kernel, _compiled_single_track_numba_kernel())
        args = self._single_track_kernel_inputs()

        python_result = _single_track_speed_profile_kernel(*args)
        compiled_result = kernel(*args)

        np.testing.assert_allclose(compiled_result[0], python_result[0], rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(compiled_result[1], python_result[1], rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(compiled_result[2], python_result[2], rtol=1e-11, atol=1e-11)
        self.assertEqual(int(compiled_result[3]), int(python_result[3]))
        self.assertAlmostEqual(float(compiled_result[4]), float(python_result[4]), places=11)

    def test_require_torch_raises_configuration_error_when_missing(self) -> None:
        """Raise configuration error when torch import fails at runtime."""
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
            torch_profile._require_torch()

    def test_compiled_solver_falls_back_when_torch_compile_fails(self) -> None:
        """Use non-compiled fallback when torch.compile raises an exception."""
        torch_profile._COMPILED_SOLVER_CACHE.clear()
        with patch.object(torch_profile, "_require_torch", return_value=_DummyTorchModule()):
            solver = torch_profile._compiled_solver(enable_compile=True)
        self.assertIs(solver, torch_profile._solve_speed_profile_torch_impl)

    def test_compiled_solver_uses_cache_without_reimporting_torch(self) -> None:
        """Reuse cached compiled solver callable on repeated requests."""
        sentinel = object()
        torch_profile._COMPILED_SOLVER_CACHE[(torch_profile.TORCH_COMPILE_MODE,)] = sentinel
        with patch.object(
            torch_profile,
            "_require_torch",
            side_effect=AssertionError("_require_torch must not be called"),
        ):
            solver = torch_profile._compiled_solver(enable_compile=True)
        self.assertIs(solver, sentinel)

    def test_numba_profile_rejects_malformed_parameter_tuple(self) -> None:
        """Reject malformed numba parameter tuples from model backends."""
        if not NUMBA_AVAILABLE:
            self.skipTest("Numba not installed")

        config = build_simulation_config(compute_backend="numba")
        with self.assertRaises(ConfigurationError):
            solve_speed_profile_numba(
                track=self.track,
                model=_InvalidNumbaModel(),
                config=config,
            )

    def test_numba_profile_rejects_model_without_numba_backend_method(self) -> None:
        """Reject models that do not expose the numba backend adapter API."""
        if not NUMBA_AVAILABLE:
            self.skipTest("Numba not installed")

        config = build_simulation_config(compute_backend="numba")
        with self.assertRaises(ConfigurationError):
            solve_speed_profile_numba(
                track=self.track,
                model=_MissingNumbaApiModel(),
                config=config,
            )

    def test_require_numba_raises_configuration_error_when_missing(self) -> None:
        """Raise configuration error when numba import fails at runtime."""
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
            numba_profile._require_numba()

    def test_compiled_numba_kernel_returns_cached_callable(self) -> None:
        """Return same callable on repeated numba kernel compilation requests."""
        if not NUMBA_AVAILABLE:
            self.skipTest("Numba not installed")

        numba_profile._COMPILED_NUMBA_KERNEL = None
        first_kernel = _compiled_numba_kernel()
        second_kernel = _compiled_numba_kernel()
        self.assertIs(first_kernel, second_kernel)

    def test_compiled_single_track_numba_kernel_returns_cached_callable(self) -> None:
        """Return same callable on repeated single_track-kernel compilation requests."""
        if not NUMBA_AVAILABLE:
            self.skipTest("Numba not installed")

        numba_profile._COMPILED_SINGLE_TRACK_NUMBA_KERNEL = None
        first_kernel = _compiled_single_track_numba_kernel()
        second_kernel = _compiled_single_track_numba_kernel()
        self.assertIs(first_kernel, second_kernel)


if __name__ == "__main__":
    unittest.main()
