"""Unit tests for performance-envelope analysis APIs."""

from __future__ import annotations

import builtins
import importlib.util
import unittest
from typing import cast
from unittest.mock import patch

import numpy as np

import pylapsim.analysis.performance_envelope as envelope_module
from pylapsim.analysis.performance_envelope import (
    PerformanceEnvelopeNumerics,
    PerformanceEnvelopePhysics,
    PerformanceEnvelopeResult,
    PerformanceEnvelopeRuntime,
    build_performance_envelope_config,
    compute_performance_envelope,
)
from pylapsim.utils.exceptions import ConfigurationError
from pylapsim.vehicle import PointMassModel, PointMassPhysics
from tests.helpers import sample_vehicle_parameters

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
NUMBA_AVAILABLE = importlib.util.find_spec("numba") is not None
PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None


def _build_point_mass_model() -> PointMassModel:
    """Build a representative point-mass model instance for tests.

    Returns:
        Configured point-mass model.
    """
    return PointMassModel(
        vehicle=sample_vehicle_parameters(),
        physics=PointMassPhysics(
            max_drive_accel=8.0,
            max_brake_accel=16.0,
            friction_coefficient=1.7,
        ),
    )


class PerformanceEnvelopeTests(unittest.TestCase):
    """Validate performance-envelope configuration and computation behavior."""

    def test_config_validation_rejects_invalid_values(self) -> None:
        """Reject invalid physical, numerical, and runtime envelope settings."""
        with self.assertRaises(ConfigurationError):
            build_performance_envelope_config(
                physics=PerformanceEnvelopePhysics(speed_min=0.0, speed_max=60.0),
            )
        with self.assertRaises(ConfigurationError):
            build_performance_envelope_config(
                physics=PerformanceEnvelopePhysics(speed_min=20.0, speed_max=20.0),
            )
        with self.assertRaises(ConfigurationError):
            build_performance_envelope_config(
                numerics=PerformanceEnvelopeNumerics(speed_samples=1),
            )
        with self.assertRaises(ConfigurationError):
            build_performance_envelope_config(
                numerics=PerformanceEnvelopeNumerics(lateral_accel_samples=1),
            )
        with self.assertRaises(ConfigurationError):
            build_performance_envelope_config(
                numerics=PerformanceEnvelopeNumerics(lateral_accel_fraction_span=0.0),
            )
        with self.assertRaises(ConfigurationError):
            build_performance_envelope_config(
                runtime=PerformanceEnvelopeRuntime(compute_backend="numpy", torch_compile=True),
            )

    def test_validation_rejects_nonfinite_physics_and_runtime_contract_violations(self) -> None:
        """Reject non-finite physical values and malformed runtime fields."""
        with self.assertRaises(ConfigurationError):
            build_performance_envelope_config(
                physics=PerformanceEnvelopePhysics(speed_min=20.0, speed_max=60.0, grade=np.nan),
            )
        with self.assertRaises(ConfigurationError):
            build_performance_envelope_config(
                physics=PerformanceEnvelopePhysics(speed_min=20.0, speed_max=60.0, banking=np.inf),
            )
        with self.assertRaises(ConfigurationError):
            build_performance_envelope_config(
                runtime=PerformanceEnvelopeRuntime(compute_backend="invalid"),
            )
        with self.assertRaises(ConfigurationError):
            build_performance_envelope_config(
                runtime=PerformanceEnvelopeRuntime(compute_backend="numpy", torch_device=""),
            )
        with self.assertRaises(ConfigurationError):
            build_performance_envelope_config(
                runtime=PerformanceEnvelopeRuntime(
                    compute_backend="numpy",
                    torch_compile=cast(bool, "yes"),
                ),
            )
        with self.assertRaises(ConfigurationError):
            build_performance_envelope_config(
                runtime=PerformanceEnvelopeRuntime(
                    compute_backend="numpy",
                    torch_device="cuda:0",
                ),
            )

    def test_compute_with_numpy_backend_returns_consistent_grid(self) -> None:
        """Compute envelope on NumPy backend and verify shape and sign conventions."""
        model = _build_point_mass_model()
        result = compute_performance_envelope(
            model=model,
            physics=PerformanceEnvelopePhysics(speed_min=20.0, speed_max=70.0),
            numerics=PerformanceEnvelopeNumerics(speed_samples=7, lateral_accel_samples=9),
        )

        self.assertEqual(result.speed.shape, (7,))
        self.assertEqual(result.lateral_accel_limit.shape, (7,))
        self.assertEqual(result.lateral_accel_fraction.shape, (9,))
        self.assertEqual(result.lateral_accel.shape, (7, 9))
        self.assertEqual(result.max_longitudinal_accel.shape, (7, 9))
        self.assertEqual(result.min_longitudinal_accel.shape, (7, 9))

        self.assertTrue(np.all(np.diff(result.speed) > 0.0))
        self.assertTrue(np.all(result.lateral_accel_limit > 0.0))
        self.assertTrue(np.all(result.max_longitudinal_accel >= result.min_longitudinal_accel))

        np.testing.assert_allclose(
            result.lateral_accel[:, 0],
            -result.lateral_accel_limit,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.lateral_accel[:, -1],
            result.lateral_accel_limit,
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertTrue(np.all(result.min_longitudinal_accel <= 0.0))

    def test_compute_accepts_explicit_config_and_rejects_mixed_inputs(self) -> None:
        """Allow config-driven setup and reject conflicting component overrides."""
        model = _build_point_mass_model()
        config = build_performance_envelope_config(
            physics=PerformanceEnvelopePhysics(speed_min=15.0, speed_max=65.0),
            numerics=PerformanceEnvelopeNumerics(speed_samples=5, lateral_accel_samples=5),
        )
        result = compute_performance_envelope(model=model, config=config)
        self.assertEqual(result.speed.shape, (5,))

        with self.assertRaises(ConfigurationError):
            compute_performance_envelope(
                model=model,
                config=config,
                physics=PerformanceEnvelopePhysics(speed_min=20.0, speed_max=70.0),
            )

    def test_result_to_numpy_stacks_expected_channels(self) -> None:
        """Stack lateral and longitudinal arrays in deterministic channel order."""
        model = _build_point_mass_model()
        result = compute_performance_envelope(
            model=model,
            numerics=PerformanceEnvelopeNumerics(speed_samples=4, lateral_accel_samples=6),
        )
        stacked = result.to_numpy()

        self.assertEqual(stacked.shape, (4, 6, 3))
        np.testing.assert_allclose(stacked[:, :, 0], result.lateral_accel, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            stacked[:, :, 1],
            result.max_longitudinal_accel,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            stacked[:, :, 2],
            result.min_longitudinal_accel,
            rtol=0.0,
            atol=0.0,
        )

    def test_result_shape_validation_rejects_invalid_grid(self) -> None:
        """Reject inconsistent output-grid dimensions at result construction time."""
        with self.assertRaises(ConfigurationError):
            PerformanceEnvelopeResult(
                speed=np.array([20.0, 40.0], dtype=float),
                lateral_accel_limit=np.array([10.0], dtype=float),
                lateral_accel_fraction=np.array([-1.0, 1.0], dtype=float),
                lateral_accel=np.zeros((2, 2), dtype=float),
                max_longitudinal_accel=np.zeros((2, 2), dtype=float),
                min_longitudinal_accel=np.zeros((2, 2), dtype=float),
            )

    def test_result_shape_validation_rejects_non_vector_axes(self) -> None:
        """Reject non-1D speed/fraction axes and mismatched signal-grid shape."""
        with self.assertRaises(ConfigurationError):
            PerformanceEnvelopeResult(
                speed=np.zeros((2, 1), dtype=float),
                lateral_accel_limit=np.zeros((2, 1), dtype=float),
                lateral_accel_fraction=np.array([-1.0, 1.0], dtype=float),
                lateral_accel=np.zeros((2, 2), dtype=float),
                max_longitudinal_accel=np.zeros((2, 2), dtype=float),
                min_longitudinal_accel=np.zeros((2, 2), dtype=float),
            )
        with self.assertRaises(ConfigurationError):
            PerformanceEnvelopeResult(
                speed=np.array([20.0, 40.0], dtype=float),
                lateral_accel_limit=np.array([10.0, 11.0], dtype=float),
                lateral_accel_fraction=np.zeros((1, 2), dtype=float),
                lateral_accel=np.zeros((2, 2), dtype=float),
                max_longitudinal_accel=np.zeros((2, 2), dtype=float),
                min_longitudinal_accel=np.zeros((2, 2), dtype=float),
            )
        with self.assertRaises(ConfigurationError):
            PerformanceEnvelopeResult(
                speed=np.array([20.0, 40.0], dtype=float),
                lateral_accel_limit=np.array([10.0, 11.0], dtype=float),
                lateral_accel_fraction=np.array([-1.0, 1.0], dtype=float),
                lateral_accel=np.zeros((2, 2), dtype=float),
                max_longitudinal_accel=np.zeros((2, 3), dtype=float),
                min_longitudinal_accel=np.zeros((2, 2), dtype=float),
            )

    def test_dataframe_export_raises_when_pandas_missing(self) -> None:
        """Raise configuration-level guidance when pandas cannot be imported."""
        model = _build_point_mass_model()
        result = compute_performance_envelope(
            model=model,
            numerics=PerformanceEnvelopeNumerics(speed_samples=3, lateral_accel_samples=3),
        )

        original_import = builtins.__import__

        def guarded_import(name: str, *args: object, **kwargs: object) -> object:
            """Block pandas import while delegating all other imports.

            Args:
                name: Module name requested by import machinery.
                *args: Positional import arguments.
                **kwargs: Keyword import arguments.

            Returns:
                Imported module for non-blocked names.

            Raises:
                ModuleNotFoundError: For pandas imports.
            """
            if name == "pandas":
                raise ModuleNotFoundError("No module named 'pandas'")
            return original_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=guarded_import),
            self.assertRaises(ConfigurationError),
        ):
            result.to_dataframe()

    def test_dataframe_export_works_with_stubbed_pandas_module(self) -> None:
        """Build dataframe output with a stub pandas module when pandas is absent."""
        model = _build_point_mass_model()
        result = compute_performance_envelope(
            model=model,
            numerics=PerformanceEnvelopeNumerics(speed_samples=2, lateral_accel_samples=3),
        )

        class _FakeFrame:
            """Minimal DataFrame stub for envelope export tests."""

            def __init__(self, data: dict[str, np.ndarray]) -> None:
                """Store frame data and derive ordered columns.

                Args:
                    data: Column-major payload passed by ``to_dataframe``.
                """
                self.data = data
                self.columns = list(data.keys())

            def __len__(self) -> int:
                """Return number of rows based on first column length.

                Returns:
                    Number of rows in the stub frame.
                """
                return int(len(next(iter(self.data.values()))))

        fake_pandas = type(
            "_FakePandas",
            (),
            {"DataFrame": staticmethod(lambda data: _FakeFrame(cast(dict[str, np.ndarray], data)))},
        )

        original_import = builtins.__import__

        def guarded_import(name: str, *args: object, **kwargs: object) -> object:
            """Inject fake pandas while delegating other imports.

            Args:
                name: Module name requested by import machinery.
                *args: Positional import arguments.
                **kwargs: Keyword import arguments.

            Returns:
                Imported module object.
            """
            if name == "pandas":
                return fake_pandas
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            frame = result.to_dataframe()

        self.assertEqual(
            frame.columns,
            [
                "speed",
                "lateral_accel_limit",
                "lateral_accel_fraction",
                "lateral_accel",
                "max_longitudinal_accel",
                "min_longitudinal_accel",
            ],
        )
        self.assertEqual(len(frame), 6)

    @unittest.skipUnless(PANDAS_AVAILABLE, "pandas not installed")
    def test_dataframe_export_contains_expected_columns(self) -> None:
        """Export long-form DataFrame with deterministic schema and row count."""
        model = _build_point_mass_model()
        result = compute_performance_envelope(
            model=model,
            numerics=PerformanceEnvelopeNumerics(speed_samples=4, lateral_accel_samples=5),
        )
        frame = result.to_dataframe()

        expected_columns = {
            "speed",
            "lateral_accel_limit",
            "lateral_accel_fraction",
            "lateral_accel",
            "max_longitudinal_accel",
            "min_longitudinal_accel",
        }
        self.assertEqual(set(frame.columns), expected_columns)
        self.assertEqual(len(frame), 20)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_torch_backend_matches_numpy_backend(self) -> None:
        """Match torch and NumPy envelopes for identical model and discretization."""
        model = _build_point_mass_model()
        physics = PerformanceEnvelopePhysics(speed_min=20.0, speed_max=80.0)
        numerics = PerformanceEnvelopeNumerics(speed_samples=9, lateral_accel_samples=11)

        numpy_result = compute_performance_envelope(
            model=model,
            physics=physics,
            numerics=numerics,
            runtime=PerformanceEnvelopeRuntime(compute_backend="numpy"),
        )
        torch_result = compute_performance_envelope(
            model=model,
            physics=physics,
            numerics=numerics,
            runtime=PerformanceEnvelopeRuntime(
                compute_backend="torch",
                torch_device="cpu",
                torch_compile=False,
            ),
        )

        np.testing.assert_allclose(torch_result.speed, numpy_result.speed, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            torch_result.lateral_accel_limit,
            numpy_result.lateral_accel_limit,
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            torch_result.lateral_accel,
            numpy_result.lateral_accel,
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            torch_result.max_longitudinal_accel,
            numpy_result.max_longitudinal_accel,
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            torch_result.min_longitudinal_accel,
            numpy_result.min_longitudinal_accel,
            rtol=1e-10,
            atol=1e-10,
        )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_torch_backend_requires_torch_model_api_methods(self) -> None:
        """Reject torch backend usage when model lacks torch-specific methods."""

        class _NoTorchModel:
            """Minimal scalar-only model used to trigger API contract errors."""

            def validate(self) -> None:
                """Provide no-op validation for protocol compatibility."""

            @staticmethod
            def lateral_accel_limit(speed: float, banking: float) -> float:
                """Return constant lateral limit.

                Args:
                    speed: Vehicle speed [m/s].
                    banking: Banking angle [rad].

                Returns:
                    Constant lateral limit [m/s^2].
                """
                del speed, banking
                return 10.0

            @staticmethod
            def max_longitudinal_accel(
                speed: float,
                lateral_accel_required: float,
                grade: float,
                banking: float,
            ) -> float:
                """Return constant forward acceleration.

                Args:
                    speed: Vehicle speed [m/s].
                    lateral_accel_required: Required lateral acceleration [m/s^2].
                    grade: Road grade ``dz/ds``.
                    banking: Banking angle [rad].

                Returns:
                    Constant forward acceleration [m/s^2].
                """
                del speed, lateral_accel_required, grade, banking
                return 2.0

            @staticmethod
            def max_longitudinal_decel(
                speed: float,
                lateral_accel_required: float,
                grade: float,
                banking: float,
            ) -> float:
                """Return constant braking deceleration magnitude.

                Args:
                    speed: Vehicle speed [m/s].
                    lateral_accel_required: Required lateral acceleration [m/s^2].
                    grade: Road grade ``dz/ds``.
                    banking: Banking angle [rad].

                Returns:
                    Constant braking deceleration magnitude [m/s^2].
                """
                del speed, lateral_accel_required, grade, banking
                return 3.0

        with self.assertRaises(ConfigurationError):
            compute_performance_envelope(
                model=_NoTorchModel(),
                runtime=PerformanceEnvelopeRuntime(compute_backend="torch"),
            )

    def test_compute_uses_scalar_lateral_limit_fallback(self) -> None:
        """Use scalar lateral-limit API when vectorized batch API is unavailable."""

        class _ScalarOnlyModel:
            """Minimal model exposing only scalar limit methods."""

            def validate(self) -> None:
                """Provide no-op validation for protocol compatibility."""

            @staticmethod
            def lateral_accel_limit(speed: float, banking: float) -> float:
                """Return speed-dependent lateral limit.

                Args:
                    speed: Vehicle speed [m/s].
                    banking: Banking angle [rad].

                Returns:
                    Lateral acceleration limit [m/s^2].
                """
                del banking
                return 5.0 + 0.01 * speed

            @staticmethod
            def max_longitudinal_accel(
                speed: float,
                lateral_accel_required: float,
                grade: float,
                banking: float,
            ) -> float:
                """Return a simple constant acceleration model.

                Args:
                    speed: Vehicle speed [m/s].
                    lateral_accel_required: Required lateral acceleration [m/s^2].
                    grade: Road grade ``dz/ds``.
                    banking: Banking angle [rad].

                Returns:
                    Constant forward acceleration [m/s^2].
                """
                del speed, lateral_accel_required, grade, banking
                return 1.5

            @staticmethod
            def max_longitudinal_decel(
                speed: float,
                lateral_accel_required: float,
                grade: float,
                banking: float,
            ) -> float:
                """Return a simple constant deceleration model.

                Args:
                    speed: Vehicle speed [m/s].
                    lateral_accel_required: Required lateral acceleration [m/s^2].
                    grade: Road grade ``dz/ds``.
                    banking: Banking angle [rad].

                Returns:
                    Constant braking deceleration [m/s^2].
                """
                del speed, lateral_accel_required, grade, banking
                return 2.0

        result = compute_performance_envelope(
            model=_ScalarOnlyModel(),
            numerics=PerformanceEnvelopeNumerics(speed_samples=4, lateral_accel_samples=4),
        )
        self.assertEqual(result.lateral_accel.shape, (4, 4))

    def test_compute_rejects_invalid_vectorized_lateral_limit_shape(self) -> None:
        """Reject batch lateral-limit model APIs that return mismatched shapes."""

        class _BadBatchModel:
            """Model with malformed vectorized lateral-limit output."""

            def validate(self) -> None:
                """Provide no-op validation for protocol compatibility."""

            @staticmethod
            def lateral_accel_limit_batch(speed: np.ndarray, banking: np.ndarray) -> np.ndarray:
                """Return malformed shape for negative-path testing.

                Args:
                    speed: Speed samples [m/s].
                    banking: Banking-angle samples [rad].

                Returns:
                    Shape-incompatible lateral limit array.
                """
                del speed, banking
                return np.array([1.0], dtype=float)

            @staticmethod
            def lateral_accel_limit(speed: float, banking: float) -> float:
                """Return fallback scalar limit.

                Args:
                    speed: Vehicle speed [m/s].
                    banking: Banking angle [rad].

                Returns:
                    Constant lateral limit [m/s^2].
                """
                del speed, banking
                return 8.0

            @staticmethod
            def max_longitudinal_accel(
                speed: float,
                lateral_accel_required: float,
                grade: float,
                banking: float,
            ) -> float:
                """Return constant acceleration for interface completeness.

                Args:
                    speed: Vehicle speed [m/s].
                    lateral_accel_required: Required lateral acceleration [m/s^2].
                    grade: Road grade ``dz/ds``.
                    banking: Banking angle [rad].

                Returns:
                    Constant acceleration [m/s^2].
                """
                del speed, lateral_accel_required, grade, banking
                return 1.0

            @staticmethod
            def max_longitudinal_decel(
                speed: float,
                lateral_accel_required: float,
                grade: float,
                banking: float,
            ) -> float:
                """Return constant braking for interface completeness.

                Args:
                    speed: Vehicle speed [m/s].
                    lateral_accel_required: Required lateral acceleration [m/s^2].
                    grade: Road grade ``dz/ds``.
                    banking: Banking angle [rad].

                Returns:
                    Constant braking [m/s^2].
                """
                del speed, lateral_accel_required, grade, banking
                return 1.0

        with self.assertRaises(ConfigurationError):
            compute_performance_envelope(
                model=_BadBatchModel(),
                numerics=PerformanceEnvelopeNumerics(speed_samples=4, lateral_accel_samples=4),
            )

    def test_runtime_validation_handles_missing_optional_backends(self) -> None:
        """Raise explicit errors when requested optional backends are unavailable."""
        original_import = builtins.__import__

        def guarded_import(name: str, *args: object, **kwargs: object) -> object:
            """Block backend imports while delegating all other imports.

            Args:
                name: Module name requested by import machinery.
                *args: Positional import arguments.
                **kwargs: Keyword import arguments.

            Returns:
                Imported module for non-blocked names.

            Raises:
                ModuleNotFoundError: For blocked backend module imports.
            """
            if name in {"numba", "torch"}:
                raise ModuleNotFoundError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            with self.assertRaises(ConfigurationError):
                PerformanceEnvelopeRuntime(compute_backend="numba").validate()
            with self.assertRaises(ConfigurationError):
                PerformanceEnvelopeRuntime(compute_backend="torch").validate()

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_runtime_validation_rejects_unavailable_cuda_device(self) -> None:
        """Reject CUDA runtime requests when torch reports no CUDA availability."""
        with patch.object(envelope_module, "_require_torch") as require_torch_mock:
            require_torch_mock.return_value = type(
                "_TorchStub",
                (),
                {"cuda": type("_CudaStub", (), {"is_available": staticmethod(lambda: False)})},
            )
            with self.assertRaises(ConfigurationError):
                PerformanceEnvelopeRuntime(
                    compute_backend="torch",
                    torch_device="cuda:0",
                ).validate()

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_torch_compile_fallback_returns_solver_when_compile_fails(self) -> None:
        """Fallback to eager solver implementation when torch.compile fails."""
        envelope_module._COMPILED_TORCH_ENVELOPE_SOLVER_CACHE.clear()
        compile_failure_stub = staticmethod(
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError)
        )
        with patch.object(envelope_module, "_require_torch") as require_torch_mock:
            require_torch_mock.return_value = type(
                "_TorchStub",
                (),
                {"compile": compile_failure_stub},
            )
            solver = envelope_module._compiled_torch_envelope_solver(enable_compile=True)
        self.assertIs(solver, envelope_module._solve_performance_envelope_torch_impl)

    @unittest.skipUnless(NUMBA_AVAILABLE, "Numba not installed")
    def test_numba_runtime_path_matches_numpy_backend(self) -> None:
        """Match numba-runtime selection against baseline NumPy envelope output."""
        model = _build_point_mass_model()
        physics = PerformanceEnvelopePhysics(speed_min=25.0, speed_max=75.0)
        numerics = PerformanceEnvelopeNumerics(speed_samples=6, lateral_accel_samples=8)

        numpy_result = compute_performance_envelope(
            model=model,
            physics=physics,
            numerics=numerics,
            runtime=PerformanceEnvelopeRuntime(compute_backend="numpy"),
        )
        numba_result = compute_performance_envelope(
            model=model,
            physics=physics,
            numerics=numerics,
            runtime=PerformanceEnvelopeRuntime(compute_backend="numba"),
        )

        np.testing.assert_allclose(numba_result.speed, numpy_result.speed, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            numba_result.lateral_accel_limit,
            numpy_result.lateral_accel_limit,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            numba_result.max_longitudinal_accel,
            numpy_result.max_longitudinal_accel,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            numba_result.min_longitudinal_accel,
            numpy_result.min_longitudinal_accel,
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
