"""Unit tests for scalar sensitivity-analysis APIs."""

from __future__ import annotations

import builtins
import importlib.util
import unittest
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import numpy as np

from apexsim.analysis.sensitivity import (
    SensitivityConfig,
    SensitivityNumerics,
    SensitivityParameter,
    SensitivityResult,
    SensitivityRuntime,
    build_sensitivity_config,
    compute_sensitivities,
)
from apexsim.simulation import build_simulation_config, solve_speed_profile_torch
from apexsim.track import build_straight_track
from apexsim.utils.exceptions import ConfigurationError

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _quadratic_objective_numpy(parameters: dict[str, float]) -> float:
    """Return a deterministic scalar objective from two scalar parameters.

    Args:
        parameters: Parameter mapping with keys ``x`` and ``y``.

    Returns:
        Objective value ``x^2 + 3y + 2``.
    """
    x = float(parameters["x"])
    y = float(parameters["y"])
    return x * x + 3.0 * y + 2.0


@dataclass(frozen=True)
class _LapTimeObjectiveTorchModel:
    """Differentiable torch model used for AD-vs-FD lap-time regression tests."""

    drive_gain: Any
    drag_gain: float = 8e-4

    def validate(self) -> None:
        """Provide protocol-compatible model validation."""

    def lateral_accel_limit_torch(self, speed: Any, banking: Any) -> Any:
        """Return large lateral capacity for straight-line objective track.

        Args:
            speed: Speed tensor [m/s].
            banking: Banking-angle tensor [rad].

        Returns:
            Lateral-acceleration limit tensor [m/s^2].
        """
        del banking
        import torch

        return torch.full_like(speed, 50.0)

    def max_longitudinal_accel_torch(
        self,
        speed: Any,
        lateral_accel_required: Any,
        grade: Any,
        banking: Any,
    ) -> Any:
        """Return differentiable acceleration model with quadratic drag.

        Args:
            speed: Speed tensor [m/s].
            lateral_accel_required: Required lateral acceleration tensor [m/s^2].
            grade: Track-grade tensor ``dz/ds``.
            banking: Banking-angle tensor [rad].

        Returns:
            Net forward-acceleration tensor [m/s^2].
        """
        del lateral_accel_required, grade, banking
        import torch

        accel = self.drive_gain - self.drag_gain * speed * speed
        return torch.clamp(accel, min=0.0)

    def max_longitudinal_decel_torch(
        self,
        speed: Any,
        lateral_accel_required: Any,
        grade: Any,
        banking: Any,
    ) -> Any:
        """Return constant deceleration capacity for backward pass feasibility.

        Args:
            speed: Speed tensor [m/s].
            lateral_accel_required: Required lateral acceleration tensor [m/s^2].
            grade: Track-grade tensor ``dz/ds``.
            banking: Banking-angle tensor [rad].

        Returns:
            Available deceleration-magnitude tensor [m/s^2].
        """
        del lateral_accel_required, grade, banking
        import torch

        return torch.full_like(speed, 12.0)


class SensitivityApiTests(unittest.TestCase):
    """Validate sensitivity API contracts and backend behavior."""

    def test_parameter_validation_rejects_invalid_inputs(self) -> None:
        """Reject malformed parameter names, kinds, values, and bounds."""
        with self.assertRaises(ConfigurationError):
            SensitivityParameter(name="", value=1.0).validate()
        with self.assertRaises(ConfigurationError):
            SensitivityParameter(name="mass", value=np.nan).validate()
        with self.assertRaises(ConfigurationError):
            SensitivityParameter(name="mass", value=1.0, kind="invalid").validate()
        with self.assertRaises(ConfigurationError):
            SensitivityParameter(
                name="mass",
                value=1.0,
                lower_bound=2.0,
                upper_bound=1.0,
            ).validate()
        with self.assertRaises(ConfigurationError):
            SensitivityParameter(name="mass", value=0.5, lower_bound=1.0).validate()
        with self.assertRaises(ConfigurationError):
            SensitivityParameter(name="mass", value=2.0, upper_bound=1.0).validate()

    def test_config_validation_rejects_invalid_runtime_and_numerics(self) -> None:
        """Reject invalid methods, schemes, and step definitions."""
        with self.assertRaises(ConfigurationError):
            build_sensitivity_config(runtime=SensitivityRuntime(method="unsupported"))
        with self.assertRaises(ConfigurationError):
            build_sensitivity_config(
                numerics=SensitivityNumerics(finite_difference_scheme="backward"),
            )
        with self.assertRaises(ConfigurationError):
            build_sensitivity_config(
                numerics=SensitivityNumerics(finite_difference_relative_step=0.0),
            )
        with self.assertRaises(ConfigurationError):
            build_sensitivity_config(
                numerics=SensitivityNumerics(finite_difference_absolute_step=0.0),
            )
        with self.assertRaises(ConfigurationError):
            build_sensitivity_config(
                runtime=SensitivityRuntime(method="finite_difference", torch_device="cuda:0"),
            )
        with self.assertRaises(ConfigurationError):
            build_sensitivity_config(
                runtime=SensitivityRuntime(
                    method="autodiff",
                    autodiff_fallback_to_finite_difference="yes",  # type: ignore[arg-type]
                ),
            )

    def test_compute_with_central_finite_difference_matches_analytic_gradient(self) -> None:
        """Estimate gradients with central finite differences."""
        parameters = [
            SensitivityParameter(name="x", value=2.0, kind="physical"),
            SensitivityParameter(name="y", value=1.0, kind="numerical"),
        ]
        result = compute_sensitivities(
            objective=_quadratic_objective_numpy,
            parameters=parameters,
            runtime=SensitivityRuntime(method="finite_difference"),
            numerics=SensitivityNumerics(
                finite_difference_scheme="central",
                finite_difference_relative_step=1e-6,
                finite_difference_absolute_step=1e-8,
            ),
        )

        self.assertEqual(result.method, "finite_difference")
        self.assertAlmostEqual(result.objective_value, 9.0, places=10)
        self.assertAlmostEqual(result.sensitivities["x"], 4.0, places=5)
        self.assertAlmostEqual(result.sensitivities["y"], 3.0, places=5)
        self.assertEqual(result.parameter_kinds["x"], "physical")
        self.assertEqual(result.parameter_kinds["y"], "numerical")

    def test_compute_with_forward_scheme_falls_back_to_backward_at_upper_bound(self) -> None:
        """Use backward difference when forward perturbation violates bounds."""
        parameter = SensitivityParameter(
            name="x",
            value=2.0,
            lower_bound=0.0,
            upper_bound=2.0,
        )

        result = compute_sensitivities(
            objective=lambda params: float(params["x"] ** 2),
            parameters=[parameter],
            runtime=SensitivityRuntime(method="finite_difference"),
            numerics=SensitivityNumerics(
                finite_difference_scheme="forward",
                finite_difference_relative_step=1e-6,
                finite_difference_absolute_step=1e-8,
            ),
        )

        self.assertEqual(result.method, "finite_difference")
        self.assertAlmostEqual(result.sensitivities["x"], 4.0, places=5)

    def test_compute_accepts_mapping_input(self) -> None:
        """Allow concise ``name -> value`` mapping for parameter definitions."""
        result = compute_sensitivities(
            objective=_quadratic_objective_numpy,
            parameters={"x": 2.0, "y": 1.0},
            runtime=SensitivityRuntime(method="finite_difference"),
            numerics=SensitivityNumerics(
                finite_difference_scheme="central",
                finite_difference_relative_step=1e-6,
                finite_difference_absolute_step=1e-8,
            ),
        )
        self.assertEqual(result.method, "finite_difference")
        self.assertAlmostEqual(result.sensitivities["x"], 4.0, places=5)
        self.assertAlmostEqual(result.sensitivities["y"], 3.0, places=5)

    def test_compute_rejects_duplicate_parameter_names(self) -> None:
        """Reject sensitivity parameter lists with duplicate names."""
        with self.assertRaises(ConfigurationError):
            compute_sensitivities(
                objective=_quadratic_objective_numpy,
                parameters=[
                    SensitivityParameter(name="x", value=1.0),
                    SensitivityParameter(name="x", value=2.0),
                ],
                runtime=SensitivityRuntime(method="finite_difference"),
            )
        with self.assertRaises(ConfigurationError):
            compute_sensitivities(
                objective=_quadratic_objective_numpy,
                parameters=[],
                runtime=SensitivityRuntime(method="finite_difference"),
            )

    def test_compute_rejects_mixed_config_and_component_inputs(self) -> None:
        """Reject passing both full config and component overrides."""
        config = SensitivityConfig()
        with self.assertRaises(ConfigurationError):
            compute_sensitivities(
                objective=_quadratic_objective_numpy,
                parameters={"x": 2.0},
                config=config,
                runtime=SensitivityRuntime(method="finite_difference"),
            )

    def test_compute_rejects_nonscalar_objective_in_finite_difference(self) -> None:
        """Reject objective outputs that are not scalar values."""
        with self.assertRaises(ConfigurationError):
            compute_sensitivities(
                objective=lambda params: np.array([params["x"], params["x"]], dtype=float),
                parameters={"x": 2.0},
                runtime=SensitivityRuntime(method="finite_difference"),
            )

    def test_autodiff_falls_back_to_finite_difference_by_default(self) -> None:
        """Fallback to finite difference when autodiff output is non-tensor."""
        result = compute_sensitivities(
            objective=lambda params: 5.0,
            parameters={"x": 2.0, "y": 1.0},
        )
        self.assertEqual(result.method, "finite_difference")
        self.assertAlmostEqual(result.objective_value, 5.0, places=12)
        self.assertAlmostEqual(result.sensitivities["x"], 0.0, places=12)
        self.assertAlmostEqual(result.sensitivities["y"], 0.0, places=12)

    def test_autodiff_error_is_raised_when_fallback_disabled(self) -> None:
        """Raise configuration error when autodiff fails and fallback is disabled."""
        with self.assertRaises(ConfigurationError):
            compute_sensitivities(
                objective=lambda params: 3.0,
                parameters={"x": 2.0},
                runtime=SensitivityRuntime(
                    method="autodiff",
                    autodiff_fallback_to_finite_difference=False,
                ),
            )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_autodiff_default_method_returns_expected_gradients(self) -> None:
        """Compute gradients with autodiff when objective is torch-native."""

        def objective(parameters: dict[str, object]) -> object:
            """Return differentiable quadratic objective.

            Args:
                parameters: Tensor-valued parameter mapping.

            Returns:
                Scalar tensor objective.
            """
            x = parameters["x"]
            y = parameters["y"]
            return x * x + 3.0 * y + 2.0

        result = compute_sensitivities(
            objective=objective,
            parameters={"x": 2.0, "y": 1.0},
        )
        self.assertEqual(result.method, "autodiff")
        self.assertAlmostEqual(result.objective_value, 9.0, places=10)
        self.assertAlmostEqual(result.sensitivities["x"], 4.0, places=10)
        self.assertAlmostEqual(result.sensitivities["y"], 3.0, places=10)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_autodiff_rejects_non_scalar_tensor_outputs_when_fallback_disabled(self) -> None:
        """Reject tensor objectives that do not produce a scalar."""
        import torch

        with self.assertRaises(ConfigurationError):
            compute_sensitivities(
                objective=lambda params: torch.stack((params["x"], params["x"] + 1.0)),
                parameters={"x": 2.0},
                runtime=SensitivityRuntime(
                    method="autodiff",
                    autodiff_fallback_to_finite_difference=False,
                ),
            )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_autodiff_matches_finite_difference_for_lap_time_objective(self) -> None:
        """Match AD and FD gradients for a real torch-backed lap-time objective."""
        track = build_straight_track(length=600.0, sample_count=301)
        config = build_simulation_config(
            compute_backend="torch",
            torch_device="cpu",
            torch_compile=False,
            max_speed=60.0,
            initial_speed=10.0,
        )

        def objective(parameters: dict[str, Any]) -> Any:
            model = _LapTimeObjectiveTorchModel(drive_gain=parameters["drive_gain"])
            result = solve_speed_profile_torch(track=track, model=model, config=config)
            return result.lap_time

        parameters = [
            SensitivityParameter(
                name="drive_gain",
                value=6.0,
                lower_bound=0.1,
                kind="physical",
            ),
        ]
        finite_difference = compute_sensitivities(
            objective=objective,
            parameters=parameters,
            runtime=SensitivityRuntime(method="finite_difference"),
            numerics=SensitivityNumerics(
                finite_difference_scheme="central",
                finite_difference_relative_step=1e-4,
                finite_difference_absolute_step=1e-6,
            ),
        )
        autodiff = compute_sensitivities(
            objective=objective,
            parameters=parameters,
            runtime=SensitivityRuntime(
                method="autodiff",
                autodiff_fallback_to_finite_difference=False,
            ),
        )

        self.assertEqual(finite_difference.method, "finite_difference")
        self.assertEqual(autodiff.method, "autodiff")
        self.assertAlmostEqual(
            autodiff.objective_value,
            finite_difference.objective_value,
            places=9,
        )
        self.assertAlmostEqual(
            autodiff.sensitivities["drive_gain"],
            finite_difference.sensitivities["drive_gain"],
            delta=1e-6,
        )

    def test_runtime_validation_handles_missing_torch_dependency(self) -> None:
        """Raise clear error when autodiff is requested without torch."""
        original_import = builtins.__import__

        def guarded_import(name: str, *args: object, **kwargs: object) -> object:
            """Block torch imports while delegating all other imports.

            Args:
                name: Module name requested by import machinery.
                *args: Positional import arguments.
                **kwargs: Keyword import arguments.

            Returns:
                Imported module for non-blocked names.

            Raises:
                ModuleNotFoundError: For blocked torch import requests.
            """
            if name == "torch":
                raise ModuleNotFoundError("No module named 'torch'")
            return original_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=guarded_import),
            self.assertRaises(ConfigurationError),
        ):
            SensitivityRuntime(method="autodiff").validate()

    def test_result_validation_rejects_mismatched_keys(self) -> None:
        """Reject result payloads with inconsistent parameter key sets."""
        with self.assertRaises(ConfigurationError):
            SensitivityResult(
                objective_value=1.0,
                sensitivities={"mass": 0.1},
                method="finite_difference",
                parameter_values={"mass": 1000.0, "cd": 0.7},
                parameter_kinds={"mass": "physical"},
            )
        with self.assertRaises(ConfigurationError):
            SensitivityResult(
                objective_value=1.0,
                sensitivities={"mass": 0.1},
                method="finite_difference",
                parameter_values={"mass": 1000.0},
                parameter_kinds={"mass": "unknown"},
            )


if __name__ == "__main__":
    unittest.main()
