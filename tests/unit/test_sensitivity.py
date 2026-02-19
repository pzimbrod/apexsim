"""Unit tests for scalar sensitivity-analysis APIs."""

from __future__ import annotations

import builtins
import importlib.util
import unittest
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import patch

import numpy as np

import apexsim.analysis.sensitivity as sensitivity_module
from apexsim.analysis.sensitivity import (
    SensitivityConfig,
    SensitivityNumerics,
    SensitivityParameter,
    SensitivityResult,
    SensitivityRuntime,
    SensitivityStudyParameter,
    build_sensitivity_config,
    compute_sensitivities,
    register_sensitivity_model_adapter,
    run_lap_sensitivity_study,
)
from apexsim.simulation import (
    TransientConfig,
    TransientRuntimeConfig,
    build_simulation_config,
    solve_speed_profile_torch,
)
from apexsim.tire import default_axle_tire_parameters
from apexsim.track import build_straight_track
from apexsim.utils.exceptions import ConfigurationError
from apexsim.vehicle import (
    PointMassPhysics,
    SingleTrackPhysics,
    build_point_mass_model,
    build_single_track_model,
)
from tests.helpers import sample_vehicle_parameters

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _patch_transient_dependency_specs() -> Any:
    """Patch transient dependency discovery for deterministic unit tests.

    Returns:
        Active patch context manager.
    """
    real_find_spec = importlib.util.find_spec
    return patch(
        "apexsim.simulation.config.importlib.util.find_spec",
        side_effect=lambda name: (
            object() if name in {"scipy", "torchdiffeq"} else real_find_spec(name)
        ),
    )


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


class SensitivityStudyApiTests(unittest.TestCase):
    """Validate high-level lap sensitivity study helpers."""

    @staticmethod
    def _single_track_model() -> Any:
        """Build baseline single-track model for study API tests.

        Returns:
            Single-track model instance.
        """
        return build_single_track_model(
            vehicle=sample_vehicle_parameters(),
            tires=default_axle_tire_parameters(),
            physics=SingleTrackPhysics(),
        )

    @staticmethod
    def _point_mass_model() -> Any:
        """Build baseline point-mass model for study API tests.

        Returns:
            Point-mass model instance.
        """
        return build_point_mass_model(
            vehicle=sample_vehicle_parameters(),
            physics=PointMassPhysics(),
        )

    def test_register_sensitivity_model_adapter_rejects_invalid_inputs(self) -> None:
        """Reject malformed model-adapter registration inputs."""
        model = self._single_track_model()
        original_adapters = dict(sensitivity_module._SENSITIVITY_MODEL_ADAPTERS)
        try:
            with self.assertRaises(ConfigurationError):
                register_sensitivity_model_adapter(
                    model_type=cast(Any, 42),
                    model_factory=build_single_track_model,
                    model_inputs_getter=lambda _: {"vehicle": sample_vehicle_parameters()},
                )
            with self.assertRaises(ConfigurationError):
                register_sensitivity_model_adapter(
                    model_type=type(model),
                    model_factory=42,  # type: ignore[arg-type]
                    model_inputs_getter=lambda _: {"vehicle": sample_vehicle_parameters()},
                )
            with self.assertRaises(ConfigurationError):
                register_sensitivity_model_adapter(
                    model_type=type(model),
                    model_factory=build_single_track_model,
                    model_inputs_getter=42,  # type: ignore[arg-type]
                )
        finally:
            sensitivity_module._SENSITIVITY_MODEL_ADAPTERS.clear()
            sensitivity_module._SENSITIVITY_MODEL_ADAPTERS.update(original_adapters)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_run_lap_sensitivity_study_rejects_blank_label(self) -> None:
        """Reject blank study labels in model-first study API."""
        track = build_straight_track(length=250.0, sample_count=101)
        model = self._single_track_model()
        config = build_simulation_config(
            compute_backend="torch",
            torch_device="cpu",
            torch_compile=False,
            max_speed=55.0,
            initial_speed=0.0,
        )
        with self.assertRaises(ConfigurationError):
            run_lap_sensitivity_study(
                track=track,
                model=model,
                simulation_config=config,
                parameters=[SensitivityStudyParameter(name="mass", target="vehicle.mass")],
                label="   ",
            )

    def test_sensitivity_study_parameter_rejects_invalid_inputs(self) -> None:
        """Reject invalid study-parameter definitions."""
        with self.assertRaises(ConfigurationError):
            SensitivityStudyParameter(name="", target="vehicle.mass").validate()
        with self.assertRaises(ConfigurationError):
            SensitivityStudyParameter(name="mass", target="").validate()
        with self.assertRaises(ConfigurationError):
            SensitivityStudyParameter(name="mass", target="vehicle..mass").validate()
        with self.assertRaises(ConfigurationError):
            SensitivityStudyParameter(
                name="mass",
                target="vehicle.mass",
                kind="unsupported",
            ).validate()
        with self.assertRaises(ConfigurationError):
            SensitivityStudyParameter(
                name="mass",
                target="vehicle.mass",
                label=" ",
            ).validate()
        with self.assertRaises(ConfigurationError):
            SensitivityStudyParameter(
                name="mass",
                target="vehicle.mass",
                relative_variation=0.0,
            ).validate()
        with self.assertRaises(ConfigurationError):
            SensitivityStudyParameter(
                name="mass",
                target="vehicle.mass",
                lower_bound=np.nan,
            ).validate()
        with self.assertRaises(ConfigurationError):
            SensitivityStudyParameter(
                name="mass",
                target="vehicle.mass",
                lower_bound=2.0,
                upper_bound=1.0,
            ).validate()

    def test_study_result_validation_rejects_inconsistent_payloads(self) -> None:
        """Reject inconsistent objective and parameter mappings in result payloads."""
        parameter = SensitivityStudyParameter(name="mass", target="vehicle.mass")
        base_result = SensitivityResult(
            objective_value=10.0,
            sensitivities={"mass": 0.1},
            method="finite_difference",
            parameter_values={"mass": 800.0},
            parameter_kinds={"mass": "physical"},
        )
        with self.assertRaises(ConfigurationError):
            sensitivity_module.SensitivityStudyResult(
                study_label=None,
                objective_order=(),
                objective_units={},
                parameters=(parameter,),
                sensitivity_results={},
            )
        with self.assertRaises(ConfigurationError):
            sensitivity_module.SensitivityStudyResult(
                study_label=None,
                objective_order=("lap_time_s",),
                objective_units={},
                parameters=(parameter,),
                sensitivity_results={"lap_time_s": base_result},
            )
        with self.assertRaises(ConfigurationError):
            sensitivity_module.SensitivityStudyResult(
                study_label=None,
                objective_order=("lap_time_s",),
                objective_units={"lap_time_s": "s"},
                parameters=(parameter,),
                sensitivity_results={
                    "lap_time_s": SensitivityResult(
                        objective_value=10.0,
                        sensitivities={"drag": 0.1},
                        method="finite_difference",
                        parameter_values={"drag": 0.9},
                        parameter_kinds={"drag": "physical"},
                    )
                },
            )

    def test_private_dot_path_helpers_cover_error_and_object_branches(self) -> None:
        """Exercise dot-path helper branches used by the study API."""
        vehicle = sample_vehicle_parameters()
        root = {"vehicle": vehicle}

        with self.assertRaises(ConfigurationError):
            sensitivity_module._resolve_dot_path(root, target="vehicle.unknown")
        with self.assertRaises(ConfigurationError):
            sensitivity_module._set_dot_path(root, target="unknown.mass", value=1.0)
        with self.assertRaises(ConfigurationError):
            sensitivity_module._set_dot_path(root, target="vehicle.unknown", value=1.0)

        class MutableContainer:
            """Simple mutable object for helper-branch coverage."""

            def __init__(self) -> None:
                self.value = 1.0

        obj_root = {"obj": MutableContainer()}
        obj_updated = sensitivity_module._set_dot_path(
            obj_root,
            target="obj.value",
            value=2.0,
        )
        self.assertEqual(obj_root["obj"].value, 1.0)
        self.assertEqual(obj_updated["obj"].value, 2.0)
        with self.assertRaises(ConfigurationError):
            sensitivity_module._set_dot_path(obj_root, target="obj.missing", value=0.0)

    def test_private_study_helpers_reject_inconsistent_calls(self) -> None:
        """Exercise study-helper error paths that are otherwise hard to trigger."""
        parameter = SensitivityStudyParameter(name="mass", target="vehicle.mass")
        with self.assertRaises(ConfigurationError):
            sensitivity_module._resolve_lap_study_sensitivity_config(
                config=SensitivityConfig(),
                numerics=None,
                runtime=SensitivityRuntime(method="finite_difference"),
                default_torch_device="cpu",
            )
        with self.assertRaises(ConfigurationError):
            sensitivity_module._normalize_lap_sensitivity_objectives(())
        self.assertEqual(
            sensitivity_module._normalize_lap_sensitivity_objectives(
                ("lap_time_s", "lap_time_s", "energy_kwh")
            ),
            ["lap_time_s", "energy_kwh"],
        )
        with self.assertRaises(ConfigurationError):
            sensitivity_module._build_study_model_inputs(
                base_inputs={"vehicle": sample_vehicle_parameters()},
                parameters={"mass": parameter},
                parameter_values={},
            )

    def test_private_model_adapter_helpers_cover_branches(self) -> None:
        """Exercise adapter-helper branches for mapping extraction and MRO lookup."""
        original_adapters = dict(sensitivity_module._SENSITIVITY_MODEL_ADAPTERS)
        original_registered = sensitivity_module._DEFAULT_MODEL_ADAPTERS_REGISTERED

        class _BaseModel:
            def __init__(self) -> None:
                self.value = 1.0

        class _DerivedModel(_BaseModel):
            pass

        try:
            sensitivity_module._SENSITIVITY_MODEL_ADAPTERS.clear()
            sensitivity_module._DEFAULT_MODEL_ADAPTERS_REGISTERED = True
            register_sensitivity_model_adapter(
                model_type=_BaseModel,
                model_factory=lambda **kwargs: kwargs,
                model_inputs_getter=lambda model: {"value": model.value},
            )
            adapter = sensitivity_module._resolve_sensitivity_model_adapter(_DerivedModel())
            self.assertIsNotNone(adapter)
            self.assertEqual(
                sensitivity_module._extract_model_attributes(
                    model=_DerivedModel(),
                    required_fields=("value",),
                )["value"],
                1.0,
            )
            with self.assertRaises(ConfigurationError):
                sensitivity_module._extract_model_attributes(
                    model=_DerivedModel(),
                    required_fields=("missing",),
                )
        finally:
            sensitivity_module._SENSITIVITY_MODEL_ADAPTERS.clear()
            sensitivity_module._SENSITIVITY_MODEL_ADAPTERS.update(original_adapters)
            sensitivity_module._DEFAULT_MODEL_ADAPTERS_REGISTERED = original_registered

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_run_lap_sensitivity_study_rejects_autodiff_torch_device_mismatch(self) -> None:
        """Reject mismatch between simulation torch device and autodiff runtime device."""
        track = build_straight_track(length=250.0, sample_count=101)
        model = self._single_track_model()
        config = build_simulation_config(
            compute_backend="torch",
            torch_device="cpu",
            torch_compile=False,
            max_speed=55.0,
            initial_speed=0.0,
        )

        with self.assertRaises(ConfigurationError):
            run_lap_sensitivity_study(
                track=track,
                model=model,
                simulation_config=config,
                parameters=[SensitivityStudyParameter(name="mass", target="vehicle.mass")],
                runtime=SensitivityRuntime(method="autodiff", torch_device="cpu:0"),
            )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_private_energy_helper_rejects_missing_or_bad_power_shapes(self) -> None:
        """Reject invalid model power hooks and shape mismatches for energy objective."""
        import torch

        track = build_straight_track(length=10.0, sample_count=4)

        @dataclass(frozen=True)
        class _Profile:
            speed: Any
            longitudinal_accel: Any

        profile = _Profile(
            speed=torch.tensor([5.0, 5.0, 5.0], dtype=torch.float64),
            longitudinal_accel=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        )

        with self.assertRaises(ConfigurationError):
            sensitivity_module._compute_lap_energy_kwh_torch(
                track=track,
                model=object(),
                speed_profile=profile,
            )

        class _BadPowerModel:
            def tractive_power(self, speed: Any, longitudinal_accel: Any) -> Any:
                del speed, longitudinal_accel
                return torch.tensor([1.0], dtype=torch.float64)

        with self.assertRaises(ConfigurationError):
            sensitivity_module._compute_lap_energy_kwh_torch(
                track=track,
                model=_BadPowerModel(),
                speed_profile=profile,
            )

        class _SinglePointTrack:
            arc_length = np.array([0.0], dtype=float)

        class _ValidPowerModel:
            def tractive_power(self, speed: Any, longitudinal_accel: Any) -> Any:
                return speed + 0.0 * longitudinal_accel

        single_point_profile = _Profile(
            speed=torch.tensor([5.0], dtype=torch.float64),
            longitudinal_accel=torch.tensor([0.0], dtype=torch.float64),
        )
        energy = sensitivity_module._compute_lap_energy_kwh_torch(
            track=_SinglePointTrack(),
            model=_ValidPowerModel(),
            speed_profile=single_point_profile,
        )
        self.assertEqual(float(energy.item()), 0.0)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_run_lap_sensitivity_study_returns_expected_table_columns(self) -> None:
        """Build long-form and pivot outputs with expected schemas."""
        track = build_straight_track(length=300.0, sample_count=151)
        model = self._single_track_model()
        config = build_simulation_config(
            compute_backend="torch",
            torch_device="cpu",
            torch_compile=False,
            max_speed=65.0,
            initial_speed=0.0,
        )
        parameters = [
            SensitivityStudyParameter(
                name="mass",
                target="vehicle.mass",
                label="Vehicle mass",
            ),
            SensitivityStudyParameter(
                name="drag_coefficient",
                target="vehicle.drag_coefficient",
                label="Drag coefficient",
            ),
        ]
        study = run_lap_sensitivity_study(
            track=track,
            model=model,
            simulation_config=config,
            parameters=parameters,
            label="unit-test-study",
        )

        long_table = study.to_dataframe()
        expected_columns = {
            "study_label",
            "objective",
            "objective_unit",
            "objective_value",
            "parameter",
            "parameter_label",
            "parameter_target",
            "parameter_kind",
            "parameter_value",
            "sensitivity_raw",
            "sensitivity_pct_per_pct",
            "variation_minus_pct",
            "variation_plus_pct",
            "predicted_objective_minus",
            "predicted_objective_plus",
            "method",
        }
        self.assertEqual(set(long_table.columns), expected_columns)
        self.assertEqual(
            set(long_table["objective"].unique().tolist()),
            {"lap_time_s", "energy_kwh"},
        )

        pivot = study.to_pivot()
        self.assertEqual(set(pivot.columns.tolist()), {"lap_time_s", "energy_kwh"})
        self.assertEqual(set(pivot.index.tolist()), {"mass", "drag_coefficient"})

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_run_lap_sensitivity_study_rejects_invalid_target(self) -> None:
        """Reject unresolved parameter target paths in study setup."""
        track = build_straight_track(length=250.0, sample_count=101)
        model = self._single_track_model()
        config = build_simulation_config(
            compute_backend="torch",
            torch_device="cpu",
            torch_compile=False,
            max_speed=55.0,
            initial_speed=0.0,
        )

        with self.assertRaises(ConfigurationError):
            run_lap_sensitivity_study(
                track=track,
                model=model,
                simulation_config=config,
                parameters=[
                    SensitivityStudyParameter(
                        name="invalid",
                        target="vehicle.not_a_field",
                    )
                ],
            )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_run_lap_sensitivity_study_rejects_unregistered_model_type(self) -> None:
        """Reject sensitivity studies for model types without registered adapters."""
        track = build_straight_track(length=250.0, sample_count=101)
        config = build_simulation_config(
            compute_backend="torch",
            torch_device="cpu",
            torch_compile=False,
            max_speed=55.0,
            initial_speed=0.0,
        )

        class _UnsupportedModel:
            """Model without registered sensitivity adapter."""

        with self.assertRaisesRegex(ConfigurationError, "register_sensitivity_model_adapter"):
            run_lap_sensitivity_study(
                track=track,
                model=_UnsupportedModel(),
                simulation_config=config,
                parameters=[SensitivityStudyParameter(name="mass", target="vehicle.mass")],
            )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_run_lap_sensitivity_study_rejects_invalid_adapter_payload(self) -> None:
        """Reject adapters whose input getters do not return mappings."""
        track = build_straight_track(length=250.0, sample_count=101)
        config = build_simulation_config(
            compute_backend="torch",
            torch_device="cpu",
            torch_compile=False,
            max_speed=55.0,
            initial_speed=0.0,
        )
        original_adapters = dict(sensitivity_module._SENSITIVITY_MODEL_ADAPTERS)
        original_registered = sensitivity_module._DEFAULT_MODEL_ADAPTERS_REGISTERED

        class _DummyModel:
            """Dummy model type for adapter payload validation tests."""

        try:
            register_sensitivity_model_adapter(
                model_type=_DummyModel,
                model_factory=lambda **_: object(),
                model_inputs_getter=lambda _: cast(Any, []),
            )
            with self.assertRaisesRegex(ConfigurationError, "must return a mapping"):
                run_lap_sensitivity_study(
                    track=track,
                    model=_DummyModel(),
                    simulation_config=config,
                    parameters=[SensitivityStudyParameter(name="mass", target="vehicle.mass")],
                )
        finally:
            sensitivity_module._SENSITIVITY_MODEL_ADAPTERS.clear()
            sensitivity_module._SENSITIVITY_MODEL_ADAPTERS.update(original_adapters)
            sensitivity_module._DEFAULT_MODEL_ADAPTERS_REGISTERED = original_registered

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_run_lap_sensitivity_study_rejects_empty_adapter_payload(self) -> None:
        """Reject adapters whose input getters return empty mappings."""
        track = build_straight_track(length=250.0, sample_count=101)
        config = build_simulation_config(
            compute_backend="torch",
            torch_device="cpu",
            torch_compile=False,
            max_speed=55.0,
            initial_speed=0.0,
        )
        original_adapters = dict(sensitivity_module._SENSITIVITY_MODEL_ADAPTERS)
        original_registered = sensitivity_module._DEFAULT_MODEL_ADAPTERS_REGISTERED

        class _DummyModel:
            """Dummy model type for adapter payload validation tests."""

        try:
            register_sensitivity_model_adapter(
                model_type=_DummyModel,
                model_factory=lambda **_: object(),
                model_inputs_getter=lambda _: {},
            )
            with self.assertRaisesRegex(ConfigurationError, "non-empty mapping"):
                run_lap_sensitivity_study(
                    track=track,
                    model=_DummyModel(),
                    simulation_config=config,
                    parameters=[SensitivityStudyParameter(name="mass", target="vehicle.mass")],
                )
        finally:
            sensitivity_module._SENSITIVITY_MODEL_ADAPTERS.clear()
            sensitivity_module._SENSITIVITY_MODEL_ADAPTERS.update(original_adapters)
            sensitivity_module._DEFAULT_MODEL_ADAPTERS_REGISTERED = original_registered

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_run_lap_sensitivity_study_wraps_model_factory_type_error(self) -> None:
        """Wrap model-factory signature errors with a study-level configuration error."""
        track = build_straight_track(length=250.0, sample_count=101)
        config = build_simulation_config(
            compute_backend="torch",
            torch_device="cpu",
            torch_compile=False,
            max_speed=55.0,
            initial_speed=0.0,
        )
        original_adapters = dict(sensitivity_module._SENSITIVITY_MODEL_ADAPTERS)
        original_registered = sensitivity_module._DEFAULT_MODEL_ADAPTERS_REGISTERED

        class _DummyModel:
            """Dummy model type with custom adapter."""

        try:
            register_sensitivity_model_adapter(
                model_type=_DummyModel,
                model_factory=lambda: object(),
                model_inputs_getter=lambda _: {"value": 1.0},
            )
            with self.assertRaisesRegex(ConfigurationError, "failed to build model"):
                run_lap_sensitivity_study(
                    track=track,
                    model=_DummyModel(),
                    simulation_config=config,
                    parameters=[SensitivityStudyParameter(name="value", target="value")],
                )
        finally:
            sensitivity_module._SENSITIVITY_MODEL_ADAPTERS.clear()
            sensitivity_module._SENSITIVITY_MODEL_ADAPTERS.update(original_adapters)
            sensitivity_module._DEFAULT_MODEL_ADAPTERS_REGISTERED = original_registered

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_run_lap_sensitivity_study_rejects_non_torch_backend(self) -> None:
        """Require torch simulation backend for AD-first lap study API."""
        track = build_straight_track(length=250.0, sample_count=101)
        model = self._single_track_model()
        numpy_config = build_simulation_config(
            compute_backend="numpy",
            max_speed=55.0,
            initial_speed=0.0,
        )

        with self.assertRaises(ConfigurationError):
            run_lap_sensitivity_study(
                track=track,
                model=model,
                simulation_config=numpy_config,
                parameters=[SensitivityStudyParameter(name="mass", target="vehicle.mass")],
            )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_run_lap_sensitivity_study_rejects_unknown_objective(self) -> None:
        """Reject objective identifiers outside the supported lap-study set."""
        track = build_straight_track(length=250.0, sample_count=101)
        model = self._single_track_model()
        config = build_simulation_config(
            compute_backend="torch",
            torch_device="cpu",
            torch_compile=False,
            max_speed=55.0,
            initial_speed=0.0,
        )

        with self.assertRaises(ConfigurationError):
            run_lap_sensitivity_study(
                track=track,
                model=model,
                simulation_config=config,
                parameters=[SensitivityStudyParameter(name="mass", target="vehicle.mass")],
                objectives=("lap_time_s", "unknown"),
            )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_run_lap_sensitivity_study_rejects_transient_optimal_control_autodiff(self) -> None:
        """Reject AD sensitivity path for transient optimal-control solver mode."""
        track = build_straight_track(length=180.0, sample_count=41)
        model = self._single_track_model()
        with _patch_transient_dependency_specs():
            config = build_simulation_config(
                compute_backend="torch",
                torch_device="cpu",
                torch_compile=False,
                max_speed=60.0,
                initial_speed=8.0,
                solver_mode="transient_oc",
                transient=TransientConfig(
                    runtime=TransientRuntimeConfig(driver_model="optimal_control"),
                ),
            )

        with self.assertRaises(ConfigurationError):
            run_lap_sensitivity_study(
                track=track,
                model=model,
                simulation_config=config,
                parameters=[SensitivityStudyParameter(name="mass", target="vehicle.mass")],
            )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_run_lap_sensitivity_study_matches_fd_for_lap_time(self) -> None:
        """Keep FD and AD derivatives consistent for a real lap-time study objective."""
        track = build_straight_track(length=350.0, sample_count=181)
        model = self._single_track_model()
        config = build_simulation_config(
            compute_backend="torch",
            torch_device="cpu",
            torch_compile=False,
            max_speed=70.0,
            initial_speed=0.0,
        )
        parameters = [
            SensitivityStudyParameter(name="mass", target="vehicle.mass"),
            SensitivityStudyParameter(name="drag_coefficient", target="vehicle.drag_coefficient"),
        ]

        autodiff = run_lap_sensitivity_study(
            track=track,
            model=model,
            simulation_config=config,
            parameters=parameters,
            objectives=("lap_time_s",),
        )
        finite_difference = run_lap_sensitivity_study(
            track=track,
            model=model,
            simulation_config=config,
            parameters=parameters,
            objectives=("lap_time_s",),
            runtime=SensitivityRuntime(method="finite_difference"),
            numerics=SensitivityNumerics(
                finite_difference_scheme="central",
                finite_difference_relative_step=1e-4,
                finite_difference_absolute_step=1e-6,
            ),
        )

        ad_result = autodiff.sensitivity_results["lap_time_s"]
        fd_result = finite_difference.sensitivity_results["lap_time_s"]
        self.assertEqual(ad_result.method, "autodiff")
        self.assertEqual(fd_result.method, "finite_difference")
        self.assertAlmostEqual(ad_result.objective_value, fd_result.objective_value, places=9)
        self.assertAlmostEqual(
            ad_result.sensitivities["mass"],
            fd_result.sensitivities["mass"],
            delta=1e-4,
        )
        self.assertAlmostEqual(
            ad_result.sensitivities["drag_coefficient"],
            fd_result.sensitivities["drag_coefficient"],
            delta=1e-4,
        )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_run_lap_sensitivity_study_energy_objective_is_autodiff(self) -> None:
        """Compute energy sensitivities through the AD-first lap study interface."""
        track = build_straight_track(length=300.0, sample_count=151)
        model = self._single_track_model()
        config = build_simulation_config(
            compute_backend="torch",
            torch_device="cpu",
            torch_compile=False,
            max_speed=65.0,
            initial_speed=0.0,
        )

        study = run_lap_sensitivity_study(
            track=track,
            model=model,
            simulation_config=config,
            parameters=[
                SensitivityStudyParameter(name="mass", target="vehicle.mass"),
                SensitivityStudyParameter(
                    name="drag_coefficient",
                    target="vehicle.drag_coefficient",
                ),
            ],
            objectives=("energy_kwh",),
        )

        result = study.sensitivity_results["energy_kwh"]
        self.assertEqual(result.method, "autodiff")
        self.assertGreater(result.objective_value, 0.0)
        self.assertTrue(np.isfinite(result.sensitivities["mass"]))
        self.assertTrue(np.isfinite(result.sensitivities["drag_coefficient"]))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_run_lap_sensitivity_study_supports_point_mass_model(self) -> None:
        """Run the high-level study API with a point-mass model and both objectives."""
        track = build_straight_track(length=320.0, sample_count=161)
        model = self._point_mass_model()
        config = build_simulation_config(
            compute_backend="torch",
            torch_device="cpu",
            torch_compile=False,
            max_speed=70.0,
            initial_speed=0.0,
        )

        study = run_lap_sensitivity_study(
            track=track,
            model=model,
            simulation_config=config,
            parameters=[
                SensitivityStudyParameter(name="mass", target="vehicle.mass"),
                SensitivityStudyParameter(
                    name="drag_coefficient",
                    target="vehicle.drag_coefficient",
                ),
                SensitivityStudyParameter(
                    name="friction_coefficient",
                    target="physics.friction_coefficient",
                ),
            ],
            label="point-mass-study",
        )

        self.assertEqual(set(study.sensitivity_results.keys()), {"lap_time_s", "energy_kwh"})
        self.assertEqual(study.sensitivity_results["lap_time_s"].method, "autodiff")
        self.assertEqual(study.sensitivity_results["energy_kwh"].method, "autodiff")
        long_table = study.to_dataframe()
        self.assertEqual(
            set(long_table["objective"].unique().tolist()),
            {"lap_time_s", "energy_kwh"},
        )
        self.assertEqual(
            set(long_table["parameter"].unique().tolist()),
            {"mass", "drag_coefficient", "friction_coefficient"},
        )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_run_lap_sensitivity_study_point_mass_ad_matches_fd(self) -> None:
        """Match point-mass AD and FD sensitivities for a real lap-time study objective."""
        track = build_straight_track(length=280.0, sample_count=141)
        model = self._point_mass_model()
        config = build_simulation_config(
            compute_backend="torch",
            torch_device="cpu",
            torch_compile=False,
            max_speed=62.0,
            initial_speed=0.0,
        )
        parameters = [
            SensitivityStudyParameter(name="mass", target="vehicle.mass"),
            SensitivityStudyParameter(name="drag_coefficient", target="vehicle.drag_coefficient"),
        ]

        autodiff = run_lap_sensitivity_study(
            track=track,
            model=model,
            simulation_config=config,
            parameters=parameters,
            objectives=("lap_time_s",),
        ).sensitivity_results["lap_time_s"]

        finite_difference = run_lap_sensitivity_study(
            track=track,
            model=model,
            simulation_config=config,
            parameters=parameters,
            objectives=("lap_time_s",),
            runtime=SensitivityRuntime(method="finite_difference"),
            numerics=SensitivityNumerics(
                finite_difference_scheme="central",
                finite_difference_relative_step=1e-4,
                finite_difference_absolute_step=1e-6,
            ),
        ).sensitivity_results["lap_time_s"]

        self.assertAlmostEqual(
            autodiff.objective_value,
            finite_difference.objective_value,
            places=9,
        )
        self.assertAlmostEqual(
            autodiff.sensitivities["mass"],
            finite_difference.sensitivities["mass"],
            delta=1e-4,
        )
        self.assertAlmostEqual(
            autodiff.sensitivities["drag_coefficient"],
            finite_difference.sensitivities["drag_coefficient"],
            delta=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
