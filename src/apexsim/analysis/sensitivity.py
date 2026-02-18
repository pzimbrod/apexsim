"""Sensitivity-analysis APIs for scalar simulation objectives."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field, is_dataclass, replace
from typing import Any, Protocol, cast

import numpy as np

from apexsim.simulation.config import DEFAULT_TORCH_DEVICE, SimulationConfig
from apexsim.track.models import TrackData
from apexsim.utils.constants import SMALL_EPS
from apexsim.utils.exceptions import ConfigurationError

DEFAULT_SENSITIVITY_METHOD = "autodiff"
DEFAULT_FD_SCHEME = "central"
DEFAULT_FD_RELATIVE_STEP = 1e-3
DEFAULT_FD_ABSOLUTE_STEP = 1e-6
DEFAULT_AUTODIFF_FALLBACK_TO_FD = True
DEFAULT_SENSITIVITY_STUDY_RELATIVE_VARIATION = 0.10
VALID_SENSITIVITY_METHODS = ("autodiff", "finite_difference")
VALID_FD_SCHEMES = ("central", "forward")
VALID_PARAMETER_KINDS = ("physical", "numerical")
DEFAULT_LAP_SENSITIVITY_OBJECTIVES = ("lap_time_s", "energy_kwh")
VALID_LAP_SENSITIVITY_OBJECTIVES = DEFAULT_LAP_SENSITIVITY_OBJECTIVES
LAP_SENSITIVITY_OBJECTIVE_UNITS = {
    "lap_time_s": "s",
    "energy_kwh": "kWh",
}


class SensitivityObjective(Protocol):
    """Protocol for scalar objective callables used in sensitivity analysis."""

    def __call__(self, parameters: Mapping[str, Any]) -> Any:
        """Evaluate objective at provided parameter values.

        Args:
            parameters: Parameter-name mapping. Values are ``float`` for finite
                differences and ``torch.Tensor`` for autodiff.

        Returns:
            Scalar objective value.
        """


SensitivityModelFactory = Callable[..., Any]


class SensitivityModelInputsGetter(Protocol):
    """Protocol for extracting reconstructable model inputs from an instance."""

    def __call__(self, model: Any) -> Mapping[str, Any]:
        """Extract baseline model inputs from a model instance.

        Args:
            model: Vehicle model instance.

        Returns:
            Mapping consumed by a registered model factory.
        """


@dataclass(frozen=True)
class _SensitivityModelAdapter:
    """Registered model adapter for model reconstruction."""

    model_factory: SensitivityModelFactory
    model_inputs_getter: SensitivityModelInputsGetter


@dataclass(frozen=True)
class _ResolvedStudyModelSpec:
    """Internal resolved model specification used by lap sensitivity studies.

    Args:
        model_factory: Callable that builds a solver-compatible vehicle model.
        model_inputs: Baseline keyword arguments passed to ``model_factory``.
        label: Optional study label included in tabular outputs.
    """

    model_factory: SensitivityModelFactory
    model_inputs: Mapping[str, Any]
    label: str | None = None

    def validate(self) -> None:
        """Validate resolved model-spec construction settings.

        Raises:
            apexsim.utils.exceptions.ConfigurationError: If model factory or
                model input map is malformed.
        """
        if not callable(self.model_factory):
            msg = "model_factory must be callable"
            raise ConfigurationError(msg)
        if not isinstance(self.model_inputs, Mapping):
            msg = "model_inputs must be a mapping"
            raise ConfigurationError(msg)
        if not self.model_inputs:
            msg = "model_inputs must not be empty"
            raise ConfigurationError(msg)
        if self.label is not None and not self.label.strip():
            msg = "label must be a non-empty string when provided"
            raise ConfigurationError(msg)


_SENSITIVITY_MODEL_ADAPTERS: dict[type[Any], _SensitivityModelAdapter] = {}
_DEFAULT_MODEL_ADAPTERS_REGISTERED = False


@dataclass(frozen=True)
class SensitivityStudyParameter:
    """Parameter specification for high-level lap sensitivity studies.

    Args:
        name: Parameter identifier used in exported sensitivity tables.
        target: Dot-path in the reconstructed model input mapping to the scalar
            value to be perturbed (for example ``vehicle.mass``).
        label: Optional human-readable label for plots/tables.
        kind: Conceptual parameter category (``physical`` or ``numerical``).
        relative_variation: Relative variation used for local +/- prediction
            columns in tabular outputs.
        lower_bound: Optional lower bound applied to local perturbations.
        upper_bound: Optional upper bound applied to local perturbations.
    """

    name: str
    target: str
    label: str | None = None
    kind: str = "physical"
    relative_variation: float = DEFAULT_SENSITIVITY_STUDY_RELATIVE_VARIATION
    lower_bound: float | None = None
    upper_bound: float | None = None

    def validate(self) -> None:
        """Validate study-parameter definition.

        Raises:
            apexsim.utils.exceptions.ConfigurationError: If target format,
                parameter category, bounds, or relative variation are invalid.
        """
        if not self.name:
            msg = "study parameter name must be a non-empty string"
            raise ConfigurationError(msg)
        if not self.target:
            msg = f"study parameter {self.name!r} target must be non-empty"
            raise ConfigurationError(msg)
        if any(not part for part in self.target.split(".")):
            msg = (
                f"study parameter {self.name!r} target {self.target!r} "
                "contains an empty path segment"
            )
            raise ConfigurationError(msg)
        if self.kind not in VALID_PARAMETER_KINDS:
            msg = (
                "study parameter kind must be one of "
                f"{VALID_PARAMETER_KINDS}, got: {self.kind!r}"
            )
            raise ConfigurationError(msg)
        if self.label is not None and not self.label.strip():
            msg = f"study parameter {self.name!r} label must be non-empty when provided"
            raise ConfigurationError(msg)
        if not np.isfinite(self.relative_variation) or self.relative_variation <= 0.0:
            msg = (
                f"study parameter {self.name!r} relative_variation must be a positive, "
                f"finite value, got: {self.relative_variation}"
            )
            raise ConfigurationError(msg)
        if self.lower_bound is not None and not np.isfinite(self.lower_bound):
            msg = f"study parameter {self.name!r} lower_bound must be finite when provided"
            raise ConfigurationError(msg)
        if self.upper_bound is not None and not np.isfinite(self.upper_bound):
            msg = f"study parameter {self.name!r} upper_bound must be finite when provided"
            raise ConfigurationError(msg)
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound >= self.upper_bound
        ):
            msg = (
                f"study parameter {self.name!r} must satisfy lower_bound < upper_bound, "
                f"got {self.lower_bound} >= {self.upper_bound}"
            )
            raise ConfigurationError(msg)


@dataclass(frozen=True)
class SensitivityStudyResult:
    """Aggregated lap sensitivity outputs for multiple objectives.

    Args:
        study_label: Optional study label propagated into tabular exports.
        objective_order: Objective identifier order used for outputs.
        objective_units: Objective-unit mapping.
        parameters: Ordered study parameters used in the run.
        sensitivity_results: Per-objective scalar sensitivity outputs.
    """

    study_label: str | None
    objective_order: tuple[str, ...]
    objective_units: dict[str, str]
    parameters: tuple[SensitivityStudyParameter, ...]
    sensitivity_results: dict[str, SensitivityResult]

    def __post_init__(self) -> None:
        """Validate structural consistency of study outputs.

        Raises:
            apexsim.utils.exceptions.ConfigurationError: If objective or
                parameter mappings are inconsistent.
        """
        if not self.objective_order:
            msg = "objective_order must not be empty"
            raise ConfigurationError(msg)
        objective_keys = set(self.objective_order)
        if objective_keys != set(self.objective_units.keys()):
            msg = (
                "objective_order and objective_units keys must match, got "
                f"{objective_keys} vs {set(self.objective_units.keys())}"
            )
            raise ConfigurationError(msg)
        if objective_keys != set(self.sensitivity_results.keys()):
            msg = (
                "objective_order and sensitivity_results keys must match, got "
                f"{objective_keys} vs {set(self.sensitivity_results.keys())}"
            )
            raise ConfigurationError(msg)

        parameter_names = tuple(parameter.name for parameter in self.parameters)
        if len(set(parameter_names)) != len(parameter_names):
            msg = "parameters contain duplicate names"
            raise ConfigurationError(msg)
        expected_parameter_names = set(parameter_names)
        for objective, result in self.sensitivity_results.items():
            result_names = set(result.sensitivities.keys())
            if result_names != expected_parameter_names:
                msg = (
                    f"sensitivity result for objective {objective!r} has parameter keys "
                    f"{result_names}, expected {expected_parameter_names}"
                )
                raise ConfigurationError(msg)

    def to_dataframe(self) -> Any:
        """Return long-form tabular sensitivity output.

        Returns:
            Pandas DataFrame containing one row per
            ``(objective, parameter)`` pair.

        Raises:
            apexsim.utils.exceptions.ConfigurationError: If pandas is not
                installed in the active environment.
        """
        try:
            import pandas as pd  # type: ignore[import-untyped]
        except ModuleNotFoundError as exc:
            msg = (
                "SensitivityStudyResult.to_dataframe requires pandas. "
                "Install with `pip install pandas`."
            )
            raise ConfigurationError(msg) from exc

        parameter_map = {parameter.name: parameter for parameter in self.parameters}
        rows: list[dict[str, Any]] = []
        study_label = self.study_label or ""

        for objective in self.objective_order:
            objective_result = self.sensitivity_results[objective]
            objective_value = float(objective_result.objective_value)
            objective_unit = self.objective_units[objective]
            for parameter_name in objective_result.sensitivities:
                parameter = parameter_map[parameter_name]
                baseline_value = float(objective_result.parameter_values[parameter_name])
                sensitivity_raw = float(objective_result.sensitivities[parameter_name])
                delta = baseline_value * float(parameter.relative_variation)
                predicted_minus = objective_value - sensitivity_raw * delta
                predicted_plus = objective_value + sensitivity_raw * delta
                if abs(objective_value) > SMALL_EPS:
                    sensitivity_pct_per_pct = (
                        sensitivity_raw * baseline_value / objective_value
                    )
                else:
                    sensitivity_pct_per_pct = np.nan

                rows.append(
                    {
                        "study_label": study_label,
                        "objective": objective,
                        "objective_unit": objective_unit,
                        "objective_value": objective_value,
                        "parameter": parameter_name,
                        "parameter_label": parameter.label or parameter.name,
                        "parameter_target": parameter.target,
                        "parameter_kind": parameter.kind,
                        "parameter_value": baseline_value,
                        "sensitivity_raw": sensitivity_raw,
                        "sensitivity_pct_per_pct": float(sensitivity_pct_per_pct),
                        "variation_minus_pct": -100.0 * float(parameter.relative_variation),
                        "variation_plus_pct": 100.0 * float(parameter.relative_variation),
                        "predicted_objective_minus": predicted_minus,
                        "predicted_objective_plus": predicted_plus,
                        "method": objective_result.method,
                    }
                )

        return pd.DataFrame(rows)

    def to_pivot(self, value_column: str = "sensitivity_pct_per_pct") -> Any:
        """Return compact ``parameter x objective`` sensitivity table.

        Args:
            value_column: DataFrame value column used for the pivot table.

        Returns:
            Pandas DataFrame with parameters as rows and objectives as columns.

        Raises:
            apexsim.utils.exceptions.ConfigurationError: If ``value_column`` is
                not present in the long-form table.
        """
        table = self.to_dataframe()
        if value_column not in table.columns:
            msg = (
                f"value_column {value_column!r} is not available in study output. "
                f"Available columns: {list(table.columns)}"
            )
            raise ConfigurationError(msg)
        pivot = table.pivot(index="parameter", columns="objective", values=value_column)
        return pivot.reindex(columns=list(self.objective_order))


@dataclass(frozen=True)
class SensitivityParameter:
    """Single scalar parameter definition for sensitivity analysis.

    Args:
        name: Parameter identifier consumed by the objective callable.
        value: Baseline scalar parameter value.
        kind: Conceptual parameter category. ``physical`` is intended for
            measurable real-world quantities; ``numerical`` for solver controls.
        lower_bound: Optional lower bound for perturbation validity.
        upper_bound: Optional upper bound for perturbation validity.
    """

    name: str
    value: float
    kind: str = "physical"
    lower_bound: float | None = None
    upper_bound: float | None = None

    def validate(self) -> None:
        """Validate parameter definition.

        Raises:
            apexsim.utils.exceptions.ConfigurationError: If name, value, kind,
                or bounds violate required constraints.
        """
        if not self.name:
            msg = "parameter name must be a non-empty string"
            raise ConfigurationError(msg)
        if self.kind not in VALID_PARAMETER_KINDS:
            msg = (
                "parameter kind must be one of "
                f"{VALID_PARAMETER_KINDS}, got: {self.kind!r}"
            )
            raise ConfigurationError(msg)
        if not np.isfinite(self.value):
            msg = f"parameter {self.name!r} must have a finite value"
            raise ConfigurationError(msg)

        if self.lower_bound is not None and not np.isfinite(self.lower_bound):
            msg = f"parameter {self.name!r} lower_bound must be finite when provided"
            raise ConfigurationError(msg)
        if self.upper_bound is not None and not np.isfinite(self.upper_bound):
            msg = f"parameter {self.name!r} upper_bound must be finite when provided"
            raise ConfigurationError(msg)

        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound >= self.upper_bound
        ):
            msg = (
                f"parameter {self.name!r} must satisfy lower_bound < upper_bound, "
                f"got {self.lower_bound} >= {self.upper_bound}"
            )
            raise ConfigurationError(msg)

        if self.lower_bound is not None and self.value < self.lower_bound:
            msg = (
                f"parameter {self.name!r} value {self.value} is below lower_bound "
                f"{self.lower_bound}"
            )
            raise ConfigurationError(msg)
        if self.upper_bound is not None and self.value > self.upper_bound:
            msg = (
                f"parameter {self.name!r} value {self.value} is above upper_bound "
                f"{self.upper_bound}"
            )
            raise ConfigurationError(msg)


@dataclass(frozen=True)
class SensitivityNumerics:
    """Numerical controls for sensitivity estimation.

    Args:
        finite_difference_scheme: Finite-difference scheme identifier
            (``central`` or ``forward``).
        finite_difference_relative_step: Relative finite-difference step scale.
        finite_difference_absolute_step: Absolute finite-difference step floor.
    """

    finite_difference_scheme: str = DEFAULT_FD_SCHEME
    finite_difference_relative_step: float = DEFAULT_FD_RELATIVE_STEP
    finite_difference_absolute_step: float = DEFAULT_FD_ABSOLUTE_STEP

    def validate(self) -> None:
        """Validate sensitivity numerical controls.

        Raises:
            apexsim.utils.exceptions.ConfigurationError: If scheme or step
                controls violate required bounds.
        """
        if self.finite_difference_scheme not in VALID_FD_SCHEMES:
            msg = (
                "finite_difference_scheme must be one of "
                f"{VALID_FD_SCHEMES}, got: {self.finite_difference_scheme!r}"
            )
            raise ConfigurationError(msg)
        if self.finite_difference_relative_step <= 0.0:
            msg = "finite_difference_relative_step must be positive"
            raise ConfigurationError(msg)
        if self.finite_difference_absolute_step <= 0.0:
            msg = "finite_difference_absolute_step must be positive"
            raise ConfigurationError(msg)

    def step_size(self, value: float) -> float:
        """Return finite-difference perturbation size for a scalar value.

        Args:
            value: Baseline parameter value.

        Returns:
            Positive perturbation size.
        """
        scale = abs(float(value))
        step = max(
            scale * self.finite_difference_relative_step,
            self.finite_difference_absolute_step,
        )
        return max(step, SMALL_EPS)


@dataclass(frozen=True)
class SensitivityRuntime:
    """Runtime controls for selecting the sensitivity backend method.

    Args:
        method: Sensitivity backend (``autodiff`` or ``finite_difference``).
        torch_device: Torch device used for autodiff parameter tensors.
        autodiff_fallback_to_finite_difference: Whether autodiff backend errors
            should fall back to finite-difference evaluation.
    """

    method: str = DEFAULT_SENSITIVITY_METHOD
    torch_device: str = DEFAULT_TORCH_DEVICE
    autodiff_fallback_to_finite_difference: bool = DEFAULT_AUTODIFF_FALLBACK_TO_FD

    def validate(self) -> None:
        """Validate sensitivity runtime controls.

        Raises:
            apexsim.utils.exceptions.ConfigurationError: If method/device
                selection is invalid or unavailable.
        """
        if self.method not in VALID_SENSITIVITY_METHODS:
            msg = (
                "method must be one of "
                f"{VALID_SENSITIVITY_METHODS}, got: {self.method!r}"
            )
            raise ConfigurationError(msg)
        if not self.torch_device:
            msg = "torch_device must be a non-empty string"
            raise ConfigurationError(msg)
        if not isinstance(self.autodiff_fallback_to_finite_difference, bool):
            msg = "autodiff_fallback_to_finite_difference must be a boolean"
            raise ConfigurationError(msg)

        if self.method != "autodiff" and self.torch_device != DEFAULT_TORCH_DEVICE:
            msg = (
                "torch_device is only meaningful for method='autodiff'. "
                "Use torch_device='cpu' for finite-difference mode."
            )
            raise ConfigurationError(msg)

        if self.method == "autodiff":
            torch = _require_torch()
            if self.torch_device.startswith("cuda") and not torch.cuda.is_available():
                msg = (
                    "torch_device requests CUDA but no CUDA device is available: "
                    f"{self.torch_device!r}"
                )
                raise ConfigurationError(msg)


@dataclass(frozen=True)
class SensitivityConfig:
    """Top-level config for scalar sensitivity computation.

    Args:
        numerics: Numerical controls for finite-difference perturbations.
        runtime: Runtime controls for backend-method selection.
    """

    numerics: SensitivityNumerics = field(default_factory=SensitivityNumerics)
    runtime: SensitivityRuntime = field(default_factory=SensitivityRuntime)

    def validate(self) -> None:
        """Validate combined sensitivity settings.

        Raises:
            apexsim.utils.exceptions.ConfigurationError: If runtime or
                numerical controls are inconsistent.
        """
        self.numerics.validate()
        self.runtime.validate()


@dataclass(frozen=True)
class SensitivityResult:
    """Sensitivity result for a scalar objective around a baseline point.

    Args:
        objective_value: Baseline objective value at provided parameters.
        sensitivities: Mapping from parameter name to local scalar sensitivity.
        method: Backend method used for the returned sensitivities.
        parameter_values: Baseline scalar values used for each parameter.
        parameter_kinds: Parameter category map (``physical``/``numerical``).
    """

    objective_value: float
    sensitivities: dict[str, float]
    method: str
    parameter_values: dict[str, float]
    parameter_kinds: dict[str, str]

    def __post_init__(self) -> None:
        """Validate result payload consistency.

        Raises:
            apexsim.utils.exceptions.ConfigurationError: If keys are mismatched
                or unsupported method/category identifiers are provided.
        """
        if self.method not in VALID_SENSITIVITY_METHODS:
            msg = f"method must be one of {VALID_SENSITIVITY_METHODS}, got {self.method!r}"
            raise ConfigurationError(msg)

        keys = set(self.sensitivities.keys())
        if keys != set(self.parameter_values.keys()):
            msg = "sensitivities and parameter_values must contain identical parameter keys"
            raise ConfigurationError(msg)
        if keys != set(self.parameter_kinds.keys()):
            msg = "sensitivities and parameter_kinds must contain identical parameter keys"
            raise ConfigurationError(msg)

        for name, kind in self.parameter_kinds.items():
            if kind not in VALID_PARAMETER_KINDS:
                msg = (
                    f"parameter {name!r} has unsupported kind {kind!r}; "
                    f"expected one of {VALID_PARAMETER_KINDS}"
                )
                raise ConfigurationError(msg)


def build_sensitivity_config(
    numerics: SensitivityNumerics | None = None,
    runtime: SensitivityRuntime | None = None,
) -> SensitivityConfig:
    """Build a validated sensitivity configuration.

    Args:
        numerics: Optional numerical controls.
        runtime: Optional runtime controls.

    Returns:
        Validated sensitivity config.
    """
    config = SensitivityConfig(
        numerics=numerics or SensitivityNumerics(),
        runtime=runtime or SensitivityRuntime(),
    )
    config.validate()
    return config


def compute_sensitivities(
    objective: SensitivityObjective,
    parameters: Sequence[SensitivityParameter] | Mapping[str, float],
    config: SensitivityConfig | None = None,
    *,
    numerics: SensitivityNumerics | None = None,
    runtime: SensitivityRuntime | None = None,
) -> SensitivityResult:
    """Compute scalar-objective sensitivities for selected input parameters.

    Either provide a full ``config`` object, or provide ``numerics`` and/or
    ``runtime`` components directly.

    Args:
        objective: Objective function returning a scalar KPI value.
        parameters: Parameter definitions or a ``name -> value`` mapping.
        config: Optional validated sensitivity config.
        numerics: Optional numerical controls (used only if ``config`` is not
            provided).
        runtime: Optional runtime controls (used only if ``config`` is not
            provided).

    Returns:
        Baseline scalar objective and local sensitivities per parameter.

    Raises:
        apexsim.utils.exceptions.ConfigurationError: If configuration or
            parameter definitions are invalid, or if objective outputs violate
            scalar/autodiff requirements.
    """
    if config is not None and any(value is not None for value in (numerics, runtime)):
        msg = "Provide either `config` or (`numerics`, `runtime`) components, not both."
        raise ConfigurationError(msg)

    resolved_config = config or build_sensitivity_config(numerics=numerics, runtime=runtime)
    resolved_config.validate()
    normalized = _normalize_parameters(parameters)

    if resolved_config.runtime.method == "autodiff":
        try:
            return _compute_sensitivities_autodiff(
                objective=objective,
                parameters=normalized,
                runtime=resolved_config.runtime,
            )
        except ConfigurationError:
            if not resolved_config.runtime.autodiff_fallback_to_finite_difference:
                raise

    return _compute_sensitivities_finite_difference(
        objective=objective,
        parameters=normalized,
        numerics=resolved_config.numerics,
    )


def register_sensitivity_model_adapter(
    *,
    model_type: type[Any],
    model_factory: SensitivityModelFactory,
    model_inputs_getter: SensitivityModelInputsGetter,
) -> None:
    """Register model reconstruction adapter for lap sensitivity studies.

    Registered adapters enable :func:`run_lap_sensitivity_study` to rebuild a
    model for each parameter perturbation without mutating the passed model.

    Args:
        model_type: Model class handled by the adapter.
        model_factory: Callable rebuilding model instances from keyword inputs.
        model_inputs_getter: Callable extracting reconstructable baseline model
            inputs from an existing model instance.

    Raises:
        apexsim.utils.exceptions.ConfigurationError: If adapter inputs are
            malformed.
    """
    if not isinstance(model_type, type):
        msg = "model_type must be a class type"
        raise ConfigurationError(msg)
    if not callable(model_factory):
        msg = "model_factory must be callable"
        raise ConfigurationError(msg)
    if not callable(model_inputs_getter):
        msg = "model_inputs_getter must be callable"
        raise ConfigurationError(msg)

    _SENSITIVITY_MODEL_ADAPTERS[model_type] = _SensitivityModelAdapter(
        model_factory=model_factory,
        model_inputs_getter=model_inputs_getter,
    )


def run_lap_sensitivity_study(
    *,
    track: TrackData,
    model: Any,
    simulation_config: SimulationConfig,
    parameters: Sequence[SensitivityStudyParameter],
    objectives: Sequence[str] = DEFAULT_LAP_SENSITIVITY_OBJECTIVES,
    label: str | None = None,
    config: SensitivityConfig | None = None,
    numerics: SensitivityNumerics | None = None,
    runtime: SensitivityRuntime | None = None,
) -> SensitivityStudyResult:
    """Run a local lap-KPI sensitivity study over selected scalar parameters.

    The study API is model-agnostic. The model is rebuilt from the provided
    model instance for each parameter perturbation and objective evaluation.

    Args:
        track: Track used for speed-profile evaluation.
        model: Baseline model instance used for adapter-based reconstruction.
        simulation_config: Simulation setup. Must use ``compute_backend='torch'``.
        parameters: Parameter definitions with dot-path targets into the model
            input map reconstructed by the registered adapter.
        objectives: Objective identifiers. Supported values are
            ``lap_time_s`` and ``energy_kwh``.
        label: Optional study label used in exported tabular outputs.
        config: Optional full sensitivity config.
        numerics: Optional sensitivity numerics (used if ``config`` is omitted).
        runtime: Optional sensitivity runtime (used if ``config`` is omitted).

    Returns:
        Multi-objective sensitivity-study result with tabular conversion helpers.

    Raises:
        apexsim.utils.exceptions.ConfigurationError: If objective names are unsupported,
            parameter targets are invalid, or backend requirements are not satisfied.
    """
    simulation_config.validate()
    track.validate()
    resolved_study_model = _resolve_lap_study_model(
        model=model,
        label=label,
    )
    resolved_study_model.validate()

    if simulation_config.runtime.compute_backend != "torch":
        msg = (
            "run_lap_sensitivity_study requires simulation_config.runtime.compute_backend='torch'"
        )
        raise ConfigurationError(msg)

    objective_names = _normalize_lap_sensitivity_objectives(objectives)
    normalized_study_parameters = _normalize_study_parameters(
        parameters=parameters,
        model_inputs=resolved_study_model.model_inputs,
    )
    resolved_config = _resolve_lap_study_sensitivity_config(
        config=config,
        numerics=numerics,
        runtime=runtime,
        default_torch_device=simulation_config.runtime.torch_device,
    )

    if (
        resolved_config.runtime.method == "autodiff"
        and resolved_config.runtime.torch_device != simulation_config.runtime.torch_device
    ):
        msg = (
            "autodiff sensitivity runtime torch_device must match simulation torch_device, "
            f"got {resolved_config.runtime.torch_device!r} vs "
            f"{simulation_config.runtime.torch_device!r}"
        )
        raise ConfigurationError(msg)

    base_parameters = [
        SensitivityParameter(
            name=parameter.name,
            value=_resolve_study_parameter_baseline_value(
                model_inputs=resolved_study_model.model_inputs,
                target=parameter.target,
                parameter_name=parameter.name,
            ),
            kind=parameter.kind,
            lower_bound=parameter.lower_bound,
            upper_bound=parameter.upper_bound,
        )
        for parameter in normalized_study_parameters
    ]

    sensitivity_results: dict[str, SensitivityResult] = {}
    for objective_name in objective_names:
        objective = _build_lap_study_objective(
            objective=objective_name,
            study_model=resolved_study_model,
            track=track,
            simulation_config=simulation_config,
            parameters=normalized_study_parameters,
        )
        sensitivity_results[objective_name] = compute_sensitivities(
            objective=objective,
            parameters=base_parameters,
            config=resolved_config,
        )

    return SensitivityStudyResult(
        study_label=resolved_study_model.label,
        objective_order=tuple(objective_names),
        objective_units={
            name: LAP_SENSITIVITY_OBJECTIVE_UNITS[name] for name in objective_names
        },
        parameters=tuple(normalized_study_parameters),
        sensitivity_results=sensitivity_results,
    )


def _resolve_lap_study_model(
    *,
    model: Any,
    label: str | None,
) -> _ResolvedStudyModelSpec:
    """Resolve model reconstruction spec from a model instance.

    Args:
        model: Baseline model instance for the sensitivity study.
        label: Optional study label for exported results.

    Returns:
        Resolved internal model spec.
    """
    if label is not None and not label.strip():
        msg = "label must be a non-empty string when provided"
        raise ConfigurationError(msg)

    adapter = _resolve_sensitivity_model_adapter(model)
    model_inputs = adapter.model_inputs_getter(model)
    if not isinstance(model_inputs, Mapping):
        msg = "model_inputs_getter must return a mapping"
        raise ConfigurationError(msg)
    if not model_inputs:
        msg = "model_inputs_getter must return a non-empty mapping"
        raise ConfigurationError(msg)

    resolved = _ResolvedStudyModelSpec(
        model_factory=adapter.model_factory,
        model_inputs=dict(model_inputs),
        label=label,
    )
    resolved.validate()
    return resolved


def _resolve_sensitivity_model_adapter(model: Any) -> _SensitivityModelAdapter:
    """Resolve registered adapter for a model instance via MRO lookup.

    Args:
        model: Model instance used for lookup.

    Returns:
        Registered model adapter for the model type.
    """
    _ensure_default_sensitivity_model_adapters()

    for model_cls in type(model).__mro__:
        adapter = _SENSITIVITY_MODEL_ADAPTERS.get(model_cls)
        if adapter is not None:
            return adapter

    msg = (
        "No sensitivity model adapter registered for model type "
        f"{type(model)!r}. Register one via "
        "`register_sensitivity_model_adapter(model_type=..., "
        "model_factory=..., model_inputs_getter=...)`."
    )
    raise ConfigurationError(msg)


def _ensure_default_sensitivity_model_adapters() -> None:
    """Register built-in model adapters once."""
    global _DEFAULT_MODEL_ADAPTERS_REGISTERED
    if _DEFAULT_MODEL_ADAPTERS_REGISTERED:
        return

    from apexsim.vehicle.point_mass_model import PointMassModel, build_point_mass_model
    from apexsim.vehicle.single_track_model import SingleTrackModel, build_single_track_model

    register_sensitivity_model_adapter(
        model_type=SingleTrackModel,
        model_factory=build_single_track_model,
        model_inputs_getter=_single_track_model_inputs_getter,
    )
    register_sensitivity_model_adapter(
        model_type=PointMassModel,
        model_factory=build_point_mass_model,
        model_inputs_getter=_point_mass_model_inputs_getter,
    )
    _DEFAULT_MODEL_ADAPTERS_REGISTERED = True


def _single_track_model_inputs_getter(model: Any) -> Mapping[str, Any]:
    """Extract reconstructable inputs from single-track model instances.

    Args:
        model: Single-track model instance.

    Returns:
        Mapping consumed by ``build_single_track_model``.
    """
    return _extract_model_attributes(
        model=model,
        required_fields=("vehicle", "tires", "physics", "numerics"),
    )


def _point_mass_model_inputs_getter(model: Any) -> Mapping[str, Any]:
    """Extract reconstructable inputs from point-mass model instances.

    Args:
        model: Point-mass model instance.

    Returns:
        Mapping consumed by ``build_point_mass_model``.
    """
    return _extract_model_attributes(
        model=model,
        required_fields=("vehicle", "physics"),
    )


def _extract_model_attributes(
    *,
    model: Any,
    required_fields: Sequence[str],
) -> Mapping[str, Any]:
    """Extract required model attributes into a mapping.

    Args:
        model: Model instance.
        required_fields: Required attribute names.

    Returns:
        Mapping with extracted attribute values.
    """
    values: dict[str, Any] = {}
    missing = [name for name in required_fields if not hasattr(model, name)]
    if missing:
        msg = (
            f"Model type {type(model)!r} is missing required attributes for "
            f"sensitivity reconstruction: {missing}"
        )
        raise ConfigurationError(msg)

    for name in required_fields:
        values[name] = getattr(model, name)
    return values


def _resolve_lap_study_sensitivity_config(
    *,
    config: SensitivityConfig | None,
    numerics: SensitivityNumerics | None,
    runtime: SensitivityRuntime | None,
    default_torch_device: str,
) -> SensitivityConfig:
    """Resolve and validate sensitivity config for high-level study calls.

    Args:
        config: Optional fully assembled sensitivity config.
        numerics: Optional sensitivity numerical controls.
        runtime: Optional sensitivity runtime controls.
        default_torch_device: Torch device used for default autodiff runtime.

    Returns:
        Validated sensitivity config for high-level lap study execution.
    """
    if config is not None and any(value is not None for value in (numerics, runtime)):
        msg = "Provide either `config` or (`numerics`, `runtime`) components, not both."
        raise ConfigurationError(msg)

    if config is not None:
        config.validate()
        return config

    resolved_runtime = runtime
    if resolved_runtime is None:
        resolved_runtime = SensitivityRuntime(
            method="autodiff",
            torch_device=default_torch_device,
            autodiff_fallback_to_finite_difference=False,
        )

    return build_sensitivity_config(numerics=numerics, runtime=resolved_runtime)


def _normalize_lap_sensitivity_objectives(objectives: Sequence[str]) -> list[str]:
    """Normalize and validate objective name sequence for lap studies.

    Args:
        objectives: Objective identifier sequence.

    Returns:
        Ordered list of unique validated objective identifiers.
    """
    names = list(objectives)
    if not names:
        msg = "at least one lap sensitivity objective must be provided"
        raise ConfigurationError(msg)

    normalized: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name not in VALID_LAP_SENSITIVITY_OBJECTIVES:
            msg = (
                "unsupported lap sensitivity objective. "
                f"Expected one of {VALID_LAP_SENSITIVITY_OBJECTIVES}, got: {name!r}"
            )
            raise ConfigurationError(msg)
        if name in seen:
            continue
        normalized.append(name)
        seen.add(name)
    return normalized


def _normalize_study_parameters(
    *,
    parameters: Sequence[SensitivityStudyParameter],
    model_inputs: Mapping[str, Any],
) -> list[SensitivityStudyParameter]:
    """Validate study parameters and ensure referenced targets exist.

    Args:
        parameters: Sequence of study parameter definitions.
        model_inputs: Baseline study-model input mapping.

    Returns:
        Ordered list of validated study parameters.
    """
    normalized = list(parameters)
    if not normalized:
        msg = "at least one sensitivity study parameter must be provided"
        raise ConfigurationError(msg)

    seen_names: set[str] = set()
    for parameter in normalized:
        parameter.validate()
        if parameter.name in seen_names:
            msg = f"duplicate study parameter name: {parameter.name!r}"
            raise ConfigurationError(msg)
        seen_names.add(parameter.name)
        _resolve_study_parameter_baseline_value(
            model_inputs=model_inputs,
            target=parameter.target,
            parameter_name=parameter.name,
        )
    return normalized


def _resolve_study_parameter_baseline_value(
    *,
    model_inputs: Mapping[str, Any],
    target: str,
    parameter_name: str,
) -> float:
    """Resolve and validate baseline scalar value for a study parameter.

    Args:
        model_inputs: Baseline model-input mapping.
        target: Dot-path into ``model_inputs``.
        parameter_name: Parameter identifier used for error reporting.

    Returns:
        Baseline scalar parameter value resolved from ``target``.
    """
    target_value = _resolve_dot_path(model_inputs, target=target)
    baseline_value = _as_scalar_float(
        target_value,
        context=(
            f"study parameter {parameter_name!r} target {target!r} "
            "must resolve to a scalar numeric value"
        ),
    )
    return baseline_value


def _build_lap_study_objective(
    *,
    objective: str,
    study_model: _ResolvedStudyModelSpec,
    track: TrackData,
    simulation_config: SimulationConfig,
    parameters: Sequence[SensitivityStudyParameter],
) -> SensitivityObjective:
    """Build objective callable for high-level lap sensitivity studies.

    Args:
        objective: Objective identifier.
        study_model: Resolved study-model bundle with factory and baseline inputs.
        track: Track used for lap simulation.
        simulation_config: Torch simulation config used for objective solves.
        parameters: Study parameter definitions.

    Returns:
        Scalar objective callable compatible with :func:`compute_sensitivities`.
    """
    if objective not in VALID_LAP_SENSITIVITY_OBJECTIVES:
        msg = (
            "unsupported lap sensitivity objective. "
            f"Expected one of {VALID_LAP_SENSITIVITY_OBJECTIVES}, got: {objective!r}"
        )
        raise ConfigurationError(msg)

    parameter_map = {parameter.name: parameter for parameter in parameters}

    def objective_fn(parameter_values: Mapping[str, Any]) -> Any:
        updated_inputs = _build_study_model_inputs(
            base_inputs=study_model.model_inputs,
            parameters=parameter_map,
            parameter_values=parameter_values,
        )
        try:
            model = study_model.model_factory(**updated_inputs)
        except TypeError as exc:
            msg = (
                "Registered sensitivity model_factory failed to build model "
                "from extracted inputs."
            )
            raise ConfigurationError(msg) from exc

        from apexsim.simulation.torch_profile import solve_speed_profile_torch

        profile = solve_speed_profile_torch(
            track=track,
            model=model,
            config=simulation_config,
        )

        if objective == "lap_time_s":
            return profile.lap_time
        return _compute_lap_energy_kwh_torch(
            track=track,
            model=model,
            speed_profile=profile,
        )

    return cast(SensitivityObjective, objective_fn)


def _build_study_model_inputs(
    *,
    base_inputs: Mapping[str, Any],
    parameters: Mapping[str, SensitivityStudyParameter],
    parameter_values: Mapping[str, Any],
) -> dict[str, Any]:
    """Build model input mapping with parameter-value overrides applied.

    Args:
        base_inputs: Baseline model-input mapping.
        parameters: Study-parameter definitions keyed by parameter name.
        parameter_values: Current parameter values consumed by objective call.

    Returns:
        Updated model-input mapping with all target overrides applied.
    """
    updated: Any = dict(base_inputs)
    missing = [name for name in parameters if name not in parameter_values]
    if missing:
        msg = f"objective parameter map is missing required keys: {missing}"
        raise ConfigurationError(msg)

    for name, definition in parameters.items():
        updated = _set_dot_path(
            updated,
            target=definition.target,
            value=parameter_values[name],
        )
    return dict(updated)


def _resolve_dot_path(root: Mapping[str, Any], *, target: str) -> Any:
    """Resolve dot-path values from mapping/dataclass/object trees.

    Args:
        root: Root mapping of model inputs.
        target: Dot-path expression.

    Returns:
        Value resolved at ``target``.
    """
    current: Any = root
    for segment in target.split("."):
        if isinstance(current, Mapping):
            if segment not in current:
                msg = f"target {target!r} segment {segment!r} not found in mapping"
                raise ConfigurationError(msg)
            current = current[segment]
            continue
        if hasattr(current, segment):
            current = getattr(current, segment)
            continue
        msg = (
            f"target {target!r} segment {segment!r} could not be resolved "
            f"on object of type {type(current)!r}"
        )
        raise ConfigurationError(msg)
    return current


def _set_dot_path(
    root: Any,
    *,
    target: str,
    value: Any,
) -> Any:
    """Return copy of ``root`` with dot-path target set to ``value``.

    Args:
        root: Root object to be copied and updated.
        target: Dot-path expression.
        value: Replacement value at the target path.

    Returns:
        Updated copy with target value set.
    """
    segments = target.split(".")
    return _set_dot_path_segments(root, segments=segments, target=target, value=value)


def _set_dot_path_segments(
    root: Any,
    *,
    segments: Sequence[str],
    target: str,
    value: Any,
) -> Any:
    """Recursively apply a dot-path update with immutable-style copies.

    Args:
        root: Current object or mapping node.
        segments: Remaining target path segments.
        target: Full target path used for error messages.
        value: Replacement value.

    Returns:
        Updated copy of ``root`` with ``value`` applied at target path.
    """
    if not segments:
        return value

    head = segments[0]
    tail = segments[1:]

    if isinstance(root, Mapping):
        if head not in root:
            msg = f"target {target!r} segment {head!r} not found in mapping"
            raise ConfigurationError(msg)
        updated = _set_dot_path_segments(
            root[head],
            segments=tail,
            target=target,
            value=value,
        )
        copied_mapping = dict(root)
        copied_mapping[head] = updated
        return copied_mapping

    if is_dataclass(root):
        if not hasattr(root, head):
            msg = (
                f"target {target!r} segment {head!r} could not be resolved "
                f"on dataclass {type(root)!r}"
            )
            raise ConfigurationError(msg)
        updated = _set_dot_path_segments(
            getattr(root, head),
            segments=tail,
            target=target,
            value=value,
        )
        return replace(cast(Any, root), **{head: updated})

    if not hasattr(root, head):
        msg = (
            f"target {target!r} segment {head!r} could not be resolved "
            f"on object type {type(root)!r}"
        )
        raise ConfigurationError(msg)

    if tail:
        updated = _set_dot_path_segments(
            getattr(root, head),
            segments=tail,
            target=target,
            value=value,
        )
    else:
        updated = value

    copied_object = copy(root)
    setattr(copied_object, head, updated)
    return copied_object


def _compute_lap_energy_kwh_torch(
    *,
    track: TrackData,
    model: Any,
    speed_profile: Any,
) -> Any:
    """Compute differentiable lap energy in ``kWh`` from torch profile outputs.

    Args:
        track: Track used for speed-profile solve.
        model: Vehicle model exposing ``tractive_power_torch``.
        speed_profile: Torch speed-profile output with speed and acceleration.

    Returns:
        Differentiable scalar energy value in ``kWh``.
    """
    torch = _require_torch()
    tractive_power_torch = getattr(model, "tractive_power_torch", None)
    if not callable(tractive_power_torch):
        msg = (
            "energy_kwh lap sensitivity objective requires model to implement "
            "`tractive_power_torch(speed, longitudinal_accel)`."
        )
        raise ConfigurationError(msg)

    speed = speed_profile.speed
    longitudinal_accel = speed_profile.longitudinal_accel
    power = tractive_power_torch(speed=speed, longitudinal_accel=longitudinal_accel)
    power_tensor = torch.as_tensor(power, dtype=speed.dtype, device=speed.device)

    if tuple(power_tensor.shape) != tuple(speed.shape):
        msg = (
            "tractive_power_torch must return tensor with same shape as speed, "
            f"expected {tuple(speed.shape)}, got {tuple(power_tensor.shape)}"
        )
        raise ConfigurationError(msg)

    arc_length = torch.as_tensor(track.arc_length, dtype=speed.dtype, device=speed.device)
    if arc_length.numel() < 2:
        return torch.zeros((), dtype=speed.dtype, device=speed.device)

    ds = arc_length[1:] - arc_length[:-1]
    speed_average = torch.clamp(0.5 * (speed[:-1] + speed[1:]), min=SMALL_EPS)
    dt = ds / speed_average
    traction_power = torch.clamp(power_tensor[:-1], min=0.0)
    energy_joule = torch.sum(traction_power * dt)
    return energy_joule / 3_600_000.0


def _normalize_parameters(
    parameters: Sequence[SensitivityParameter] | Mapping[str, float],
) -> list[SensitivityParameter]:
    """Normalize parameter input into validated ordered definitions.

    Args:
        parameters: Parameter definitions or mapping from name to value.

    Returns:
        Ordered list of validated sensitivity parameters.

    Raises:
        apexsim.utils.exceptions.ConfigurationError: If no parameters are
            provided, if names are duplicated, or if any definition is invalid.
    """
    if isinstance(parameters, Mapping):
        normalized = [
            SensitivityParameter(name=name, value=float(value))
            for name, value in parameters.items()
        ]
    else:
        normalized = list(parameters)

    if not normalized:
        msg = "at least one sensitivity parameter must be provided"
        raise ConfigurationError(msg)

    seen: set[str] = set()
    for parameter in normalized:
        parameter.validate()
        if parameter.name in seen:
            msg = f"duplicate parameter name: {parameter.name!r}"
            raise ConfigurationError(msg)
        seen.add(parameter.name)

    return normalized


def _require_torch() -> Any:
    """Import torch lazily and fail with a configuration-level message.

    Returns:
        Imported ``torch`` module.

    Raises:
        apexsim.utils.exceptions.ConfigurationError: If torch is not installed.
    """
    try:
        import torch
    except ModuleNotFoundError as exc:
        msg = (
            "autodiff sensitivity method requires PyTorch. "
            "Install with `pip install -e '.[torch]'`."
        )
        raise ConfigurationError(msg) from exc
    return torch


def _as_scalar_float(value: Any, *, context: str) -> float:
    """Convert objective outputs into plain scalar ``float`` values.

    Args:
        value: Objective output.
        context: Error-context prefix used in raised messages.

    Returns:
        Scalar objective value as ``float``.

    Raises:
        apexsim.utils.exceptions.ConfigurationError: If value is not scalar.
    """
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return float(value.item())
        msg = f"{context} must return a scalar, got ndarray with shape {value.shape}"
        raise ConfigurationError(msg)

    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            item_value = item_method()
        except Exception as exc:  # pragma: no cover - defensive guard
            msg = f"{context} returned a non-scalar value of type {type(value)!r}"
            raise ConfigurationError(msg) from exc
        if isinstance(item_value, (float, int, np.floating, np.integer)):
            return float(item_value)

    msg = f"{context} must return a scalar numeric value, got {type(value)!r}"
    raise ConfigurationError(msg)


def _objective_float(
    objective: SensitivityObjective,
    parameters: Mapping[str, float],
) -> float:
    """Evaluate objective on scalar parameter mapping and convert to float.

    Args:
        objective: Scalar objective callable.
        parameters: Scalar-valued parameter mapping.

    Returns:
        Scalar objective value as float.
    """
    value = objective(parameters)
    return _as_scalar_float(value, context="objective")


def _compute_sensitivities_finite_difference(
    objective: SensitivityObjective,
    parameters: Sequence[SensitivityParameter],
    numerics: SensitivityNumerics,
) -> SensitivityResult:
    """Compute sensitivities with finite differences.

    Args:
        objective: Scalar objective callable.
        parameters: Parameter definitions.
        numerics: Finite-difference numerical controls.

    Returns:
        Finite-difference sensitivity result.
    """
    base_parameters = {parameter.name: float(parameter.value) for parameter in parameters}
    base_value = _objective_float(objective, base_parameters)
    sensitivities: dict[str, float] = {}

    for parameter in parameters:
        step_candidate = numerics.step_size(parameter.value)
        mode, step = _finite_difference_mode(parameter, step_candidate, numerics)
        if mode == "central":
            plus = dict(base_parameters)
            minus = dict(base_parameters)
            plus[parameter.name] = parameter.value + step
            minus[parameter.name] = parameter.value - step
            value_plus = _objective_float(objective, plus)
            value_minus = _objective_float(objective, minus)
            sensitivities[parameter.name] = (value_plus - value_minus) / (2.0 * step)
        elif mode == "forward":
            plus = dict(base_parameters)
            plus[parameter.name] = parameter.value + step
            value_plus = _objective_float(objective, plus)
            sensitivities[parameter.name] = (value_plus - base_value) / step
        else:
            minus = dict(base_parameters)
            minus[parameter.name] = parameter.value - step
            value_minus = _objective_float(objective, minus)
            sensitivities[parameter.name] = (base_value - value_minus) / step

    return SensitivityResult(
        objective_value=base_value,
        sensitivities=sensitivities,
        method="finite_difference",
        parameter_values=base_parameters,
        parameter_kinds={parameter.name: parameter.kind for parameter in parameters},
    )


def _finite_difference_mode(
    parameter: SensitivityParameter,
    step_candidate: float,
    numerics: SensitivityNumerics,
) -> tuple[str, float]:
    """Resolve perturbation mode and step considering parameter bounds.

    Args:
        parameter: Parameter definition with optional bounds.
        step_candidate: Preferred positive perturbation magnitude.
        numerics: Finite-difference numerical controls.

    Returns:
        Tuple ``(mode, step)`` where mode is ``central``, ``forward``, or
        ``backward``.

    Raises:
        apexsim.utils.exceptions.ConfigurationError: If bounds leave no valid
            perturbation direction.
    """
    value = float(parameter.value)
    plus_margin = np.inf if parameter.upper_bound is None else parameter.upper_bound - value
    minus_margin = np.inf if parameter.lower_bound is None else value - parameter.lower_bound

    can_plus = plus_margin > 0.0
    can_minus = minus_margin > 0.0
    if not can_plus and not can_minus:
        msg = (
            f"parameter {parameter.name!r} cannot be perturbed within bounds "
            f"[{parameter.lower_bound}, {parameter.upper_bound}]"
        )
        raise ConfigurationError(msg)

    if numerics.finite_difference_scheme == "central":
        if can_plus and can_minus:
            step = min(step_candidate, plus_margin, minus_margin)
            return "central", max(float(step), SMALL_EPS)
        if can_plus:
            step = min(step_candidate, plus_margin)
            return "forward", max(float(step), SMALL_EPS)
        step = min(step_candidate, minus_margin)
        return "backward", max(float(step), SMALL_EPS)

    if can_plus:
        step = min(step_candidate, plus_margin)
        return "forward", max(float(step), SMALL_EPS)
    step = min(step_candidate, minus_margin)
    return "backward", max(float(step), SMALL_EPS)


def _compute_sensitivities_autodiff(
    objective: SensitivityObjective,
    parameters: Sequence[SensitivityParameter],
    runtime: SensitivityRuntime,
) -> SensitivityResult:
    """Compute sensitivities with PyTorch automatic differentiation.

    Args:
        objective: Differentiable scalar objective callable.
        parameters: Parameter definitions.
        runtime: Autodiff runtime controls.

    Returns:
        Autodiff sensitivity result.

    Raises:
        apexsim.utils.exceptions.ConfigurationError: If objective output is not
            a scalar tensor or if gradients cannot be evaluated.
    """
    torch = _require_torch()
    tensors = {
        parameter.name: torch.tensor(
            float(parameter.value),
            dtype=torch.float64,
            device=runtime.torch_device,
            requires_grad=True,
        )
        for parameter in parameters
    }

    objective_value = objective(tensors)
    if not isinstance(objective_value, torch.Tensor):
        msg = (
            "autodiff sensitivity method requires objective to return a torch.Tensor scalar, "
            f"got {type(objective_value)!r}"
        )
        raise ConfigurationError(msg)
    if objective_value.numel() != 1:
        msg = (
            "autodiff sensitivity method requires scalar objective output, "
            f"got tensor shape {tuple(objective_value.shape)}"
        )
        raise ConfigurationError(msg)

    scalar_objective = objective_value.reshape(())
    if not scalar_objective.is_floating_point():
        scalar_objective = scalar_objective.to(torch.float64)
    if not scalar_objective.requires_grad:
        msg = (
            "autodiff objective output does not require gradients. "
            "Ensure objective uses differentiable torch operations on input parameters."
        )
        raise ConfigurationError(msg)

    try:
        scalar_objective.backward()
    except RuntimeError as exc:
        msg = f"autodiff backward pass failed: {exc}"
        raise ConfigurationError(msg) from exc

    sensitivities: dict[str, float] = {}
    for parameter in parameters:
        gradient = tensors[parameter.name].grad
        sensitivities[parameter.name] = float(0.0 if gradient is None else gradient.item())

    return SensitivityResult(
        objective_value=float(scalar_objective.detach().cpu().item()),
        sensitivities=sensitivities,
        method="autodiff",
        parameter_values={parameter.name: float(parameter.value) for parameter in parameters},
        parameter_kinds={parameter.name: parameter.kind for parameter in parameters},
    )
