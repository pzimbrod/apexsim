"""Sensitivity-analysis APIs for scalar simulation objectives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from apexsim.simulation.config import DEFAULT_TORCH_DEVICE
from apexsim.utils.constants import SMALL_EPS
from apexsim.utils.exceptions import ConfigurationError

DEFAULT_SENSITIVITY_METHOD = "autodiff"
DEFAULT_FD_SCHEME = "central"
DEFAULT_FD_RELATIVE_STEP = 1e-3
DEFAULT_FD_ABSOLUTE_STEP = 1e-6
DEFAULT_AUTODIFF_FALLBACK_TO_FD = True
VALID_SENSITIVITY_METHODS = ("autodiff", "finite_difference")
VALID_FD_SCHEMES = ("central", "forward")
VALID_PARAMETER_KINDS = ("physical", "numerical")


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
