"""Velocity-dependent performance-envelope computation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from pylapsim.simulation.config import (
    DEFAULT_COMPUTE_BACKEND,
    DEFAULT_ENABLE_TORCH_COMPILE,
    DEFAULT_TORCH_DEVICE,
    VALID_COMPUTE_BACKENDS,
)
from pylapsim.simulation.model_api import VehicleModel
from pylapsim.utils.exceptions import ConfigurationError

DEFAULT_ENVELOPE_SPEED_MIN = 10.0
DEFAULT_ENVELOPE_SPEED_MAX = 90.0
DEFAULT_ENVELOPE_GRADE = 0.0
DEFAULT_ENVELOPE_BANKING = 0.0
DEFAULT_ENVELOPE_SPEED_SAMPLES = 41
DEFAULT_ENVELOPE_LATERAL_ACCEL_SAMPLES = 41
DEFAULT_LATERAL_ACCEL_FRACTION_SPAN = 1.0
TORCH_COMPILE_MODE = "reduce-overhead"

_COMPILED_TORCH_ENVELOPE_SOLVER_CACHE: dict[tuple[str], Any] = {}


@dataclass(frozen=True)
class PerformanceEnvelopePhysics:
    """Physical operating-point definitions for envelope generation.

    Args:
        speed_min: Lower speed bound for the envelope sweep [m/s].
        speed_max: Upper speed bound for the envelope sweep [m/s].
        grade: Constant road grade used for all envelope samples ``dz/ds`` [-].
        banking: Constant road banking used for all envelope samples [rad].
    """

    speed_min: float = DEFAULT_ENVELOPE_SPEED_MIN
    speed_max: float = DEFAULT_ENVELOPE_SPEED_MAX
    grade: float = DEFAULT_ENVELOPE_GRADE
    banking: float = DEFAULT_ENVELOPE_BANKING

    def validate(self) -> None:
        """Validate physical operating-point definitions.

        Raises:
            pylapsim.utils.exceptions.ConfigurationError: If speed bounds are
                invalid or non-finite values are provided.
        """
        if self.speed_min <= 0.0:
            msg = "speed_min must be positive"
            raise ConfigurationError(msg)
        if self.speed_max <= self.speed_min:
            msg = "speed_max must be greater than speed_min"
            raise ConfigurationError(msg)
        if not np.isfinite(self.grade):
            msg = "grade must be finite"
            raise ConfigurationError(msg)
        if not np.isfinite(self.banking):
            msg = "banking must be finite"
            raise ConfigurationError(msg)


@dataclass(frozen=True)
class PerformanceEnvelopeNumerics:
    """Numerical discretization controls for envelope generation.

    Args:
        speed_samples: Number of speed support points used for the sweep.
        lateral_accel_samples: Number of lateral-demand support points used per
            speed sample.
        lateral_accel_fraction_span: Symmetric lateral-demand span relative to
            available lateral limit. ``1.0`` means ``[-ay_max, ay_max]``.
    """

    speed_samples: int = DEFAULT_ENVELOPE_SPEED_SAMPLES
    lateral_accel_samples: int = DEFAULT_ENVELOPE_LATERAL_ACCEL_SAMPLES
    lateral_accel_fraction_span: float = DEFAULT_LATERAL_ACCEL_FRACTION_SPAN

    def validate(self) -> None:
        """Validate numerical envelope discretization settings.

        Raises:
            pylapsim.utils.exceptions.ConfigurationError: If sample counts or
                span values violate required bounds.
        """
        if self.speed_samples < 2:
            msg = "speed_samples must be at least 2"
            raise ConfigurationError(msg)
        if self.lateral_accel_samples < 2:
            msg = "lateral_accel_samples must be at least 2"
            raise ConfigurationError(msg)
        if self.lateral_accel_fraction_span <= 0.0:
            msg = "lateral_accel_fraction_span must be positive"
            raise ConfigurationError(msg)


@dataclass(frozen=True)
class PerformanceEnvelopeRuntime:
    """Runtime controls for envelope generation backend selection.

    Args:
        compute_backend: Numerical backend identifier (``numpy``, ``numba``,
            or ``torch``).
        torch_device: Torch device identifier. For non-torch backends this
            must remain ``"cpu"``.
        torch_compile: Enable ``torch.compile`` when backend is ``torch``.
    """

    compute_backend: str = DEFAULT_COMPUTE_BACKEND
    torch_device: str = DEFAULT_TORCH_DEVICE
    torch_compile: bool = DEFAULT_ENABLE_TORCH_COMPILE

    def validate(self) -> None:
        """Validate runtime controls and backend availability.

        Raises:
            pylapsim.utils.exceptions.ConfigurationError: If backend selection
                or backend-specific runtime settings are invalid.
        """
        if self.compute_backend not in VALID_COMPUTE_BACKENDS:
            msg = (
                "compute_backend must be one of "
                f"{VALID_COMPUTE_BACKENDS}, got: {self.compute_backend!r}"
            )
            raise ConfigurationError(msg)
        if not self.torch_device:
            msg = "torch_device must be a non-empty string"
            raise ConfigurationError(msg)
        if not isinstance(self.torch_compile, bool):
            msg = "torch_compile must be a boolean"
            raise ConfigurationError(msg)

        if self.compute_backend != "torch":
            if self.torch_device != DEFAULT_TORCH_DEVICE:
                msg = (
                    "torch_device is only meaningful for compute_backend='torch'. "
                    "Use torch_device='cpu' for numpy/numba backends."
                )
                raise ConfigurationError(msg)
            if self.torch_compile:
                msg = "torch_compile can only be enabled for compute_backend='torch'"
                raise ConfigurationError(msg)

        if self.compute_backend == "numba":
            self._validate_numba_runtime()
        if self.compute_backend == "torch":
            self._validate_torch_runtime()

    @staticmethod
    def _validate_numba_runtime() -> None:
        """Validate that numba is available for runtime selection.

        Raises:
            pylapsim.utils.exceptions.ConfigurationError: If numba is not
                installed in the active environment.
        """
        try:
            import numba  # type: ignore[import-untyped]  # noqa: F401
        except ModuleNotFoundError as exc:
            msg = (
                "compute_backend='numba' requires Numba. "
                "Install with `pip install -e '.[numba]'` or add `numba` to your environment."
            )
            raise ConfigurationError(msg) from exc

    def _validate_torch_runtime(self) -> None:
        """Validate that the configured torch runtime is available.

        Raises:
            pylapsim.utils.exceptions.ConfigurationError: If torch is not
                installed or selected CUDA device is unavailable.
        """
        torch = _require_torch()
        if self.torch_device.startswith("cuda") and not torch.cuda.is_available():
            msg = (
                "torch_device requests CUDA but no CUDA device is available: "
                f"{self.torch_device!r}"
            )
            raise ConfigurationError(msg)


@dataclass(frozen=True)
class PerformanceEnvelopeConfig:
    """Top-level config for performance-envelope generation.

    Args:
        physics: Physical operating-point definitions for the envelope.
        numerics: Numerical discretization controls for the envelope sweep.
        runtime: Runtime backend controls for model evaluation.
    """

    physics: PerformanceEnvelopePhysics = field(default_factory=PerformanceEnvelopePhysics)
    numerics: PerformanceEnvelopeNumerics = field(default_factory=PerformanceEnvelopeNumerics)
    runtime: PerformanceEnvelopeRuntime = field(default_factory=PerformanceEnvelopeRuntime)

    def validate(self) -> None:
        """Validate combined envelope settings.

        Raises:
            pylapsim.utils.exceptions.ConfigurationError: If any physical,
                numerical, or runtime setting is invalid.
        """
        self.physics.validate()
        self.numerics.validate()
        self.runtime.validate()


@dataclass(frozen=True)
class PerformanceEnvelopeResult:
    """Computed velocity-dependent G-G envelope arrays.

    Args:
        speed: Speed support points [m/s], shape ``(n_speed,)``.
        lateral_accel_limit: Lateral acceleration limits at each speed [m/s^2],
            shape ``(n_speed,)``.
        lateral_accel_fraction: Normalized lateral-demand support points used
            for each speed sample, shape ``(n_lateral,)``.
        lateral_accel: Lateral-demand matrix [m/s^2], shape
            ``(n_speed, n_lateral)``.
        max_longitudinal_accel: Maximum net forward acceleration matrix [m/s^2],
            shape ``(n_speed, n_lateral)``.
        min_longitudinal_accel: Minimum net longitudinal acceleration matrix
            (negative for braking) [m/s^2], shape ``(n_speed, n_lateral)``.
    """

    speed: np.ndarray
    lateral_accel_limit: np.ndarray
    lateral_accel_fraction: np.ndarray
    lateral_accel: np.ndarray
    max_longitudinal_accel: np.ndarray
    min_longitudinal_accel: np.ndarray

    def __post_init__(self) -> None:
        """Validate array shapes after initialization."""
        expected_speed_shape = self.speed.shape
        if self.lateral_accel_limit.shape != expected_speed_shape:
            msg = (
                "lateral_accel_limit must match speed shape: "
                f"expected {expected_speed_shape}, got {self.lateral_accel_limit.shape}"
            )
            raise ConfigurationError(msg)

        if self.speed.ndim != 1:
            msg = f"speed must be one-dimensional, got {self.speed.ndim} dimensions"
            raise ConfigurationError(msg)
        if self.lateral_accel_fraction.ndim != 1:
            msg = (
                "lateral_accel_fraction must be one-dimensional, got "
                f"{self.lateral_accel_fraction.ndim} dimensions"
            )
            raise ConfigurationError(msg)

        expected_grid_shape = (self.speed.size, self.lateral_accel_fraction.size)
        for signal_name, signal in (
            ("lateral_accel", self.lateral_accel),
            ("max_longitudinal_accel", self.max_longitudinal_accel),
            ("min_longitudinal_accel", self.min_longitudinal_accel),
        ):
            if signal.shape != expected_grid_shape:
                msg = (
                    f"{signal_name} must have shape {expected_grid_shape}, "
                    f"got {signal.shape}"
                )
                raise ConfigurationError(msg)

    def to_numpy(self) -> np.ndarray:
        """Return a stacked NumPy tensor representation.

        Returns:
            Tensor with shape ``(n_speed, n_lateral, 3)`` and channels ordered
            as ``(lateral_accel, max_longitudinal_accel, min_longitudinal_accel)``.
        """
        return np.stack(
            (
                self.lateral_accel,
                self.max_longitudinal_accel,
                self.min_longitudinal_accel,
            ),
            axis=-1,
        )

    def to_dataframe(self) -> Any:
        """Return a long-form pandas representation of the envelope.

        Returns:
            Pandas DataFrame with columns ``speed``, ``lateral_accel_limit``,
            ``lateral_accel_fraction``, ``lateral_accel``,
            ``max_longitudinal_accel``, and ``min_longitudinal_accel``.

        Raises:
            pylapsim.utils.exceptions.ConfigurationError: If pandas is not
                installed in the active environment.
        """
        try:
            import pandas as pd  # type: ignore[import-untyped]
        except ModuleNotFoundError as exc:
            msg = (
                "PerformanceEnvelopeResult.to_dataframe requires pandas. "
                "Install with `pip install pandas`."
            )
            raise ConfigurationError(msg) from exc

        n_speed = self.speed.size
        n_lateral = self.lateral_accel_fraction.size

        speed_grid = np.repeat(self.speed[:, np.newaxis], n_lateral, axis=1)
        lateral_limit_grid = np.repeat(self.lateral_accel_limit[:, np.newaxis], n_lateral, axis=1)
        lateral_fraction_grid = np.repeat(
            self.lateral_accel_fraction[np.newaxis, :],
            n_speed,
            axis=0,
        )

        return pd.DataFrame(
            {
                "speed": speed_grid.ravel(),
                "lateral_accel_limit": lateral_limit_grid.ravel(),
                "lateral_accel_fraction": lateral_fraction_grid.ravel(),
                "lateral_accel": self.lateral_accel.ravel(),
                "max_longitudinal_accel": self.max_longitudinal_accel.ravel(),
                "min_longitudinal_accel": self.min_longitudinal_accel.ravel(),
            }
        )


def build_performance_envelope_config(
    physics: PerformanceEnvelopePhysics | None = None,
    numerics: PerformanceEnvelopeNumerics | None = None,
    runtime: PerformanceEnvelopeRuntime | None = None,
) -> PerformanceEnvelopeConfig:
    """Build a validated performance-envelope config.

    Args:
        physics: Optional physical operating-point definitions.
        numerics: Optional numerical discretization controls.
        runtime: Optional runtime backend controls.

    Returns:
        Fully validated performance-envelope configuration.

    Raises:
        pylapsim.utils.exceptions.ConfigurationError: If assembled settings are
            inconsistent.
    """
    config = PerformanceEnvelopeConfig(
        physics=physics or PerformanceEnvelopePhysics(),
        numerics=numerics or PerformanceEnvelopeNumerics(),
        runtime=runtime or PerformanceEnvelopeRuntime(),
    )
    config.validate()
    return config


def _require_torch() -> Any:
    """Import torch lazily and fail with a configuration-level message.

    Returns:
        Imported ``torch`` module.

    Raises:
        pylapsim.utils.exceptions.ConfigurationError: If torch is not installed.
    """
    try:
        import torch
    except ModuleNotFoundError as exc:
        msg = (
            "Torch backend requested but PyTorch is not installed. "
            "Install with `pip install -e '.[torch]'`."
        )
        raise ConfigurationError(msg) from exc
    return torch


def _lateral_accel_limit_batch(
    model: VehicleModel,
    speed: np.ndarray,
    banking: np.ndarray,
) -> np.ndarray:
    """Evaluate lateral acceleration limits for vectorized operating points.

    Args:
        model: Vehicle model backend implementing ``VehicleModel``.
        speed: Speed samples [m/s].
        banking: Banking-angle samples [rad].

    Returns:
        Lateral acceleration limit samples [m/s^2].

    Raises:
        pylapsim.utils.exceptions.ConfigurationError: If a vectorized model
            implementation returns an incompatible array shape.
    """
    batch_method: Any = getattr(model, "lateral_accel_limit_batch", None)
    if callable(batch_method):
        lateral_limit = np.asarray(batch_method(speed=speed, banking=banking), dtype=float)
        if lateral_limit.shape != speed.shape:
            msg = (
                "lateral_accel_limit_batch must return an array with shape "
                f"{speed.shape}, got {lateral_limit.shape}"
            )
            raise ConfigurationError(msg)
        return lateral_limit

    return np.array(
        [
            model.lateral_accel_limit(speed=float(speed[idx]), banking=float(banking[idx]))
            for idx in range(speed.size)
        ],
        dtype=float,
    )


def _compute_longitudinal_limits_numpy(
    model: VehicleModel,
    speed: np.ndarray,
    lateral_accel_required: np.ndarray,
    grade: float,
    banking: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute longitudinal acceleration and braking envelopes on NumPy path.

    Args:
        model: Vehicle model backend implementing ``VehicleModel``.
        speed: Speed support points [m/s], shape ``(n_speed,)``.
        lateral_accel_required: Required lateral acceleration magnitudes [m/s^2],
            shape ``(n_speed, n_lateral)``.
        grade: Constant road grade ``dz/ds`` used for all points.
        banking: Constant road banking angle [rad] used for all points.

    Returns:
        Tuple ``(max_longitudinal_accel, min_longitudinal_accel)`` with shape
        ``(n_speed, n_lateral)``.
    """
    max_longitudinal_accel = np.empty_like(lateral_accel_required, dtype=float)
    min_longitudinal_accel = np.empty_like(lateral_accel_required, dtype=float)

    max_accel_fn = model.max_longitudinal_accel
    max_decel_fn = model.max_longitudinal_decel

    for speed_idx in range(speed.size):
        speed_value = float(speed[speed_idx])
        for lateral_idx in range(lateral_accel_required.shape[1]):
            ay_required = float(lateral_accel_required[speed_idx, lateral_idx])

            max_longitudinal_accel[speed_idx, lateral_idx] = max_accel_fn(
                speed=speed_value,
                lateral_accel_required=ay_required,
                grade=grade,
                banking=banking,
            )
            max_decel = max_decel_fn(
                speed=speed_value,
                lateral_accel_required=ay_required,
                grade=grade,
                banking=banking,
            )
            min_longitudinal_accel[speed_idx, lateral_idx] = -float(max_decel)

    return max_longitudinal_accel, min_longitudinal_accel


def _validate_torch_model_api(model: VehicleModel) -> None:
    """Validate torch-specific model API requirements.

    Args:
        model: Vehicle model backend implementing ``VehicleModel``.

    Raises:
        pylapsim.utils.exceptions.ConfigurationError: If required torch API
            methods are missing on the provided model.
    """
    required_methods = (
        "lateral_accel_limit_torch",
        "max_longitudinal_accel_torch",
        "max_longitudinal_decel_torch",
    )
    missing = [name for name in required_methods if not callable(getattr(model, name, None))]
    if missing:
        msg = "Model does not implement required torch backend methods: " f"{missing}"
        raise ConfigurationError(msg)


def _compiled_torch_envelope_solver(enable_compile: bool) -> Any:
    """Return torch envelope solver callable with optional compile acceleration.

    Args:
        enable_compile: Whether ``torch.compile`` should be attempted.

    Returns:
        Callable implementing the torch envelope solve routine.
    """
    if not enable_compile:
        return _solve_performance_envelope_torch_impl

    cache_key = (TORCH_COMPILE_MODE,)
    cached = _COMPILED_TORCH_ENVELOPE_SOLVER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    torch = _require_torch()
    try:
        compiled = torch.compile(
            _solve_performance_envelope_torch_impl,
            mode=TORCH_COMPILE_MODE,
            fullgraph=False,
            dynamic=False,
        )
    except Exception:
        compiled = _solve_performance_envelope_torch_impl

    _COMPILED_TORCH_ENVELOPE_SOLVER_CACHE[cache_key] = compiled
    return compiled


def _solve_performance_envelope_torch_impl(
    model: VehicleModel,
    config: PerformanceEnvelopeConfig,
) -> PerformanceEnvelopeResult:
    """Solve performance envelope with torch tensor backend.

    Args:
        model: Vehicle model backend implementing torch-specific methods.
        config: Validated performance-envelope configuration.

    Returns:
        Computed performance-envelope result.

    Raises:
        pylapsim.utils.exceptions.ConfigurationError: If torch model methods
            return shape-mismatched outputs.
    """
    torch = _require_torch()

    physics = config.physics
    numerics = config.numerics
    runtime = config.runtime

    dtype = torch.float64
    device = runtime.torch_device
    torch_model = cast(Any, model)

    speed = torch.linspace(
        physics.speed_min,
        physics.speed_max,
        steps=int(numerics.speed_samples),
        dtype=dtype,
        device=device,
    )
    banking = torch.full_like(speed, float(physics.banking))
    grade = torch.full_like(speed, float(physics.grade))

    lateral_limit_torch = torch_model.lateral_accel_limit_torch(
        speed=speed,
        banking=banking,
    )
    if lateral_limit_torch.shape != speed.shape:
        msg = (
            "lateral_accel_limit_torch must return an array with shape "
            f"{tuple(speed.shape)}, got {tuple(lateral_limit_torch.shape)}"
        )
        raise ConfigurationError(msg)

    lateral_fraction = torch.linspace(
        -float(numerics.lateral_accel_fraction_span),
        float(numerics.lateral_accel_fraction_span),
        steps=int(numerics.lateral_accel_samples),
        dtype=dtype,
        device=device,
    )
    lateral_accel = lateral_limit_torch[:, None] * lateral_fraction[None, :]
    lateral_required = torch.abs(lateral_accel)

    speed_grid = speed[:, None].expand(-1, lateral_fraction.shape[0])
    grade_grid = grade[:, None].expand_as(speed_grid)
    banking_grid = banking[:, None].expand_as(speed_grid)

    max_accel_torch = torch_model.max_longitudinal_accel_torch(
        speed=speed_grid,
        lateral_accel_required=lateral_required,
        grade=grade_grid,
        banking=banking_grid,
    )
    max_decel_torch = torch_model.max_longitudinal_decel_torch(
        speed=speed_grid,
        lateral_accel_required=lateral_required,
        grade=grade_grid,
        banking=banking_grid,
    )

    expected_shape = lateral_required.shape
    if max_accel_torch.shape != expected_shape:
        msg = (
            "max_longitudinal_accel_torch must return an array with shape "
            f"{tuple(expected_shape)}, got {tuple(max_accel_torch.shape)}"
        )
        raise ConfigurationError(msg)
    if max_decel_torch.shape != expected_shape:
        msg = (
            "max_longitudinal_decel_torch must return an array with shape "
            f"{tuple(expected_shape)}, got {tuple(max_decel_torch.shape)}"
        )
        raise ConfigurationError(msg)

    speed_np = np.asarray(speed.detach().cpu().numpy(), dtype=float)
    lateral_limit_np = np.asarray(lateral_limit_torch.detach().cpu().numpy(), dtype=float)
    lateral_fraction_np = np.asarray(lateral_fraction.detach().cpu().numpy(), dtype=float)
    lateral_accel_np = np.asarray(lateral_accel.detach().cpu().numpy(), dtype=float)
    max_accel_np = np.asarray(max_accel_torch.detach().cpu().numpy(), dtype=float)
    min_accel_np = -np.asarray(max_decel_torch.detach().cpu().numpy(), dtype=float)

    return PerformanceEnvelopeResult(
        speed=speed_np,
        lateral_accel_limit=lateral_limit_np,
        lateral_accel_fraction=lateral_fraction_np,
        lateral_accel=lateral_accel_np,
        max_longitudinal_accel=max_accel_np,
        min_longitudinal_accel=min_accel_np,
    )


def _solve_performance_envelope_numpy(
    model: VehicleModel,
    config: PerformanceEnvelopeConfig,
) -> PerformanceEnvelopeResult:
    """Solve performance envelope using NumPy model API path.

    Args:
        model: Vehicle model backend implementing ``VehicleModel``.
        config: Validated performance-envelope configuration.

    Returns:
        Computed performance-envelope result.
    """
    physics = config.physics
    numerics = config.numerics

    speed = np.linspace(physics.speed_min, physics.speed_max, numerics.speed_samples, dtype=float)
    banking = np.full_like(speed, physics.banking, dtype=float)
    lateral_accel_limit = _lateral_accel_limit_batch(model=model, speed=speed, banking=banking)

    lateral_accel_fraction = np.linspace(
        -numerics.lateral_accel_fraction_span,
        numerics.lateral_accel_fraction_span,
        numerics.lateral_accel_samples,
        dtype=float,
    )
    lateral_accel = lateral_accel_limit[:, np.newaxis] * lateral_accel_fraction[np.newaxis, :]
    lateral_accel_required = np.abs(lateral_accel)

    max_longitudinal_accel, min_longitudinal_accel = _compute_longitudinal_limits_numpy(
        model=model,
        speed=speed,
        lateral_accel_required=lateral_accel_required,
        grade=float(physics.grade),
        banking=float(physics.banking),
    )

    return PerformanceEnvelopeResult(
        speed=speed,
        lateral_accel_limit=np.asarray(lateral_accel_limit, dtype=float),
        lateral_accel_fraction=lateral_accel_fraction,
        lateral_accel=lateral_accel,
        max_longitudinal_accel=np.asarray(max_longitudinal_accel, dtype=float),
        min_longitudinal_accel=np.asarray(min_longitudinal_accel, dtype=float),
    )


def compute_performance_envelope(
    model: VehicleModel,
    config: PerformanceEnvelopeConfig | None = None,
    *,
    physics: PerformanceEnvelopePhysics | None = None,
    numerics: PerformanceEnvelopeNumerics | None = None,
    runtime: PerformanceEnvelopeRuntime | None = None,
) -> PerformanceEnvelopeResult:
    """Compute a velocity-dependent G-G performance envelope for a vehicle model.

    Either provide a full ``config`` object, or provide ``physics``,
    ``numerics``, and/or ``runtime`` components directly.

    Args:
        model: Vehicle model backend implementing ``VehicleModel``.
        config: Optional pre-built performance-envelope configuration.
        physics: Optional physical operating-point settings (only used when
            ``config`` is not provided).
        numerics: Optional numerical envelope discretization controls (only used
            when ``config`` is not provided).
        runtime: Optional backend runtime controls (only used when ``config``
            is not provided).

    Returns:
        Computed performance-envelope result containing speed support, lateral
        limits, and longitudinal accel/decel limits over the sampled G-G domain.

    Raises:
        pylapsim.utils.exceptions.ConfigurationError: If configuration is
            invalid, incompatible options are provided, or backend-specific
            model API requirements are not met.
    """
    if config is not None and any(value is not None for value in (physics, numerics, runtime)):
        msg = "Provide either `config` or (`physics`, `numerics`, `runtime`) components, not both."
        raise ConfigurationError(msg)

    resolved_config = config or build_performance_envelope_config(
        physics=physics,
        numerics=numerics,
        runtime=runtime,
    )

    model.validate()
    resolved_config.validate()

    if resolved_config.runtime.compute_backend == "torch":
        _validate_torch_model_api(model)
        solver = _compiled_torch_envelope_solver(
            enable_compile=resolved_config.runtime.torch_compile
        )
        result = solver(model=model, config=resolved_config)
        return cast(PerformanceEnvelopeResult, result)

    return _solve_performance_envelope_numpy(model=model, config=resolved_config)
