"""Torch-backed speed-profile solver for accelerated and differentiable studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from apexsim.simulation._profile_core import (
    ProfileOps,
    SpeedProfileCoreCallbacks,
    SpeedProfileCoreInputs,
    resolve_profile_start_speed,
    solve_speed_profile_core,
)
from apexsim.simulation.config import SimulationConfig
from apexsim.simulation.profile import SpeedProfileResult
from apexsim.track.models import TrackData
from apexsim.utils.exceptions import ConfigurationError

TORCH_COMPILE_MODE = "reduce-overhead"
_COMPILED_SOLVER_CACHE: dict[tuple[str], Any] = {}


class TorchSpeedModel(Protocol):
    """Protocol for vehicle models that provide torch-native speed constraints."""

    def validate(self) -> None:
        """Validate model parameters before solver execution."""

    def lateral_accel_limit_torch(self, speed: Any, banking: Any) -> Any:
        """Return lateral-acceleration limits for tensor-valued operating points.

        Args:
            speed: Speed tensor [m/s].
            banking: Banking-angle tensor [rad].

        Returns:
            Lateral acceleration limit tensor [m/s^2].
        """

    def max_longitudinal_accel_torch(
        self,
        speed: Any,
        lateral_accel_required: Any,
        grade: Any,
        banking: Any,
    ) -> Any:
        """Return net forward acceleration limits along path tangent.

        Args:
            speed: Speed tensor [m/s].
            lateral_accel_required: Required lateral acceleration tensor [m/s^2].
            grade: Track-grade tensor ``dz/ds``.
            banking: Banking-angle tensor [rad].

        Returns:
            Net forward-acceleration limit tensor [m/s^2].
        """

    def max_longitudinal_decel_torch(
        self,
        speed: Any,
        lateral_accel_required: Any,
        grade: Any,
        banking: Any,
    ) -> Any:
        """Return available deceleration magnitudes along path tangent.

        Args:
            speed: Speed tensor [m/s].
            lateral_accel_required: Required lateral acceleration tensor [m/s^2].
            grade: Track-grade tensor ``dz/ds``.
            banking: Banking-angle tensor [rad].

        Returns:
            Available deceleration-magnitude tensor [m/s^2].
        """


@dataclass(frozen=True)
class TorchSpeedProfileResult:
    """Differentiable torch speed-profile output tensors.

    Args:
        speed: Converged speed trace along track arc length [m/s].
        longitudinal_accel: Net longitudinal acceleration trace [m/s^2].
        lateral_accel: Lateral acceleration trace [m/s^2].
        lateral_envelope_iterations: Number of fixed-point iterations used for
            lateral envelope convergence.
        lap_time: Integrated lap time over one track traversal [s].
    """

    speed: Any
    longitudinal_accel: Any
    lateral_accel: Any
    lateral_envelope_iterations: int
    lap_time: Any

    def to_numpy(self) -> SpeedProfileResult:
        """Convert torch tensor result to ``SpeedProfileResult``.

        Returns:
            NumPy-based speed-profile result detached from autograd graph.
        """
        return SpeedProfileResult(
            speed=np.asarray(self.speed.detach().cpu().numpy(), dtype=float),
            longitudinal_accel=np.asarray(
                self.longitudinal_accel.detach().cpu().numpy(),
                dtype=float,
            ),
            lateral_accel=np.asarray(self.lateral_accel.detach().cpu().numpy(), dtype=float),
            lateral_envelope_iterations=int(self.lateral_envelope_iterations),
            lap_time=float(self.lap_time.detach().cpu().item()),
        )


def _build_torch_profile_ops(*, torch: Any, dtype: Any, device: str) -> ProfileOps:
    """Build torch operation bundle for the shared profile core.

    Args:
        torch: Imported torch module.
        dtype: Tensor dtype used for solver computations.
        device: Target torch device string.

    Returns:
        ``ProfileOps`` instance backed by torch tensor primitives.
    """

    def _to_value(value: Any) -> Any:
        return torch.as_tensor(value, dtype=dtype, device=device)

    return ProfileOps(
        full=lambda size, value: torch.full((size,), value, dtype=dtype, device=device),
        copy=lambda value: value.clone(),
        scalar=lambda value: _to_value(value),
        abs=lambda value: torch.abs(value),
        maximum=lambda left, right: torch.maximum(left, _to_value(right)),
        minimum=lambda left, right: torch.minimum(left, _to_value(right)),
        clip=lambda value, low, high: torch.clamp(value, min=low, max=high),
        sqrt=lambda value: torch.sqrt(value),
        where=lambda condition, left, right: torch.where(condition, left, right),
        stack=lambda values: torch.stack(values),
        cat_tail=lambda core: torch.cat((core, core[-1:])),
        zeros_like=lambda ref: torch.zeros_like(ref),
        max=lambda value: torch.max(value),
        sum=lambda value: torch.sum(value),
        to_float=lambda value: float(value.detach().cpu().item()),
    )


def _require_torch() -> Any:
    """Import torch lazily and fail with a configuration-level message.

    Returns:
        Imported `torch` module.

    Raises:
        apexsim.utils.exceptions.ConfigurationError: If torch is not installed.
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


def _compiled_solver(
    enable_compile: bool,
) -> Any:
    """Return torch solver callable with optional compile acceleration.

    Args:
        enable_compile: Whether ``torch.compile`` should be attempted.

    Returns:
        Callable implementing the torch profile solve routine.
    """
    if not enable_compile:
        return _solve_speed_profile_torch_impl

    cache_key = (TORCH_COMPILE_MODE,)
    cached = _COMPILED_SOLVER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    torch = _require_torch()
    try:
        compiled = torch.compile(
            _solve_speed_profile_torch_impl,
            mode=TORCH_COMPILE_MODE,
            fullgraph=False,
            dynamic=False,
        )
    except Exception:
        compiled = _solve_speed_profile_torch_impl

    _COMPILED_SOLVER_CACHE[cache_key] = compiled
    return compiled


def _validate_torch_solver_inputs(
    track: TrackData,
    model: TorchSpeedModel,
    config: SimulationConfig,
) -> None:
    """Validate common preconditions for torch-backed profile solvers.

    Args:
        track: Track geometry and derived arc-length-domain quantities.
        model: Vehicle model implementing torch-native speed-limit methods.
        config: Solver runtime and numerical controls.

    Raises:
        apexsim.utils.exceptions.ConfigurationError: If backend selection is
            incompatible with the provided model or runtime settings.
    """
    if config.runtime.compute_backend != "torch":
        msg = "torch profile solvers require runtime.compute_backend='torch'"
        raise ConfigurationError(msg)

    track.validate()
    model.validate()
    config.validate()

    required_methods = (
        "lateral_accel_limit_torch",
        "max_longitudinal_accel_torch",
        "max_longitudinal_decel_torch",
    )
    missing = [name for name in required_methods if not hasattr(model, name)]
    if missing:
        msg = "Model does not implement required torch backend methods: " f"{missing}"
        raise ConfigurationError(msg)


def _solve_speed_profile_torch_impl(
    track: TrackData,
    model: TorchSpeedModel,
    config: SimulationConfig,
) -> TorchSpeedProfileResult:
    """Solve speed profile on torch backend without validation checks.

    Args:
        track: Track geometry and derived arc-length-domain quantities.
        model: Vehicle model implementing torch-native speed-limit methods.
        config: Solver runtime and numerical controls.

    Returns:
        Converged speed profile and integrated lap metrics as torch tensors.
    """
    torch = _require_torch()

    device = config.runtime.torch_device
    dtype = torch.float64

    arc_length = torch.as_tensor(track.arc_length, dtype=dtype, device=device)
    inputs = SpeedProfileCoreInputs(
        ds=arc_length[1:] - arc_length[:-1],
        curvature=torch.as_tensor(track.curvature, dtype=dtype, device=device),
        grade=torch.as_tensor(track.grade, dtype=dtype, device=device),
        banking=torch.as_tensor(track.banking, dtype=dtype, device=device),
        max_speed=float(config.runtime.max_speed),
        min_speed=float(config.numerics.min_speed),
        start_speed=resolve_profile_start_speed(
            max_speed=float(config.runtime.max_speed),
            initial_speed=config.runtime.initial_speed,
        ),
        lateral_envelope_max_iterations=int(config.numerics.lateral_envelope_max_iterations),
        lateral_envelope_convergence_tolerance=float(
            config.numerics.lateral_envelope_convergence_tolerance
        ),
    )
    callbacks = SpeedProfileCoreCallbacks(
        lateral_accel_limit=model.lateral_accel_limit_torch,
        max_longitudinal_accel=model.max_longitudinal_accel_torch,
        max_longitudinal_decel=model.max_longitudinal_decel_torch,
    )
    core = solve_speed_profile_core(
        inputs=inputs,
        callbacks=callbacks,
        ops=_build_torch_profile_ops(torch=torch, dtype=dtype, device=device),
    )

    return TorchSpeedProfileResult(
        speed=core.speed,
        longitudinal_accel=core.longitudinal_accel,
        lateral_accel=core.lateral_accel,
        lateral_envelope_iterations=core.lateral_envelope_iterations,
        lap_time=core.lap_time,
    )


def solve_speed_profile_torch(
    track: TrackData,
    model: TorchSpeedModel,
    config: SimulationConfig,
) -> TorchSpeedProfileResult:
    """Solve lap speed profile with a torch-backed differentiable backend.

    Args:
        track: Track geometry and derived arc-length-domain quantities.
        model: Vehicle model implementing torch-native speed-limit methods.
        config: Solver runtime and numerical controls.

    Returns:
        Differentiable tensor-valued speed-profile result.

    Raises:
        apexsim.utils.exceptions.ConfigurationError: If backend selection is
            incompatible with the provided model or runtime settings.
    """
    _validate_torch_solver_inputs(track=track, model=model, config=config)

    if config.runtime.torch_compile:
        msg = (
            "solve_speed_profile_torch does not support torch_compile=True. "
            "Disable torch_compile for the torch simulation backend."
        )
        raise ConfigurationError(msg)

    return _solve_speed_profile_torch_impl(track=track, model=model, config=config)
