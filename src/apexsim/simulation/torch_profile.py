"""Torch-backed speed-profile solver for accelerated and differentiable studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from apexsim.simulation.config import SimulationConfig
from apexsim.simulation.profile import SpeedProfileResult
from apexsim.track.models import TrackData
from apexsim.utils.constants import SMALL_EPS
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


def _lateral_speed_limit_torch(
    torch: Any,
    curvature_abs: Any,
    lateral_accel_limit: Any,
    max_speed: float,
) -> Any:
    """Compute lateral-speed envelope in torch.

    Args:
        torch: Imported torch module.
        curvature_abs: Absolute curvature tensor [1/m].
        lateral_accel_limit: Lateral acceleration limit tensor [m/s^2].
        max_speed: Runtime maximum speed [m/s].

    Returns:
        Speed-limit tensor from lateral envelope constraints [m/s].
    """
    max_speed_tensor = torch.full_like(curvature_abs, float(max_speed))

    safe_denominator = torch.clamp(curvature_abs, min=SMALL_EPS)
    safe_numerator = torch.clamp(lateral_accel_limit, min=0.0)
    lateral_limited_speed = torch.sqrt(safe_numerator / safe_denominator)

    bounded_speed = torch.minimum(lateral_limited_speed, max_speed_tensor)
    return torch.where(curvature_abs > SMALL_EPS, bounded_speed, max_speed_tensor)


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
    curvature = torch.as_tensor(track.curvature, dtype=dtype, device=device)
    curvature_abs = torch.abs(curvature)
    grade = torch.as_tensor(track.grade, dtype=dtype, device=device)
    banking = torch.as_tensor(track.banking, dtype=dtype, device=device)

    n = int(arc_length.numel())
    ds = arc_length[1:] - arc_length[:-1]

    min_speed = float(config.numerics.min_speed)
    max_speed = float(config.runtime.max_speed)
    start_speed = (
        max_speed
        if config.runtime.initial_speed is None
        else float(config.runtime.initial_speed)
    )

    max_speed_tensor = torch.as_tensor(max_speed, dtype=dtype, device=device)
    start_speed_tensor = torch.as_tensor(start_speed, dtype=dtype, device=device)
    min_speed_squared = min_speed * min_speed

    lateral_accel_limit_torch = model.lateral_accel_limit_torch
    max_longitudinal_accel_torch = model.max_longitudinal_accel_torch
    max_longitudinal_decel_torch = model.max_longitudinal_decel_torch

    v_lat = torch.full((n,), max_speed, dtype=dtype, device=device)
    lateral_envelope_iterations = 0

    for iteration_idx in range(config.numerics.lateral_envelope_max_iterations):
        previous_v_lat = v_lat
        ay_limit = lateral_accel_limit_torch(speed=v_lat, banking=banking)

        v_lat = _lateral_speed_limit_torch(
            torch=torch,
            curvature_abs=curvature_abs,
            lateral_accel_limit=ay_limit,
            max_speed=max_speed,
        )
        v_lat = torch.clamp(v_lat, min=min_speed, max=max_speed)

        lateral_envelope_iterations = iteration_idx + 1
        max_delta_speed = torch.max(torch.abs(v_lat - previous_v_lat))
        if float(max_delta_speed.item()) <= config.numerics.lateral_envelope_convergence_tolerance:
            break

    forward_speeds: list[Any] = [
        torch.minimum(v_lat[0], torch.minimum(start_speed_tensor, max_speed_tensor))
    ]
    for idx in range(n - 1):
        speed_value = forward_speeds[-1]
        ay_required = speed_value * speed_value * curvature_abs[idx]
        net_accel = max_longitudinal_accel_torch(
            speed=speed_value,
            lateral_accel_required=ay_required,
            grade=grade[idx],
            banking=banking[idx],
        )

        next_speed_squared = speed_value * speed_value + 2.0 * net_accel * ds[idx]
        v_candidate = torch.sqrt(torch.clamp(next_speed_squared, min=min_speed_squared))

        bounded = torch.minimum(v_lat[idx + 1], v_candidate)
        bounded = torch.minimum(bounded, max_speed_tensor)
        forward_speeds.append(bounded)

    v_forward = torch.stack(forward_speeds)

    backward_reverse: list[Any] = [v_forward[-1]]
    for idx in range(n - 2, -1, -1):
        speed_value = backward_reverse[-1]
        ay_required = speed_value * speed_value * curvature_abs[idx + 1]
        available_decel = max_longitudinal_decel_torch(
            speed=speed_value,
            lateral_accel_required=ay_required,
            grade=grade[idx + 1],
            banking=banking[idx + 1],
        )

        entry_speed_squared = speed_value * speed_value + 2.0 * available_decel * ds[idx]
        v_entry = torch.sqrt(torch.clamp(entry_speed_squared, min=min_speed_squared))

        bounded = torch.minimum(v_forward[idx], v_entry)
        bounded = torch.minimum(bounded, v_lat[idx])
        bounded = torch.minimum(bounded, max_speed_tensor)
        backward_reverse.append(bounded)

    v_profile = torch.stack(backward_reverse[::-1])

    if n > 1:
        ax_core = (v_profile[1:] * v_profile[1:] - v_profile[:-1] * v_profile[:-1]) / (2.0 * ds)
        ax = torch.cat((ax_core, ax_core[-1:]))
    else:
        ax = torch.zeros_like(v_profile)

    ay = v_profile * v_profile * curvature

    segment_speed_avg = torch.clamp(0.5 * (v_profile[:-1] + v_profile[1:]), min=SMALL_EPS)
    lap_time = torch.sum(ds / segment_speed_avg)

    return TorchSpeedProfileResult(
        speed=v_profile,
        longitudinal_accel=ax,
        lateral_accel=ay,
        lateral_envelope_iterations=lateral_envelope_iterations,
        lap_time=lap_time,
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
