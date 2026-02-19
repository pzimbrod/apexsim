"""Shared core algorithm for quasi-static speed-profile solving."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from apexsim.utils.constants import SMALL_EPS


@dataclass(frozen=True)
class ProfileOps:
    """Backend operation bundle consumed by the shared solver core.

    Each callable maps one mathematical primitive to backend-specific
    implementation details (NumPy or Torch).
    """

    full: Callable[[int, float], Any]
    copy: Callable[[Any], Any]
    scalar: Callable[[float], Any]
    abs: Callable[[Any], Any]
    maximum: Callable[[Any, Any], Any]
    minimum: Callable[[Any, Any], Any]
    clip: Callable[[Any, float, float], Any]
    sqrt: Callable[[Any], Any]
    where: Callable[[Any, Any, Any], Any]
    stack: Callable[[list[Any]], Any]
    cat_tail: Callable[[Any], Any]
    zeros_like: Callable[[Any], Any]
    max: Callable[[Any], Any]
    sum: Callable[[Any], Any]
    to_float: Callable[[Any], float]


@dataclass(frozen=True)
class SpeedProfileCoreResult:
    """Backend-agnostic quasi-static solver result."""

    speed: Any
    longitudinal_accel: Any
    lateral_accel: Any
    lateral_envelope_iterations: int
    lap_time: Any


@dataclass(frozen=True)
class SpeedProfileCoreInputs:
    """Static data and limits consumed by the shared profile core."""

    ds: Any
    curvature: Any
    grade: Any
    banking: Any
    max_speed: float
    min_speed: float
    start_speed: float
    lateral_envelope_max_iterations: int
    lateral_envelope_convergence_tolerance: float


@dataclass(frozen=True)
class SpeedProfileCoreCallbacks:
    """Model callback bundle consumed by the shared profile core."""

    lateral_accel_limit: Callable[..., Any]
    max_longitudinal_accel: Callable[..., Any]
    max_longitudinal_decel: Callable[..., Any]


def resolve_profile_start_speed(
    *,
    max_speed: float,
    initial_speed: float | None,
) -> float:
    """Resolve quasi-static profile entry speed from runtime settings.

    Args:
        max_speed: Runtime speed ceiling [m/s].
        initial_speed: Optional explicit start speed [m/s].

    Returns:
        Effective initial speed used at the first profile sample [m/s].
    """
    return float(max_speed if initial_speed is None else initial_speed)


def lateral_speed_limit_core(
    *,
    curvature_abs: Any,
    lateral_accel_limit: Any,
    max_speed: float,
    ops: ProfileOps,
) -> Any:
    """Compute lateral speed limit from curvature and lateral accel capacity.

    Args:
        curvature_abs: Absolute curvature signal [1/m].
        lateral_accel_limit: Lateral acceleration limit signal [m/s^2].
        max_speed: Runtime hard speed limit [m/s].
        ops: Backend operation bundle.

    Returns:
        Backend-typed lateral speed limit signal [m/s].
    """
    max_speed_value = ops.scalar(max_speed)
    safe_curvature = ops.maximum(curvature_abs, SMALL_EPS)
    safe_ay = ops.maximum(lateral_accel_limit, 0.0)
    lateral_limited_speed = ops.sqrt(safe_ay / safe_curvature)
    bounded_speed = ops.minimum(lateral_limited_speed, max_speed_value)
    return ops.where(curvature_abs > SMALL_EPS, bounded_speed, max_speed_value)


def solve_speed_profile_core(
    *,
    inputs: SpeedProfileCoreInputs,
    callbacks: SpeedProfileCoreCallbacks,
    ops: ProfileOps,
) -> SpeedProfileCoreResult:
    """Solve quasi-static profile with a backend-agnostic core algorithm.

    Args:
        inputs: Track samples and runtime limits for one solve.
        callbacks: Vehicle-model callbacks for lateral/longitudinal limits.
        ops: Backend operation bundle.

    Returns:
        Backend-typed speed profile, accelerations, iteration count, and lap time.
    """
    n = int(inputs.curvature.shape[0])
    curvature_abs = ops.abs(inputs.curvature)

    max_speed_value = ops.scalar(inputs.max_speed)
    start_speed_value = ops.scalar(inputs.start_speed)
    min_speed_squared = inputs.min_speed * inputs.min_speed

    v_lat = ops.full(n, inputs.max_speed)
    lateral_envelope_iterations = 0

    for iteration_idx in range(inputs.lateral_envelope_max_iterations):
        previous_v_lat = ops.copy(v_lat)
        ay_limit = callbacks.lateral_accel_limit(speed=v_lat, banking=inputs.banking)
        v_lat = lateral_speed_limit_core(
            curvature_abs=curvature_abs,
            lateral_accel_limit=ay_limit,
            max_speed=inputs.max_speed,
            ops=ops,
        )
        v_lat = ops.clip(v_lat, inputs.min_speed, inputs.max_speed)
        lateral_envelope_iterations = iteration_idx + 1
        max_delta_speed = ops.max(ops.abs(v_lat - previous_v_lat))
        if (
            ops.to_float(max_delta_speed)
            <= inputs.lateral_envelope_convergence_tolerance
        ):
            break

    forward_speeds: list[Any] = [
        ops.minimum(v_lat[0], ops.minimum(start_speed_value, max_speed_value))
    ]
    for idx in range(n - 1):
        speed_value = forward_speeds[-1]
        ay_required = speed_value * speed_value * curvature_abs[idx]
        net_accel = callbacks.max_longitudinal_accel(
            speed=speed_value,
            lateral_accel_required=ay_required,
            grade=inputs.grade[idx],
            banking=inputs.banking[idx],
        )
        next_speed_squared = speed_value * speed_value + 2.0 * net_accel * inputs.ds[idx]
        v_candidate = ops.sqrt(ops.maximum(next_speed_squared, min_speed_squared))
        bounded = ops.minimum(v_lat[idx + 1], v_candidate)
        bounded = ops.minimum(bounded, max_speed_value)
        forward_speeds.append(bounded)
    v_forward = ops.stack(forward_speeds)

    backward_reverse: list[Any] = [v_forward[-1]]
    for idx in range(n - 2, -1, -1):
        speed_value = backward_reverse[-1]
        ay_required = speed_value * speed_value * curvature_abs[idx + 1]
        available_decel = callbacks.max_longitudinal_decel(
            speed=speed_value,
            lateral_accel_required=ay_required,
            grade=inputs.grade[idx + 1],
            banking=inputs.banking[idx + 1],
        )
        entry_speed_squared = speed_value * speed_value + 2.0 * available_decel * inputs.ds[idx]
        v_entry = ops.sqrt(ops.maximum(entry_speed_squared, min_speed_squared))
        bounded = ops.minimum(v_forward[idx], v_entry)
        bounded = ops.minimum(bounded, v_lat[idx])
        bounded = ops.minimum(bounded, max_speed_value)
        backward_reverse.append(bounded)
    v_profile = ops.stack(backward_reverse[::-1])

    if n > 1:
        ax_core = (
            v_profile[1:] * v_profile[1:] - v_profile[:-1] * v_profile[:-1]
        ) / (2.0 * inputs.ds)
        ax = ops.cat_tail(ax_core)
    else:
        ax = ops.zeros_like(v_profile)

    ay = v_profile * v_profile * inputs.curvature
    segment_speed_avg = ops.maximum(
        0.5 * (v_profile[:-1] + v_profile[1:]),
        SMALL_EPS,
    )
    lap_time = ops.sum(inputs.ds / segment_speed_avg)

    return SpeedProfileCoreResult(
        speed=v_profile,
        longitudinal_accel=ax,
        lateral_accel=ay,
        lateral_envelope_iterations=lateral_envelope_iterations,
        lap_time=lap_time,
    )
