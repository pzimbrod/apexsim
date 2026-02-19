"""Shared helpers for transient PID-based solver paths."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from apexsim.utils.constants import SMALL_EPS


def resolve_initial_speed(*, max_speed: float, initial_speed: float | None) -> float:
    """Resolve transient start speed from runtime settings.

    Args:
        max_speed: Runtime hard speed limit [m/s].
        initial_speed: Optional explicit start speed [m/s].

    Returns:
        Effective initial speed used to initialize transient rollout [m/s].
    """
    return float(max_speed if initial_speed is None else initial_speed)


def clamp_integral_scalar(value: float, limit: float) -> float:
    """Clamp scalar PID integrator state to a symmetric finite interval.

    Args:
        value: Current scalar integrator state.
        limit: Symmetric absolute clamp limit.

    Returns:
        Clamped scalar integrator state.
    """
    return float(np.clip(value, -limit, limit))


def clamp_integral_value(
    *,
    value: Any,
    limit: float,
    clamp_fn: Callable[[Any, Any, Any], Any],
    as_value_fn: Callable[[float], Any],
) -> Any:
    """Clamp generic PID integrator state using backend-specific clamp ops.

    Args:
        value: Integrator state value in backend-native type.
        limit: Symmetric absolute clamp limit.
        clamp_fn: Backend clamp callable ``clamp_fn(value, low, high)``.
        as_value_fn: Cast helper from float to backend-native scalar.

    Returns:
        Clamped integrator state in backend-native type.
    """
    limit_value = as_value_fn(limit)
    return clamp_fn(value, -limit_value, limit_value)


def pid_error_derivative(
    *,
    error: Any,
    previous_error: Any,
    dt: Any,
    denominator_floor: float,
    clamp_min_fn: Callable[[Any, Any], Any],
    as_value_fn: Callable[[float], Any],
) -> Any:
    """Return stable PID error derivative with backend-specific min clamp.

    Args:
        error: Current control error.
        previous_error: Previous-step control error.
        dt: Current integration step [s].
        denominator_floor: Minimum denominator floor to avoid division by zero.
        clamp_min_fn: Backend clamp-min callable ``clamp_min_fn(value, min)``.
        as_value_fn: Cast helper from float to backend-native scalar.

    Returns:
        Backend-native error derivative estimate.
    """
    safe_dt = clamp_min_fn(dt, as_value_fn(denominator_floor))
    return (error - previous_error) / safe_dt


def bounded_pid_command(
    *,
    reference_feedforward: Any,
    kp: Any,
    ki: Any,
    kd: Any,
    error: Any,
    error_integral: Any,
    error_derivative: Any,
    lower_limit: Any,
    upper_limit: Any,
    clamp_fn: Callable[[Any, Any, Any], Any],
) -> Any:
    """Return bounded PID command with additive feedforward term.

    Args:
        reference_feedforward: Feedforward base command.
        kp: Proportional gain.
        ki: Integral gain.
        kd: Derivative gain.
        error: Current control error.
        error_integral: Clamped integral error state.
        error_derivative: Error derivative estimate.
        lower_limit: Lower command saturation bound.
        upper_limit: Upper command saturation bound.
        clamp_fn: Backend clamp callable ``clamp_fn(value, low, high)``.

    Returns:
        Saturated PID command in backend-native type.
    """
    commanded = (
        reference_feedforward
        + kp * error
        + ki * error_integral
        + kd * error_derivative
    )
    return clamp_fn(commanded, lower_limit, upper_limit)


def segment_time_partition(
    *,
    segment_length: float,
    speed: float,
    min_time_step: float,
    max_time_step: float,
    max_integration_step: float | None = None,
) -> tuple[float, float, int]:
    """Split one arc-length segment time into stable scalar sub-steps.

    Args:
        segment_length: Segment arc length [m].
        speed: Representative segment speed [m/s].
        min_time_step: Lower bound for per-segment travel time [s].
        max_time_step: Upper bound for per-segment travel time [s].
        max_integration_step: Optional upper bound for integration substep [s].

    Returns:
        Tuple of total segment time [s], integration substep [s], and substep
        count.
    """
    raw_time = segment_length / max(abs(speed), SMALL_EPS)
    total_time = float(np.clip(raw_time, min_time_step, max_time_step))
    step_limit = max_time_step
    if max_integration_step is not None:
        step_limit = min(step_limit, max_integration_step)
    step_limit = max(step_limit, SMALL_EPS)
    substeps = max(1, int(np.ceil(total_time / step_limit)))
    integration_step = total_time / substeps
    return float(total_time), float(integration_step), int(substeps)


def segment_time_partition_torch(
    *,
    torch: Any,
    segment_length: Any,
    speed: Any,
    min_time_step: float,
    max_time_step: float,
    max_integration_step: float | None = None,
) -> tuple[Any, Any, int]:
    """Split one arc-length segment time into stable torch sub-steps.

    Args:
        torch: Imported torch module.
        segment_length: Segment arc length tensor/scalar [m].
        speed: Representative segment speed tensor/scalar [m/s].
        min_time_step: Lower bound for per-segment travel time [s].
        max_time_step: Upper bound for per-segment travel time [s].
        max_integration_step: Optional upper bound for integration substep [s].

    Returns:
        Tuple of total segment time tensor [s], integration substep tensor [s],
        and integer substep count.
    """
    safe_speed = torch.clamp(torch.abs(speed), min=SMALL_EPS)
    raw_time = segment_length / safe_speed
    total_time = torch.clamp(raw_time, min=min_time_step, max=max_time_step)

    step_limit = max_time_step
    if max_integration_step is not None:
        step_limit = min(step_limit, max_integration_step)
    step_limit = max(step_limit, SMALL_EPS)

    total_seconds = float(total_time.detach().cpu().item())
    substeps = max(1, int(np.ceil(total_seconds / step_limit)))
    integration_step = total_time / substeps
    return total_time, integration_step, int(substeps)
