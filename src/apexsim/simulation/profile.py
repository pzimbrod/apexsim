"""Forward/backward speed profile solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from apexsim.simulation.config import SimulationConfig
from apexsim.simulation.model_api import VehicleModel
from apexsim.track.models import TrackData
from apexsim.utils.constants import SMALL_EPS
from apexsim.utils.exceptions import ConfigurationError


@dataclass(frozen=True)
class SpeedProfileResult:
    """Speed profile along track arc length.

    `lateral_envelope_iterations` reports how many fixed-point iterations were
    required to converge the lateral speed envelope.

    Args:
        speed: Converged speed trace along track arc length [m/s].
        longitudinal_accel: Net longitudinal acceleration trace [m/s^2].
        lateral_accel: Lateral acceleration trace [m/s^2].
        lateral_envelope_iterations: Number of fixed-point iterations used for
            lateral envelope convergence.
        lap_time: Integrated lap time over one track traversal [s].
    """

    speed: np.ndarray
    longitudinal_accel: np.ndarray
    lateral_accel: np.ndarray
    lateral_envelope_iterations: int
    lap_time: float


def _segment_dt(segment_length: float, start_speed: float, end_speed: float) -> float:
    """Compute segment travel time from adjacent speed samples.

    Args:
        segment_length: Segment length [m].
        start_speed: Segment entry speed [m/s].
        end_speed: Segment exit speed [m/s].

    Returns:
        Segment traversal time [s] using trapezoidal average speed.
    """
    v_avg = max(0.5 * (start_speed + end_speed), SMALL_EPS)
    return segment_length / v_avg


def _lateral_accel_limit_batch(
    model: VehicleModel,
    speed: np.ndarray,
    banking: np.ndarray,
) -> np.ndarray:
    """Evaluate lateral acceleration limits over all track samples.

    The solver uses an optional vectorized model API when available. If a model
    does not provide `lateral_accel_limit_batch`, it falls back to scalar calls.

    Args:
        model: Vehicle model backend.
        speed: Speed samples [m/s].
        banking: Banking-angle samples [rad].

    Returns:
        Lateral acceleration limit samples [m/s^2].

    Raises:
        apexsim.utils.exceptions.ConfigurationError: If a vectorized model
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


def _lateral_speed_limit_batch(
    curvature: np.ndarray,
    lateral_accel_limit: np.ndarray,
    max_speed: float,
) -> np.ndarray:
    """Compute speed limit from lateral envelope constraints for all samples.

    Args:
        curvature: Curvature samples [1/m].
        lateral_accel_limit: Lateral acceleration capability samples [m/s^2].
        max_speed: Global hard speed cap [m/s].

    Returns:
        Lateral-envelope speed limits [m/s].
    """
    kappa_abs = np.abs(curvature)
    safe_kappa = np.maximum(kappa_abs, SMALL_EPS)
    safe_ay = np.maximum(lateral_accel_limit, 0.0)

    lateral_limited_speed = np.sqrt(safe_ay / safe_kappa)
    bounded_speed = np.minimum(lateral_limited_speed, max_speed)
    return np.where(kappa_abs > SMALL_EPS, bounded_speed, max_speed)


def solve_speed_profile(
    track: TrackData,
    model: VehicleModel,
    config: SimulationConfig,
) -> SpeedProfileResult:
    """Solve lap speed profile with lateral and longitudinal constraints.

    The algorithm is a quasi-steady forward/backward solver in arc-length domain:
    1. Solve a fixed-point lateral speed envelope `v_lat[s]` via the model API.
    2. Forward pass enforces acceleration feasibility.
    3. Backward pass enforces braking feasibility.
    4. Integrate segment times to obtain lap time.

    See `docs/SOLVER.md` for the full mathematical derivation.

    Args:
        track: Track geometry and derived arc-length-domain quantities.
        model: Vehicle-model backend implementing ``VehicleModel``.
        config: Global simulation limits and solver iteration settings.

    Returns:
        Converged speed profile, derived accelerations, envelope iteration count,
        and integrated lap time.

    Raises:
        apexsim.utils.exceptions.TrackDataError: If ``track`` is invalid.
        apexsim.utils.exceptions.ConfigurationError: If model or solver
            configuration is invalid.
    """
    track.validate()
    model.validate()
    config.validate()

    n = track.arc_length.size
    ds = np.diff(track.arc_length)

    max_speed = config.runtime.max_speed
    start_speed = (
        max_speed
        if config.runtime.initial_speed is None
        else float(config.runtime.initial_speed)
    )
    min_speed = config.numerics.min_speed
    min_speed_squared = min_speed * min_speed
    banking = track.banking
    grade = track.grade
    curvature = track.curvature
    curvature_abs = np.abs(curvature)

    v_lat = np.full(n, max_speed, dtype=float)
    lateral_envelope_iterations = 0
    for iteration_idx in range(config.numerics.lateral_envelope_max_iterations):
        previous_v_lat = np.copy(v_lat)

        ay_limit = _lateral_accel_limit_batch(model=model, speed=v_lat, banking=banking)
        v_lat = _lateral_speed_limit_batch(
            curvature=curvature,
            lateral_accel_limit=ay_limit,
            max_speed=max_speed,
        )
        v_lat = np.clip(v_lat, min_speed, max_speed)

        lateral_envelope_iterations = iteration_idx + 1
        max_delta_speed = float(np.max(np.abs(v_lat - previous_v_lat)))
        if max_delta_speed <= config.numerics.lateral_envelope_convergence_tolerance:
            break

    max_longitudinal_accel = model.max_longitudinal_accel
    max_longitudinal_decel = model.max_longitudinal_decel

    v_forward = np.copy(v_lat)
    v_forward[0] = min(v_forward[0], start_speed, max_speed)
    for idx in range(n - 1):
        speed_value = float(v_forward[idx])
        ay_required = speed_value * speed_value * float(curvature_abs[idx])
        net_accel = max_longitudinal_accel(
            speed=speed_value,
            lateral_accel_required=ay_required,
            grade=float(grade[idx]),
            banking=float(banking[idx]),
        )

        next_speed_squared = speed_value * speed_value + 2.0 * net_accel * float(ds[idx])
        v_candidate = float(np.sqrt(max(next_speed_squared, min_speed_squared)))
        v_forward[idx + 1] = min(v_forward[idx + 1], v_candidate, v_lat[idx + 1], max_speed)

    v_profile = np.copy(v_forward)
    for idx in range(n - 2, -1, -1):
        speed_value = float(v_profile[idx + 1])
        ay_required = speed_value * speed_value * float(curvature_abs[idx + 1])
        available_decel = max_longitudinal_decel(
            speed=speed_value,
            lateral_accel_required=ay_required,
            grade=float(grade[idx + 1]),
            banking=float(banking[idx + 1]),
        )

        entry_speed_squared = speed_value * speed_value + 2.0 * available_decel * float(ds[idx])
        v_entry = float(np.sqrt(max(entry_speed_squared, min_speed_squared)))
        v_profile[idx] = min(v_profile[idx], v_entry, v_lat[idx], max_speed)

    ax = np.zeros(n, dtype=float)
    if n > 1:
        ax[:-1] = (v_profile[1:] * v_profile[1:] - v_profile[:-1] * v_profile[:-1]) / (2.0 * ds)
        ax[-1] = ax[-2]

    ay = v_profile * v_profile * curvature

    segment_speed_avg = np.maximum(0.5 * (v_profile[:-1] + v_profile[1:]), SMALL_EPS)
    lap_time = float(np.sum(ds / segment_speed_avg))

    return SpeedProfileResult(
        speed=v_profile,
        longitudinal_accel=ax,
        lateral_accel=ay,
        lateral_envelope_iterations=lateral_envelope_iterations,
        lap_time=lap_time,
    )
