"""Numba-backed speed-profile solver for CPU-optimized point-mass studies."""

from __future__ import annotations

import math
from typing import Any, Protocol, cast

import numpy as np

from pylapsim.simulation.config import SimulationConfig
from pylapsim.simulation.profile import SpeedProfileResult
from pylapsim.track.models import TrackData
from pylapsim.utils.constants import GRAVITY, SMALL_EPS
from pylapsim.utils.exceptions import ConfigurationError

NumbaProfileParameters = tuple[float, float, float, float, float, float]
_COMPILED_NUMBA_KERNEL: Any | None = None


class NumbaSpeedModel(Protocol):
    """Protocol for vehicle models that provide numba-compatible constants."""

    def validate(self) -> None:
        """Validate model parameters before solver execution."""

    def numba_speed_profile_parameters(self) -> NumbaProfileParameters:
        """Return scalar model parameters required by the numba kernel.

        Returns:
            Tuple ``(mass, downforce_scale, drag_scale, friction_coefficient,
            max_drive_accel, max_brake_accel)``.
        """


def _require_numba() -> Any:
    """Import numba lazily and fail with a configuration-level message.

    Returns:
        Imported ``numba`` module.

    Raises:
        pylapsim.utils.exceptions.ConfigurationError: If numba is not installed.
    """
    try:
        import numba  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        msg = (
            "Numba backend requested but numba is not installed. "
            "Install with `pip install -e '.[numba]'`."
        )
        raise ConfigurationError(msg) from exc
    return numba


def _point_mass_speed_profile_kernel(
    arc_length: np.ndarray,
    curvature: np.ndarray,
    grade: np.ndarray,
    banking: np.ndarray,
    mass: float,
    downforce_scale: float,
    drag_scale: float,
    friction_coefficient: float,
    max_drive_accel: float,
    max_brake_accel: float,
    max_speed: float,
    min_speed: float,
    lateral_iterations_limit: int,
    lateral_convergence_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Solve the point-mass profile in arc-length domain.

    This function is compiled with ``numba.njit`` at runtime.

    Args:
        arc_length: Cumulative arc-length samples [m].
        curvature: Signed curvature samples [1/m].
        grade: Track grade samples ``dz/ds`` [-].
        banking: Banking-angle samples [rad].
        mass: Vehicle mass [kg].
        downforce_scale: Quadratic downforce coefficient [N/(m/s)^2].
        drag_scale: Quadratic drag-force coefficient [N/(m/s)^2].
        friction_coefficient: Isotropic tire-road friction coefficient (-).
        max_drive_accel: Maximum drive acceleration limit [m/s^2].
        max_brake_accel: Maximum braking deceleration magnitude [m/s^2].
        max_speed: Global speed cap [m/s].
        min_speed: Numerical speed floor [m/s].
        lateral_iterations_limit: Maximum lateral-envelope fixed-point iterations.
        lateral_convergence_tolerance: Convergence tolerance for lateral
            envelope fixed-point update [m/s].

    Returns:
        Tuple ``(speed, longitudinal_accel, lateral_accel,
        lateral_iterations, lap_time)``.
    """
    n = arc_length.shape[0]
    ds = np.empty(n - 1, dtype=np.float64)
    for idx in range(n - 1):
        ds[idx] = arc_length[idx + 1] - arc_length[idx]

    curvature_abs = np.empty(n, dtype=np.float64)
    for idx in range(n):
        curvature_abs[idx] = abs(curvature[idx])

    min_speed_squared = min_speed * min_speed

    v_lat = np.empty(n, dtype=np.float64)
    for idx in range(n):
        v_lat[idx] = max_speed

    lateral_iterations = 0
    for iteration_idx in range(lateral_iterations_limit):
        max_delta_speed = 0.0
        for idx in range(n):
            speed_non_negative = max(v_lat[idx], 0.0)
            speed_squared = speed_non_negative * speed_non_negative

            downforce = downforce_scale * speed_squared
            normal_accel = max(GRAVITY + downforce / mass, SMALL_EPS)
            ay_tire = friction_coefficient * normal_accel
            ay_limit = max(ay_tire + GRAVITY * math.sin(banking[idx]), SMALL_EPS)

            kappa = curvature_abs[idx]
            if kappa > SMALL_EPS:
                v_limit = math.sqrt(max(ay_limit, 0.0) / max(kappa, SMALL_EPS))
                v_next = min(v_limit, max_speed)
            else:
                v_next = max_speed
            v_next = max(v_next, min_speed)

            delta_speed = abs(v_next - v_lat[idx])
            if delta_speed > max_delta_speed:
                max_delta_speed = delta_speed
            v_lat[idx] = v_next

        lateral_iterations = iteration_idx + 1
        if max_delta_speed <= lateral_convergence_tolerance:
            break

    v_forward = v_lat.copy()
    v_forward[0] = min(v_forward[0], max_speed)

    for idx in range(n - 1):
        speed_value = v_forward[idx]
        speed_non_negative = max(speed_value, 0.0)
        speed_squared = speed_non_negative * speed_non_negative

        downforce = downforce_scale * speed_squared
        normal_accel = max(GRAVITY + downforce / mass, SMALL_EPS)
        ay_tire = friction_coefficient * normal_accel
        ay_limit = max(ay_tire + GRAVITY * math.sin(banking[idx]), SMALL_EPS)

        ay_required = speed_value * speed_value * curvature_abs[idx]
        usage = min(abs(ay_required) / ay_limit, 1.0)
        circle_scale = math.sqrt(max(0.0, 1.0 - usage * usage))

        tire_limit = min(ay_tire, max_drive_accel)
        tire_accel = tire_limit * circle_scale

        drag_accel = drag_scale * speed_squared / mass
        grade_accel = GRAVITY * grade[idx]
        net_accel = tire_accel - drag_accel - grade_accel

        next_speed_squared = speed_value * speed_value + 2.0 * net_accel * ds[idx]
        v_candidate = math.sqrt(max(next_speed_squared, min_speed_squared))

        bounded = min(v_forward[idx + 1], v_candidate)
        bounded = min(bounded, v_lat[idx + 1])
        bounded = min(bounded, max_speed)
        v_forward[idx + 1] = bounded

    v_profile = v_forward.copy()
    for idx in range(n - 2, -1, -1):
        speed_value = v_profile[idx + 1]
        speed_non_negative = max(speed_value, 0.0)
        speed_squared = speed_non_negative * speed_non_negative

        downforce = downforce_scale * speed_squared
        normal_accel = max(GRAVITY + downforce / mass, SMALL_EPS)
        ay_tire = friction_coefficient * normal_accel
        ay_limit = max(ay_tire + GRAVITY * math.sin(banking[idx + 1]), SMALL_EPS)

        ay_required = speed_value * speed_value * curvature_abs[idx + 1]
        usage = min(abs(ay_required) / ay_limit, 1.0)
        circle_scale = math.sqrt(max(0.0, 1.0 - usage * usage))

        tire_limit = min(ay_tire, max_brake_accel)
        tire_brake = tire_limit * circle_scale

        drag_accel = drag_scale * speed_squared / mass
        grade_accel = GRAVITY * grade[idx + 1]
        available_decel = max(tire_brake + drag_accel + grade_accel, 0.0)

        entry_speed_squared = speed_value * speed_value + 2.0 * available_decel * ds[idx]
        v_entry = math.sqrt(max(entry_speed_squared, min_speed_squared))

        bounded = min(v_profile[idx], v_entry)
        bounded = min(bounded, v_lat[idx])
        bounded = min(bounded, max_speed)
        v_profile[idx] = bounded

    ax = np.zeros(n, dtype=np.float64)
    if n > 1:
        for idx in range(n - 1):
            ax[idx] = (
                v_profile[idx + 1] * v_profile[idx + 1] - v_profile[idx] * v_profile[idx]
            ) / (2.0 * ds[idx])
        ax[-1] = ax[-2]

    ay = np.empty(n, dtype=np.float64)
    for idx in range(n):
        ay[idx] = v_profile[idx] * v_profile[idx] * curvature[idx]

    lap_time = 0.0
    for idx in range(n - 1):
        segment_average_speed = max(0.5 * (v_profile[idx] + v_profile[idx + 1]), SMALL_EPS)
        lap_time += ds[idx] / segment_average_speed

    return v_profile, ax, ay, lateral_iterations, lap_time


def _compiled_numba_kernel() -> Any:
    """Return cached ``njit``-compiled numba kernel callable.

    Returns:
        Compiled numba function object.
    """
    global _COMPILED_NUMBA_KERNEL
    if _COMPILED_NUMBA_KERNEL is not None:
        return _COMPILED_NUMBA_KERNEL

    numba = _require_numba()
    _COMPILED_NUMBA_KERNEL = numba.njit(cache=False)(_point_mass_speed_profile_kernel)
    return _COMPILED_NUMBA_KERNEL


def solve_speed_profile_numba(
    track: TrackData,
    model: NumbaSpeedModel,
    config: SimulationConfig,
) -> SpeedProfileResult:
    """Solve lap speed profile with a numba-backed CPU backend.

    Args:
        track: Track geometry and derived arc-length-domain quantities.
        model: Vehicle model implementing ``numba_speed_profile_parameters``.
        config: Solver runtime and numerical controls.

    Returns:
        Converged speed profile and integrated lap metrics.

    Raises:
        pylapsim.utils.exceptions.ConfigurationError: If backend selection is
            incompatible with the provided model or runtime settings.
    """
    if config.runtime.compute_backend != "numba":
        msg = "solve_speed_profile_numba requires runtime.compute_backend='numba'"
        raise ConfigurationError(msg)

    track.validate()
    model.validate()
    config.validate()

    param_method: Any = getattr(model, "numba_speed_profile_parameters", None)
    if not callable(param_method):
        msg = (
            "Model does not implement required numba backend method: "
            "numba_speed_profile_parameters"
        )
        raise ConfigurationError(msg)

    params = cast(NumbaProfileParameters, param_method())
    if len(params) != 6:
        msg = "numba_speed_profile_parameters must return exactly 6 scalar values"
        raise ConfigurationError(msg)

    (
        mass,
        downforce_scale,
        drag_scale,
        friction_coefficient,
        max_drive_accel,
        max_brake_accel,
    ) = params

    kernel = _compiled_numba_kernel()
    speed, longitudinal_accel, lateral_accel, lateral_iterations, lap_time = kernel(
        np.asarray(track.arc_length, dtype=np.float64),
        np.asarray(track.curvature, dtype=np.float64),
        np.asarray(track.grade, dtype=np.float64),
        np.asarray(track.banking, dtype=np.float64),
        float(mass),
        float(downforce_scale),
        float(drag_scale),
        float(friction_coefficient),
        float(max_drive_accel),
        float(max_brake_accel),
        float(config.runtime.max_speed),
        float(config.numerics.min_speed),
        int(config.numerics.lateral_envelope_max_iterations),
        float(config.numerics.lateral_envelope_convergence_tolerance),
    )

    return SpeedProfileResult(
        speed=np.asarray(speed, dtype=float),
        longitudinal_accel=np.asarray(longitudinal_accel, dtype=float),
        lateral_accel=np.asarray(lateral_accel, dtype=float),
        lateral_envelope_iterations=int(lateral_iterations),
        lap_time=float(lap_time),
    )
