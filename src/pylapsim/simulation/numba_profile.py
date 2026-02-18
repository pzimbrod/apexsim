"""Numba-backed speed-profile solver for CPU-optimized vehicle studies."""

from __future__ import annotations

import math
from typing import Any, Protocol, cast

import numpy as np

from pylapsim.simulation.config import SimulationConfig
from pylapsim.simulation.profile import SpeedProfileResult
from pylapsim.track.models import TrackData
from pylapsim.utils.constants import GRAVITY, SMALL_EPS
from pylapsim.utils.exceptions import ConfigurationError

NumbaPointMassProfileParameters = tuple[float, float, float, float, float, float]
NumbaSingleTrackProfileParameters = tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    int,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]
NumbaProfileParameters = NumbaPointMassProfileParameters | NumbaSingleTrackProfileParameters

POINT_MASS_PARAMETER_COUNT = 6
SINGLE_TRACK_PARAMETER_COUNT = 25

_COMPILED_NUMBA_KERNEL: Any | None = None
_COMPILED_SINGLE_TRACK_NUMBA_KERNEL: Any | None = None


class NumbaSpeedModel(Protocol):
    """Protocol for vehicle models that provide numba-compatible constants."""

    def validate(self) -> None:
        """Validate model parameters before solver execution."""

    def numba_speed_profile_parameters(self) -> NumbaProfileParameters:
        """Return scalar model parameters required by the numba kernel.

        Returns:
            Backend parameter tuple consumed by numba kernels.
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


def _single_track_speed_profile_kernel(
    arc_length: np.ndarray,
    curvature: np.ndarray,
    grade: np.ndarray,
    banking: np.ndarray,
    mass: float,
    downforce_scale: float,
    drag_scale: float,
    front_downforce_share: float,
    front_weight_fraction: float,
    max_drive_accel: float,
    max_brake_accel: float,
    peak_slip_angle: float,
    min_lateral_accel_limit: float,
    lateral_limit_max_iterations: int,
    lateral_limit_convergence_tolerance: float,
    front_b: float,
    front_c: float,
    front_d: float,
    front_e: float,
    front_reference_load: float,
    front_load_sensitivity: float,
    front_min_mu_scale: float,
    rear_b: float,
    rear_c: float,
    rear_d: float,
    rear_e: float,
    rear_reference_load: float,
    rear_load_sensitivity: float,
    rear_min_mu_scale: float,
    max_speed: float,
    min_speed: float,
    lateral_envelope_iterations_limit: int,
    lateral_envelope_convergence_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Solve the single-track profile in arc-length domain.

    Args:
        arc_length: Cumulative arc-length samples [m].
        curvature: Signed curvature samples [1/m].
        grade: Track grade samples ``dz/ds`` [-].
        banking: Banking-angle samples [rad].
        mass: Vehicle mass [kg].
        downforce_scale: Total downforce coefficient [N/(m/s)^2].
        drag_scale: Quadratic drag-force coefficient [N/(m/s)^2].
        front_downforce_share: Fraction of downforce on front axle [-].
        front_weight_fraction: Static front weight fraction [-].
        max_drive_accel: Maximum drive acceleration limit [m/s^2].
        max_brake_accel: Maximum braking deceleration magnitude [m/s^2].
        peak_slip_angle: Slip angle used for lateral-force envelope [rad].
        min_lateral_accel_limit: Lower bound on lateral acceleration [m/s^2].
        lateral_limit_max_iterations: Fixed-point iterations for per-point
            lateral-limit solve.
        lateral_limit_convergence_tolerance: Convergence threshold for
            per-point lateral-limit solve [m/s^2].
        front_b: Front-tire Pacejka ``B`` parameter.
        front_c: Front-tire Pacejka ``C`` parameter.
        front_d: Front-tire Pacejka ``D`` parameter.
        front_e: Front-tire Pacejka ``E`` parameter.
        front_reference_load: Front reference normal load [N].
        front_load_sensitivity: Front friction load-sensitivity slope [-].
        front_min_mu_scale: Front lower bound for friction scaling [-].
        rear_b: Rear-tire Pacejka ``B`` parameter.
        rear_c: Rear-tire Pacejka ``C`` parameter.
        rear_d: Rear-tire Pacejka ``D`` parameter.
        rear_e: Rear-tire Pacejka ``E`` parameter.
        rear_reference_load: Rear reference normal load [N].
        rear_load_sensitivity: Rear friction load-sensitivity slope [-].
        rear_min_mu_scale: Rear lower bound for friction scaling [-].
        max_speed: Global speed cap [m/s].
        min_speed: Numerical speed floor [m/s].
        lateral_envelope_iterations_limit: Maximum outer lateral-envelope
            fixed-point iterations.
        lateral_envelope_convergence_tolerance: Outer lateral-envelope
            convergence threshold [m/s].

    Returns:
        Tuple ``(speed, longitudinal_accel, lateral_accel, lateral_iterations,
        lap_time)``.
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
    for iteration_idx in range(lateral_envelope_iterations_limit):
        max_delta_speed = 0.0
        for idx in range(n):
            speed_non_negative = max(v_lat[idx], 0.0)
            speed_squared = speed_non_negative * speed_non_negative

            downforce_total = downforce_scale * speed_squared
            front_downforce = downforce_total * front_downforce_share

            weight = mass * GRAVITY
            total_vertical_load = weight + downforce_total
            front_static_load = weight * front_weight_fraction

            min_axle_load = 2.0 * SMALL_EPS
            front_axle_raw = front_static_load + front_downforce
            front_axle_load = min(
                max(front_axle_raw, min_axle_load),
                total_vertical_load - min_axle_load,
            )
            rear_axle_load = total_vertical_load - front_axle_load

            front_tire_load = max(front_axle_load * 0.5, SMALL_EPS)
            rear_tire_load = max(rear_axle_load * 0.5, SMALL_EPS)

            ay_banking = GRAVITY * math.sin(banking[idx])
            ay_estimate = min_lateral_accel_limit

            for _ in range(lateral_limit_max_iterations):
                front_slip_term = front_b * peak_slip_angle
                front_nonlinear = front_slip_term - front_e * (
                    front_slip_term - math.atan(front_slip_term)
                )
                front_shape = math.sin(front_c * math.atan(front_nonlinear))
                front_load_delta = (front_tire_load - front_reference_load) / front_reference_load
                front_mu_scale = max(
                    1.0 + front_load_sensitivity * front_load_delta,
                    front_min_mu_scale,
                )
                fy_front = 2.0 * front_d * front_mu_scale * front_tire_load * front_shape

                rear_slip_term = rear_b * peak_slip_angle
                rear_nonlinear = rear_slip_term - rear_e * (
                    rear_slip_term - math.atan(rear_slip_term)
                )
                rear_shape = math.sin(rear_c * math.atan(rear_nonlinear))
                rear_load_delta = (rear_tire_load - rear_reference_load) / rear_reference_load
                rear_mu_scale = max(
                    1.0 + rear_load_sensitivity * rear_load_delta,
                    rear_min_mu_scale,
                )
                fy_rear = 2.0 * rear_d * rear_mu_scale * rear_tire_load * rear_shape

                ay_tire = (fy_front + fy_rear) / mass
                ay_next = max(min_lateral_accel_limit, ay_tire + ay_banking)
                if abs(ay_next - ay_estimate) <= lateral_limit_convergence_tolerance:
                    ay_estimate = ay_next
                    break
                ay_estimate = ay_next

            ay_limit = max(ay_estimate, SMALL_EPS)

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
        if max_delta_speed <= lateral_envelope_convergence_tolerance:
            break

    v_forward = v_lat.copy()
    v_forward[0] = min(v_forward[0], max_speed)

    for idx in range(n - 1):
        speed_value = v_forward[idx]
        speed_non_negative = max(speed_value, 0.0)
        speed_squared = speed_non_negative * speed_non_negative

        downforce_total = downforce_scale * speed_squared
        front_downforce = downforce_total * front_downforce_share

        weight = mass * GRAVITY
        total_vertical_load = weight + downforce_total
        front_static_load = weight * front_weight_fraction

        min_axle_load = 2.0 * SMALL_EPS
        front_axle_raw = front_static_load + front_downforce
        front_axle_load = min(
            max(front_axle_raw, min_axle_load),
            total_vertical_load - min_axle_load,
        )
        rear_axle_load = total_vertical_load - front_axle_load

        front_tire_load = max(front_axle_load * 0.5, SMALL_EPS)
        rear_tire_load = max(rear_axle_load * 0.5, SMALL_EPS)

        ay_banking = GRAVITY * math.sin(banking[idx])
        ay_estimate = min_lateral_accel_limit

        for _ in range(lateral_limit_max_iterations):
            front_slip_term = front_b * peak_slip_angle
            front_nonlinear = front_slip_term - front_e * (
                front_slip_term - math.atan(front_slip_term)
            )
            front_shape = math.sin(front_c * math.atan(front_nonlinear))
            front_load_delta = (front_tire_load - front_reference_load) / front_reference_load
            front_mu_scale = max(
                1.0 + front_load_sensitivity * front_load_delta,
                front_min_mu_scale,
            )
            fy_front = 2.0 * front_d * front_mu_scale * front_tire_load * front_shape

            rear_slip_term = rear_b * peak_slip_angle
            rear_nonlinear = rear_slip_term - rear_e * (rear_slip_term - math.atan(rear_slip_term))
            rear_shape = math.sin(rear_c * math.atan(rear_nonlinear))
            rear_load_delta = (rear_tire_load - rear_reference_load) / rear_reference_load
            rear_mu_scale = max(
                1.0 + rear_load_sensitivity * rear_load_delta,
                rear_min_mu_scale,
            )
            fy_rear = 2.0 * rear_d * rear_mu_scale * rear_tire_load * rear_shape

            ay_tire = (fy_front + fy_rear) / mass
            ay_next = max(min_lateral_accel_limit, ay_tire + ay_banking)
            if abs(ay_next - ay_estimate) <= lateral_limit_convergence_tolerance:
                ay_estimate = ay_next
                break
            ay_estimate = ay_next

        ay_limit = max(ay_estimate, SMALL_EPS)

        ay_required = speed_value * speed_value * curvature_abs[idx]
        usage = min(abs(ay_required) / ay_limit, 1.0)
        circle_scale = math.sqrt(max(0.0, 1.0 - usage * usage))

        tire_accel = max_drive_accel * circle_scale

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

        downforce_total = downforce_scale * speed_squared
        front_downforce = downforce_total * front_downforce_share

        weight = mass * GRAVITY
        total_vertical_load = weight + downforce_total
        front_static_load = weight * front_weight_fraction

        min_axle_load = 2.0 * SMALL_EPS
        front_axle_raw = front_static_load + front_downforce
        front_axle_load = min(
            max(front_axle_raw, min_axle_load),
            total_vertical_load - min_axle_load,
        )
        rear_axle_load = total_vertical_load - front_axle_load

        front_tire_load = max(front_axle_load * 0.5, SMALL_EPS)
        rear_tire_load = max(rear_axle_load * 0.5, SMALL_EPS)

        ay_banking = GRAVITY * math.sin(banking[idx + 1])
        ay_estimate = min_lateral_accel_limit

        for _ in range(lateral_limit_max_iterations):
            front_slip_term = front_b * peak_slip_angle
            front_nonlinear = front_slip_term - front_e * (
                front_slip_term - math.atan(front_slip_term)
            )
            front_shape = math.sin(front_c * math.atan(front_nonlinear))
            front_load_delta = (front_tire_load - front_reference_load) / front_reference_load
            front_mu_scale = max(
                1.0 + front_load_sensitivity * front_load_delta,
                front_min_mu_scale,
            )
            fy_front = 2.0 * front_d * front_mu_scale * front_tire_load * front_shape

            rear_slip_term = rear_b * peak_slip_angle
            rear_nonlinear = rear_slip_term - rear_e * (rear_slip_term - math.atan(rear_slip_term))
            rear_shape = math.sin(rear_c * math.atan(rear_nonlinear))
            rear_load_delta = (rear_tire_load - rear_reference_load) / rear_reference_load
            rear_mu_scale = max(
                1.0 + rear_load_sensitivity * rear_load_delta,
                rear_min_mu_scale,
            )
            fy_rear = 2.0 * rear_d * rear_mu_scale * rear_tire_load * rear_shape

            ay_tire = (fy_front + fy_rear) / mass
            ay_next = max(min_lateral_accel_limit, ay_tire + ay_banking)
            if abs(ay_next - ay_estimate) <= lateral_limit_convergence_tolerance:
                ay_estimate = ay_next
                break
            ay_estimate = ay_next

        ay_limit = max(ay_estimate, SMALL_EPS)

        ay_required = speed_value * speed_value * curvature_abs[idx + 1]
        usage = min(abs(ay_required) / ay_limit, 1.0)
        circle_scale = math.sqrt(max(0.0, 1.0 - usage * usage))

        tire_brake = max_brake_accel * circle_scale

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
    """Return cached ``njit``-compiled point-mass numba kernel callable.

    Returns:
        Compiled numba function object.
    """
    global _COMPILED_NUMBA_KERNEL
    if _COMPILED_NUMBA_KERNEL is not None:
        return _COMPILED_NUMBA_KERNEL

    numba = _require_numba()
    _COMPILED_NUMBA_KERNEL = numba.njit(cache=False)(_point_mass_speed_profile_kernel)
    return _COMPILED_NUMBA_KERNEL


def _compiled_single_track_numba_kernel() -> Any:
    """Return cached ``njit``-compiled single-track numba kernel callable.

    Returns:
        Compiled numba function object.
    """
    global _COMPILED_SINGLE_TRACK_NUMBA_KERNEL
    if _COMPILED_SINGLE_TRACK_NUMBA_KERNEL is not None:
        return _COMPILED_SINGLE_TRACK_NUMBA_KERNEL

    numba = _require_numba()
    _COMPILED_SINGLE_TRACK_NUMBA_KERNEL = numba.njit(cache=False)(
        _single_track_speed_profile_kernel
    )
    return _COMPILED_SINGLE_TRACK_NUMBA_KERNEL


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

    params = cast(tuple[float | int, ...], param_method())
    param_count = len(params)

    if param_count == POINT_MASS_PARAMETER_COUNT:
        (
            mass,
            downforce_scale,
            drag_scale,
            friction_coefficient,
            max_drive_accel,
            max_brake_accel,
        ) = cast(NumbaPointMassProfileParameters, params)
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
    elif param_count == SINGLE_TRACK_PARAMETER_COUNT:
        (
            mass,
            downforce_scale,
            drag_scale,
            front_downforce_share,
            front_weight_fraction,
            max_drive_accel,
            max_brake_accel,
            peak_slip_angle,
            single_track_min_lateral_accel_limit,
            single_track_lateral_limit_max_iterations,
            single_track_lateral_limit_convergence_tolerance,
            front_b,
            front_c,
            front_d,
            front_e,
            front_reference_load,
            front_load_sensitivity,
            front_min_mu_scale,
            rear_b,
            rear_c,
            rear_d,
            rear_e,
            rear_reference_load,
            rear_load_sensitivity,
            rear_min_mu_scale,
        ) = cast(NumbaSingleTrackProfileParameters, params)
        kernel = _compiled_single_track_numba_kernel()
        speed, longitudinal_accel, lateral_accel, lateral_iterations, lap_time = kernel(
            np.asarray(track.arc_length, dtype=np.float64),
            np.asarray(track.curvature, dtype=np.float64),
            np.asarray(track.grade, dtype=np.float64),
            np.asarray(track.banking, dtype=np.float64),
            float(mass),
            float(downforce_scale),
            float(drag_scale),
            float(front_downforce_share),
            float(front_weight_fraction),
            float(max_drive_accel),
            float(max_brake_accel),
            float(peak_slip_angle),
            float(single_track_min_lateral_accel_limit),
            int(single_track_lateral_limit_max_iterations),
            float(single_track_lateral_limit_convergence_tolerance),
            float(front_b),
            float(front_c),
            float(front_d),
            float(front_e),
            float(front_reference_load),
            float(front_load_sensitivity),
            float(front_min_mu_scale),
            float(rear_b),
            float(rear_c),
            float(rear_d),
            float(rear_e),
            float(rear_reference_load),
            float(rear_load_sensitivity),
            float(rear_min_mu_scale),
            float(config.runtime.max_speed),
            float(config.numerics.min_speed),
            int(config.numerics.lateral_envelope_max_iterations),
            float(config.numerics.lateral_envelope_convergence_tolerance),
        )
    else:
        msg = (
            "numba_speed_profile_parameters returned unsupported parameter count: "
            f"{param_count}. Expected {POINT_MASS_PARAMETER_COUNT} (point-mass) "
            f"or {SINGLE_TRACK_PARAMETER_COUNT} (single-track)."
        )
        raise ConfigurationError(msg)

    return SpeedProfileResult(
        speed=np.asarray(speed, dtype=float),
        longitudinal_accel=np.asarray(longitudinal_accel, dtype=float),
        lateral_accel=np.asarray(lateral_accel, dtype=float),
        lateral_envelope_iterations=int(lateral_iterations),
        lap_time=float(lap_time),
    )
