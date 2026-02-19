"""NumPy/SciPy transient optimal-control lap solver."""

from __future__ import annotations

import sys
from dataclasses import replace
from typing import Any

import numpy as np

from apexsim.simulation.config import SimulationConfig
from apexsim.simulation.integrator import rk4_step
from apexsim.simulation.profile import solve_speed_profile
from apexsim.simulation.transient_common import (
    PidSpeedSchedule,
    TransientPidGainSchedulingConfig,
    TransientProfileResult,
    segment_time_step,
)
from apexsim.track.models import TrackData
from apexsim.utils.constants import SMALL_EPS
from apexsim.utils.exceptions import ConfigurationError
from apexsim.vehicle.single_track.dynamics import ControlInput, VehicleState

_PROGRESS_BAR_WIDTH = 30
_TRACK_PROGRESS_FRACTION_STEP = 0.10
_TRACK_PROGRESS_EVAL_STRIDE = 20
_MAX_TRANSIENT_SIDESLIP_RATIO = 0.35

_DEFAULT_SCHEDULE_SPEED_NODES_MPS = (0.0, 10.0, 20.0, 35.0, 55.0)
_SCHEDULE_REFERENCE_SPEED_MPS = 20.0
_SCHEDULE_MIN_SPEED_FOR_STEER_MPS = 3.0
_SCHEDULE_LONGITUDINAL_SCALE_MIN = 0.55
_SCHEDULE_LONGITUDINAL_SCALE_MAX = 1.45
_SCHEDULE_LONGITUDINAL_D_SCALE_MIN = 0.60
_SCHEDULE_LONGITUDINAL_D_SCALE_MAX = 1.35
_SCHEDULE_STEER_KP_SCALE_MIN = 0.35
_SCHEDULE_STEER_KP_SCALE_MAX = 2.40
_SCHEDULE_STEER_KI_SCALE_MIN = 0.35
_SCHEDULE_STEER_KI_SCALE_MAX = 2.20
_SCHEDULE_STEER_KD_SCALE_MIN = 0.40
_SCHEDULE_STEER_KD_SCALE_MAX = 2.10
_SCHEDULE_STEER_VY_SCALE_MIN = 0.75
_SCHEDULE_STEER_VY_SCALE_MAX = 2.00
_SCHEDULE_STEER_VY_GRADIENT = 0.50


def _clamp_integral(value: float, limit: float) -> float:
    """Clamp PID integrator state to a symmetric finite interval.

    Args:
        value: Integrator state value.
        limit: Absolute integrator limit.

    Returns:
        Clamped integrator state.
    """
    return float(np.clip(value, -limit, limit))


def _resolve_schedule_speed_nodes(
    *,
    max_speed: float,
    speed_nodes_mps: tuple[float, ...] | None = None,
) -> tuple[float, ...]:
    """Build validated schedule speed nodes up to ``max_speed``.

    Args:
        max_speed: Runtime speed cap [m/s].
        speed_nodes_mps: Optional explicit node set.

    Returns:
        Monotonic node tuple suitable for interpolation.
    """
    if speed_nodes_mps is not None:
        schedule = PidSpeedSchedule(speed_nodes_mps=speed_nodes_mps, values=speed_nodes_mps)
        schedule.validate()
        return tuple(float(value) for value in speed_nodes_mps)

    if max_speed <= 0.0:
        return (0.0, 1.0)

    nodes = [0.0]
    nodes.extend(
        float(value)
        for value in _DEFAULT_SCHEDULE_SPEED_NODES_MPS[1:]
        if float(value) < float(max_speed)
    )
    if float(max_speed) > nodes[-1]:
        nodes.append(float(max_speed))
    if len(nodes) == 1:
        nodes.append(max(float(max_speed), SMALL_EPS))
    return tuple(nodes)


def _longitudinal_authority(model: Any, speed_mps: float) -> float:
    """Return average accel/decel authority on flat road at ``speed_mps``.

    Args:
        model: Vehicle model implementing accel/decel limits.
        speed_mps: Evaluation speed [m/s].

    Returns:
        Effective symmetric longitudinal acceleration authority [m/s^2].
    """
    accel = float(
        model.max_longitudinal_accel(
            speed=float(speed_mps),
            lateral_accel_required=0.0,
            grade=0.0,
            banking=0.0,
        )
    )
    decel = float(
        model.max_longitudinal_decel(
            speed=float(speed_mps),
            lateral_accel_required=0.0,
            grade=0.0,
            banking=0.0,
        )
    )
    return max(0.5 * (accel + decel), SMALL_EPS)


def _schedule_or_default(
    *,
    schedule: PidSpeedSchedule | None,
    default_value: float,
    speed_mps: float,
) -> float:
    """Evaluate schedule at speed or return default scalar.

    Args:
        schedule: Optional speed schedule.
        default_value: Fallback scalar value when ``schedule`` is ``None``.
        speed_mps: Evaluation speed [m/s].

    Returns:
        Scheduled or fallback gain value.
    """
    if schedule is None:
        return float(default_value)
    return float(schedule.evaluate(speed_mps))


def build_physics_informed_pid_gain_scheduling(
    *,
    model: Any,
    numerics: Any,
    max_speed: float,
    speed_nodes_mps: tuple[float, ...] | None = None,
) -> TransientPidGainSchedulingConfig:
    """Build deterministic speed-dependent PID schedules from physics heuristics.

    The generated schedule uses only model-based longitudinal authority on flat
    road and speed normalization. It is deterministic and independent of track
    geometry.

    Args:
        model: Vehicle model used for longitudinal authority estimation.
        numerics: Object exposing baseline scalar PID gains.
        max_speed: Runtime speed cap [m/s].
        speed_nodes_mps: Optional custom speed nodes [m/s]. If omitted, uses
            the package default node set.

    Returns:
        Fully specified gain-scheduling table set.
    """
    nodes = _resolve_schedule_speed_nodes(
        max_speed=float(max_speed),
        speed_nodes_mps=speed_nodes_mps,
    )
    reference_speed = float(
        np.clip(
            _SCHEDULE_REFERENCE_SPEED_MPS,
            _SCHEDULE_MIN_SPEED_FOR_STEER_MPS,
            max(float(max_speed), _SCHEDULE_MIN_SPEED_FOR_STEER_MPS),
        )
    )
    authority_reference = _longitudinal_authority(model=model, speed_mps=reference_speed)

    kp0_long = float(numerics.pid_longitudinal_kp)
    ki0_long = float(numerics.pid_longitudinal_ki)
    kd0_long = float(numerics.pid_longitudinal_kd)
    kp0_steer = float(numerics.pid_steer_kp)
    ki0_steer = float(numerics.pid_steer_ki)
    kd0_steer = float(numerics.pid_steer_kd)
    vy0_steer = float(numerics.pid_steer_vy_damping)

    kp_long_values: list[float] = []
    ki_long_values: list[float] = []
    kd_long_values: list[float] = []
    kp_steer_values: list[float] = []
    ki_steer_values: list[float] = []
    kd_steer_values: list[float] = []
    vy_steer_values: list[float] = []

    # Longitudinal gains scale with available accel/decel authority. Steering
    # gains decrease with speed while lateral-velocity damping increases.
    for speed_value in nodes:
        authority = _longitudinal_authority(model=model, speed_mps=float(speed_value))
        authority_scale = float(
            np.clip(
                authority / max(authority_reference, SMALL_EPS),
                _SCHEDULE_LONGITUDINAL_SCALE_MIN,
                _SCHEDULE_LONGITUDINAL_SCALE_MAX,
            )
        )
        kp_long_values.append(kp0_long * authority_scale)
        ki_long_values.append(ki0_long * authority_scale)
        kd_long_values.append(
            kd0_long
            * float(
                np.clip(
                    np.sqrt(max(authority_scale, SMALL_EPS)),
                    _SCHEDULE_LONGITUDINAL_D_SCALE_MIN,
                    _SCHEDULE_LONGITUDINAL_D_SCALE_MAX,
                )
            )
        )

        speed_scale = reference_speed / max(float(speed_value), _SCHEDULE_MIN_SPEED_FOR_STEER_MPS)
        kp_steer_values.append(
            kp0_steer
            * float(
                np.clip(
                    speed_scale,
                    _SCHEDULE_STEER_KP_SCALE_MIN,
                    _SCHEDULE_STEER_KP_SCALE_MAX,
                )
            )
        )
        ki_steer_values.append(
            ki0_steer
            * float(
                np.clip(
                    speed_scale,
                    _SCHEDULE_STEER_KI_SCALE_MIN,
                    _SCHEDULE_STEER_KI_SCALE_MAX,
                )
            )
        )
        kd_steer_values.append(
            kd0_steer
            * float(
                np.clip(
                    np.sqrt(max(speed_scale, SMALL_EPS)),
                    _SCHEDULE_STEER_KD_SCALE_MIN,
                    _SCHEDULE_STEER_KD_SCALE_MAX,
                )
            )
        )
        vy_steer_values.append(
            vy0_steer
            * float(
                np.clip(
                    1.0
                    + _SCHEDULE_STEER_VY_GRADIENT
                    * float(speed_value)
                    / max(reference_speed, SMALL_EPS),
                    _SCHEDULE_STEER_VY_SCALE_MIN,
                    _SCHEDULE_STEER_VY_SCALE_MAX,
                )
            )
        )

    scheduling = TransientPidGainSchedulingConfig(
        longitudinal_kp=PidSpeedSchedule(nodes, tuple(kp_long_values)),
        longitudinal_ki=PidSpeedSchedule(nodes, tuple(ki_long_values)),
        longitudinal_kd=PidSpeedSchedule(nodes, tuple(kd_long_values)),
        steer_kp=PidSpeedSchedule(nodes, tuple(kp_steer_values)),
        steer_ki=PidSpeedSchedule(nodes, tuple(ki_steer_values)),
        steer_kd=PidSpeedSchedule(nodes, tuple(kd_steer_values)),
        steer_vy_damping=PidSpeedSchedule(nodes, tuple(vy_steer_values)),
    )
    scheduling.validate()
    return scheduling


def _resolve_pid_gain_scheduling(
    *,
    model: Any,
    config: SimulationConfig,
) -> TransientPidGainSchedulingConfig | None:
    """Resolve active PID scheduling strategy from config.

    Args:
        model: Vehicle model used for physics-informed schedule generation.
        config: Simulation configuration.

    Returns:
        Active PID gain scheduling config or ``None`` when scheduling is off.
    """
    numerics = config.transient.numerics
    mode = numerics.pid_gain_scheduling_mode
    if mode == "off":
        return None
    if mode == "physics_informed":
        return build_physics_informed_pid_gain_scheduling(
            model=model,
            numerics=numerics,
            max_speed=float(config.runtime.max_speed),
        )
    return numerics.pid_gain_scheduling


def _require_scipy_optimize() -> Any:
    """Import scipy.optimize lazily and fail with clear guidance.

    Returns:
        Imported ``scipy.optimize`` module.

    Raises:
        apexsim.utils.exceptions.ConfigurationError: If SciPy is not installed.
    """
    try:
        import scipy.optimize as optimize  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        msg = (
            "Transient OC solver for numpy/numba backends requires SciPy. "
            "Install or update dependencies with `pip install -e .`."
        )
        raise ConfigurationError(msg) from exc
    return optimize


def _render_progress_line(
    *,
    prefix: str,
    fraction: float,
    suffix: str,
    final: bool = False,
) -> None:
    """Render one in-place text progress line to stderr.

    Args:
        prefix: Prefix shown before the progress bar.
        fraction: Progress fraction in ``[0, 1]``.
        suffix: Additional information shown after percentage.
        final: If ``True``, terminate the line with a newline.
    """
    clamped = float(np.clip(fraction, 0.0, 1.0))
    filled = int(clamped * _PROGRESS_BAR_WIDTH)
    bar = "#" * filled + "-" * (_PROGRESS_BAR_WIDTH - filled)
    end = "\n" if final else ""
    print(
        f"\r{prefix} [{bar}] {100.0 * clamped:5.1f}% {suffix}",
        end=end,
        file=sys.stderr,
        flush=True,
    )


def _maybe_emit_track_progress(
    *,
    progress_prefix: str | None,
    segment_idx: int,
    segment_count: int,
    next_fraction_threshold: float,
) -> float:
    """Emit throttled track-integration progress updates.

    Args:
        progress_prefix: Progress prefix or ``None`` to disable reporting.
        segment_idx: Current segment index (0-based).
        segment_count: Total number of segments.
        next_fraction_threshold: Next fraction threshold triggering output.

    Returns:
        Updated next fraction threshold.
    """
    if progress_prefix is None or segment_count <= 0:
        return next_fraction_threshold

    completed = segment_idx + 1
    fraction = completed / segment_count
    is_final = completed == segment_count
    if fraction < next_fraction_threshold and not is_final:
        return next_fraction_threshold

    _render_progress_line(
        prefix=progress_prefix,
        fraction=fraction,
        suffix=f"segment {completed}/{segment_count}",
        final=is_final,
    )

    threshold = next_fraction_threshold
    while threshold <= fraction:
        threshold += _TRACK_PROGRESS_FRACTION_STEP
    return threshold


def _is_single_track_model(model: Any) -> bool:
    """Return whether model exposes single-track transient dynamics internals.

    Args:
        model: Solver model instance.

    Returns:
        ``True`` if single-track transient dynamics are available.
    """
    return (
        hasattr(model, "_dynamics")
        and hasattr(model, "physics")
        and hasattr(model.physics, "max_steer_angle")
        and hasattr(model.physics, "max_steer_rate")
    )


def _build_control_mesh_positions(
    *,
    sample_count: int,
    control_interval: int,
) -> np.ndarray:
    """Build evenly spaced control-node positions for optimization.

    Args:
        sample_count: Number of full track samples.
        control_interval: Desired spacing between control nodes in samples.

    Returns:
        Monotonic control-node positions over sample indices.
    """
    if sample_count <= 1:
        return np.zeros(1, dtype=float)
    control_count = int(np.ceil((sample_count - 1) / max(control_interval, 1))) + 1
    control_count = min(max(control_count, 2), sample_count)
    return np.linspace(0.0, float(sample_count - 1), control_count, dtype=float)


def _sample_seed_on_mesh(seed: np.ndarray, mesh_positions: np.ndarray) -> np.ndarray:
    """Sample a full-resolution seed signal onto the control mesh.

    Args:
        seed: Full-resolution seed signal.
        mesh_positions: Control-node positions in sample index coordinates.

    Returns:
        Seed values at control-node positions.
    """
    sample_positions = np.arange(seed.size, dtype=float)
    return np.asarray(np.interp(mesh_positions, sample_positions, seed), dtype=float)


def _expand_mesh_controls(
    *,
    node_values: np.ndarray,
    sample_count: int,
    mesh_positions: np.ndarray,
) -> np.ndarray:
    """Expand control-node values to full track resolution by interpolation.

    Args:
        node_values: Control values at mesh nodes.
        sample_count: Full sample count.
        mesh_positions: Control-node positions in sample index coordinates.

    Returns:
        Full-resolution control signal.
    """
    if sample_count <= 1:
        return np.asarray(node_values[:1], dtype=float)
    sample_positions = np.arange(sample_count, dtype=float)
    return np.asarray(np.interp(sample_positions, mesh_positions, node_values), dtype=float)


def _bounded_artanh(value: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Return stable inverse-tanh transform for bounded control seeds.

    Args:
        value: Input vector in ``[-1, 1]``.
        eps: Clipping epsilon away from the interval boundaries.

    Returns:
        Inverse-tanh transformed values.
    """
    clipped = np.clip(value, -1.0 + eps, 1.0 - eps)
    return np.asarray(np.arctanh(clipped), dtype=float)


def _transient_reference_profile(
    track: TrackData,
    model: Any,
    config: SimulationConfig,
) -> Any:
    """Solve quasi-static reference profile used by PID driver control.

    Args:
        track: Track geometry.
        model: Solver model instance.
        config: Simulation config.

    Returns:
        Quasi-static speed-profile solution.
    """
    quasi_config = replace(
        config,
        runtime=replace(
            config.runtime,
            solver_mode="quasi_static",
            enable_transient_refinement=False,
        ),
    )
    return solve_speed_profile(track=track, model=model, config=quasi_config)


def _decode_point_mass_controls(
    model: Any,
    raw_ax: np.ndarray,
    *,
    sample_count: int | None = None,
    mesh_positions: np.ndarray | None = None,
) -> np.ndarray:
    """Decode bounded point-mass acceleration controls from raw variables.

    Args:
        model: Point-mass model.
        raw_ax: Raw unconstrained optimization variables.
        sample_count: Optional full track sample count.
        mesh_positions: Optional control-node positions in sample index
            coordinates.

    Returns:
        Bounded acceleration command vector [m/s^2].
    """
    upper = float(model.envelope_physics.max_drive_accel)
    lower = -float(model.envelope_physics.max_brake_accel)
    midpoint = 0.5 * (upper + lower)
    half_span = 0.5 * (upper - lower)
    node_values = np.asarray(midpoint + half_span * np.tanh(raw_ax), dtype=float)
    if sample_count is None:
        return node_values
    if mesh_positions is None:
        if node_values.size != sample_count:
            msg = (
                "sample_count provided without mesh_positions requires raw "
                "controls to match sample count"
            )
            raise ConfigurationError(msg)
        return node_values
    return _expand_mesh_controls(
        node_values=node_values,
        sample_count=sample_count,
        mesh_positions=mesh_positions,
    )


def _encode_point_mass_controls(
    model: Any,
    ax_seed: np.ndarray,
    *,
    mesh_positions: np.ndarray | None = None,
) -> np.ndarray:
    """Encode point-mass acceleration seed into raw optimization variables.

    Args:
        model: Point-mass model.
        ax_seed: Seed acceleration signal [m/s^2].
        mesh_positions: Optional control-node positions in sample index
            coordinates.

    Returns:
        Raw unconstrained optimization variables.
    """
    upper = float(model.envelope_physics.max_drive_accel)
    lower = -float(model.envelope_physics.max_brake_accel)
    midpoint = 0.5 * (upper + lower)
    half_span = max(0.5 * (upper - lower), SMALL_EPS)
    seed_values = (
        _sample_seed_on_mesh(ax_seed, mesh_positions)
        if mesh_positions is not None
        else np.asarray(ax_seed, dtype=float)
    )
    normalized = (seed_values - midpoint) / half_span
    return np.asarray(_bounded_artanh(normalized), dtype=float)


def _decode_single_track_controls(
    model: Any,
    raw_controls: np.ndarray,
    *,
    sample_count: int | None = None,
    mesh_positions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode bounded single-track acceleration and steering controls.

    Args:
        model: Single-track model.
        raw_controls: Raw unconstrained control vector of length ``2 * n``.
        sample_count: Optional full track sample count.
        mesh_positions: Optional control-node positions in sample index
            coordinates.

    Returns:
        Tuple ``(ax_cmd, steer_target)``.
    """
    n = raw_controls.size // 2
    raw_ax = raw_controls[:n]
    raw_steer = raw_controls[n:]

    ax_cmd = _decode_point_mass_controls(
        model=model,
        raw_ax=raw_ax,
        sample_count=sample_count,
        mesh_positions=mesh_positions,
    )
    steer_nodes = float(model.physics.max_steer_angle) * np.tanh(raw_steer)
    if sample_count is None:
        return ax_cmd, np.asarray(steer_nodes, dtype=float)
    if mesh_positions is None:
        if steer_nodes.size != sample_count:
            msg = (
                "sample_count provided without mesh_positions requires steering "
                "raw controls to match sample count"
            )
            raise ConfigurationError(msg)
        return ax_cmd, np.asarray(steer_nodes, dtype=float)
    steer_target = _expand_mesh_controls(
        node_values=np.asarray(steer_nodes, dtype=float),
        sample_count=sample_count,
        mesh_positions=mesh_positions,
    )
    return ax_cmd, steer_target


def _encode_single_track_controls(
    model: Any,
    ax_seed: np.ndarray,
    steer_seed: np.ndarray,
    *,
    mesh_positions: np.ndarray | None = None,
) -> np.ndarray:
    """Encode single-track seed signals into raw optimization variables.

    Args:
        model: Single-track model.
        ax_seed: Seed acceleration signal [m/s^2].
        steer_seed: Seed steering signal [rad].
        mesh_positions: Optional control-node positions in sample index
            coordinates.

    Returns:
        Raw unconstrained optimization vector.
    """
    ax_raw = _encode_point_mass_controls(
        model=model,
        ax_seed=ax_seed,
        mesh_positions=mesh_positions,
    )
    max_angle = max(float(model.physics.max_steer_angle), SMALL_EPS)
    steer_values = (
        _sample_seed_on_mesh(steer_seed, mesh_positions)
        if mesh_positions is not None
        else np.asarray(steer_seed, dtype=float)
    )
    steer_raw = _bounded_artanh(steer_values / max_angle)
    return np.concatenate((ax_raw, steer_raw))


def _simulate_point_mass_controls(
    *,
    track: TrackData,
    model: Any,
    config: SimulationConfig,
    ax_signal: np.ndarray,
    track_progress_prefix: str | None = None,
) -> TransientProfileResult:
    """Simulate point-mass transient lap from bounded control sequence.

    Args:
        track: Track geometry.
        model: Point-mass model.
        config: Simulation config.
        ax_signal: Bounded acceleration-command sequence [m/s^2].
        track_progress_prefix: Optional progress prefix shown during
            arc-length integration.

    Returns:
        Transient profile result.
    """
    n = track.arc_length.size
    ds = np.diff(track.arc_length)

    speed = np.zeros(n, dtype=float)
    longitudinal_accel = np.zeros(n, dtype=float)
    lateral_accel = np.zeros(n, dtype=float)
    time = np.zeros(n, dtype=float)
    vx = np.zeros(n, dtype=float)
    vy = np.zeros(n, dtype=float)
    yaw_rate = np.zeros(n, dtype=float)
    steer_cmd = np.zeros(n, dtype=float)
    ax_cmd = np.zeros(n, dtype=float)

    min_time_step = config.transient.numerics.min_time_step
    max_time_step = config.transient.numerics.max_time_step
    max_speed = float(config.runtime.max_speed)
    start_speed = (
        max_speed
        if config.runtime.initial_speed is None
        else float(config.runtime.initial_speed)
    )

    speed[0] = max(start_speed, 0.0)
    vx[0] = speed[0]

    lateral_penalty = 0.0
    next_track_progress = _TRACK_PROGRESS_FRACTION_STEP
    for idx in range(n - 1):
        speed_value = float(speed[idx])
        curvature = float(track.curvature[idx])
        ay_required = speed_value * speed_value * abs(curvature)
        ay_limit = float(
            model.lateral_accel_limit(
                speed=speed_value,
                banking=float(track.banking[idx]),
            )
        )
        lateral_penalty += max(0.0, ay_required - ay_limit) ** 2

        max_accel = float(
            model.max_longitudinal_accel(
                speed=speed_value,
                lateral_accel_required=ay_required,
                grade=float(track.grade[idx]),
                banking=float(track.banking[idx]),
            )
        )
        max_brake = float(
            model.max_longitudinal_decel(
                speed=speed_value,
                lateral_accel_required=ay_required,
                grade=float(track.grade[idx]),
                banking=float(track.banking[idx]),
            )
        )
        commanded = float(np.clip(ax_signal[idx], -max_brake, max_accel))
        ax_cmd[idx] = commanded
        longitudinal_accel[idx] = commanded
        lateral_accel[idx] = speed_value * speed_value * curvature

        dt = segment_time_step(
            segment_length=float(ds[idx]),
            speed=max(speed_value, SMALL_EPS),
            min_time_step=min_time_step,
            max_time_step=max_time_step,
        )
        next_speed = np.clip(speed_value + commanded * dt, 0.0, max_speed)
        speed[idx + 1] = float(next_speed)
        vx[idx + 1] = speed[idx + 1]
        time[idx + 1] = time[idx] + dt
        next_track_progress = _maybe_emit_track_progress(
            progress_prefix=track_progress_prefix,
            segment_idx=idx,
            segment_count=n - 1,
            next_fraction_threshold=next_track_progress,
        )

    if n > 1:
        longitudinal_accel[-1] = longitudinal_accel[-2]
        lateral_accel[-1] = speed[-1] * speed[-1] * float(track.curvature[-1])
        ax_cmd[-1] = ax_cmd[-2]

    lap_time = float(time[-1])
    smooth_penalty = float(np.sum(np.diff(ax_cmd) ** 2))
    objective = (
        lap_time
        + config.transient.numerics.lateral_constraint_weight * lateral_penalty
        + config.transient.numerics.control_smoothness_weight * smooth_penalty
    )

    return TransientProfileResult(
        speed=speed,
        longitudinal_accel=longitudinal_accel,
        lateral_accel=lateral_accel,
        lap_time=lap_time,
        time=time,
        vx=vx,
        vy=vy,
        yaw_rate=yaw_rate,
        steer_cmd=steer_cmd,
        ax_cmd=ax_cmd,
        objective_value=float(objective),
    )


def _simulate_single_track_controls(
    *,
    track: TrackData,
    model: Any,
    config: SimulationConfig,
    ax_signal: np.ndarray,
    steer_target: np.ndarray,
    track_progress_prefix: str | None = None,
) -> TransientProfileResult:
    """Simulate single-track transient lap from bounded control sequences.

    Args:
        track: Track geometry.
        model: Single-track model.
        config: Simulation config.
        ax_signal: Bounded acceleration-command sequence [m/s^2].
        steer_target: Bounded steering-target sequence [rad].
        track_progress_prefix: Optional progress prefix shown during
            arc-length integration.

    Returns:
        Transient profile result.
    """
    n = track.arc_length.size
    ds = np.diff(track.arc_length)

    speed = np.zeros(n, dtype=float)
    longitudinal_accel = np.zeros(n, dtype=float)
    lateral_accel = np.zeros(n, dtype=float)
    time = np.zeros(n, dtype=float)
    vx = np.zeros(n, dtype=float)
    vy = np.zeros(n, dtype=float)
    yaw_rate = np.zeros(n, dtype=float)
    steer_cmd = np.zeros(n, dtype=float)
    ax_cmd = np.zeros(n, dtype=float)

    max_speed = float(config.runtime.max_speed)
    start_speed = (
        max_speed
        if config.runtime.initial_speed is None
        else float(config.runtime.initial_speed)
    )
    min_time_step = config.transient.numerics.min_time_step
    max_time_step = config.transient.numerics.max_time_step
    method = config.transient.numerics.integration_method
    max_steer_angle = float(model.physics.max_steer_angle)
    max_steer_rate = float(model.physics.max_steer_rate)

    vx[0] = max(start_speed, SMALL_EPS)
    vy[0] = 0.0
    yaw_rate[0] = vx[0] * float(track.curvature[0])
    speed[0] = float(vx[0])
    steer_cmd[0] = float(np.clip(steer_target[0], -max_steer_angle, max_steer_angle))

    dynamics = model._dynamics
    lateral_penalty = 0.0
    tracking_penalty = 0.0
    next_track_progress = _TRACK_PROGRESS_FRACTION_STEP

    for idx in range(n - 1):
        progress_speed = max(float(vx[idx]), SMALL_EPS)
        body_speed = max(float(np.hypot(vx[idx], vy[idx])), SMALL_EPS)
        curvature = float(track.curvature[idx])
        ay_required = progress_speed * progress_speed * abs(curvature)
        ay_limit = float(
            model.lateral_accel_limit(
                speed=body_speed,
                banking=float(track.banking[idx]),
            )
        )
        lateral_penalty += max(0.0, ay_required - ay_limit) ** 2
        tracking_penalty += (
            (yaw_rate[idx] - progress_speed * curvature) ** 2 + vy[idx] * vy[idx]
        )

        max_accel = float(
            model.max_longitudinal_accel(
                speed=body_speed,
                lateral_accel_required=ay_required,
                grade=float(track.grade[idx]),
                banking=float(track.banking[idx]),
            )
        )
        max_brake = float(
            model.max_longitudinal_decel(
                speed=body_speed,
                lateral_accel_required=ay_required,
                grade=float(track.grade[idx]),
                banking=float(track.banking[idx]),
            )
        )
        commanded = float(np.clip(ax_signal[idx], -max_brake, max_accel))
        ax_cmd[idx] = commanded
        longitudinal_accel[idx] = commanded
        lateral_accel[idx] = progress_speed * progress_speed * curvature

        dt = segment_time_step(
            segment_length=float(ds[idx]),
            speed=progress_speed,
            min_time_step=min_time_step,
            max_time_step=max_time_step,
        )
        max_delta_steer = max_steer_rate * dt
        steer_prev = steer_cmd[idx]
        steer_rate_limited = np.clip(
            steer_target[idx],
            steer_prev - max_delta_steer,
            steer_prev + max_delta_steer,
        )
        steer_cmd[idx] = float(np.clip(steer_rate_limited, -max_steer_angle, max_steer_angle))

        state = np.array([vx[idx], vy[idx], yaw_rate[idx]], dtype=np.float64)

        steer_sample = float(steer_cmd[idx])
        command_sample = float(commanded)

        def rhs(
            _: float,
            values: np.ndarray,
            steer_value: float = steer_sample,
            longitudinal_command: float = command_sample,
        ) -> np.ndarray:
            state_value = VehicleState(
                vx=float(values[0]),
                vy=float(values[1]),
                yaw_rate=float(values[2]),
            )
            control_value = ControlInput(
                steer=steer_value,
                longitudinal_accel_cmd=longitudinal_command,
            )
            derivatives = dynamics.derivatives(state_value, control_value)
            return np.array(
                [derivatives.vx, derivatives.vy, derivatives.yaw_rate],
                dtype=np.float64,
            )

        if method == "euler":
            next_state = state + dt * rhs(0.0, state)
        else:
            next_state = rk4_step(rhs=rhs, time=0.0, state=state, dtime=dt)

        vx[idx + 1] = float(np.clip(next_state[0], SMALL_EPS, max_speed))
        vy_limit = _MAX_TRANSIENT_SIDESLIP_RATIO * max(vx[idx + 1], SMALL_EPS)
        vy[idx + 1] = float(np.clip(next_state[1], -vy_limit, vy_limit))
        yaw_rate[idx + 1] = float(next_state[2])
        speed[idx + 1] = float(vx[idx + 1])
        time[idx + 1] = time[idx] + dt
        next_track_progress = _maybe_emit_track_progress(
            progress_prefix=track_progress_prefix,
            segment_idx=idx,
            segment_count=n - 1,
            next_fraction_threshold=next_track_progress,
        )

    if n > 1:
        longitudinal_accel[-1] = longitudinal_accel[-2]
        lateral_accel[-1] = speed[-1] * speed[-1] * float(track.curvature[-1])
        steer_cmd[-1] = steer_cmd[-2]
        ax_cmd[-1] = ax_cmd[-2]

    lap_time = float(time[-1])
    smooth_penalty = float(
        np.sum(np.diff(ax_cmd) ** 2) + np.sum(np.diff(steer_cmd) ** 2)
    )
    objective = (
        lap_time
        + config.transient.numerics.lateral_constraint_weight * lateral_penalty
        + config.transient.numerics.tracking_weight * tracking_penalty
        + config.transient.numerics.control_smoothness_weight * smooth_penalty
    )

    return TransientProfileResult(
        speed=speed,
        longitudinal_accel=longitudinal_accel,
        lateral_accel=lateral_accel,
        lap_time=lap_time,
        time=time,
        vx=vx,
        vy=vy,
        yaw_rate=yaw_rate,
        steer_cmd=steer_cmd,
        ax_cmd=ax_cmd,
        objective_value=float(objective),
    )


def _simulate_point_mass_pid_driver(
    *,
    track: TrackData,
    model: Any,
    config: SimulationConfig,
    reference_speed: np.ndarray,
    reference_ax: np.ndarray,
    track_progress_prefix: str | None = None,
) -> TransientProfileResult:
    """Simulate point-mass transient lap with closed-loop PID longitudinal control.

    Args:
        track: Track geometry.
        model: Point-mass model.
        config: Simulation config.
        reference_speed: Reference speed profile [m/s].
        reference_ax: Reference longitudinal-acceleration profile [m/s^2].
        track_progress_prefix: Optional progress prefix shown during
            arc-length integration.

    Returns:
        Transient profile result.
    """
    n = track.arc_length.size
    ds = np.diff(track.arc_length)

    speed = np.zeros(n, dtype=float)
    longitudinal_accel = np.zeros(n, dtype=float)
    lateral_accel = np.zeros(n, dtype=float)
    time = np.zeros(n, dtype=float)
    vx = np.zeros(n, dtype=float)
    vy = np.zeros(n, dtype=float)
    yaw_rate = np.zeros(n, dtype=float)
    steer_cmd = np.zeros(n, dtype=float)
    ax_cmd = np.zeros(n, dtype=float)

    max_speed = float(config.runtime.max_speed)
    start_speed = (
        max_speed
        if config.runtime.initial_speed is None
        else float(config.runtime.initial_speed)
    )
    min_time_step = float(config.transient.numerics.min_time_step)
    max_time_step = float(config.transient.numerics.max_time_step)
    kp = float(config.transient.numerics.pid_longitudinal_kp)
    ki = float(config.transient.numerics.pid_longitudinal_ki)
    kd = float(config.transient.numerics.pid_longitudinal_kd)
    integral_limit = float(config.transient.numerics.pid_longitudinal_integral_limit)
    gain_scheduling = _resolve_pid_gain_scheduling(model=model, config=config)

    speed[0] = max(start_speed, 0.0)
    vx[0] = speed[0]

    speed_error_integral = 0.0
    previous_speed_error = 0.0
    lateral_penalty = 0.0
    next_track_progress = _TRACK_PROGRESS_FRACTION_STEP
    for idx in range(n - 1):
        speed_value = float(speed[idx])
        curvature = float(track.curvature[idx])
        ay_required = speed_value * speed_value * abs(curvature)
        ay_limit = float(
            model.lateral_accel_limit(
                speed=speed_value,
                banking=float(track.banking[idx]),
            )
        )
        lateral_penalty += max(0.0, ay_required - ay_limit) ** 2

        max_accel = float(
            model.max_longitudinal_accel(
                speed=speed_value,
                lateral_accel_required=ay_required,
                grade=float(track.grade[idx]),
                banking=float(track.banking[idx]),
            )
        )
        max_brake = float(
            model.max_longitudinal_decel(
                speed=speed_value,
                lateral_accel_required=ay_required,
                grade=float(track.grade[idx]),
                banking=float(track.banking[idx]),
            )
        )

        dt = segment_time_step(
            segment_length=float(ds[idx]),
            speed=max(speed_value, SMALL_EPS),
            min_time_step=min_time_step,
            max_time_step=max_time_step,
        )
        speed_error = float(reference_speed[idx] - speed_value)
        speed_error_integral = _clamp_integral(
            speed_error_integral + speed_error * dt,
            integral_limit,
        )
        speed_error_derivative = (speed_error - previous_speed_error) / max(dt, SMALL_EPS)
        previous_speed_error = speed_error
        kp_value = _schedule_or_default(
            schedule=(
                gain_scheduling.longitudinal_kp
                if gain_scheduling is not None
                else None
            ),
            default_value=kp,
            speed_mps=speed_value,
        )
        ki_value = _schedule_or_default(
            schedule=(
                gain_scheduling.longitudinal_ki
                if gain_scheduling is not None
                else None
            ),
            default_value=ki,
            speed_mps=speed_value,
        )
        kd_value = _schedule_or_default(
            schedule=(
                gain_scheduling.longitudinal_kd
                if gain_scheduling is not None
                else None
            ),
            default_value=kd,
            speed_mps=speed_value,
        )

        commanded_unbounded = (
            float(reference_ax[idx])
            + kp_value * speed_error
            + ki_value * speed_error_integral
            + kd_value * speed_error_derivative
        )
        commanded = float(np.clip(commanded_unbounded, -max_brake, max_accel))
        ax_cmd[idx] = commanded
        longitudinal_accel[idx] = commanded
        lateral_accel[idx] = speed_value * speed_value * curvature

        next_speed = np.clip(speed_value + commanded * dt, 0.0, max_speed)
        speed[idx + 1] = float(next_speed)
        vx[idx + 1] = speed[idx + 1]
        time[idx + 1] = time[idx] + dt
        next_track_progress = _maybe_emit_track_progress(
            progress_prefix=track_progress_prefix,
            segment_idx=idx,
            segment_count=n - 1,
            next_fraction_threshold=next_track_progress,
        )

    if n > 1:
        longitudinal_accel[-1] = longitudinal_accel[-2]
        lateral_accel[-1] = speed[-1] * speed[-1] * float(track.curvature[-1])
        ax_cmd[-1] = ax_cmd[-2]

    lap_time = float(time[-1])
    smooth_penalty = float(np.sum(np.diff(ax_cmd) ** 2))
    objective = (
        lap_time
        + config.transient.numerics.lateral_constraint_weight * lateral_penalty
        + config.transient.numerics.control_smoothness_weight * smooth_penalty
    )

    return TransientProfileResult(
        speed=speed,
        longitudinal_accel=longitudinal_accel,
        lateral_accel=lateral_accel,
        lap_time=lap_time,
        time=time,
        vx=vx,
        vy=vy,
        yaw_rate=yaw_rate,
        steer_cmd=steer_cmd,
        ax_cmd=ax_cmd,
        objective_value=float(objective),
    )


def _simulate_single_track_pid_driver(
    *,
    track: TrackData,
    model: Any,
    config: SimulationConfig,
    reference_speed: np.ndarray,
    reference_ax: np.ndarray,
    reference_steer: np.ndarray,
    track_progress_prefix: str | None = None,
) -> TransientProfileResult:
    """Simulate single-track transient lap with PID longitudinal/steering control.

    Args:
        track: Track geometry.
        model: Single-track model.
        config: Simulation config.
        reference_speed: Reference speed profile [m/s].
        reference_ax: Reference longitudinal-acceleration profile [m/s^2].
        reference_steer: Reference steering profile [rad].
        track_progress_prefix: Optional progress prefix shown during
            arc-length integration.

    Returns:
        Transient profile result.
    """
    n = track.arc_length.size
    ds = np.diff(track.arc_length)

    speed = np.zeros(n, dtype=float)
    longitudinal_accel = np.zeros(n, dtype=float)
    lateral_accel = np.zeros(n, dtype=float)
    time = np.zeros(n, dtype=float)
    vx = np.zeros(n, dtype=float)
    vy = np.zeros(n, dtype=float)
    yaw_rate = np.zeros(n, dtype=float)
    steer_cmd = np.zeros(n, dtype=float)
    ax_cmd = np.zeros(n, dtype=float)

    max_speed = float(config.runtime.max_speed)
    start_speed = (
        max_speed
        if config.runtime.initial_speed is None
        else float(config.runtime.initial_speed)
    )
    min_time_step = float(config.transient.numerics.min_time_step)
    max_time_step = float(config.transient.numerics.max_time_step)
    method = config.transient.numerics.integration_method
    max_steer_angle = float(model.physics.max_steer_angle)
    max_steer_rate = float(model.physics.max_steer_rate)

    longitudinal_kp = float(config.transient.numerics.pid_longitudinal_kp)
    longitudinal_ki = float(config.transient.numerics.pid_longitudinal_ki)
    longitudinal_kd = float(config.transient.numerics.pid_longitudinal_kd)
    longitudinal_integral_limit = float(
        config.transient.numerics.pid_longitudinal_integral_limit
    )

    steer_kp = float(config.transient.numerics.pid_steer_kp)
    steer_ki = float(config.transient.numerics.pid_steer_ki)
    steer_kd = float(config.transient.numerics.pid_steer_kd)
    steer_vy_damping = float(config.transient.numerics.pid_steer_vy_damping)
    steer_integral_limit = float(config.transient.numerics.pid_steer_integral_limit)
    gain_scheduling = _resolve_pid_gain_scheduling(model=model, config=config)

    vx[0] = max(start_speed, SMALL_EPS)
    vy[0] = 0.0
    yaw_rate[0] = vx[0] * float(track.curvature[0])
    speed[0] = float(vx[0])
    steer_cmd[0] = float(np.clip(reference_steer[0], -max_steer_angle, max_steer_angle))

    dynamics = model._dynamics
    lateral_penalty = 0.0
    tracking_penalty = 0.0
    longitudinal_integral = 0.0
    steer_integral = 0.0
    previous_speed_error = 0.0
    previous_yaw_error = 0.0
    next_track_progress = _TRACK_PROGRESS_FRACTION_STEP

    for idx in range(n - 1):
        progress_speed = max(float(vx[idx]), SMALL_EPS)
        body_speed = max(float(np.hypot(vx[idx], vy[idx])), SMALL_EPS)
        curvature = float(track.curvature[idx])
        ay_required = progress_speed * progress_speed * abs(curvature)
        ay_limit = float(
            model.lateral_accel_limit(
                speed=body_speed,
                banking=float(track.banking[idx]),
            )
        )
        lateral_penalty += max(0.0, ay_required - ay_limit) ** 2
        tracking_penalty += (
            (yaw_rate[idx] - progress_speed * curvature) ** 2 + vy[idx] * vy[idx]
        )

        max_accel = float(
            model.max_longitudinal_accel(
                speed=body_speed,
                lateral_accel_required=ay_required,
                grade=float(track.grade[idx]),
                banking=float(track.banking[idx]),
            )
        )
        max_brake = float(
            model.max_longitudinal_decel(
                speed=body_speed,
                lateral_accel_required=ay_required,
                grade=float(track.grade[idx]),
                banking=float(track.banking[idx]),
            )
        )

        dt = segment_time_step(
            segment_length=float(ds[idx]),
            speed=progress_speed,
            min_time_step=min_time_step,
            max_time_step=max_time_step,
        )

        speed_error = float(reference_speed[idx] - progress_speed)
        longitudinal_integral = _clamp_integral(
            longitudinal_integral + speed_error * dt,
            longitudinal_integral_limit,
        )
        speed_error_derivative = (speed_error - previous_speed_error) / max(dt, SMALL_EPS)
        previous_speed_error = speed_error
        long_kp_value = _schedule_or_default(
            schedule=(
                gain_scheduling.longitudinal_kp
                if gain_scheduling is not None
                else None
            ),
            default_value=longitudinal_kp,
            speed_mps=progress_speed,
        )
        long_ki_value = _schedule_or_default(
            schedule=(
                gain_scheduling.longitudinal_ki
                if gain_scheduling is not None
                else None
            ),
            default_value=longitudinal_ki,
            speed_mps=progress_speed,
        )
        long_kd_value = _schedule_or_default(
            schedule=(
                gain_scheduling.longitudinal_kd
                if gain_scheduling is not None
                else None
            ),
            default_value=longitudinal_kd,
            speed_mps=progress_speed,
        )
        commanded_unbounded = (
            float(reference_ax[idx])
            + long_kp_value * speed_error
            + long_ki_value * longitudinal_integral
            + long_kd_value * speed_error_derivative
        )
        commanded = float(np.clip(commanded_unbounded, -max_brake, max_accel))
        ax_cmd[idx] = commanded
        longitudinal_accel[idx] = commanded
        lateral_accel[idx] = progress_speed * progress_speed * curvature

        yaw_rate_reference = float(reference_speed[idx] * curvature)
        yaw_error = yaw_rate_reference - float(yaw_rate[idx])
        steer_integral = _clamp_integral(
            steer_integral + yaw_error * dt,
            steer_integral_limit,
        )
        yaw_error_derivative = (yaw_error - previous_yaw_error) / max(dt, SMALL_EPS)
        previous_yaw_error = yaw_error
        steer_kp_value = _schedule_or_default(
            schedule=(gain_scheduling.steer_kp if gain_scheduling is not None else None),
            default_value=steer_kp,
            speed_mps=progress_speed,
        )
        steer_ki_value = _schedule_or_default(
            schedule=(gain_scheduling.steer_ki if gain_scheduling is not None else None),
            default_value=steer_ki,
            speed_mps=progress_speed,
        )
        steer_kd_value = _schedule_or_default(
            schedule=(gain_scheduling.steer_kd if gain_scheduling is not None else None),
            default_value=steer_kd,
            speed_mps=progress_speed,
        )
        steer_vy_damping_value = _schedule_or_default(
            schedule=(
                gain_scheduling.steer_vy_damping
                if gain_scheduling is not None
                else None
            ),
            default_value=steer_vy_damping,
            speed_mps=progress_speed,
        )
        steer_target = (
            float(reference_steer[idx])
            + steer_kp_value * yaw_error
            + steer_ki_value * steer_integral
            + steer_kd_value * yaw_error_derivative
            - steer_vy_damping_value * float(vy[idx])
        )
        steer_target = float(np.clip(steer_target, -max_steer_angle, max_steer_angle))

        max_delta_steer = max_steer_rate * dt
        steer_prev = steer_cmd[idx]
        steer_rate_limited = np.clip(
            steer_target,
            steer_prev - max_delta_steer,
            steer_prev + max_delta_steer,
        )
        steer_cmd[idx] = float(np.clip(steer_rate_limited, -max_steer_angle, max_steer_angle))

        state = np.array([vx[idx], vy[idx], yaw_rate[idx]], dtype=np.float64)
        steer_sample = float(steer_cmd[idx])
        command_sample = float(commanded)

        def rhs(
            _: float,
            values: np.ndarray,
            steer_value: float = steer_sample,
            longitudinal_command: float = command_sample,
        ) -> np.ndarray:
            state_value = VehicleState(
                vx=float(values[0]),
                vy=float(values[1]),
                yaw_rate=float(values[2]),
            )
            control_value = ControlInput(
                steer=steer_value,
                longitudinal_accel_cmd=longitudinal_command,
            )
            derivatives = dynamics.derivatives(state_value, control_value)
            return np.array(
                [derivatives.vx, derivatives.vy, derivatives.yaw_rate],
                dtype=np.float64,
            )

        if method == "euler":
            next_state = state + dt * rhs(0.0, state)
        else:
            next_state = rk4_step(rhs=rhs, time=0.0, state=state, dtime=dt)

        vx[idx + 1] = float(np.clip(next_state[0], SMALL_EPS, max_speed))
        vy_limit = _MAX_TRANSIENT_SIDESLIP_RATIO * max(vx[idx + 1], SMALL_EPS)
        vy[idx + 1] = float(np.clip(next_state[1], -vy_limit, vy_limit))
        yaw_rate[idx + 1] = float(next_state[2])
        speed[idx + 1] = float(vx[idx + 1])
        time[idx + 1] = time[idx] + dt
        next_track_progress = _maybe_emit_track_progress(
            progress_prefix=track_progress_prefix,
            segment_idx=idx,
            segment_count=n - 1,
            next_fraction_threshold=next_track_progress,
        )

    if n > 1:
        longitudinal_accel[-1] = longitudinal_accel[-2]
        lateral_accel[-1] = speed[-1] * speed[-1] * float(track.curvature[-1])
        steer_cmd[-1] = steer_cmd[-2]
        ax_cmd[-1] = ax_cmd[-2]

    lap_time = float(time[-1])
    smooth_penalty = float(
        np.sum(np.diff(ax_cmd) ** 2) + np.sum(np.diff(steer_cmd) ** 2)
    )
    objective = (
        lap_time
        + config.transient.numerics.lateral_constraint_weight * lateral_penalty
        + config.transient.numerics.tracking_weight * tracking_penalty
        + config.transient.numerics.control_smoothness_weight * smooth_penalty
    )

    return TransientProfileResult(
        speed=speed,
        longitudinal_accel=longitudinal_accel,
        lateral_accel=lateral_accel,
        lap_time=lap_time,
        time=time,
        vx=vx,
        vy=vy,
        yaw_rate=yaw_rate,
        steer_cmd=steer_cmd,
        ax_cmd=ax_cmd,
        objective_value=float(objective),
    )


def solve_transient_lap_numpy(
    *,
    track: TrackData,
    model: Any,
    config: SimulationConfig,
) -> TransientProfileResult:
    """Solve transient OC lap problem for numpy backend.

    Args:
        track: Track geometry.
        model: Vehicle model.
        config: Simulation configuration.

    Returns:
        Transient profile result.
    """
    track.validate()
    model.validate()
    config.validate()

    backend_label = str(config.runtime.compute_backend)
    show_iteration_progress = config.transient.runtime.verbosity >= 1
    show_track_progress = config.transient.runtime.verbosity >= 2
    driver_model = config.transient.runtime.driver_model
    is_single_track = _is_single_track_model(model)
    model_label = "single_track" if is_single_track else "point_mass"

    reference_profile = _transient_reference_profile(
        track=track,
        model=model,
        config=config,
    )
    reference_speed = np.asarray(reference_profile.speed, dtype=float)
    reference_ax = np.asarray(reference_profile.longitudinal_accel, dtype=float)
    if is_single_track:
        reference_steer = np.arctan(model.vehicle.wheelbase * track.curvature)
        reference_steer = np.clip(
            reference_steer,
            -float(model.physics.max_steer_angle),
            float(model.physics.max_steer_angle),
        )
    else:
        reference_steer = np.zeros_like(reference_ax)

    if driver_model == "pid":
        if show_iteration_progress:
            _render_progress_line(
                prefix=f"Transient PID ({backend_label})",
                fraction=0.0,
                suffix=f"model {model_label}",
                final=False,
            )
        track_progress_prefix = (
            f"Transient track ({backend_label}, {model_label}, pid)"
            if show_track_progress
            else None
        )
        if is_single_track:
            profile = _simulate_single_track_pid_driver(
                track=track,
                model=model,
                config=config,
                reference_speed=reference_speed,
                reference_ax=reference_ax,
                reference_steer=np.asarray(reference_steer, dtype=float),
                track_progress_prefix=track_progress_prefix,
            )
        else:
            profile = _simulate_point_mass_pid_driver(
                track=track,
                model=model,
                config=config,
                reference_speed=reference_speed,
                reference_ax=reference_ax,
                track_progress_prefix=track_progress_prefix,
            )
        if show_iteration_progress:
            _render_progress_line(
                prefix=f"Transient PID ({backend_label})",
                fraction=1.0,
                suffix=f"done objective {profile.objective_value:.4f}",
                final=True,
            )
        return profile

    optimize = _require_scipy_optimize()
    sample_count = track.arc_length.size
    mesh_positions = _build_control_mesh_positions(
        sample_count=sample_count,
        control_interval=int(config.transient.numerics.control_interval),
    )
    control_node_count = int(mesh_positions.size)

    ax_seed = reference_ax
    steer_seed = np.asarray(reference_steer, dtype=float)
    if is_single_track:
        initial_raw = _encode_single_track_controls(
            model=model,
            ax_seed=ax_seed,
            steer_seed=steer_seed,
            mesh_positions=mesh_positions,
        )
    else:
        initial_raw = _encode_point_mass_controls(
            model=model,
            ax_seed=ax_seed,
            mesh_positions=mesh_positions,
        )

    max_iterations = int(config.transient.numerics.max_iterations)
    objective_evaluations = 0
    latest_objective = float("nan")
    iteration_count = 0

    def objective(raw_controls: np.ndarray) -> float:
        nonlocal objective_evaluations
        nonlocal latest_objective
        objective_evaluations += 1
        enable_track_progress = show_track_progress and (
            objective_evaluations == 1
            or objective_evaluations % _TRACK_PROGRESS_EVAL_STRIDE == 0
        )
        track_progress_prefix = (
            f"Transient track ({backend_label}, {model_label}, "
            f"eval {objective_evaluations}, nodes {control_node_count})"
            if enable_track_progress
            else None
        )
        if is_single_track:
            ax_signal, steer_target = _decode_single_track_controls(
                model,
                raw_controls,
                sample_count=sample_count,
                mesh_positions=mesh_positions,
            )
            profile = _simulate_single_track_controls(
                track=track,
                model=model,
                config=config,
                ax_signal=ax_signal,
                steer_target=steer_target,
                track_progress_prefix=track_progress_prefix,
            )
            latest_objective = float(profile.objective_value)
            return latest_objective

        ax_signal = _decode_point_mass_controls(
            model=model,
            raw_ax=raw_controls,
            sample_count=sample_count,
            mesh_positions=mesh_positions,
        )
        profile = _simulate_point_mass_controls(
            track=track,
            model=model,
            config=config,
            ax_signal=ax_signal,
            track_progress_prefix=track_progress_prefix,
        )
        latest_objective = float(profile.objective_value)
        return latest_objective

    def optimization_callback(_: np.ndarray) -> None:
        nonlocal iteration_count
        if not show_iteration_progress:
            return
        iteration_count += 1
        _render_progress_line(
            prefix=f"Transient OC ({backend_label})",
            fraction=iteration_count / max(max_iterations, 1),
            suffix=(
                f"iter {iteration_count}/{max_iterations} "
                f"evals {objective_evaluations} "
                f"objective {latest_objective:.4f}"
            ),
            final=False,
        )

    result = optimize.minimize(
        objective,
        x0=np.asarray(initial_raw, dtype=float),
        method="L-BFGS-B",
        callback=optimization_callback,
        options={
            "maxiter": max_iterations,
            "ftol": float(config.transient.numerics.tolerance),
            "disp": bool(config.transient.runtime.verbosity > 0),
        },
    )
    if show_iteration_progress:
        completed_iterations = max(iteration_count, int(getattr(result, "nit", 0)))
        _render_progress_line(
            prefix=f"Transient OC ({backend_label})",
            fraction=completed_iterations / max(max_iterations, 1),
            suffix=(
                f"iter {completed_iterations}/{max_iterations} "
                f"evals {objective_evaluations} "
                f"objective {float(getattr(result, 'fun', latest_objective)):.4f}"
            ),
            final=True,
        )

    best_controls = (
        np.asarray(result.x, dtype=float)
        if np.isfinite(result.fun)
        else np.asarray(initial_raw, dtype=float)
    )

    if is_single_track:
        ax_signal, steer_target = _decode_single_track_controls(
            model,
            best_controls,
            sample_count=sample_count,
            mesh_positions=mesh_positions,
        )
        return _simulate_single_track_controls(
            track=track,
            model=model,
            config=config,
            ax_signal=ax_signal,
            steer_target=steer_target,
        )

    ax_signal = _decode_point_mass_controls(
        model=model,
        raw_ax=best_controls,
        sample_count=sample_count,
        mesh_positions=mesh_positions,
    )
    return _simulate_point_mass_controls(
        track=track,
        model=model,
        config=config,
        ax_signal=ax_signal,
    )
