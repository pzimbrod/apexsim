"""Torch-backed transient optimal-control lap solver."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from apexsim.simulation._progress import (
    DEFAULT_TRACK_PROGRESS_FRACTION_STEP,
)
from apexsim.simulation._progress import (
    maybe_emit_track_progress as _maybe_emit_track_progress,
)
from apexsim.simulation._progress import (
    render_progress_line as _render_progress_line,
)
from apexsim.simulation._transient_controls_core import (
    ControlInterpolationMap,
    build_control_interpolation_map,
    build_control_mesh_positions,
)
from apexsim.simulation._transient_controls_core import (
    build_control_node_count as _build_control_node_count,
)
from apexsim.simulation._transient_pid_core import (
    resolve_initial_speed,
)
from apexsim.simulation._transient_pid_core import (
    segment_time_partition_torch as _segment_time_partition_torch,
)
from apexsim.simulation.config import SimulationConfig
from apexsim.simulation.torch_profile import solve_speed_profile_torch
from apexsim.simulation.transient_common import (
    PidSpeedSchedule,
    TransientPidGainSchedulingConfig,
    TransientProfileResult,
)
from apexsim.track.models import TrackData
from apexsim.utils.constants import SMALL_EPS
from apexsim.utils.exceptions import ConfigurationError
from apexsim.vehicle._backend_physics_core import (
    roll_stiffness_front_share_numpy,
    single_track_wheel_loads_torch,
)

_TRACK_PROGRESS_FRACTION_STEP = DEFAULT_TRACK_PROGRESS_FRACTION_STEP
_TRACK_PROGRESS_EVAL_STRIDE = 20
_MAX_TRANSIENT_SIDESLIP_RATIO = 0.35
_MAX_SINGLE_TRACK_PID_INTEGRATION_STEP = 0.05


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
            "Torch transient OC solver requires PyTorch. "
            "Install with `pip install -e '.[torch]'`."
        )
        raise ConfigurationError(msg) from exc
    return torch


def _require_torchdiffeq() -> Any:
    """Import torchdiffeq lazily and fail with clear guidance.

    Returns:
        Imported ``torchdiffeq`` module.

    Raises:
        apexsim.utils.exceptions.ConfigurationError: If torchdiffeq is not installed.
    """
    try:
        import torchdiffeq  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        msg = (
            "Torch transient OC solver requires torchdiffeq. "
            "Install or update dependencies with `pip install -e .`."
        )
        raise ConfigurationError(msg) from exc
    return torchdiffeq


@dataclass(frozen=True)
class TorchTransientProfileResult:
    """Differentiable torch transient-lap result tensors.

    Args:
        speed: Total speed trace [m/s].
        longitudinal_accel: Net longitudinal acceleration trace [m/s^2].
        lateral_accel: Lateral acceleration trace [m/s^2].
        lap_time: Integrated lap time [s].
        time: Cumulative time trace [s].
        vx: Body-frame longitudinal speed trace [m/s].
        vy: Body-frame lateral speed trace [m/s].
        yaw_rate: Yaw-rate trace [rad/s].
        steer_cmd: Steering command trace [rad].
        ax_cmd: Longitudinal acceleration command trace [m/s^2].
        objective_value: Final OC objective value.
    """

    speed: Any
    longitudinal_accel: Any
    lateral_accel: Any
    lap_time: Any
    time: Any
    vx: Any
    vy: Any
    yaw_rate: Any
    steer_cmd: Any
    ax_cmd: Any
    objective_value: Any

    def to_numpy(self) -> TransientProfileResult:
        """Convert differentiable tensors to NumPy arrays.

        Returns:
            NumPy transient profile result detached from autograd graph.
        """
        return TransientProfileResult(
            speed=np.asarray(self.speed.detach().cpu().numpy(), dtype=float),
            longitudinal_accel=np.asarray(
                self.longitudinal_accel.detach().cpu().numpy(),
                dtype=float,
            ),
            lateral_accel=np.asarray(self.lateral_accel.detach().cpu().numpy(), dtype=float),
            lap_time=float(self.lap_time.detach().cpu().item()),
            time=np.asarray(self.time.detach().cpu().numpy(), dtype=float),
            vx=np.asarray(self.vx.detach().cpu().numpy(), dtype=float),
            vy=np.asarray(self.vy.detach().cpu().numpy(), dtype=float),
            yaw_rate=np.asarray(self.yaw_rate.detach().cpu().numpy(), dtype=float),
            steer_cmd=np.asarray(self.steer_cmd.detach().cpu().numpy(), dtype=float),
            ax_cmd=np.asarray(self.ax_cmd.detach().cpu().numpy(), dtype=float),
            objective_value=float(self.objective_value.detach().cpu().item()),
        )


def _from_numpy_profile(
    *,
    torch: Any,
    profile: TransientProfileResult,
    device: str,
) -> TorchTransientProfileResult:
    """Build torch transient result tensors from NumPy transient output.

    Args:
        torch: Imported torch module.
        profile: NumPy transient profile.
        device: Torch device identifier.

    Returns:
        Torch-backed transient profile result.
    """
    dtype = torch.float64
    return TorchTransientProfileResult(
        speed=torch.as_tensor(profile.speed, dtype=dtype, device=device),
        longitudinal_accel=torch.as_tensor(
            profile.longitudinal_accel,
            dtype=dtype,
            device=device,
        ),
        lateral_accel=torch.as_tensor(profile.lateral_accel, dtype=dtype, device=device),
        lap_time=torch.as_tensor(profile.lap_time, dtype=dtype, device=device),
        time=torch.as_tensor(profile.time, dtype=dtype, device=device),
        vx=torch.as_tensor(profile.vx, dtype=dtype, device=device),
        vy=torch.as_tensor(profile.vy, dtype=dtype, device=device),
        yaw_rate=torch.as_tensor(profile.yaw_rate, dtype=dtype, device=device),
        steer_cmd=torch.as_tensor(profile.steer_cmd, dtype=dtype, device=device),
        ax_cmd=torch.as_tensor(profile.ax_cmd, dtype=dtype, device=device),
        objective_value=torch.as_tensor(profile.objective_value, dtype=dtype, device=device),
    )


def _is_single_track_model(model: Any) -> bool:
    """Return whether model exposes single-track transient dynamics internals.

    Args:
        model: Solver model instance.

    Returns:
        ``True`` if single-track transient dynamics are available.
    """
    return (
        hasattr(model, "_backend_magic_formula_lateral")
        and (
            hasattr(model, "_backend_wheel_loads")
            or hasattr(model, "_backend_axle_tire_loads")
        )
        and hasattr(model, "tires")
        and hasattr(model, "physics")
        and hasattr(model.physics, "max_steer_angle")
        and hasattr(model.physics, "max_steer_rate")
    )


def _resolve_pid_gain_scheduling_torch(
    *,
    model: Any,
    config: SimulationConfig,
) -> TransientPidGainSchedulingConfig | None:
    """Resolve active PID gain scheduling strategy for torch transient PID mode.

    Args:
        model: Vehicle model used for optional physics-informed scheduling.
        config: Simulation configuration.

    Returns:
        Active scheduling config, or ``None`` when disabled.
    """
    mode = config.transient.numerics.pid_gain_scheduling_mode
    if mode == "off":
        return None
    if mode == "physics_informed":
        from apexsim.simulation.transient_numpy import (
            build_physics_informed_pid_gain_scheduling,
        )

        return build_physics_informed_pid_gain_scheduling(
            model=model,
            numerics=config.transient.numerics,
            max_speed=float(config.runtime.max_speed),
        )
    return config.transient.numerics.pid_gain_scheduling


def _evaluate_pid_schedule_torch(
    *,
    torch: Any,
    schedule: PidSpeedSchedule,
    speed_mps: Any,
    dtype: Any,
    device: str,
) -> Any:
    """Evaluate PWL PID schedule with torch ops and boundary clamping.

    Args:
        torch: Imported torch module.
        schedule: Speed-dependent schedule.
        speed_mps: Speed tensor/scalar [m/s].
        dtype: Target torch dtype.
        device: Target torch device.

    Returns:
        Interpolated schedule value tensor.
    """
    speed = torch.as_tensor(speed_mps, dtype=dtype, device=device)
    nodes = torch.as_tensor(schedule.speed_nodes_mps, dtype=dtype, device=device)
    values = torch.as_tensor(schedule.values, dtype=dtype, device=device)
    clamped_speed = torch.clamp(speed, min=nodes[0], max=nodes[-1])

    upper_index = torch.bucketize(clamped_speed, nodes)
    upper_index = torch.clamp(upper_index, min=1, max=int(nodes.numel()) - 1)
    lower_index = upper_index - 1

    node_lower = nodes[lower_index]
    node_upper = nodes[upper_index]
    value_lower = values[lower_index]
    value_upper = values[upper_index]
    denominator = torch.clamp(node_upper - node_lower, min=SMALL_EPS)
    fraction = (clamped_speed - node_lower) / denominator
    return value_lower + fraction * (value_upper - value_lower)


def _schedule_or_default_torch(
    *,
    torch: Any,
    schedule: PidSpeedSchedule | None,
    default_value: float,
    speed_mps: Any,
    dtype: Any,
    device: str,
) -> Any:
    """Evaluate schedule in torch or return scalar default as torch tensor.

    Args:
        torch: Imported torch module.
        schedule: Optional speed-dependent schedule.
        default_value: Fallback scalar gain.
        speed_mps: Evaluation speed [m/s].
        dtype: Target torch dtype.
        device: Target torch device.

    Returns:
        Gain value tensor.
    """
    if schedule is None:
        return torch.as_tensor(default_value, dtype=dtype, device=device)
    return _evaluate_pid_schedule_torch(
        torch=torch,
        schedule=schedule,
        speed_mps=speed_mps,
        dtype=dtype,
        device=device,
    )


def _clamp_integral_torch(*, torch: Any, value: Any, limit: float, dtype: Any, device: str) -> Any:
    """Clamp PID integrator state symmetrically.

    Args:
        torch: Imported torch module.
        value: Integrator state tensor.
        limit: Absolute integrator bound.
        dtype: Target torch dtype.
        device: Target torch device.

    Returns:
        Clamped integrator state tensor.
    """
    limit_tensor = torch.as_tensor(limit, dtype=dtype, device=device)
    return torch.clamp(value, min=-limit_tensor, max=limit_tensor)


def _rk4_step_torch(*, rhs: Any, state: Any, dtime: Any) -> Any:
    """Advance state by one RK4 step using differentiable torch operations.

    Args:
        rhs: State derivative callable ``rhs(state)``.
        state: Current state tensor.
        dtime: Integration step tensor [s].

    Returns:
        Next state tensor.
    """
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dtime * k1)
    k3 = rhs(state + 0.5 * dtime * k2)
    k4 = rhs(state + dtime * k3)
    return state + (dtime / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _sample_seed_nodes_torch(
    *,
    torch: Any,
    seed: Any,
    control_count: int,
) -> Any:
    """Sample a full-resolution seed tensor onto control nodes.

    Args:
        torch: Imported torch module.
        seed: Full-resolution seed tensor.
        control_count: Desired control-node count.

    Returns:
        Seed tensor on control-node support.
    """
    if int(seed.numel()) == control_count:
        return seed
    seed_1d = seed.reshape(1, 1, -1)
    nodes = torch.nn.functional.interpolate(
        seed_1d,
        size=control_count,
        mode="linear",
        align_corners=True,
    )
    return nodes.reshape(-1)


def _expand_control_nodes_torch(
    *,
    torch: Any,
    node_values: Any,
    sample_count: int,
) -> Any:
    """Expand control-node values to full track resolution.

    Args:
        torch: Imported torch module.
        node_values: Control values at node support.
        sample_count: Desired full sample count.

    Returns:
        Full-resolution control tensor.
    """
    if int(node_values.numel()) == sample_count:
        return node_values
    node_1d = node_values.reshape(1, 1, -1)
    expanded = torch.nn.functional.interpolate(
        node_1d,
        size=sample_count,
        mode="linear",
        align_corners=True,
    )
    return expanded.reshape(-1)


def _expand_control_nodes_torch_with_map(
    *,
    torch: Any,
    node_values: Any,
    interpolation_map: ControlInterpolationMap,
    dtype: Any,
    device: str,
) -> Any:
    """Expand node controls via a precomputed shared interpolation map.

    Args:
        torch: Imported torch module.
        node_values: Control values on node support.
        interpolation_map: Precomputed mesh-to-sample interpolation map.
        dtype: Target tensor dtype.
        device: Target torch device string.

    Returns:
        Full-resolution control tensor on sample support.
    """
    left_index = torch.as_tensor(
        interpolation_map.left_index,
        dtype=torch.int64,
        device=device,
    )
    right_index = torch.as_tensor(
        interpolation_map.right_index,
        dtype=torch.int64,
        device=device,
    )
    right_weight = torch.as_tensor(
        interpolation_map.right_weight,
        dtype=dtype,
        device=device,
    )
    left = node_values[left_index]
    right = node_values[right_index]
    return left * (1.0 - right_weight) + right * right_weight


def _bounded_artanh_torch(torch: Any, value: Any, eps: float = 1e-6) -> Any:
    """Return stable inverse-tanh transform for bounded control seeds.

    Args:
        torch: Imported torch module.
        value: Tensor values in ``[-1, 1]``.
        eps: Boundary clipping epsilon.

    Returns:
        Inverse-tanh transformed tensor.
    """
    clipped = torch.clamp(value, min=-1.0 + eps, max=1.0 - eps)
    return torch.atanh(clipped)


def _decode_point_mass_controls_torch(
    torch: Any,
    model: Any,
    raw_ax: Any,
    *,
    sample_count: int | None = None,
    interpolation_map: ControlInterpolationMap | None = None,
) -> Any:
    """Decode bounded point-mass acceleration controls from raw variables.

    Args:
        torch: Imported torch module.
        model: Point-mass model.
        raw_ax: Raw unconstrained optimization tensor.
        sample_count: Optional full track sample count.
        interpolation_map: Optional precomputed mesh-to-sample interpolation
            map shared with NumPy solver paths.

    Returns:
        Bounded acceleration command tensor [m/s^2].
    """
    upper = float(model.envelope_physics.max_drive_accel)
    lower = -float(model.envelope_physics.max_brake_accel)
    midpoint = 0.5 * (upper + lower)
    half_span = 0.5 * (upper - lower)
    node_values = midpoint + half_span * torch.tanh(raw_ax)
    if sample_count is None:
        return node_values
    if interpolation_map is not None:
        return _expand_control_nodes_torch_with_map(
            torch=torch,
            node_values=node_values,
            interpolation_map=interpolation_map,
            dtype=node_values.dtype,
            device=str(node_values.device),
        )
    return _expand_control_nodes_torch(
        torch=torch,
        node_values=node_values,
        sample_count=sample_count,
    )


def _encode_point_mass_controls_torch(
    torch: Any,
    model: Any,
    ax_seed: Any,
    *,
    control_count: int | None = None,
) -> Any:
    """Encode point-mass acceleration seed into raw optimization variables.

    Args:
        torch: Imported torch module.
        model: Point-mass model.
        ax_seed: Seed acceleration tensor [m/s^2].
        control_count: Optional control-node count for optimization variables.

    Returns:
        Raw unconstrained optimization tensor.
    """
    upper = float(model.envelope_physics.max_drive_accel)
    lower = -float(model.envelope_physics.max_brake_accel)
    midpoint = 0.5 * (upper + lower)
    half_span = max(0.5 * (upper - lower), SMALL_EPS)
    seed_values = (
        _sample_seed_nodes_torch(
            torch=torch,
            seed=ax_seed,
            control_count=control_count,
        )
        if control_count is not None
        else ax_seed
    )
    normalized = (seed_values - midpoint) / half_span
    return _bounded_artanh_torch(torch, normalized)


def _decode_single_track_controls_torch(
    torch: Any,
    model: Any,
    raw_controls: Any,
    *,
    sample_count: int | None = None,
    interpolation_map: ControlInterpolationMap | None = None,
) -> tuple[Any, Any]:
    """Decode bounded single-track acceleration and steering controls.

    Args:
        torch: Imported torch module.
        model: Single-track model.
        raw_controls: Raw unconstrained optimization tensor of length ``2 * n``.
        sample_count: Optional full track sample count.
        interpolation_map: Optional precomputed mesh-to-sample interpolation
            map shared with NumPy solver paths.

    Returns:
        Tuple ``(ax_cmd, steer_target)``.
    """
    n = int(raw_controls.numel() // 2)
    raw_ax = raw_controls[:n]
    raw_steer = raw_controls[n:]
    ax_cmd = _decode_point_mass_controls_torch(
        torch,
        model,
        raw_ax,
        sample_count=sample_count,
        interpolation_map=interpolation_map,
    )
    steer_nodes = float(model.physics.max_steer_angle) * torch.tanh(raw_steer)
    if sample_count is None:
        return ax_cmd, steer_nodes
    if interpolation_map is not None:
        steer_target = _expand_control_nodes_torch_with_map(
            torch=torch,
            node_values=steer_nodes,
            interpolation_map=interpolation_map,
            dtype=steer_nodes.dtype,
            device=str(steer_nodes.device),
        )
        return ax_cmd, steer_target
    steer_target = _expand_control_nodes_torch(
        torch=torch,
        node_values=steer_nodes,
        sample_count=sample_count,
    )
    return ax_cmd, steer_target


def _encode_single_track_controls_torch(
    torch: Any,
    model: Any,
    ax_seed: Any,
    steer_seed: Any,
    *,
    control_count: int | None = None,
) -> Any:
    """Encode single-track seed signals into raw optimization variables.

    Args:
        torch: Imported torch module.
        model: Single-track model.
        ax_seed: Seed acceleration tensor [m/s^2].
        steer_seed: Seed steering tensor [rad].
        control_count: Optional control-node count for optimization variables.

    Returns:
        Raw unconstrained optimization tensor.
    """
    ax_raw = _encode_point_mass_controls_torch(
        torch,
        model,
        ax_seed,
        control_count=control_count,
    )
    max_angle = max(float(model.physics.max_steer_angle), SMALL_EPS)
    steer_values = (
        _sample_seed_nodes_torch(
            torch=torch,
            seed=steer_seed,
            control_count=control_count,
        )
        if control_count is not None
        else steer_seed
    )
    steer_raw = _bounded_artanh_torch(torch, steer_values / max_angle)
    return torch.cat((ax_raw, steer_raw), dim=0)


def _transient_seed_profile_torch(
    *,
    track: TrackData,
    model: Any,
    config: SimulationConfig,
    torch: Any,
) -> tuple[Any, Any]:
    """Build initial acceleration/steering seeds from quasi-static torch profile.

    Args:
        track: Track geometry.
        model: Solver model instance.
        config: Simulation config.
        torch: Imported torch module.

    Returns:
        Tuple ``(ax_seed, steer_seed)``.
    """
    quasi_config = replace(
        config,
        runtime=replace(
            config.runtime,
            solver_mode="quasi_static",
            enable_transient_refinement=False,
        ),
    )
    profile = solve_speed_profile_torch(track=track, model=model, config=quasi_config)

    ax_seed = profile.longitudinal_accel.detach()
    if _is_single_track_model(model):
        curvature = torch.as_tensor(
            track.curvature,
            dtype=ax_seed.dtype,
            device=ax_seed.device,
        )
        steer_seed = torch.atan(model.vehicle.wheelbase * curvature)
        steer_seed = torch.clamp(
            steer_seed,
            min=-float(model.physics.max_steer_angle),
            max=float(model.physics.max_steer_angle),
        )
    else:
        steer_seed = torch.zeros_like(ax_seed)
    return ax_seed, steer_seed


def _transient_reference_profile_torch(
    *,
    track: TrackData,
    model: Any,
    config: SimulationConfig,
) -> Any:
    """Solve quasi-static torch profile used as reference for PID transient mode.

    Args:
        track: Track geometry.
        model: Solver model.
        config: Full simulation configuration.

    Returns:
        Quasi-static torch speed-profile result.
    """
    quasi_config = replace(
        config,
        runtime=replace(
            config.runtime,
            solver_mode="quasi_static",
            enable_transient_refinement=False,
        ),
    )
    return solve_speed_profile_torch(track=track, model=model, config=quasi_config)


def _single_track_wheel_loads_torch(
    *,
    torch: Any,
    model: Any,
    speed: Any,
    longitudinal_accel: Any,
    lateral_accel: Any,
) -> tuple[Any, Any, Any, Any]:
    """Estimate wheel loads with transient load transfer terms.

    Args:
        torch: Imported torch module.
        model: Single-track model.
        speed: Body-speed tensor [m/s].
        longitudinal_accel: Net longitudinal acceleration command [m/s^2].
        lateral_accel: Kinematic lateral-acceleration estimate [m/s^2].

    Returns:
        Tuple ``(front_left, front_right, rear_left, rear_right)`` [N].
    """
    front_roll_share = roll_stiffness_front_share_numpy(
        front_spring_rate=model.vehicle.front_spring_rate,
        rear_spring_rate=model.vehicle.rear_spring_rate,
        front_arb_distribution=model.vehicle.front_arb_distribution,
    )
    (
        _front_axle_load,
        _rear_axle_load,
        front_left,
        front_right,
        rear_left,
        rear_right,
    ) = single_track_wheel_loads_torch(
        torch=torch,
        speed=speed,
        mass=model.vehicle.mass,
        downforce_scale=model._downforce_scale,
        front_downforce_share=model._front_downforce_share,
        front_weight_fraction=model.vehicle.front_weight_fraction,
        longitudinal_accel=longitudinal_accel,
        lateral_accel=lateral_accel,
        cg_height=model.vehicle.cg_height,
        wheelbase=model.vehicle.wheelbase,
        front_track=model.vehicle.front_track,
        rear_track=model.vehicle.rear_track,
        front_roll_stiffness_share=front_roll_share,
    )
    return front_left, front_right, rear_left, rear_right


def _single_track_derivatives_torch(
    *,
    torch: Any,
    model: Any,
    state: Any,
    steer: Any,
    longitudinal_accel_cmd: Any,
) -> Any:
    """Compute single-track dynamics derivatives for torch state tensors.

    Args:
        torch: Imported torch module.
        model: Single-track model.
        state: State tensor ``[vx, vy, yaw_rate]``.
        steer: Steering angle tensor [rad].
        longitudinal_accel_cmd: Longitudinal acceleration command tensor [m/s^2].

    Returns:
        State-derivative tensor ``[dvx, dvy, dyaw_rate]``.
    """
    vx = state[0]
    vy = state[1]
    yaw_rate = state[2]

    dtype = state.dtype
    device = state.device
    u = torch.clamp(torch.abs(vx), min=0.5)
    cg_to_front = torch.as_tensor(model.vehicle.cg_to_front_axle, dtype=dtype, device=device)
    cg_to_rear = torch.as_tensor(model.vehicle.cg_to_rear_axle, dtype=dtype, device=device)

    alpha_front = steer - torch.atan2(vy + cg_to_front * yaw_rate, u)
    alpha_rear = -torch.atan2(vy - cg_to_rear * yaw_rate, u)

    speed = torch.sqrt(vx * vx + vy * vy)
    lateral_accel_estimate = vx * yaw_rate
    front_left, front_right, rear_left, rear_right = _single_track_wheel_loads_torch(
        torch=torch,
        model=model,
        speed=speed,
        longitudinal_accel=longitudinal_accel_cmd,
        lateral_accel=lateral_accel_estimate,
    )
    fy_front = model._backend_magic_formula_lateral(
        torch=torch,
        slip_angle=alpha_front,
        normal_load=front_left,
        params=model.tires.front,
    ) + model._backend_magic_formula_lateral(
        torch=torch,
        slip_angle=alpha_front,
        normal_load=front_right,
        params=model.tires.front,
    )
    fy_rear = model._backend_magic_formula_lateral(
        torch=torch,
        slip_angle=alpha_rear,
        normal_load=rear_left,
        params=model.tires.rear,
    ) + model._backend_magic_formula_lateral(
        torch=torch,
        slip_angle=alpha_rear,
        normal_load=rear_right,
        params=model.tires.rear,
    )

    yaw_moment = cg_to_front * fy_front * torch.cos(steer) - cg_to_rear * fy_rear
    mass = torch.as_tensor(model.vehicle.mass, dtype=dtype, device=device)
    # Longitudinal command is interpreted as net acceleration at the path tangent;
    # drag is already accounted for in acceleration-limit calculations.
    longitudinal_force = mass * longitudinal_accel_cmd
    dvx = (longitudinal_force - fy_front * torch.sin(steer) + mass * vy * yaw_rate) / mass
    dvy = (fy_rear + fy_front * torch.cos(steer) - mass * vx * yaw_rate) / mass
    yaw_inertia = torch.as_tensor(model.vehicle.yaw_inertia, dtype=dtype, device=device)
    dyaw = yaw_moment / yaw_inertia
    return torch.stack((dvx, dvy, dyaw))


def _simulate_point_mass_pid_torch(
    *,
    torch: Any,
    track: TrackData,
    model: Any,
    config: SimulationConfig,
    reference_speed: Any,
    reference_ax: Any,
    track_progress_prefix: str | None = None,
) -> TorchTransientProfileResult:
    """Simulate point-mass transient lap with differentiable PID control.

    Args:
        torch: Imported torch module.
        track: Track geometry.
        model: Point-mass model.
        config: Simulation config.
        reference_speed: Quasi-static reference speed [m/s].
        reference_ax: Quasi-static reference longitudinal acceleration [m/s^2].
        track_progress_prefix: Optional in-place progress-bar prefix.

    Returns:
        Differentiable transient profile result.
    """
    dtype = torch.float64
    device = config.runtime.torch_device
    curvature = torch.as_tensor(track.curvature, dtype=dtype, device=device)
    curvature_abs = torch.abs(curvature)
    grade = torch.as_tensor(track.grade, dtype=dtype, device=device)
    banking = torch.as_tensor(track.banking, dtype=dtype, device=device)
    ds = torch.as_tensor(np.diff(track.arc_length), dtype=dtype, device=device)

    max_speed = float(config.runtime.max_speed)
    start_speed = resolve_initial_speed(
        max_speed=max_speed,
        initial_speed=config.runtime.initial_speed,
    )

    min_time_step = float(config.transient.numerics.min_time_step)
    max_time_step = float(config.transient.numerics.max_time_step)
    kp = float(config.transient.numerics.pid_longitudinal_kp)
    ki = float(config.transient.numerics.pid_longitudinal_ki)
    kd = float(config.transient.numerics.pid_longitudinal_kd)
    integral_limit = float(config.transient.numerics.pid_longitudinal_integral_limit)
    gain_scheduling = _resolve_pid_gain_scheduling_torch(model=model, config=config)

    speed_values = [
        torch.as_tensor(start_speed, dtype=dtype, device=device),
    ]
    time_values = [torch.zeros((), dtype=dtype, device=device)]
    ax_values: list[Any] = []
    ay_values: list[Any] = []
    ax_cmd_values: list[Any] = []

    speed_error_integral = torch.zeros((), dtype=dtype, device=device)
    previous_speed_error = torch.zeros((), dtype=dtype, device=device)
    lateral_penalty = torch.zeros((), dtype=dtype, device=device)
    next_track_progress = _TRACK_PROGRESS_FRACTION_STEP

    for idx in range(int(ds.numel())):
        speed = speed_values[-1]
        ay_required = speed * speed * curvature_abs[idx]
        ay_limit = model.lateral_accel_limit_torch(speed=speed, banking=banking[idx])
        lateral_penalty = lateral_penalty + torch.relu(ay_required - ay_limit) ** 2

        max_accel = model.max_longitudinal_accel_torch(
            speed=speed,
            lateral_accel_required=ay_required,
            grade=grade[idx],
            banking=banking[idx],
        )
        max_brake = model.max_longitudinal_decel_torch(
            speed=speed,
            lateral_accel_required=ay_required,
            grade=grade[idx],
            banking=banking[idx],
        )

        dt_total, integration_dt, integration_substeps = _segment_time_partition_torch(
            torch=torch,
            segment_length=ds[idx],
            speed=speed,
            min_time_step=min_time_step,
            max_time_step=max_time_step,
        )
        speed_error = reference_speed[idx] - speed
        speed_error_integral = _clamp_integral_torch(
            torch=torch,
            value=speed_error_integral + speed_error * dt_total,
            limit=integral_limit,
            dtype=dtype,
            device=device,
        )
        speed_error_derivative = (speed_error - previous_speed_error) / torch.clamp(
            dt_total,
            min=SMALL_EPS,
        )
        previous_speed_error = speed_error

        kp_value = _schedule_or_default_torch(
            torch=torch,
            schedule=(
                gain_scheduling.longitudinal_kp if gain_scheduling is not None else None
            ),
            default_value=kp,
            speed_mps=speed,
            dtype=dtype,
            device=device,
        )
        ki_value = _schedule_or_default_torch(
            torch=torch,
            schedule=(
                gain_scheduling.longitudinal_ki if gain_scheduling is not None else None
            ),
            default_value=ki,
            speed_mps=speed,
            dtype=dtype,
            device=device,
        )
        kd_value = _schedule_or_default_torch(
            torch=torch,
            schedule=(
                gain_scheduling.longitudinal_kd if gain_scheduling is not None else None
            ),
            default_value=kd,
            speed_mps=speed,
            dtype=dtype,
            device=device,
        )
        commanded_unbounded = (
            reference_ax[idx]
            + kp_value * speed_error
            + ki_value * speed_error_integral
            + kd_value * speed_error_derivative
        )
        commanded = torch.minimum(torch.maximum(commanded_unbounded, -max_brake), max_accel)
        ax_cmd_values.append(commanded)
        ax_values.append(commanded)
        ay_values.append(speed * speed * curvature[idx])

        next_speed = speed
        for _ in range(integration_substeps):
            next_speed = torch.clamp(
                next_speed + commanded * integration_dt,
                min=0.0,
                max=max_speed,
            )
        speed_values.append(next_speed)
        time_values.append(time_values[-1] + dt_total)
        next_track_progress = _maybe_emit_track_progress(
            progress_prefix=track_progress_prefix,
            segment_idx=idx,
            segment_count=int(ds.numel()),
            next_fraction_threshold=next_track_progress,
        )

    if int(ds.numel()) > 0:
        ax_values.append(ax_values[-1])
        ay_values.append(speed_values[-1] * speed_values[-1] * curvature[-1])
        ax_cmd_values.append(ax_cmd_values[-1])
    else:
        zero = torch.zeros((), dtype=dtype, device=device)
        ax_values.append(zero)
        ay_values.append(zero)
        ax_cmd_values.append(zero)

    speed_tensor = torch.stack(speed_values)
    time_tensor = torch.stack(time_values)
    ax_tensor = torch.stack(ax_values)
    ay_tensor = torch.stack(ay_values)
    ax_cmd_tensor = torch.stack(ax_cmd_values)
    zeros = torch.zeros_like(speed_tensor)

    smooth_penalty = torch.sum((ax_cmd_tensor[1:] - ax_cmd_tensor[:-1]) ** 2)
    objective = (
        time_tensor[-1]
        + config.transient.numerics.lateral_constraint_weight * lateral_penalty
        + config.transient.numerics.control_smoothness_weight * smooth_penalty
    )

    return TorchTransientProfileResult(
        speed=speed_tensor,
        longitudinal_accel=ax_tensor,
        lateral_accel=ay_tensor,
        lap_time=time_tensor[-1],
        time=time_tensor,
        vx=speed_tensor,
        vy=zeros,
        yaw_rate=zeros,
        steer_cmd=zeros,
        ax_cmd=ax_cmd_tensor,
        objective_value=objective,
    )


def _simulate_single_track_pid_torch(
    *,
    torch: Any,
    track: TrackData,
    model: Any,
    config: SimulationConfig,
    reference_speed: Any,
    reference_ax: Any,
    reference_steer: Any,
    track_progress_prefix: str | None = None,
) -> TorchTransientProfileResult:
    """Simulate single-track transient lap with differentiable PID control.

    Args:
        torch: Imported torch module.
        track: Track geometry.
        model: Single-track model.
        config: Simulation config.
        reference_speed: Quasi-static reference speed [m/s].
        reference_ax: Quasi-static reference longitudinal acceleration [m/s^2].
        reference_steer: Quasi-static reference steering [rad].
        track_progress_prefix: Optional in-place progress-bar prefix.

    Returns:
        Differentiable transient profile result.
    """
    dtype = torch.float64
    device = config.runtime.torch_device
    curvature = torch.as_tensor(track.curvature, dtype=dtype, device=device)
    curvature_abs = torch.abs(curvature)
    grade = torch.as_tensor(track.grade, dtype=dtype, device=device)
    banking = torch.as_tensor(track.banking, dtype=dtype, device=device)
    ds = torch.as_tensor(np.diff(track.arc_length), dtype=dtype, device=device)

    max_speed = float(config.runtime.max_speed)
    start_speed = resolve_initial_speed(
        max_speed=max_speed,
        initial_speed=config.runtime.initial_speed,
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
    gain_scheduling = _resolve_pid_gain_scheduling_torch(model=model, config=config)

    state = torch.stack(
        (
            torch.as_tensor(max(start_speed, SMALL_EPS), dtype=dtype, device=device),
            torch.zeros((), dtype=dtype, device=device),
            torch.as_tensor(
                max(start_speed, SMALL_EPS) * float(track.curvature[0]),
                dtype=dtype,
                device=device,
            ),
        )
    )
    state_values = [state]
    speed_values = [torch.clamp(state[0], min=SMALL_EPS, max=max_speed)]
    time_values = [torch.zeros((), dtype=dtype, device=device)]
    ax_values: list[Any] = []
    ay_values: list[Any] = []
    steer_values: list[Any] = [
        torch.clamp(reference_steer[0], min=-max_steer_angle, max=max_steer_angle)
    ]
    ax_cmd_values: list[Any] = []

    lateral_penalty = torch.zeros((), dtype=dtype, device=device)
    tracking_penalty = torch.zeros((), dtype=dtype, device=device)
    longitudinal_integral = torch.zeros((), dtype=dtype, device=device)
    steer_integral = torch.zeros((), dtype=dtype, device=device)
    previous_speed_error = torch.zeros((), dtype=dtype, device=device)
    previous_yaw_error = torch.zeros((), dtype=dtype, device=device)
    next_track_progress = _TRACK_PROGRESS_FRACTION_STEP

    for idx in range(int(ds.numel())):
        state = state_values[-1]
        progress_speed = torch.clamp(state[0], min=SMALL_EPS, max=max_speed)
        body_speed = torch.sqrt(state[0] * state[0] + state[1] * state[1])
        ay_required = progress_speed * progress_speed * curvature_abs[idx]
        ay_limit = model.lateral_accel_limit_torch(speed=body_speed, banking=banking[idx])
        lateral_penalty = lateral_penalty + torch.relu(ay_required - ay_limit) ** 2
        tracking_penalty = (
            tracking_penalty
            + (state[2] - progress_speed * curvature[idx]) ** 2
            + state[1] ** 2
        )

        max_accel = model.max_longitudinal_accel_torch(
            speed=body_speed,
            lateral_accel_required=ay_required,
            grade=grade[idx],
            banking=banking[idx],
        )
        max_brake = model.max_longitudinal_decel_torch(
            speed=body_speed,
            lateral_accel_required=ay_required,
            grade=grade[idx],
            banking=banking[idx],
        )

        dt_total, integration_dt, integration_substeps = _segment_time_partition_torch(
            torch=torch,
            segment_length=ds[idx],
            speed=progress_speed,
            min_time_step=min_time_step,
            max_time_step=max_time_step,
            max_integration_step=_MAX_SINGLE_TRACK_PID_INTEGRATION_STEP,
        )
        speed_error = reference_speed[idx] - progress_speed
        longitudinal_integral = _clamp_integral_torch(
            torch=torch,
            value=longitudinal_integral + speed_error * dt_total,
            limit=longitudinal_integral_limit,
            dtype=dtype,
            device=device,
        )
        speed_error_derivative = (speed_error - previous_speed_error) / torch.clamp(
            dt_total,
            min=SMALL_EPS,
        )
        previous_speed_error = speed_error

        long_kp_value = _schedule_or_default_torch(
            torch=torch,
            schedule=(
                gain_scheduling.longitudinal_kp if gain_scheduling is not None else None
            ),
            default_value=longitudinal_kp,
            speed_mps=progress_speed,
            dtype=dtype,
            device=device,
        )
        long_ki_value = _schedule_or_default_torch(
            torch=torch,
            schedule=(
                gain_scheduling.longitudinal_ki if gain_scheduling is not None else None
            ),
            default_value=longitudinal_ki,
            speed_mps=progress_speed,
            dtype=dtype,
            device=device,
        )
        long_kd_value = _schedule_or_default_torch(
            torch=torch,
            schedule=(
                gain_scheduling.longitudinal_kd if gain_scheduling is not None else None
            ),
            default_value=longitudinal_kd,
            speed_mps=progress_speed,
            dtype=dtype,
            device=device,
        )
        commanded_unbounded = (
            reference_ax[idx]
            + long_kp_value * speed_error
            + long_ki_value * longitudinal_integral
            + long_kd_value * speed_error_derivative
        )
        commanded = torch.minimum(torch.maximum(commanded_unbounded, -max_brake), max_accel)
        ax_cmd_values.append(commanded)
        ax_values.append(commanded)
        ay_values.append(progress_speed * progress_speed * curvature[idx])

        yaw_rate_reference = reference_speed[idx] * curvature[idx]
        yaw_error = yaw_rate_reference - state[2]
        steer_integral = _clamp_integral_torch(
            torch=torch,
            value=steer_integral + yaw_error * dt_total,
            limit=steer_integral_limit,
            dtype=dtype,
            device=device,
        )
        yaw_error_derivative = (yaw_error - previous_yaw_error) / torch.clamp(
            dt_total,
            min=SMALL_EPS,
        )
        previous_yaw_error = yaw_error

        steer_kp_value = _schedule_or_default_torch(
            torch=torch,
            schedule=(gain_scheduling.steer_kp if gain_scheduling is not None else None),
            default_value=steer_kp,
            speed_mps=progress_speed,
            dtype=dtype,
            device=device,
        )
        steer_ki_value = _schedule_or_default_torch(
            torch=torch,
            schedule=(gain_scheduling.steer_ki if gain_scheduling is not None else None),
            default_value=steer_ki,
            speed_mps=progress_speed,
            dtype=dtype,
            device=device,
        )
        steer_kd_value = _schedule_or_default_torch(
            torch=torch,
            schedule=(gain_scheduling.steer_kd if gain_scheduling is not None else None),
            default_value=steer_kd,
            speed_mps=progress_speed,
            dtype=dtype,
            device=device,
        )
        steer_vy_damping_value = _schedule_or_default_torch(
            torch=torch,
            schedule=(
                gain_scheduling.steer_vy_damping if gain_scheduling is not None else None
            ),
            default_value=steer_vy_damping,
            speed_mps=progress_speed,
            dtype=dtype,
            device=device,
        )
        steer_target_value = (
            reference_steer[idx]
            + steer_kp_value * yaw_error
            + steer_ki_value * steer_integral
            + steer_kd_value * yaw_error_derivative
            - steer_vy_damping_value * state[1]
        )
        steer_target_value = torch.clamp(
            steer_target_value,
            min=-max_steer_angle,
            max=max_steer_angle,
        )

        steer_prev = steer_values[-1]
        steer_delta = max_steer_rate * dt_total
        steer_value = torch.clamp(
            steer_target_value,
            min=steer_prev - steer_delta,
            max=steer_prev + steer_delta,
        )
        steer_value = torch.clamp(steer_value, min=-max_steer_angle, max=max_steer_angle)
        steer_values.append(steer_value)

        def rhs(
            current_state: Any,
            steer_input: Any = steer_value,
            commanded_input: Any = commanded,
        ) -> Any:
            return _single_track_derivatives_torch(
                torch=torch,
                model=model,
                state=current_state,
                steer=steer_input,
                longitudinal_accel_cmd=commanded_input,
            )

        next_state = state
        for _ in range(integration_substeps):
            if method == "euler":
                next_state = next_state + integration_dt * rhs(next_state)
            else:
                next_state = _rk4_step_torch(
                    rhs=rhs,
                    state=next_state,
                    dtime=integration_dt,
                )

        next_vx = torch.clamp(next_state[0], min=SMALL_EPS, max=max_speed)
        vy_limit = _MAX_TRANSIENT_SIDESLIP_RATIO * torch.clamp(next_vx, min=SMALL_EPS)
        next_vy = torch.clamp(next_state[1], min=-vy_limit, max=vy_limit)
        next_state = torch.stack((next_vx, next_vy, next_state[2]))
        state_values.append(next_state)
        speed_values.append(torch.clamp(next_state[0], min=SMALL_EPS, max=max_speed))
        time_values.append(time_values[-1] + dt_total)
        next_track_progress = _maybe_emit_track_progress(
            progress_prefix=track_progress_prefix,
            segment_idx=idx,
            segment_count=int(ds.numel()),
            next_fraction_threshold=next_track_progress,
        )

    if int(ds.numel()) > 0:
        ax_values.append(ax_values[-1])
        ay_values.append(speed_values[-1] * speed_values[-1] * curvature[-1])
        ax_cmd_values.append(ax_cmd_values[-1])
    else:
        zero = torch.zeros((), dtype=dtype, device=device)
        ax_values.append(zero)
        ay_values.append(zero)
        ax_cmd_values.append(zero)

    state_tensor = torch.stack(state_values)
    speed_tensor = torch.stack(speed_values)
    time_tensor = torch.stack(time_values)
    ax_tensor = torch.stack(ax_values)
    ay_tensor = torch.stack(ay_values)
    steer_tensor = torch.stack(steer_values[: speed_tensor.shape[0]])
    ax_cmd_tensor = torch.stack(ax_cmd_values)

    smooth_penalty = torch.sum((ax_cmd_tensor[1:] - ax_cmd_tensor[:-1]) ** 2) + torch.sum(
        (steer_tensor[1:] - steer_tensor[:-1]) ** 2
    )
    objective = (
        time_tensor[-1]
        + config.transient.numerics.lateral_constraint_weight * lateral_penalty
        + config.transient.numerics.tracking_weight * tracking_penalty
        + config.transient.numerics.control_smoothness_weight * smooth_penalty
    )

    return TorchTransientProfileResult(
        speed=speed_tensor,
        longitudinal_accel=ax_tensor,
        lateral_accel=ay_tensor,
        lap_time=time_tensor[-1],
        time=time_tensor,
        vx=state_tensor[:, 0],
        vy=state_tensor[:, 1],
        yaw_rate=state_tensor[:, 2],
        steer_cmd=steer_tensor,
        ax_cmd=ax_cmd_tensor,
        objective_value=objective,
    )


def _simulate_point_mass_torch(
    *,
    torch: Any,
    track: TrackData,
    model: Any,
    config: SimulationConfig,
    ax_signal: Any,
    track_progress_prefix: str | None = None,
) -> TorchTransientProfileResult:
    """Simulate point-mass transient lap from bounded control sequence.

    Args:
        torch: Imported torch module.
        track: Track geometry.
        model: Point-mass model.
        config: Simulation configuration.
        ax_signal: Bounded acceleration command tensor [m/s^2].
        track_progress_prefix: Optional progress prefix shown during
            arc-length integration.

    Returns:
        Differentiable transient result.
    """
    dtype = torch.float64
    device = config.runtime.torch_device
    curvature = torch.as_tensor(track.curvature, dtype=dtype, device=device)
    curvature_abs = torch.abs(curvature)
    grade = torch.as_tensor(track.grade, dtype=dtype, device=device)
    banking = torch.as_tensor(track.banking, dtype=dtype, device=device)
    ds = torch.as_tensor(np.diff(track.arc_length), dtype=dtype, device=device)

    max_speed = float(config.runtime.max_speed)
    start_speed = resolve_initial_speed(
        max_speed=max_speed,
        initial_speed=config.runtime.initial_speed,
    )

    min_time_step = float(config.transient.numerics.min_time_step)
    max_time_step = float(config.transient.numerics.max_time_step)

    speed_values = [
        torch.as_tensor(start_speed, dtype=dtype, device=device),
    ]
    time_values = [torch.zeros((), dtype=dtype, device=device)]
    ax_values: list[Any] = []
    ay_values: list[Any] = []
    ax_cmd_values: list[Any] = []

    lateral_penalty = torch.zeros((), dtype=dtype, device=device)
    next_track_progress = _TRACK_PROGRESS_FRACTION_STEP
    for idx in range(int(ds.numel())):
        speed = speed_values[-1]
        ay_required = speed * speed * curvature_abs[idx]
        ay_limit = model.lateral_accel_limit_torch(speed=speed, banking=banking[idx])
        lateral_penalty = lateral_penalty + torch.relu(ay_required - ay_limit) ** 2

        max_accel = model.max_longitudinal_accel_torch(
            speed=speed,
            lateral_accel_required=ay_required,
            grade=grade[idx],
            banking=banking[idx],
        )
        max_brake = model.max_longitudinal_decel_torch(
            speed=speed,
            lateral_accel_required=ay_required,
            grade=grade[idx],
            banking=banking[idx],
        )
        commanded = torch.minimum(torch.maximum(ax_signal[idx], -max_brake), max_accel)
        ax_cmd_values.append(commanded)
        ax_values.append(commanded)
        ay_values.append(speed * speed * curvature[idx])

        dt = ds[idx] / torch.clamp(torch.abs(speed), min=SMALL_EPS)
        dt = torch.clamp(dt, min=min_time_step, max=max_time_step)
        next_speed = torch.clamp(speed + commanded * dt, min=0.0, max=max_speed)
        speed_values.append(next_speed)
        time_values.append(time_values[-1] + dt)
        next_track_progress = _maybe_emit_track_progress(
            progress_prefix=track_progress_prefix,
            segment_idx=idx,
            segment_count=int(ds.numel()),
            next_fraction_threshold=next_track_progress,
        )

    ax_values.append(ax_values[-1])
    ay_values.append(speed_values[-1] * speed_values[-1] * curvature[-1])
    ax_cmd_values.append(ax_cmd_values[-1])

    speed_tensor = torch.stack(speed_values)
    time_tensor = torch.stack(time_values)
    ax_tensor = torch.stack(ax_values)
    ay_tensor = torch.stack(ay_values)
    ax_cmd_tensor = torch.stack(ax_cmd_values)
    zeros = torch.zeros_like(speed_tensor)

    smooth_penalty = torch.sum((ax_cmd_tensor[1:] - ax_cmd_tensor[:-1]) ** 2)
    objective = (
        time_tensor[-1]
        + config.transient.numerics.lateral_constraint_weight * lateral_penalty
        + config.transient.numerics.control_smoothness_weight * smooth_penalty
    )

    return TorchTransientProfileResult(
        speed=speed_tensor,
        longitudinal_accel=ax_tensor,
        lateral_accel=ay_tensor,
        lap_time=time_tensor[-1],
        time=time_tensor,
        vx=speed_tensor,
        vy=zeros,
        yaw_rate=zeros,
        steer_cmd=zeros,
        ax_cmd=ax_cmd_tensor,
        objective_value=objective,
    )


def _simulate_single_track_torch(
    *,
    torch: Any,
    odeint: Any,
    track: TrackData,
    model: Any,
    config: SimulationConfig,
    ax_signal: Any,
    steer_target: Any,
    track_progress_prefix: str | None = None,
) -> TorchTransientProfileResult:
    """Simulate single-track transient lap from bounded control sequences.

    Args:
        torch: Imported torch module.
        odeint: ``torchdiffeq.odeint`` callable.
        track: Track geometry.
        model: Single-track model.
        config: Simulation configuration.
        ax_signal: Bounded acceleration-command tensor [m/s^2].
        steer_target: Bounded steering-target tensor [rad].
        track_progress_prefix: Optional progress prefix shown during
            arc-length integration.

    Returns:
        Differentiable transient result.
    """
    dtype = torch.float64
    device = config.runtime.torch_device
    curvature = torch.as_tensor(track.curvature, dtype=dtype, device=device)
    curvature_abs = torch.abs(curvature)
    grade = torch.as_tensor(track.grade, dtype=dtype, device=device)
    banking = torch.as_tensor(track.banking, dtype=dtype, device=device)
    ds = torch.as_tensor(np.diff(track.arc_length), dtype=dtype, device=device)

    max_speed = float(config.runtime.max_speed)
    start_speed = resolve_initial_speed(
        max_speed=max_speed,
        initial_speed=config.runtime.initial_speed,
    )
    min_time_step = float(config.transient.numerics.min_time_step)
    max_time_step = float(config.transient.numerics.max_time_step)
    max_steer_angle = float(model.physics.max_steer_angle)
    max_steer_rate = float(model.physics.max_steer_rate)

    state = torch.tensor(
        [max(start_speed, SMALL_EPS), 0.0, max(start_speed, SMALL_EPS) * track.curvature[0]],
        dtype=dtype,
        device=device,
    )

    state_values = [state]
    speed_values = [torch.clamp(state[0], min=SMALL_EPS, max=max_speed)]
    time_values = [torch.zeros((), dtype=dtype, device=device)]
    ax_values: list[Any] = []
    ay_values: list[Any] = []
    steer_values: list[Any] = [
        torch.clamp(steer_target[0], min=-max_steer_angle, max=max_steer_angle)
    ]
    ax_cmd_values: list[Any] = []

    method = config.transient.numerics.integration_method
    ode_method = "euler" if method == "euler" else "rk4"

    lateral_penalty = torch.zeros((), dtype=dtype, device=device)
    tracking_penalty = torch.zeros((), dtype=dtype, device=device)
    next_track_progress = _TRACK_PROGRESS_FRACTION_STEP

    for idx in range(int(ds.numel())):
        state = state_values[-1]
        progress_speed = torch.clamp(state[0], min=SMALL_EPS, max=max_speed)
        body_speed = torch.sqrt(state[0] * state[0] + state[1] * state[1])
        ay_required = progress_speed * progress_speed * curvature_abs[idx]
        ay_limit = model.lateral_accel_limit_torch(speed=body_speed, banking=banking[idx])
        lateral_penalty = lateral_penalty + torch.relu(ay_required - ay_limit) ** 2
        tracking_penalty = (
            tracking_penalty
            + (state[2] - progress_speed * curvature[idx]) ** 2
            + state[1] ** 2
        )

        max_accel = model.max_longitudinal_accel_torch(
            speed=body_speed,
            lateral_accel_required=ay_required,
            grade=grade[idx],
            banking=banking[idx],
        )
        max_brake = model.max_longitudinal_decel_torch(
            speed=body_speed,
            lateral_accel_required=ay_required,
            grade=grade[idx],
            banking=banking[idx],
        )
        commanded = torch.minimum(torch.maximum(ax_signal[idx], -max_brake), max_accel)
        ax_cmd_values.append(commanded)
        ax_values.append(commanded)
        ay_values.append(progress_speed * progress_speed * curvature[idx])

        dt = ds[idx] / torch.clamp(progress_speed, min=SMALL_EPS)
        dt = torch.clamp(dt, min=min_time_step, max=max_time_step)
        steer_prev = steer_values[-1]
        steer_delta = max_steer_rate * dt
        steer_value = torch.clamp(
            steer_target[idx],
            min=steer_prev - steer_delta,
            max=steer_prev + steer_delta,
        )
        steer_value = torch.clamp(steer_value, min=-max_steer_angle, max=max_steer_angle)
        steer_values.append(steer_value)

        steer_sample = steer_value
        command_sample = commanded

        def rhs(
            _: Any,
            values: Any,
            steer_input: Any = steer_sample,
            longitudinal_command: Any = command_sample,
        ) -> Any:
            return _single_track_derivatives_torch(
                torch=torch,
                model=model,
                state=values,
                steer=steer_input,
                longitudinal_accel_cmd=longitudinal_command,
            )

        integration_times = torch.stack(
            (
                torch.zeros((), dtype=dtype, device=device),
                dt,
            )
        )
        next_state = odeint(rhs, state, integration_times, method=ode_method)[-1]
        next_vx = torch.clamp(next_state[0], min=SMALL_EPS, max=max_speed)
        vy_limit = _MAX_TRANSIENT_SIDESLIP_RATIO * torch.clamp(next_vx, min=SMALL_EPS)
        next_vy = torch.clamp(next_state[1], min=-vy_limit, max=vy_limit)
        next_state = torch.stack((next_vx, next_vy, next_state[2]))
        state_values.append(next_state)
        speed_values.append(torch.clamp(next_state[0], min=SMALL_EPS, max=max_speed))
        time_values.append(time_values[-1] + dt)
        next_track_progress = _maybe_emit_track_progress(
            progress_prefix=track_progress_prefix,
            segment_idx=idx,
            segment_count=int(ds.numel()),
            next_fraction_threshold=next_track_progress,
        )

    ax_values.append(ax_values[-1])
    ay_values.append(speed_values[-1] * speed_values[-1] * curvature[-1])
    ax_cmd_values.append(ax_cmd_values[-1])

    state_tensor = torch.stack(state_values)
    speed_tensor = torch.stack(speed_values)
    time_tensor = torch.stack(time_values)
    ax_tensor = torch.stack(ax_values)
    ay_tensor = torch.stack(ay_values)
    steer_tensor = torch.stack(steer_values[: speed_tensor.shape[0]])
    ax_cmd_tensor = torch.stack(ax_cmd_values)

    smooth_penalty = torch.sum((ax_cmd_tensor[1:] - ax_cmd_tensor[:-1]) ** 2) + torch.sum(
        (steer_tensor[1:] - steer_tensor[:-1]) ** 2
    )
    objective = (
        time_tensor[-1]
        + config.transient.numerics.lateral_constraint_weight * lateral_penalty
        + config.transient.numerics.tracking_weight * tracking_penalty
        + config.transient.numerics.control_smoothness_weight * smooth_penalty
    )

    return TorchTransientProfileResult(
        speed=speed_tensor,
        longitudinal_accel=ax_tensor,
        lateral_accel=ay_tensor,
        lap_time=time_tensor[-1],
        time=time_tensor,
        vx=state_tensor[:, 0],
        vy=state_tensor[:, 1],
        yaw_rate=state_tensor[:, 2],
        steer_cmd=steer_tensor,
        ax_cmd=ax_cmd_tensor,
        objective_value=objective,
    )


def solve_transient_lap_torch(
    *,
    track: TrackData,
    model: Any,
    config: SimulationConfig,
) -> TorchTransientProfileResult:
    """Solve transient OC lap problem for torch backend.

    Args:
        track: Track geometry.
        model: Vehicle model.
        config: Simulation config.

    Returns:
        Differentiable transient profile result.
    """
    torch = _require_torch()

    track.validate()
    model.validate()
    config.validate()
    if config.runtime.compute_backend != "torch":
        msg = "solve_transient_lap_torch requires runtime.compute_backend='torch'"
        raise ConfigurationError(msg)

    if config.transient.runtime.deterministic_seed is not None:
        torch.manual_seed(int(config.transient.runtime.deterministic_seed))

    dtype = torch.float64
    device = config.runtime.torch_device
    is_single_track = _is_single_track_model(model)
    model_label = "single_track" if is_single_track else "point_mass"
    show_iteration_progress = config.transient.runtime.verbosity >= 1
    show_track_progress = config.transient.runtime.verbosity >= 2

    if config.transient.runtime.driver_model == "pid":
        if show_iteration_progress:
            _render_progress_line(
                prefix="Transient PID (torch)",
                fraction=0.0,
                suffix=f"model {model_label}",
                final=False,
            )
        reference_profile = _transient_reference_profile_torch(
            track=track,
            model=model,
            config=config,
        )
        reference_speed = reference_profile.speed
        reference_ax = reference_profile.longitudinal_accel
        if is_single_track:
            curvature = torch.as_tensor(track.curvature, dtype=dtype, device=device)
            reference_steer = torch.atan(model.vehicle.wheelbase * curvature)
            reference_steer = torch.clamp(
                reference_steer,
                min=-float(model.physics.max_steer_angle),
                max=float(model.physics.max_steer_angle),
            )
        else:
            reference_steer = torch.zeros_like(reference_ax)
        track_progress_prefix = (
            f"Transient track (torch, {model_label}, pid)" if show_track_progress else None
        )
        if is_single_track:
            profile = _simulate_single_track_pid_torch(
                torch=torch,
                track=track,
                model=model,
                config=config,
                reference_speed=reference_speed,
                reference_ax=reference_ax,
                reference_steer=reference_steer,
                track_progress_prefix=track_progress_prefix,
            )
        else:
            profile = _simulate_point_mass_pid_torch(
                torch=torch,
                track=track,
                model=model,
                config=config,
                reference_speed=reference_speed,
                reference_ax=reference_ax,
                track_progress_prefix=track_progress_prefix,
            )
        if show_iteration_progress:
            _render_progress_line(
                prefix="Transient PID (torch)",
                fraction=1.0,
                suffix=f"done objective {float(profile.objective_value.detach().cpu().item()):.4f}",
                final=True,
            )
        return profile

    odeint = _require_torchdiffeq().odeint

    ax_seed, steer_seed = _transient_seed_profile_torch(
        track=track,
        model=model,
        config=config,
        torch=torch,
    )

    sample_count = int(track.arc_length.size)
    control_count = _build_control_node_count(
        sample_count=sample_count,
        control_interval=int(config.transient.numerics.control_interval),
    )
    mesh_positions = build_control_mesh_positions(
        sample_count=sample_count,
        control_interval=int(config.transient.numerics.control_interval),
    )
    interpolation_map = build_control_interpolation_map(
        sample_count=sample_count,
        mesh_positions=mesh_positions,
    )
    max_iterations = int(config.transient.numerics.max_iterations)
    objective_evaluations = 0
    latest_objective = float("nan")
    if is_single_track:
        init_raw = _encode_single_track_controls_torch(
            torch=torch,
            model=model,
            ax_seed=ax_seed,
            steer_seed=steer_seed,
            control_count=control_count,
        )
    else:
        init_raw = _encode_point_mass_controls_torch(
            torch=torch,
            model=model,
            ax_seed=ax_seed,
            control_count=control_count,
        )

    raw_controls = torch.nn.Parameter(init_raw.to(dtype=dtype, device=device))
    learning_rate = float(config.transient.numerics.optimizer_learning_rate)
    lbfgs_chunk_size = max(1, int(config.transient.numerics.optimizer_lbfgs_max_iter))
    lbfgs_steps = max(1, int(np.ceil(max_iterations / lbfgs_chunk_size)))
    lbfgs = torch.optim.LBFGS(
        [raw_controls],
        lr=learning_rate,
        max_iter=lbfgs_chunk_size,
        line_search_fn="strong_wolfe",
    )

    def evaluate_profile() -> TorchTransientProfileResult:
        nonlocal objective_evaluations
        nonlocal latest_objective
        objective_evaluations += 1
        enable_track_progress = show_track_progress and (
            objective_evaluations == 1
            or objective_evaluations % _TRACK_PROGRESS_EVAL_STRIDE == 0
        )
        track_progress_prefix = (
            (
                "Transient track "
                f"(torch, {model_label}, eval {objective_evaluations}, nodes {control_count})"
            )
            if enable_track_progress
            else None
        )
        if is_single_track:
            ax_signal, steer_target = _decode_single_track_controls_torch(
                torch=torch,
                model=model,
                raw_controls=raw_controls,
                sample_count=sample_count,
                interpolation_map=interpolation_map,
            )
            return _simulate_single_track_torch(
                torch=torch,
                odeint=odeint,
                track=track,
                model=model,
                config=config,
                ax_signal=ax_signal,
                steer_target=steer_target,
                track_progress_prefix=track_progress_prefix,
            )
        ax_signal = _decode_point_mass_controls_torch(
            torch=torch,
            model=model,
            raw_ax=raw_controls,
            sample_count=sample_count,
            interpolation_map=interpolation_map,
        )
        return _simulate_point_mass_torch(
            torch=torch,
            track=track,
            model=model,
            config=config,
            ax_signal=ax_signal,
            track_progress_prefix=track_progress_prefix,
        )

    converged = False
    previous_loss: float | None = None
    completed_iterations = 0
    for step_idx in range(lbfgs_steps):
        def closure() -> Any:
            lbfgs.zero_grad()
            profile = evaluate_profile()
            loss = profile.objective_value
            loss.backward()
            return loss

        loss = lbfgs.step(closure)
        if not torch.isfinite(loss):
            break
        current_loss = float(loss.detach().cpu().item())
        latest_objective = current_loss
        completed_iterations = min(max_iterations, (step_idx + 1) * lbfgs_chunk_size)
        if show_iteration_progress:
            _render_progress_line(
                prefix="Transient OC (torch)",
                fraction=completed_iterations / max(max_iterations, 1),
                suffix=(
                    f"iter {completed_iterations}/{max_iterations} "
                    f"evals {objective_evaluations} "
                    f"objective {current_loss:.4f}"
                ),
                final=False,
            )
        if (
            previous_loss is not None
            and abs(previous_loss - current_loss) <= config.transient.numerics.tolerance
        ):
            converged = True
            break
        previous_loss = current_loss

    if not converged:
        adam = torch.optim.Adam([raw_controls], lr=learning_rate)
        adam_steps = int(config.transient.numerics.optimizer_adam_steps)
        for step_idx in range(adam_steps):
            adam.zero_grad()
            profile = evaluate_profile()
            loss = profile.objective_value
            if not torch.isfinite(loss):
                break
            loss.backward()
            adam.step()
            latest_objective = float(loss.detach().cpu().item())
            if show_iteration_progress:
                _render_progress_line(
                    prefix="Transient OC (torch/adam)",
                    fraction=(step_idx + 1) / max(adam_steps, 1),
                    suffix=(
                        f"step {step_idx + 1}/{adam_steps} "
                        f"evals {objective_evaluations} "
                        f"objective {latest_objective:.4f}"
                    ),
                    final=False,
                )

    final_profile = evaluate_profile()
    if show_iteration_progress:
        final_objective = float(final_profile.objective_value.detach().cpu().item())
        _render_progress_line(
            prefix="Transient OC (torch)",
            fraction=1.0,
            suffix=(
                f"done evals {objective_evaluations} "
                f"objective {final_objective:.4f}"
            ),
            final=True,
        )

    return final_profile
