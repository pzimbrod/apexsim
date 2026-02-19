"""Shared backend-agnostic physics primitives used across vehicle models."""

from __future__ import annotations

from typing import Any

import numpy as np

from apexsim.utils.constants import GRAVITY, SMALL_EPS


def downforce_total_numpy(
    *,
    speed: np.ndarray | float,
    downforce_scale: float,
) -> np.ndarray:
    """Return aerodynamic downforce for scalar/vector speed input.

    Args:
        speed: Vehicle speed [m/s].
        downforce_scale: Quadratic downforce scale factor [N/(m/s)^2].

    Returns:
        Aerodynamic downforce [N].
    """
    speed_non_negative = np.maximum(np.asarray(speed, dtype=float), 0.0)
    return np.asarray(downforce_scale * speed_non_negative * speed_non_negative, dtype=float)


def downforce_total_torch(
    *,
    torch: Any,
    speed: Any,
    downforce_scale: Any,
) -> Any:
    """Return aerodynamic downforce for torch tensor speed input.

    Args:
        torch: Imported torch module.
        speed: Vehicle speed tensor [m/s].
        downforce_scale: Quadratic downforce scale tensor/scalar [N/(m/s)^2].

    Returns:
        Aerodynamic downforce tensor [N].
    """
    speed_tensor = torch.as_tensor(speed)
    scale_tensor = torch.as_tensor(
        downforce_scale,
        dtype=speed_tensor.dtype,
        device=speed_tensor.device,
    )
    speed_non_negative = torch.clamp(speed_tensor, min=0.0)
    return scale_tensor * speed_non_negative * speed_non_negative


def drag_force_numpy(
    *,
    speed: np.ndarray | float,
    drag_force_scale: float,
) -> np.ndarray:
    """Return aerodynamic drag force for scalar/vector speed input.

    Args:
        speed: Vehicle speed [m/s].
        drag_force_scale: Quadratic drag scale factor [N/(m/s)^2].

    Returns:
        Aerodynamic drag force [N].
    """
    speed_non_negative = np.maximum(np.asarray(speed, dtype=float), 0.0)
    return np.asarray(drag_force_scale * speed_non_negative * speed_non_negative, dtype=float)


def drag_force_torch(
    *,
    torch: Any,
    speed: Any,
    drag_force_scale: Any,
) -> Any:
    """Return aerodynamic drag force for torch tensor speed input.

    Args:
        torch: Imported torch module.
        speed: Vehicle speed tensor [m/s].
        drag_force_scale: Quadratic drag scale tensor/scalar [N/(m/s)^2].

    Returns:
        Aerodynamic drag force tensor [N].
    """
    speed_tensor = torch.as_tensor(speed)
    scale_tensor = torch.as_tensor(
        drag_force_scale,
        dtype=speed_tensor.dtype,
        device=speed_tensor.device,
    )
    speed_non_negative = torch.clamp(speed_tensor, min=0.0)
    return scale_tensor * speed_non_negative * speed_non_negative


def normal_accel_limit_numpy(
    *,
    speed: np.ndarray | float,
    downforce_scale: float,
    mass: float,
) -> np.ndarray:
    """Return normal-acceleration budget from gravity and aero downforce.

    Args:
        speed: Vehicle speed [m/s].
        downforce_scale: Quadratic downforce scale factor [N/(m/s)^2].
        mass: Vehicle mass [kg].

    Returns:
        Normal acceleration budget [m/s^2].
    """
    downforce = downforce_total_numpy(speed=speed, downforce_scale=downforce_scale)
    return np.maximum(GRAVITY + downforce / mass, SMALL_EPS)


def normal_accel_limit_torch(
    *,
    torch: Any,
    speed: Any,
    downforce_scale: Any,
    mass: Any,
) -> Any:
    """Return normal-acceleration budget from gravity and aero downforce.

    Args:
        torch: Imported torch module.
        speed: Vehicle speed tensor [m/s].
        downforce_scale: Quadratic downforce scale tensor/scalar [N/(m/s)^2].
        mass: Vehicle mass tensor/scalar [kg].

    Returns:
        Normal acceleration budget tensor [m/s^2].
    """
    downforce = downforce_total_torch(
        torch=torch,
        speed=speed,
        downforce_scale=downforce_scale,
    )
    mass_tensor = torch.as_tensor(
        mass,
        dtype=downforce.dtype,
        device=downforce.device,
    )
    return torch.clamp(GRAVITY + downforce / mass_tensor, min=SMALL_EPS)


def friction_circle_scale_numpy(
    *,
    lateral_accel_required: np.ndarray | float,
    lateral_accel_limit: np.ndarray | float,
) -> np.ndarray:
    """Return longitudinal friction-circle utilization scale in ``[0, 1]``.

    Args:
        lateral_accel_required: Required lateral acceleration [m/s^2].
        lateral_accel_limit: Available lateral acceleration limit [m/s^2].

    Returns:
        Remaining longitudinal friction utilization factor.
    """
    required = np.asarray(lateral_accel_required, dtype=float)
    limit = np.asarray(lateral_accel_limit, dtype=float)
    safe_limit = np.maximum(limit, SMALL_EPS)
    usage = np.clip(np.abs(required) / safe_limit, 0.0, 1.0)
    return np.asarray(np.sqrt(np.clip(1.0 - usage * usage, 0.0, 1.0)), dtype=float)


def friction_circle_scale_torch(
    *,
    torch: Any,
    lateral_accel_required: Any,
    lateral_accel_limit: Any,
) -> Any:
    """Return longitudinal friction-circle utilization scale in ``[0, 1]``.

    Args:
        torch: Imported torch module.
        lateral_accel_required: Required lateral acceleration tensor [m/s^2].
        lateral_accel_limit: Available lateral acceleration tensor [m/s^2].

    Returns:
        Remaining longitudinal friction utilization factor tensor.
    """
    safe_limit = torch.clamp(lateral_accel_limit, min=SMALL_EPS)
    usage = torch.clamp(torch.abs(lateral_accel_required) / safe_limit, min=0.0, max=1.0)
    return torch.sqrt(torch.clamp(1.0 - usage * usage, min=0.0, max=1.0))


def tractive_power_numpy(
    *,
    speed: np.ndarray | float,
    longitudinal_accel: np.ndarray | float,
    mass: float,
    drag_force_scale: float,
) -> np.ndarray:
    """Return tractive power from speed and longitudinal acceleration.

    Args:
        speed: Vehicle speed [m/s].
        longitudinal_accel: Longitudinal acceleration [m/s^2].
        mass: Vehicle mass [kg].
        drag_force_scale: Quadratic drag scale factor [N/(m/s)^2].

    Returns:
        Tractive power [W].
    """
    speed_array = np.asarray(speed, dtype=float)
    accel_array = np.asarray(longitudinal_accel, dtype=float)
    drag_force = drag_force_numpy(speed=speed_array, drag_force_scale=drag_force_scale)
    tractive_force = mass * accel_array + drag_force
    return np.asarray(tractive_force * speed_array, dtype=float)


def tractive_power_torch(
    *,
    torch: Any,
    speed: Any,
    longitudinal_accel: Any,
    mass: Any,
    drag_force_scale: Any,
) -> Any:
    """Return tractive power from speed and longitudinal acceleration.

    Args:
        torch: Imported torch module.
        speed: Vehicle speed tensor [m/s].
        longitudinal_accel: Longitudinal acceleration tensor [m/s^2].
        mass: Vehicle mass tensor/scalar [kg].
        drag_force_scale: Quadratic drag scale tensor/scalar [N/(m/s)^2].

    Returns:
        Tractive power tensor [W].
    """
    speed_tensor = torch.as_tensor(speed)
    accel_tensor = torch.as_tensor(
        longitudinal_accel,
        dtype=speed_tensor.dtype,
        device=speed_tensor.device,
    )
    drag_force = drag_force_torch(
        torch=torch,
        speed=speed_tensor,
        drag_force_scale=drag_force_scale,
    )
    mass_tensor = torch.as_tensor(
        mass,
        dtype=accel_tensor.dtype,
        device=accel_tensor.device,
    )
    tractive_force = mass_tensor * accel_tensor + drag_force
    return tractive_force * speed_tensor


def axle_tire_loads_numpy(
    *,
    speed: np.ndarray | float,
    mass: float,
    downforce_scale: float,
    front_downforce_share: float,
    front_weight_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-tire front/rear normal loads from static and aero distribution.

    Args:
        speed: Vehicle speed [m/s].
        mass: Vehicle mass [kg].
        downforce_scale: Quadratic downforce scale factor [N/(m/s)^2].
        front_downforce_share: Front axle share of total aero downforce.
        front_weight_fraction: Front axle static weight fraction.

    Returns:
        Tuple of front/rear per-tire normal loads [N].
    """
    downforce_total = downforce_total_numpy(speed=speed, downforce_scale=downforce_scale)
    front_downforce = downforce_total * front_downforce_share

    weight = mass * GRAVITY
    total_vertical_load = weight + downforce_total
    front_static_load = weight * front_weight_fraction

    min_axle_load = 2.0 * SMALL_EPS
    front_axle_raw = front_static_load + front_downforce
    front_axle_load = np.clip(
        front_axle_raw,
        min_axle_load,
        total_vertical_load - min_axle_load,
    )
    rear_axle_load = total_vertical_load - front_axle_load

    front_tire_load = np.maximum(front_axle_load * 0.5, SMALL_EPS)
    rear_tire_load = np.maximum(rear_axle_load * 0.5, SMALL_EPS)
    return np.asarray(front_tire_load, dtype=float), np.asarray(rear_tire_load, dtype=float)


def axle_tire_loads_torch(
    *,
    torch: Any,
    speed: Any,
    mass: Any,
    downforce_scale: Any,
    front_downforce_share: Any,
    front_weight_fraction: Any,
) -> tuple[Any, Any]:
    """Return per-tire front/rear normal loads from static and aero distribution.

    Args:
        torch: Imported torch module.
        speed: Vehicle speed tensor [m/s].
        mass: Vehicle mass tensor/scalar [kg].
        downforce_scale: Quadratic downforce scale tensor/scalar [N/(m/s)^2].
        front_downforce_share: Front axle share of total aero downforce.
        front_weight_fraction: Front axle static weight fraction.

    Returns:
        Tuple of front/rear per-tire normal-load tensors [N].
    """
    downforce_total = downforce_total_torch(
        torch=torch,
        speed=speed,
        downforce_scale=downforce_scale,
    )
    front_share_tensor = torch.as_tensor(
        front_downforce_share,
        dtype=downforce_total.dtype,
        device=downforce_total.device,
    )
    front_downforce = downforce_total * front_share_tensor

    mass_tensor = torch.as_tensor(
        mass,
        dtype=downforce_total.dtype,
        device=downforce_total.device,
    )
    front_weight_fraction_tensor = torch.as_tensor(
        front_weight_fraction,
        dtype=downforce_total.dtype,
        device=downforce_total.device,
    )
    weight = mass_tensor * GRAVITY
    total_vertical_load = weight + downforce_total
    front_static_load = weight * front_weight_fraction_tensor

    min_axle_load = 2.0 * SMALL_EPS
    front_axle_raw = front_static_load + front_downforce
    min_axle_load_tensor = torch.full_like(total_vertical_load, min_axle_load)
    max_front_axle_load = torch.clamp(total_vertical_load - min_axle_load, min=min_axle_load)
    front_axle_load = torch.minimum(
        torch.maximum(front_axle_raw, min_axle_load_tensor),
        max_front_axle_load,
    )
    rear_axle_load = total_vertical_load - front_axle_load

    front_tire_load = torch.clamp(front_axle_load * 0.5, min=SMALL_EPS)
    rear_tire_load = torch.clamp(rear_axle_load * 0.5, min=SMALL_EPS)
    return front_tire_load, rear_tire_load


ROLL_STIFFNESS_FRONT_SHARE_MIN = 0.05
ROLL_STIFFNESS_FRONT_SHARE_MAX = 0.95


def roll_stiffness_front_share_numpy(
    *,
    front_spring_rate: float,
    rear_spring_rate: float,
    front_arb_distribution: float,
) -> float:
    """Return bounded front-axle roll-stiffness share.

    Args:
        front_spring_rate: Front spring rate [N/m].
        rear_spring_rate: Rear spring rate [N/m].
        front_arb_distribution: Front anti-roll-bar distribution in ``[0, 1]``.

    Returns:
        Bounded front roll-stiffness share.
    """
    spring_sum = max(front_spring_rate + rear_spring_rate, SMALL_EPS)
    spring_share = front_spring_rate / spring_sum
    blended_share = 0.5 * (spring_share + front_arb_distribution)
    return float(
        np.clip(
            blended_share,
            ROLL_STIFFNESS_FRONT_SHARE_MIN,
            ROLL_STIFFNESS_FRONT_SHARE_MAX,
        )
    )


def _split_axle_load_numpy(
    *,
    axle_load: np.ndarray | float,
    lateral_transfer: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Split axle load into left/right wheel loads with load-preserving saturation.

    Args:
        axle_load: Total axle normal load [N].
        lateral_transfer: Signed lateral load transfer on the axle [N].

    Returns:
        Tuple ``(left_load, right_load)`` [N].
    """
    axle = np.asarray(axle_load, dtype=float)
    transfer = np.asarray(lateral_transfer, dtype=float)
    bounded_axle = np.maximum(axle, 2.0 * SMALL_EPS)
    max_transfer = np.maximum(bounded_axle - 2.0 * SMALL_EPS, 0.0)
    bounded_transfer = np.clip(transfer, -max_transfer, max_transfer)
    left = 0.5 * (bounded_axle - bounded_transfer)
    right = bounded_axle - left
    return np.asarray(left, dtype=float), np.asarray(right, dtype=float)


def single_track_wheel_loads_numpy(
    *,
    speed: np.ndarray | float,
    mass: float,
    downforce_scale: float,
    front_downforce_share: float,
    front_weight_fraction: float,
    longitudinal_accel: np.ndarray | float,
    lateral_accel: np.ndarray | float,
    cg_height: float,
    wheelbase: float,
    front_track: float,
    rear_track: float,
    front_roll_stiffness_share: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate axle and wheel normal loads for single-track force modeling.

    Args:
        speed: Vehicle speed [m/s].
        mass: Vehicle mass [kg].
        downforce_scale: Quadratic downforce scale factor [N/(m/s)^2].
        front_downforce_share: Front axle share of total aero downforce.
        front_weight_fraction: Front axle static weight fraction.
        longitudinal_accel: Longitudinal acceleration [m/s^2].
        lateral_accel: Lateral acceleration [m/s^2].
        cg_height: CoG height [m].
        wheelbase: Wheelbase [m].
        front_track: Front track width [m].
        rear_track: Rear track width [m].
        front_roll_stiffness_share: Front roll-stiffness share in ``[0, 1]``.

    Returns:
        Tuple ``(front_axle, rear_axle, front_left, front_right, rear_left, rear_right)`` [N].
    """
    speed_array = np.asarray(speed, dtype=float)
    longitudinal_array = np.asarray(longitudinal_accel, dtype=float)
    lateral_array = np.asarray(lateral_accel, dtype=float)
    speed_array, longitudinal_array, lateral_array = np.broadcast_arrays(
        speed_array,
        longitudinal_array,
        lateral_array,
    )

    speed_non_negative = np.maximum(speed_array, 0.0)
    speed_squared = speed_non_negative * speed_non_negative
    downforce_total = downforce_scale * speed_squared
    front_downforce = downforce_total * front_downforce_share

    weight = mass * GRAVITY
    total_vertical_load = weight + downforce_total
    front_static_load = weight * front_weight_fraction
    longitudinal_transfer = mass * longitudinal_array * cg_height / max(wheelbase, SMALL_EPS)

    front_axle_raw = front_static_load + front_downforce - longitudinal_transfer
    min_axle_load = 2.0 * SMALL_EPS
    front_axle_load = np.clip(
        front_axle_raw,
        min_axle_load,
        total_vertical_load - min_axle_load,
    )
    rear_axle_load = total_vertical_load - front_axle_load

    total_roll_moment = mass * lateral_array * cg_height
    front_transfer = front_roll_stiffness_share * total_roll_moment / max(
        front_track,
        SMALL_EPS,
    )
    rear_transfer = (1.0 - front_roll_stiffness_share) * total_roll_moment / max(
        rear_track,
        SMALL_EPS,
    )
    front_left, front_right = _split_axle_load_numpy(
        axle_load=front_axle_load,
        lateral_transfer=front_transfer,
    )
    rear_left, rear_right = _split_axle_load_numpy(
        axle_load=rear_axle_load,
        lateral_transfer=rear_transfer,
    )
    return (
        np.asarray(front_axle_load, dtype=float),
        np.asarray(rear_axle_load, dtype=float),
        front_left,
        front_right,
        rear_left,
        rear_right,
    )


def _split_axle_load_torch(
    *,
    torch: Any,
    axle_load: Any,
    lateral_transfer: Any,
) -> tuple[Any, Any]:
    """Split axle load into left/right wheel loads with load-preserving saturation.

    Args:
        torch: Imported torch module.
        axle_load: Total axle normal-load tensor [N].
        lateral_transfer: Signed lateral transfer tensor [N].

    Returns:
        Tuple ``(left_load, right_load)`` [N].
    """
    axle = torch.as_tensor(axle_load)
    transfer = torch.as_tensor(lateral_transfer, dtype=axle.dtype, device=axle.device)
    bounded_axle = torch.clamp(axle, min=2.0 * SMALL_EPS)
    max_transfer = torch.clamp(bounded_axle - 2.0 * SMALL_EPS, min=0.0)
    bounded_transfer = torch.minimum(
        torch.maximum(transfer, -max_transfer),
        max_transfer,
    )
    left = 0.5 * (bounded_axle - bounded_transfer)
    right = bounded_axle - left
    return left, right


def single_track_wheel_loads_torch(
    *,
    torch: Any,
    speed: Any,
    mass: Any,
    downforce_scale: Any,
    front_downforce_share: Any,
    front_weight_fraction: Any,
    longitudinal_accel: Any,
    lateral_accel: Any,
    cg_height: Any,
    wheelbase: Any,
    front_track: Any,
    rear_track: Any,
    front_roll_stiffness_share: Any,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Estimate axle and wheel normal loads for single-track force modeling.

    Args:
        torch: Imported torch module.
        speed: Speed tensor [m/s].
        mass: Vehicle mass tensor/scalar [kg].
        downforce_scale: Quadratic downforce scale tensor/scalar [N/(m/s)^2].
        front_downforce_share: Front axle share of total aero downforce.
        front_weight_fraction: Front axle static weight fraction.
        longitudinal_accel: Longitudinal acceleration tensor [m/s^2].
        lateral_accel: Lateral acceleration tensor [m/s^2].
        cg_height: CoG height tensor/scalar [m].
        wheelbase: Wheelbase tensor/scalar [m].
        front_track: Front track width tensor/scalar [m].
        rear_track: Rear track width tensor/scalar [m].
        front_roll_stiffness_share: Front roll-stiffness share in ``[0, 1]``.

    Returns:
        Tuple ``(front_axle, rear_axle, front_left, front_right, rear_left, rear_right)`` [N].
    """
    speed_tensor = torch.as_tensor(speed)
    dtype = speed_tensor.dtype
    device = speed_tensor.device

    mass_tensor = torch.as_tensor(mass, dtype=dtype, device=device)
    downforce_scale_tensor = torch.as_tensor(downforce_scale, dtype=dtype, device=device)
    front_share_tensor = torch.as_tensor(
        front_downforce_share,
        dtype=dtype,
        device=device,
    )
    front_weight_fraction_tensor = torch.as_tensor(
        front_weight_fraction,
        dtype=dtype,
        device=device,
    )
    longitudinal_tensor = torch.as_tensor(
        longitudinal_accel,
        dtype=dtype,
        device=device,
    )
    lateral_tensor = torch.as_tensor(
        lateral_accel,
        dtype=dtype,
        device=device,
    )
    cg_height_tensor = torch.as_tensor(cg_height, dtype=dtype, device=device)
    wheelbase_tensor = torch.as_tensor(wheelbase, dtype=dtype, device=device)
    front_track_tensor = torch.as_tensor(front_track, dtype=dtype, device=device)
    rear_track_tensor = torch.as_tensor(rear_track, dtype=dtype, device=device)
    front_roll_share_tensor = torch.as_tensor(
        front_roll_stiffness_share,
        dtype=dtype,
        device=device,
    )

    speed_non_negative = torch.clamp(speed_tensor, min=0.0)
    speed_squared = speed_non_negative * speed_non_negative
    downforce_total = downforce_scale_tensor * speed_squared
    front_downforce = downforce_total * front_share_tensor

    weight = mass_tensor * GRAVITY
    total_vertical_load = weight + downforce_total
    front_static_load = weight * front_weight_fraction_tensor
    longitudinal_transfer = mass_tensor * longitudinal_tensor * cg_height_tensor / torch.clamp(
        wheelbase_tensor,
        min=SMALL_EPS,
    )

    min_axle_load = 2.0 * SMALL_EPS
    front_axle_raw = front_static_load + front_downforce - longitudinal_transfer
    min_axle_load_tensor = torch.full_like(total_vertical_load, min_axle_load)
    max_front_axle_load = torch.clamp(total_vertical_load - min_axle_load, min=min_axle_load)
    front_axle_load = torch.minimum(
        torch.maximum(front_axle_raw, min_axle_load_tensor),
        max_front_axle_load,
    )
    rear_axle_load = total_vertical_load - front_axle_load

    total_roll_moment = mass_tensor * lateral_tensor * cg_height_tensor
    front_transfer = front_roll_share_tensor * total_roll_moment / torch.clamp(
        front_track_tensor,
        min=SMALL_EPS,
    )
    rear_transfer = (1.0 - front_roll_share_tensor) * total_roll_moment / torch.clamp(
        rear_track_tensor,
        min=SMALL_EPS,
    )
    front_left, front_right = _split_axle_load_torch(
        torch=torch,
        axle_load=front_axle_load,
        lateral_transfer=front_transfer,
    )
    rear_left, rear_right = _split_axle_load_torch(
        torch=torch,
        axle_load=rear_axle_load,
        lateral_transfer=rear_transfer,
    )
    return (
        front_axle_load,
        rear_axle_load,
        torch.clamp(front_left, min=SMALL_EPS),
        torch.clamp(front_right, min=SMALL_EPS),
        torch.clamp(rear_left, min=SMALL_EPS),
        torch.clamp(rear_right, min=SMALL_EPS),
    )
