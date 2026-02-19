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
