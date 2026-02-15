"""Quasi-static normal-load estimation for bicycle-based simulation."""

from __future__ import annotations

from dataclasses import dataclass

from lap_time_sim.utils.constants import GRAVITY_MPS2, SMALL_EPS
from lap_time_sim.vehicle.aero import aero_forces
from lap_time_sim.vehicle.params import VehicleParameters

ROLL_STIFFNESS_FRONT_SHARE_MIN = 0.05
ROLL_STIFFNESS_FRONT_SHARE_MAX = 0.95
MIN_WHEEL_NORMAL_LOAD_N = SMALL_EPS


@dataclass(frozen=True)
class NormalLoadState:
    """Front/rear axle and wheel normal loads."""

    front_axle_n: float
    rear_axle_n: float
    front_left_n: float
    front_right_n: float
    rear_left_n: float
    rear_right_n: float


def _roll_stiffness_front_share(vehicle: VehicleParameters) -> float:
    """Blend spring and ARB balance into a bounded front-roll-share value.

    Args:
        vehicle: Vehicle parameter set.

    Returns:
        Front axle share of roll stiffness as a bounded fraction.
    """
    spring_share = vehicle.spring_rate_front_npm / (
        vehicle.spring_rate_front_npm + vehicle.spring_rate_rear_npm
    )
    blended_share = 0.5 * (spring_share + vehicle.arb_distribution_front)
    return min(
        max(blended_share, ROLL_STIFFNESS_FRONT_SHARE_MIN),
        ROLL_STIFFNESS_FRONT_SHARE_MAX,
    )


def _split_axle_load(axle_load_n: float, lateral_transfer_n: float) -> tuple[float, float]:
    """Split axle load into left/right wheel loads while preserving total load.

    The transfer term is saturated so neither wheel load becomes negative.

    Args:
        axle_load_n: Total normal load on an axle in Newton.
        lateral_transfer_n: Signed lateral load transfer on that axle in Newton.

    Returns:
        Tuple ``(left_load_n, right_load_n)`` preserving axle total load.
    """
    min_axle_load = 2.0 * MIN_WHEEL_NORMAL_LOAD_N
    bounded_axle_load = max(axle_load_n, min_axle_load)
    max_transfer = max(bounded_axle_load - min_axle_load, 0.0)
    bounded_transfer = min(max(lateral_transfer_n, -max_transfer), max_transfer)

    left_load = 0.5 * (bounded_axle_load - bounded_transfer)
    right_load = bounded_axle_load - left_load
    return left_load, right_load


def estimate_normal_loads(
    vehicle: VehicleParameters,
    speed_mps: float,
    longitudinal_accel_mps2: float,
    lateral_accel_mps2: float,
) -> NormalLoadState:
    """Estimate normal load distribution with aero and basic load transfer.

    Args:
        vehicle: Vehicle parameter set.
        speed_mps: Vehicle speed in m/s.
        longitudinal_accel_mps2: Net longitudinal acceleration in m/s^2.
        lateral_accel_mps2: Lateral acceleration in m/s^2.

    Returns:
        Axle and wheel normal loads in Newton.

    Raises:
        lap_time_sim.utils.exceptions.ConfigurationError: If vehicle parameters
            are invalid.
    """
    vehicle.validate()
    aero = aero_forces(vehicle, speed_mps)

    weight_n = vehicle.mass_kg * GRAVITY_MPS2
    total_vertical_load_n = weight_n + aero.downforce_n
    front_static_n = weight_n * vehicle.static_front_weight_fraction

    longitudinal_transfer_n = (
        vehicle.mass_kg * longitudinal_accel_mps2 * vehicle.h_cg_m / vehicle.wheelbase_m
    )

    front_axle_raw_n = front_static_n + aero.front_downforce_n - longitudinal_transfer_n
    min_axle_load = 2.0 * MIN_WHEEL_NORMAL_LOAD_N
    front_axle_n = min(
        max(front_axle_raw_n, min_axle_load),
        total_vertical_load_n - min_axle_load,
    )
    rear_axle_n = total_vertical_load_n - front_axle_n

    total_roll_moment_nm = vehicle.mass_kg * lateral_accel_mps2 * vehicle.h_cg_m
    front_share = _roll_stiffness_front_share(vehicle)

    front_transfer_n = front_share * total_roll_moment_nm / vehicle.track_front_m
    rear_transfer_n = (1.0 - front_share) * total_roll_moment_nm / vehicle.track_rear_m

    front_left_n, front_right_n = _split_axle_load(front_axle_n, front_transfer_n)
    rear_left_n, rear_right_n = _split_axle_load(rear_axle_n, rear_transfer_n)

    return NormalLoadState(
        front_axle_n=front_axle_n,
        rear_axle_n=rear_axle_n,
        front_left_n=front_left_n,
        front_right_n=front_right_n,
        rear_left_n=rear_left_n,
        rear_right_n=rear_right_n,
    )
