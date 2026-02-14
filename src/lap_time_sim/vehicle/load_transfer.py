"""Quasi-static normal-load estimation for bicycle-based simulation."""

from __future__ import annotations

from dataclasses import dataclass

from lap_time_sim.utils.constants import GRAVITY_MPS2
from lap_time_sim.vehicle.aero import aero_forces
from lap_time_sim.vehicle.params import VehicleParameters


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
    spring_share = vehicle.spring_rate_front_npm / (
        vehicle.spring_rate_front_npm + vehicle.spring_rate_rear_npm
    )
    blended_share = 0.5 * (spring_share + vehicle.arb_distribution_front)
    return min(max(blended_share, 0.05), 0.95)


def estimate_normal_loads(
    vehicle: VehicleParameters,
    speed_mps: float,
    longitudinal_accel_mps2: float,
    lateral_accel_mps2: float,
) -> NormalLoadState:
    """Estimate normal load distribution with aero and basic load transfer."""
    vehicle.validate()
    aero = aero_forces(vehicle, speed_mps)

    weight_n = vehicle.mass_kg * GRAVITY_MPS2
    front_static_n = weight_n * vehicle.static_front_weight_fraction
    rear_static_n = weight_n - front_static_n

    longitudinal_transfer_n = (
        vehicle.mass_kg * longitudinal_accel_mps2 * vehicle.h_cg_m / vehicle.wheelbase_m
    )

    front_axle_n = front_static_n + aero.front_downforce_n - longitudinal_transfer_n
    rear_axle_n = rear_static_n + aero.rear_downforce_n + longitudinal_transfer_n

    total_roll_moment_nm = vehicle.mass_kg * lateral_accel_mps2 * vehicle.h_cg_m
    front_share = _roll_stiffness_front_share(vehicle)

    front_transfer_n = front_share * total_roll_moment_nm / vehicle.track_front_m
    rear_transfer_n = (1.0 - front_share) * total_roll_moment_nm / vehicle.track_rear_m

    front_left_n = max(10.0, front_axle_n / 2.0 - front_transfer_n / 2.0)
    front_right_n = max(10.0, front_axle_n - front_left_n)
    rear_left_n = max(10.0, rear_axle_n / 2.0 - rear_transfer_n / 2.0)
    rear_right_n = max(10.0, rear_axle_n - rear_left_n)

    return NormalLoadState(
        front_axle_n=front_axle_n,
        rear_axle_n=rear_axle_n,
        front_left_n=front_left_n,
        front_right_n=front_right_n,
        rear_left_n=rear_left_n,
        rear_right_n=rear_right_n,
    )
