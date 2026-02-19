"""Quasi-static normal-load estimation for single_track-based simulation."""

from __future__ import annotations

from dataclasses import dataclass

from apexsim.vehicle._backend_physics_core import (
    roll_stiffness_front_share_numpy,
    single_track_wheel_loads_numpy,
)
from apexsim.vehicle.params import VehicleParameters


@dataclass(frozen=True)
class NormalLoadState:
    """Front/rear axle and wheel normal loads.

    Args:
        front_axle_load: Total front-axle normal load [N].
        rear_axle_load: Total rear-axle normal load [N].
        front_left_load: Front-left wheel normal load [N].
        front_right_load: Front-right wheel normal load [N].
        rear_left_load: Rear-left wheel normal load [N].
        rear_right_load: Rear-right wheel normal load [N].
    """

    front_axle_load: float
    rear_axle_load: float
    front_left_load: float
    front_right_load: float
    rear_left_load: float
    rear_right_load: float

def estimate_normal_loads(
    vehicle: VehicleParameters,
    speed: float,
    longitudinal_accel: float,
    lateral_accel: float,
) -> NormalLoadState:
    """Estimate normal load distribution with aero and basic load transfer.

    Args:
        vehicle: Vehicle parameter set.
        speed: Vehicle speed [m/s].
        longitudinal_accel: Net longitudinal acceleration [m/s^2].
        lateral_accel: Lateral acceleration [m/s^2].

    Returns:
        Axle and wheel normal loads [N].

    Raises:
        apexsim.utils.exceptions.ConfigurationError: If vehicle parameters
            are invalid.
    """
    vehicle.validate()
    front_share = roll_stiffness_front_share_numpy(
        front_spring_rate=vehicle.front_spring_rate,
        rear_spring_rate=vehicle.rear_spring_rate,
        front_arb_distribution=vehicle.front_arb_distribution,
    )
    (
        front_axle_load,
        rear_axle_load,
        front_left_load,
        front_right_load,
        rear_left_load,
        rear_right_load,
    ) = single_track_wheel_loads_numpy(
        speed=speed,
        mass=vehicle.mass,
        downforce_scale=0.5 * vehicle.air_density * vehicle.lift_coefficient * vehicle.frontal_area,
        front_downforce_share=min(
            max((vehicle.cg_to_rear_axle + vehicle.cop_position) / vehicle.wheelbase, 0.0),
            1.0,
        ),
        front_weight_fraction=vehicle.front_weight_fraction,
        longitudinal_accel=longitudinal_accel,
        lateral_accel=lateral_accel,
        cg_height=vehicle.cg_height,
        wheelbase=vehicle.wheelbase,
        front_track=vehicle.front_track,
        rear_track=vehicle.rear_track,
        front_roll_stiffness_share=front_share,
    )

    return NormalLoadState(
        front_axle_load=float(front_axle_load),
        rear_axle_load=float(rear_axle_load),
        front_left_load=float(front_left_load),
        front_right_load=float(front_right_load),
        rear_left_load=float(rear_left_load),
        rear_right_load=float(rear_right_load),
    )
