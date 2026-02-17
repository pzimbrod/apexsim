"""Quasi-static normal-load estimation for bicycle-based simulation."""

from __future__ import annotations

from dataclasses import dataclass

from pylapsim.utils.constants import GRAVITY, SMALL_EPS
from pylapsim.vehicle.aero import aero_forces
from pylapsim.vehicle.params import VehicleParameters

ROLL_STIFFNESS_FRONT_SHARE_MIN = 0.05
ROLL_STIFFNESS_FRONT_SHARE_MAX = 0.95
MIN_WHEEL_NORMAL_LOAD = SMALL_EPS


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


def _roll_stiffness_front_share(vehicle: VehicleParameters) -> float:
    """Blend spring and ARB balance into a bounded front-roll-share value.

    Args:
        vehicle: Vehicle parameter set.

    Returns:
        Front axle share of roll stiffness as a bounded fraction.
    """
    spring_share = vehicle.front_spring_rate / (
        vehicle.front_spring_rate + vehicle.rear_spring_rate
    )
    blended_share = 0.5 * (spring_share + vehicle.front_arb_distribution)
    return min(
        max(blended_share, ROLL_STIFFNESS_FRONT_SHARE_MIN),
        ROLL_STIFFNESS_FRONT_SHARE_MAX,
    )


def _split_axle_load(axle_load: float, lateral_transfer: float) -> tuple[float, float]:
    """Split axle load into left/right wheel loads while preserving total load.

    The transfer term is saturated so neither wheel load becomes negative.

    Args:
        axle_load: Total normal load on an axle [N].
        lateral_transfer: Signed lateral load transfer on that axle [N].

    Returns:
        Tuple ``(left_load, right_load)`` preserving axle total load.
    """
    min_axle_load = 2.0 * MIN_WHEEL_NORMAL_LOAD
    bounded_axle_load = max(axle_load, min_axle_load)
    max_transfer = max(bounded_axle_load - min_axle_load, 0.0)
    bounded_transfer = min(max(lateral_transfer, -max_transfer), max_transfer)

    left_load = 0.5 * (bounded_axle_load - bounded_transfer)
    right_load = bounded_axle_load - left_load
    return left_load, right_load


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
        pylapsim.utils.exceptions.ConfigurationError: If vehicle parameters
            are invalid.
    """
    vehicle.validate()
    aero = aero_forces(vehicle, speed)

    weight = vehicle.mass * GRAVITY
    total_vertical_load = weight + aero.downforce
    front_static_load = weight * vehicle.front_weight_fraction

    longitudinal_transfer = (
        vehicle.mass * longitudinal_accel * vehicle.cg_height / vehicle.wheelbase
    )

    front_axle_raw = front_static_load + aero.front_downforce - longitudinal_transfer
    min_axle_load = 2.0 * MIN_WHEEL_NORMAL_LOAD
    front_axle_load = min(
        max(front_axle_raw, min_axle_load),
        total_vertical_load - min_axle_load,
    )
    rear_axle_load = total_vertical_load - front_axle_load

    total_roll_moment = vehicle.mass * lateral_accel * vehicle.cg_height
    front_share = _roll_stiffness_front_share(vehicle)

    front_transfer = front_share * total_roll_moment / vehicle.front_track
    rear_transfer = (1.0 - front_share) * total_roll_moment / vehicle.rear_track

    front_left_load, front_right_load = _split_axle_load(front_axle_load, front_transfer)
    rear_left_load, rear_right_load = _split_axle_load(rear_axle_load, rear_transfer)

    return NormalLoadState(
        front_axle_load=front_axle_load,
        rear_axle_load=rear_axle_load,
        front_left_load=front_left_load,
        front_right_load=front_right_load,
        rear_left_load=rear_left_load,
        rear_right_load=rear_right_load,
    )
