"""Aerodynamic force calculations."""

from __future__ import annotations

from dataclasses import dataclass

from lap_time_sim.vehicle.params import VehicleParameters


@dataclass(frozen=True)
class AeroForces:
    """Aerodynamic loads at a given speed.

    Args:
        drag_n: Aerodynamic drag force opposing motion (N).
        downforce_n: Total aerodynamic downforce (N).
        front_downforce_n: Front-axle share of aerodynamic downforce (N).
        rear_downforce_n: Rear-axle share of aerodynamic downforce (N).
    """

    drag_n: float
    downforce_n: float
    front_downforce_n: float
    rear_downforce_n: float


def aero_forces(vehicle: VehicleParameters, speed_mps: float) -> AeroForces:
    """Compute drag and downforce with axle distribution from CoP location.

    Args:
        vehicle: Vehicle parameter set containing aerodynamic coefficients.
        speed_mps: Vehicle speed in m/s.

    Returns:
        Drag and downforce quantities in Newton, including axle split.

    Raises:
        lap_time_sim.utils.exceptions.ConfigurationError: If vehicle parameters
            are invalid.
    """
    vehicle.validate()
    speed_sq = max(speed_mps, 0.0) ** 2
    q = 0.5 * vehicle.air_density * speed_sq

    drag_n = q * vehicle.drag_coefficient * vehicle.frontal_area
    downforce_n = q * vehicle.lift_coefficient * vehicle.frontal_area

    cop_from_rear = vehicle.cg_to_rear_axle + vehicle.cop_position
    front_share = min(max(cop_from_rear / vehicle.wheelbase, 0.0), 1.0)

    front_downforce_n = downforce_n * front_share
    rear_downforce_n = downforce_n - front_downforce_n

    return AeroForces(
        drag_n=drag_n,
        downforce_n=downforce_n,
        front_downforce_n=front_downforce_n,
        rear_downforce_n=rear_downforce_n,
    )
