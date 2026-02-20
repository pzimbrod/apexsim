"""Aerodynamic force calculations."""

from __future__ import annotations

from dataclasses import dataclass

from apexsim.vehicle.params import VehicleParameters

MAX_AERO_SPEED_MPS = 300.0


@dataclass(frozen=True)
class AeroForces:
    """Aerodynamic loads at a given speed.

    Args:
        drag: Aerodynamic drag force opposing motion [N].
        downforce: Total aerodynamic downforce [N].
        front_downforce: Front-axle share of aerodynamic downforce [N].
        rear_downforce: Rear-axle share of aerodynamic downforce [N].
    """

    drag: float
    downforce: float
    front_downforce: float
    rear_downforce: float


def aero_forces(vehicle: VehicleParameters, speed: float) -> AeroForces:
    """Compute drag and downforce with axle distribution from CoP location.

    Args:
        vehicle: Vehicle parameter set containing aerodynamic coefficients.
        speed: Vehicle speed [m/s].

    Returns:
        Drag and downforce quantities [N], including axle split.

    Raises:
        apexsim.utils.exceptions.ConfigurationError: If vehicle parameters
            are invalid.
    """
    vehicle.validate()
    # Defensive clipping prevents overflow for numerically unstable transient states.
    bounded_speed = min(max(speed, 0.0), MAX_AERO_SPEED_MPS)
    speed_sq = bounded_speed * bounded_speed
    q = 0.5 * vehicle.air_density * speed_sq

    drag = q * vehicle.drag_coefficient * vehicle.frontal_area
    downforce = q * vehicle.lift_coefficient * vehicle.frontal_area

    cop_from_rear = vehicle.cg_to_rear_axle + vehicle.cop_position
    front_share = min(max(cop_from_rear / vehicle.wheelbase, 0.0), 1.0)

    front_downforce = downforce * front_share
    rear_downforce = downforce - front_downforce

    return AeroForces(
        drag=drag,
        downforce=downforce,
        front_downforce=front_downforce,
        rear_downforce=rear_downforce,
    )
