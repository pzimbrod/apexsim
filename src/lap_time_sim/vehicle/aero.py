"""Aerodynamic force calculations."""

from __future__ import annotations

from dataclasses import dataclass

from lap_time_sim.vehicle.params import VehicleParameters


@dataclass(frozen=True)
class AeroForces:
    """Aerodynamic loads at a given speed."""

    drag_n: float
    downforce_n: float
    front_downforce_n: float
    rear_downforce_n: float


def aero_forces(vehicle: VehicleParameters, speed_mps: float) -> AeroForces:
    """Compute drag and downforce with axle distribution from CoP location."""
    vehicle.validate()
    speed_sq = max(speed_mps, 0.0) ** 2
    q = 0.5 * vehicle.air_density_kgpm3 * speed_sq

    drag_n = q * vehicle.c_d * vehicle.frontal_area_m2
    downforce_n = q * vehicle.c_l * vehicle.frontal_area_m2

    cop_from_rear = vehicle.cg_to_rear_axle_m + vehicle.cop_position_m
    front_share = min(max(cop_from_rear / vehicle.wheelbase_m, 0.0), 1.0)

    front_downforce_n = downforce_n * front_share
    rear_downforce_n = downforce_n - front_downforce_n

    return AeroForces(
        drag_n=drag_n,
        downforce_n=downforce_n,
        front_downforce_n=front_downforce_n,
        rear_downforce_n=rear_downforce_n,
    )
