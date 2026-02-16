"""Shared test helpers."""

from __future__ import annotations

from lap_time_sim.utils.constants import AIR_DENSITY_KGPM3
from lap_time_sim.vehicle.params import VehicleParameters


def sample_vehicle_parameters() -> VehicleParameters:
    """Create a representative high-downforce vehicle parameter set.

    Returns:
        Vehicle parameter set used by unit and integration tests.
    """
    return VehicleParameters(
        mass_kg=798.0,
        yaw_inertia_kgm2=1120.0,
        h_cg_m=0.31,
        wheelbase_m=3.60,
        track_front_m=1.60,
        track_rear_m=1.55,
        static_front_weight_fraction=0.46,
        cop_position_m=0.10,
        c_l=3.20,
        c_d=0.90,
        frontal_area_m2=1.50,
        roll_rate_nm_per_deg=4200.0,
        spring_rate_front_npm=180000.0,
        spring_rate_rear_npm=165000.0,
        arb_distribution_front=0.55,
        ride_height_front_m=0.030,
        ride_height_rear_m=0.050,
        air_density_kgpm3=AIR_DENSITY_KGPM3,
    )
