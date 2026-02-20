"""Shared helpers for Spa-Francorchamps example scripts."""

from __future__ import annotations

from pathlib import Path

from apexsim.utils.constants import STANDARD_AIR_DENSITY
from apexsim.vehicle import VehicleParameters


def example_vehicle_parameters() -> VehicleParameters:
    """Create explicit vehicle parameters used by Spa example scripts.

    Returns:
        Vehicle parameter set for Spa example runs.
    """
    return VehicleParameters(
        mass=798.0,
        yaw_inertia=1120.0,
        cg_height=0.31,
        wheelbase=3.60,
        front_track=1.60,
        rear_track=1.55,
        front_weight_fraction=0.46,
        cop_position=0.10,
        lift_coefficient=3.20,
        drag_coefficient=0.90,
        frontal_area=1.50,
        roll_rate=4200.0,
        front_spring_rate=180000.0,
        rear_spring_rate=165000.0,
        front_arb_distribution=0.55,
        arb_roll_stiffness_fraction=0.5,
        front_ride_height=0.030,
        rear_ride_height=0.050,
        air_density=STANDARD_AIR_DENSITY,
    )


def project_root() -> Path:
    """Return project root path for scripts located in ``examples/spa``.

    Returns:
        Project root directory path.
    """
    return Path(__file__).resolve().parents[2]


def spa_track_path() -> Path:
    """Return canonical Spa track CSV path.

    Returns:
        Absolute path to ``data/spa_francorchamps.csv``.
    """
    return project_root() / "data" / "spa_francorchamps.csv"


def spa_output_root() -> Path:
    """Return canonical Spa examples output root.

    Returns:
        Path to ``examples/output/spa``.
    """
    return project_root() / "examples" / "output" / "spa"
