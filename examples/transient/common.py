"""Shared helpers for transient-solver example scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from apexsim.utils.constants import STANDARD_AIR_DENSITY
from apexsim.vehicle import VehicleParameters


def example_vehicle_parameters() -> VehicleParameters:
    """Create representative vehicle parameters for transient examples.

    Returns:
        Vehicle parameter set used across transient example scripts.
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
    """Return project root path for scripts located in ``examples/transient``.

    Returns:
        Project root directory path.
    """
    return Path(__file__).resolve().parents[2]


def transient_output_root() -> Path:
    """Return canonical transient examples output root.

    Returns:
        Path to ``examples/output/transient``.
    """
    return project_root() / "examples" / "output" / "transient"


def export_transient_trace_csv(
    *,
    arc_length: np.ndarray,
    time: np.ndarray,
    speed: np.ndarray,
    longitudinal_accel: np.ndarray,
    lateral_accel: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    yaw_rate: np.ndarray,
    steer_cmd: np.ndarray,
    ax_cmd: np.ndarray,
    path: Path,
) -> None:
    """Export transient traces for post-processing outside ApexSim.

    Args:
        arc_length: Track arc-length support [m].
        time: Cumulative time trace [s].
        speed: Speed trace [m/s].
        longitudinal_accel: Longitudinal acceleration [m/s^2].
        lateral_accel: Lateral acceleration [m/s^2].
        vx: Body-frame longitudinal speed [m/s].
        vy: Body-frame lateral speed [m/s].
        yaw_rate: Yaw-rate trace [rad/s].
        steer_cmd: Steering command trace [rad].
        ax_cmd: Longitudinal command trace [m/s^2].
        path: Destination CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table = np.column_stack(
        (
            arc_length,
            time,
            speed,
            longitudinal_accel,
            lateral_accel,
            vx,
            vy,
            yaw_rate,
            steer_cmd,
            ax_cmd,
        )
    )
    np.savetxt(
        path,
        table,
        delimiter=",",
        header=(
            "arc_length_m,time_s,speed_mps,longitudinal_accel_mps2,"
            "lateral_accel_mps2,vx_mps,vy_mps,yaw_rate_radps,"
            "steer_cmd_rad,ax_cmd_mps2"
        ),
        comments="",
    )
