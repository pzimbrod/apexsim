"""Forward/backward speed profile solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lap_time_sim.simulation.config import SimulationConfig
from lap_time_sim.simulation.envelope import lateral_accel_limit, lateral_speed_limit
from lap_time_sim.tire.models import AxleTireParameters
from lap_time_sim.track.models import TrackData
from lap_time_sim.utils.constants import GRAVITY_MPS2, SMALL_EPS
from lap_time_sim.vehicle.aero import aero_forces
from lap_time_sim.vehicle.params import VehicleParameters


@dataclass(frozen=True)
class SpeedProfileResult:
    """Speed profile along track arc length."""

    speed_mps: np.ndarray
    ax_mps2: np.ndarray
    ay_mps2: np.ndarray
    lap_time_s: float


def _friction_circle_scale(ay_required: float, ay_limit: float) -> float:
    if ay_limit <= SMALL_EPS:
        return 0.0
    usage = min(abs(ay_required) / ay_limit, 1.0)
    return float(np.sqrt(max(0.0, 1.0 - usage * usage)))


def _segment_dt(ds_m: float, v0_mps: float, v1_mps: float) -> float:
    v_avg = max(0.5 * (v0_mps + v1_mps), SMALL_EPS)
    return ds_m / v_avg


def solve_speed_profile(
    track: TrackData,
    vehicle: VehicleParameters,
    tires: AxleTireParameters,
    config: SimulationConfig,
) -> SpeedProfileResult:
    """Solve lap speed profile with lateral and longitudinal constraints."""
    track.validate()
    vehicle.validate()
    tires.validate()
    config.validate()

    n = track.s_m.size
    ds = np.diff(track.s_m)

    v_lat = np.full(n, config.max_speed_mps, dtype=float)
    for _ in range(6):
        for idx in range(n):
            ay_lim = lateral_accel_limit(
                vehicle,
                tires,
                speed_mps=v_lat[idx],
                banking_rad=float(track.banking_rad[idx]),
            )
            v_lat[idx] = max(
                config.min_speed_mps,
                lateral_speed_limit(float(track.curvature_1pm[idx]), ay_lim, config.max_speed_mps),
            )

    v_forward = np.copy(v_lat)
    v_forward[0] = min(v_forward[0], config.max_speed_mps)
    for idx in range(n - 1):
        ay_req = v_forward[idx] * v_forward[idx] * abs(track.curvature_1pm[idx])
        ay_lim = lateral_accel_limit(
            vehicle,
            tires,
            speed_mps=v_forward[idx],
            banking_rad=float(track.banking_rad[idx]),
        )
        circle_scale = _friction_circle_scale(ay_req, ay_lim)
        tire_accel = config.max_drive_accel_mps2 * circle_scale

        drag_accel = aero_forces(vehicle, v_forward[idx]).drag_n / vehicle.mass_kg
        grade_accel = GRAVITY_MPS2 * float(track.grade[idx])
        net_accel = tire_accel - drag_accel - grade_accel

        next_speed_sq = v_forward[idx] ** 2 + 2.0 * net_accel * ds[idx]
        v_candidate = float(np.sqrt(max(next_speed_sq, config.min_speed_mps**2)))
        v_forward[idx + 1] = min(
            v_forward[idx + 1],
            v_candidate,
            v_lat[idx + 1],
            config.max_speed_mps,
        )

    v_profile = np.copy(v_forward)
    for idx in range(n - 2, -1, -1):
        ay_req = v_profile[idx + 1] * v_profile[idx + 1] * abs(track.curvature_1pm[idx + 1])
        ay_lim = lateral_accel_limit(
            vehicle,
            tires,
            speed_mps=v_profile[idx + 1],
            banking_rad=float(track.banking_rad[idx + 1]),
        )
        circle_scale = _friction_circle_scale(ay_req, ay_lim)
        tire_brake = config.max_brake_accel_mps2 * circle_scale

        drag_accel = aero_forces(vehicle, v_profile[idx + 1]).drag_n / vehicle.mass_kg
        grade_accel = GRAVITY_MPS2 * float(track.grade[idx + 1])
        available_decel = tire_brake + drag_accel + grade_accel

        entry_speed_sq = v_profile[idx + 1] ** 2 + 2.0 * available_decel * ds[idx]
        v_entry = float(np.sqrt(max(entry_speed_sq, config.min_speed_mps**2)))
        v_profile[idx] = min(v_profile[idx], v_entry, v_lat[idx], config.max_speed_mps)

    ax = np.zeros(n, dtype=float)
    for idx in range(n - 1):
        ax[idx] = (v_profile[idx + 1] ** 2 - v_profile[idx] ** 2) / (2.0 * ds[idx])
    ax[-1] = ax[-2]

    ay = v_profile * v_profile * track.curvature_1pm

    lap_time = 0.0
    for idx in range(n - 1):
        lap_time += _segment_dt(float(ds[idx]), float(v_profile[idx]), float(v_profile[idx + 1]))

    return SpeedProfileResult(speed_mps=v_profile, ax_mps2=ax, ay_mps2=ay, lap_time_s=lap_time)
