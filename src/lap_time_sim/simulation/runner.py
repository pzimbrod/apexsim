"""End-to-end lap simulation orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lap_time_sim.simulation.config import SimulationConfig
from lap_time_sim.simulation.model_api import LapTimeVehicleModel
from lap_time_sim.simulation.profile import solve_speed_profile
from lap_time_sim.track.models import TrackData

MIN_AVERAGE_SPEED_FOR_DT_MPS = 1e-6


@dataclass(frozen=True)
class LapSimulationResult:
    """Simulation output arrays and integrated metrics."""

    track: TrackData
    speed_mps: np.ndarray
    ax_mps2: np.ndarray
    ay_mps2: np.ndarray
    yaw_moment_nm: np.ndarray
    front_axle_load_n: np.ndarray
    rear_axle_load_n: np.ndarray
    power_w: np.ndarray
    energy_j: float
    lap_time_s: float


def _compute_energy(power_w: np.ndarray, speed_mps: np.ndarray, s_m: np.ndarray) -> float:
    """Integrate positive tractive power along the lap.

    Args:
        power_w: Instantaneous tractive power trace in Watt.
        speed_mps: Speed trace in m/s.
        s_m: Cumulative arc-length samples in meters.

    Returns:
        Integrated positive traction energy in Joule.
    """
    ds = np.diff(s_m)
    dt = ds / np.maximum(0.5 * (speed_mps[:-1] + speed_mps[1:]), MIN_AVERAGE_SPEED_FOR_DT_MPS)
    traction_power = np.maximum(power_w[:-1], 0.0)
    return float(np.sum(traction_power * dt))


def simulate_lap(
    track: TrackData,
    model: LapTimeVehicleModel,
    config: SimulationConfig,
) -> LapSimulationResult:
    """Run quasi-steady lap simulation against a vehicle-model API backend.

    Args:
        track: Track geometry and derived arc-length-domain quantities.
        model: Vehicle-model backend implementing ``LapTimeVehicleModel``.
        config: Solver configuration containing runtime and numerical controls.

    Returns:
        Full lap simulation result including profile arrays and diagnostics.

    Raises:
        lap_time_sim.utils.exceptions.TrackDataError: If track data is invalid.
        lap_time_sim.utils.exceptions.ConfigurationError: If model or solver
            configuration is invalid.
    """
    profile = solve_speed_profile(track=track, model=model, config=config)

    n = track.s_m.size
    yaw_moment_nm = np.zeros(n, dtype=float)
    front_axle_load_n = np.zeros(n, dtype=float)
    rear_axle_load_n = np.zeros(n, dtype=float)
    power_w = np.zeros(n, dtype=float)

    for idx in range(n):
        speed = float(profile.speed_mps[idx])
        ax = float(profile.ax_mps2[idx])
        ay = float(profile.ay_mps2[idx])
        kappa = float(track.curvature_1pm[idx])
        diagnostics = model.diagnostics(
            speed_mps=speed,
            ax_mps2=ax,
            ay_mps2=ay,
            curvature_1pm=kappa,
        )
        yaw_moment_nm[idx] = diagnostics.yaw_moment_nm
        front_axle_load_n[idx] = diagnostics.front_axle_load_n
        rear_axle_load_n[idx] = diagnostics.rear_axle_load_n
        power_w[idx] = diagnostics.power_w

    energy_j = _compute_energy(power_w, profile.speed_mps, track.s_m)

    return LapSimulationResult(
        track=track,
        speed_mps=profile.speed_mps,
        ax_mps2=profile.ax_mps2,
        ay_mps2=profile.ay_mps2,
        yaw_moment_nm=yaw_moment_nm,
        front_axle_load_n=front_axle_load_n,
        rear_axle_load_n=rear_axle_load_n,
        power_w=power_w,
        energy_j=energy_j,
        lap_time_s=profile.lap_time_s,
    )
