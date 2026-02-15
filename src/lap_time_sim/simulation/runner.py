"""End-to-end lap simulation orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lap_time_sim.simulation.config import SimulationConfig
from lap_time_sim.simulation.profile import solve_speed_profile
from lap_time_sim.tire.models import AxleTireParameters
from lap_time_sim.track.models import TrackData
from lap_time_sim.vehicle.aero import aero_forces
from lap_time_sim.vehicle.bicycle import BicycleModel, ControlInput, VehicleState
from lap_time_sim.vehicle.load_transfer import estimate_normal_loads
from lap_time_sim.vehicle.params import VehicleParameters

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
    ds = np.diff(s_m)
    dt = ds / np.maximum(0.5 * (speed_mps[:-1] + speed_mps[1:]), MIN_AVERAGE_SPEED_FOR_DT_MPS)
    traction_power = np.maximum(power_w[:-1], 0.0)
    return float(np.sum(traction_power * dt))


def simulate_lap(
    track: TrackData,
    vehicle: VehicleParameters,
    tires: AxleTireParameters,
    config: SimulationConfig | None = None,
) -> LapSimulationResult:
    """Run quasi-steady lap simulation and return all primary outputs."""
    sim_config = config or SimulationConfig()
    profile = solve_speed_profile(track=track, vehicle=vehicle, tires=tires, config=sim_config)

    model = BicycleModel(vehicle, tires)

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
        steer = float(np.arctan(vehicle.wheelbase_m * kappa))

        state = VehicleState(vx_mps=speed, vy_mps=0.0, yaw_rate_rps=speed * kappa)
        control = ControlInput(steer_rad=steer, longitudinal_accel_cmd_mps2=ax)
        fb = model.force_balance(state, control)
        yaw_moment_nm[idx] = fb.yaw_moment_nm

        loads = estimate_normal_loads(
            vehicle,
            speed_mps=speed,
            longitudinal_accel_mps2=ax,
            lateral_accel_mps2=ay,
        )
        front_axle_load_n[idx] = loads.front_axle_n
        rear_axle_load_n[idx] = loads.rear_axle_n

        aero = aero_forces(vehicle, speed)
        tractive_force_n = vehicle.mass_kg * ax + aero.drag_n
        power_w[idx] = tractive_force_n * speed

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
