"""End-to-end lap simulation orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pylapsim.simulation.config import SimulationConfig
from pylapsim.simulation.model_api import VehicleModel
from pylapsim.simulation.profile import solve_speed_profile
from pylapsim.track.models import TrackData

MIN_AVERAGE_SPEED_FOR_TIME_STEP = 1e-6


@dataclass(frozen=True)
class LapResult:
    """Simulation output arrays and integrated metrics.

    Args:
        track: Track geometry used for the simulation.
        speed: Converged speed trace along arc length [m/s].
        longitudinal_accel: Net longitudinal acceleration trace [m/s^2].
        lateral_accel: Lateral acceleration trace [m/s^2].
        yaw_moment: Yaw moment trace from model diagnostics [N*m].
        front_axle_load: Front-axle normal load trace [N].
        rear_axle_load: Rear-axle normal load trace [N].
        power: Tractive power trace [W].
        energy: Integrated positive tractive energy [J].
        lap_time: Integrated lap time [s].
    """

    track: TrackData
    speed: np.ndarray
    longitudinal_accel: np.ndarray
    lateral_accel: np.ndarray
    yaw_moment: np.ndarray
    front_axle_load: np.ndarray
    rear_axle_load: np.ndarray
    power: np.ndarray
    energy: float
    lap_time: float


def _compute_energy(power: np.ndarray, speed: np.ndarray, arc_length: np.ndarray) -> float:
    """Integrate positive tractive power along the lap.

    Args:
        power: Instantaneous tractive power trace [W].
        speed: Speed trace [m/s].
        arc_length: Cumulative arc-length samples [m].

    Returns:
        Integrated positive traction energy [J].
    """
    ds = np.diff(arc_length)
    dt = ds / np.maximum(0.5 * (speed[:-1] + speed[1:]), MIN_AVERAGE_SPEED_FOR_TIME_STEP)
    traction_power = np.maximum(power[:-1], 0.0)
    return float(np.sum(traction_power * dt))


def simulate_lap(
    track: TrackData,
    model: VehicleModel,
    config: SimulationConfig,
) -> LapResult:
    """Run quasi-steady lap simulation against a vehicle-model API backend.

    Args:
        track: Track geometry and derived arc-length-domain quantities.
        model: Vehicle-model backend implementing ``VehicleModel``.
        config: Solver configuration containing runtime and numerical controls.

    Returns:
        Full lap simulation result including profile arrays and diagnostics.

    Raises:
        pylapsim.utils.exceptions.TrackDataError: If track data is invalid.
        pylapsim.utils.exceptions.ConfigurationError: If model or solver
            configuration is invalid.
    """
    profile = solve_speed_profile(track=track, model=model, config=config)

    n = track.arc_length.size
    yaw_moment = np.zeros(n, dtype=float)
    front_axle_load = np.zeros(n, dtype=float)
    rear_axle_load = np.zeros(n, dtype=float)
    power = np.zeros(n, dtype=float)

    for idx in range(n):
        speed = float(profile.speed[idx])
        ax = float(profile.longitudinal_accel[idx])
        ay = float(profile.lateral_accel[idx])
        kappa = float(track.curvature[idx])
        diagnostics = model.diagnostics(
            speed=speed,
            longitudinal_accel=ax,
            lateral_accel=ay,
            curvature=kappa,
        )
        yaw_moment[idx] = diagnostics.yaw_moment
        front_axle_load[idx] = diagnostics.front_axle_load
        rear_axle_load[idx] = diagnostics.rear_axle_load
        power[idx] = diagnostics.power

    energy = _compute_energy(power, profile.speed, track.arc_length)

    return LapResult(
        track=track,
        speed=profile.speed,
        longitudinal_accel=profile.longitudinal_accel,
        lateral_accel=profile.lateral_accel,
        yaw_moment=yaw_moment,
        front_axle_load=front_axle_load,
        rear_axle_load=rear_axle_load,
        power=power,
        energy=energy,
        lap_time=profile.lap_time,
    )
