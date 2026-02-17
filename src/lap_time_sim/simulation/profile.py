"""Forward/backward speed profile solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lap_time_sim.simulation.config import SimulationConfig
from lap_time_sim.simulation.envelope import lateral_speed_limit
from lap_time_sim.simulation.model_api import LapTimeVehicleModel
from lap_time_sim.track.models import TrackData
from lap_time_sim.utils.constants import SMALL_EPS


@dataclass(frozen=True)
class SpeedProfileResult:
    """Speed profile along track arc length.

    `lateral_envelope_iterations` reports how many fixed-point iterations were
    required to converge the lateral speed envelope.

    Args:
        speed_mps: Converged speed trace along track arc length (m/s).
        ax_mps2: Net longitudinal acceleration trace (m/s^2).
        ay_mps2: Lateral acceleration trace (m/s^2).
        lateral_envelope_iterations: Number of fixed-point iterations used for
            lateral envelope convergence.
        lap_time_s: Integrated lap time over one track traversal (s).
    """

    speed_mps: np.ndarray
    ax_mps2: np.ndarray
    ay_mps2: np.ndarray
    lateral_envelope_iterations: int
    lap_time_s: float


def _segment_dt(ds_m: float, v0_mps: float, v1_mps: float) -> float:
    """Compute segment travel time from adjacent speed samples.

    Args:
        ds_m: Segment length in meters.
        v0_mps: Segment entry speed in m/s.
        v1_mps: Segment exit speed in m/s.

    Returns:
        Segment traversal time in seconds using trapezoidal average speed.
    """
    v_avg = max(0.5 * (v0_mps + v1_mps), SMALL_EPS)
    return ds_m / v_avg


def solve_speed_profile(
    track: TrackData,
    model: LapTimeVehicleModel,
    config: SimulationConfig,
) -> SpeedProfileResult:
    """Solve lap speed profile with lateral and longitudinal constraints.

    The algorithm is a quasi-steady forward/backward solver in arc-length domain:
    1. Solve a fixed-point lateral speed envelope `v_lat(s)` via the model API.
    2. Forward pass enforces acceleration feasibility.
    3. Backward pass enforces braking feasibility.
    4. Integrate segment times to obtain lap time.

    See `docs/SOLVER.md` for the full mathematical derivation.

    Args:
        track: Track geometry and derived arc-length-domain quantities.
        model: Vehicle-model backend implementing ``LapTimeVehicleModel``.
        config: Global simulation limits and solver iteration settings.

    Returns:
        Converged speed profile, derived accelerations, envelope iteration count,
        and integrated lap time.

    Raises:
        lap_time_sim.utils.exceptions.TrackDataError: If ``track`` is invalid.
        lap_time_sim.utils.exceptions.ConfigurationError: If model or solver
            configuration is invalid.
    """
    track.validate()
    model.validate()
    config.validate()

    n = track.s_m.size
    ds = np.diff(track.s_m)

    v_lat = np.full(n, config.runtime.max_speed, dtype=float)
    lateral_envelope_iterations = 0
    for iteration_idx in range(config.numerics.lateral_envelope_max_iterations):
        previous_v_lat = np.copy(v_lat)
        for idx in range(n):
            ay_lim = model.lateral_accel_limit(
                speed_mps=v_lat[idx],
                banking_rad=float(track.banking_rad[idx]),
            )
            v_lat[idx] = max(
                config.numerics.min_speed,
                lateral_speed_limit(
                    float(track.curvature_1pm[idx]),
                    ay_lim,
                    config.runtime.max_speed,
                ),
            )
        lateral_envelope_iterations = iteration_idx + 1
        max_delta_mps = float(np.max(np.abs(v_lat - previous_v_lat)))
        if max_delta_mps <= config.numerics.lateral_envelope_convergence_tolerance:
            break

    v_forward = np.copy(v_lat)
    v_forward[0] = min(v_forward[0], config.runtime.max_speed)
    for idx in range(n - 1):
        ay_req = v_forward[idx] * v_forward[idx] * abs(track.curvature_1pm[idx])
        net_accel = model.max_longitudinal_accel(
            speed_mps=float(v_forward[idx]),
            ay_required_mps2=float(ay_req),
            grade=float(track.grade[idx]),
            banking_rad=float(track.banking_rad[idx]),
        )

        next_speed_sq = v_forward[idx] ** 2 + 2.0 * net_accel * ds[idx]
        v_candidate = float(np.sqrt(max(next_speed_sq, config.numerics.min_speed**2)))
        v_forward[idx + 1] = min(
            v_forward[idx + 1],
            v_candidate,
            v_lat[idx + 1],
            config.runtime.max_speed,
        )

    v_profile = np.copy(v_forward)
    for idx in range(n - 2, -1, -1):
        ay_req = v_profile[idx + 1] * v_profile[idx + 1] * abs(track.curvature_1pm[idx + 1])
        available_decel = model.max_longitudinal_decel(
            speed_mps=float(v_profile[idx + 1]),
            ay_required_mps2=float(ay_req),
            grade=float(track.grade[idx + 1]),
            banking_rad=float(track.banking_rad[idx + 1]),
        )

        entry_speed_sq = v_profile[idx + 1] ** 2 + 2.0 * available_decel * ds[idx]
        v_entry = float(np.sqrt(max(entry_speed_sq, config.numerics.min_speed**2)))
        v_profile[idx] = min(v_profile[idx], v_entry, v_lat[idx], config.runtime.max_speed)

    ax = np.zeros(n, dtype=float)
    for idx in range(n - 1):
        ax[idx] = (v_profile[idx + 1] ** 2 - v_profile[idx] ** 2) / (2.0 * ds[idx])
    ax[-1] = ax[-2]

    ay = v_profile * v_profile * track.curvature_1pm

    lap_time = 0.0
    for idx in range(n - 1):
        lap_time += _segment_dt(float(ds[idx]), float(v_profile[idx]), float(v_profile[idx + 1]))

    return SpeedProfileResult(
        speed_mps=v_profile,
        ax_mps2=ax,
        ay_mps2=ay,
        lateral_envelope_iterations=lateral_envelope_iterations,
        lap_time_s=lap_time,
    )
