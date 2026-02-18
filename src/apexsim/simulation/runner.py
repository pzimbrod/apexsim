"""End-to-end lap simulation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from apexsim.simulation.config import SimulationConfig
from apexsim.simulation.model_api import VehicleModel
from apexsim.simulation.profile import SpeedProfileResult, solve_speed_profile
from apexsim.track.models import TrackData
from apexsim.utils.exceptions import ConfigurationError

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


def _solve_profile(
    track: TrackData,
    model: VehicleModel,
    config: SimulationConfig,
) -> SpeedProfileResult:
    """Dispatch speed-profile solve based on configured compute backend.

    Args:
        track: Track geometry and derived arc-length-domain quantities.
        model: Vehicle-model backend implementing ``VehicleModel``.
        config: Solver configuration containing runtime and numerical controls.

    Returns:
        Backend solver result with speed and acceleration traces.
    """
    if config.runtime.compute_backend == "torch":
        from apexsim.simulation.torch_profile import TorchSpeedModel, solve_speed_profile_torch

        torch_model = cast(TorchSpeedModel, model)
        return solve_speed_profile_torch(track=track, model=torch_model, config=config)

    if config.runtime.compute_backend == "numba":
        from apexsim.simulation.numba_profile import NumbaSpeedModel, solve_speed_profile_numba

        numba_model = cast(NumbaSpeedModel, model)
        return solve_speed_profile_numba(track=track, model=numba_model, config=config)

    return solve_speed_profile(track=track, model=model, config=config)


def _compute_diagnostics(
    track: TrackData,
    model: VehicleModel,
    profile: SpeedProfileResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model diagnostics for all profile samples.

    Args:
        track: Track geometry used for the solved speed profile.
        model: Vehicle-model backend implementing ``VehicleModel``.
        profile: Solved profile containing speed and acceleration traces.

    Returns:
        Tuple ``(yaw_moment, front_axle_load, rear_axle_load, power)``.

    Raises:
        apexsim.utils.exceptions.ConfigurationError: If a vectorized
            diagnostics implementation returns shape-mismatched outputs.
    """
    expected_shape = profile.speed.shape
    batch_method: Any = getattr(model, "diagnostics_batch", None)
    if callable(batch_method):
        diagnostics_batch = batch_method(
            speed=profile.speed,
            longitudinal_accel=profile.longitudinal_accel,
            lateral_accel=profile.lateral_accel,
            curvature=track.curvature,
        )
        if len(diagnostics_batch) != 4:
            msg = (
                "diagnostics_batch must return four arrays "
                "(yaw_moment, front_axle_load, rear_axle_load, power)"
            )
            raise ConfigurationError(msg)

        yaw_moment, front_axle_load, rear_axle_load, power = (
            np.asarray(diagnostics_batch[0], dtype=float),
            np.asarray(diagnostics_batch[1], dtype=float),
            np.asarray(diagnostics_batch[2], dtype=float),
            np.asarray(diagnostics_batch[3], dtype=float),
        )
        for signal_name, signal in (
            ("yaw_moment", yaw_moment),
            ("front_axle_load", front_axle_load),
            ("rear_axle_load", rear_axle_load),
            ("power", power),
        ):
            if signal.shape != expected_shape:
                msg = (
                    "diagnostics_batch returned mismatched shape for "
                    f"{signal_name}: expected {expected_shape}, got {signal.shape}"
                )
                raise ConfigurationError(msg)
        return yaw_moment, front_axle_load, rear_axle_load, power

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

    return yaw_moment, front_axle_load, rear_axle_load, power


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
        apexsim.utils.exceptions.TrackDataError: If track data is invalid.
        apexsim.utils.exceptions.ConfigurationError: If model or solver
            configuration is invalid.
    """
    profile = _solve_profile(track=track, model=model, config=config)

    yaw_moment, front_axle_load, rear_axle_load, power = _compute_diagnostics(
        track=track,
        model=model,
        profile=profile,
    )

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
