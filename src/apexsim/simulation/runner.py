"""End-to-end lap simulation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from apexsim.simulation.config import SimulationConfig
from apexsim.simulation.model_api import VehicleModel
from apexsim.simulation.profile import SpeedProfileResult, solve_speed_profile
from apexsim.simulation.transient_common import TransientProfileResult
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
        yaw_moment: Yaw-moment trace [N*m]. In quasi-static mode this is zero by
            model assumption. In transient mode this is reported as the dynamic
            yaw residual ``I_z * dr/dt``.
        front_axle_load: Front-axle normal load trace [N].
        rear_axle_load: Rear-axle normal load trace [N].
        power: Tractive power trace [W].
        energy: Integrated positive tractive energy [J].
        lap_time: Integrated lap time [s].
        solver_mode: Solver mode used for this lap result.
        time: Optional cumulative time trace [s] for transient solver mode.
        vx: Optional body-frame longitudinal speed trace [m/s].
        vy: Optional body-frame lateral speed trace [m/s].
        yaw_rate: Optional yaw-rate trace [rad/s].
        steer_cmd: Optional steering command trace [rad].
        ax_cmd: Optional longitudinal acceleration command trace [m/s^2].
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
    solver_mode: str = "quasi_static"
    time: np.ndarray | None = None
    vx: np.ndarray | None = None
    vy: np.ndarray | None = None
    yaw_rate: np.ndarray | None = None
    steer_cmd: np.ndarray | None = None
    ax_cmd: np.ndarray | None = None


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


def _resolve_yaw_inertia(model: VehicleModel) -> float:
    """Resolve model yaw inertia for transient yaw-moment residual evaluation.

    Args:
        model: Vehicle model instance.

    Returns:
        Positive yaw inertia [kg*m^2], or ``0.0`` if unavailable/invalid.
    """
    vehicle = getattr(model, "vehicle", None)
    yaw_inertia = getattr(vehicle, "yaw_inertia", None)
    if yaw_inertia is None:
        return 0.0
    yaw_inertia_value = float(yaw_inertia)
    if not np.isfinite(yaw_inertia_value) or yaw_inertia_value <= 0.0:
        return 0.0
    return yaw_inertia_value


def _compute_transient_yaw_moment_residual(
    *,
    model: VehicleModel,
    transient_profile: TransientProfileResult,
) -> np.ndarray:
    """Compute transient yaw-moment residual from solved yaw dynamics.

    The residual is defined as ``M_z = I_z * dr/dt`` and is used as the transient
    yaw-moment output to align post-processing with dynamic equilibrium checks.

    Args:
        model: Vehicle model instance.
        transient_profile: Transient solver profile with ``time`` and ``yaw_rate``.

    Returns:
        Yaw-moment residual trace [N*m], or zeros if inputs are invalid.
    """
    yaw_rate = np.asarray(transient_profile.yaw_rate, dtype=float)
    if yaw_rate.ndim != 1:
        return np.zeros_like(yaw_rate, dtype=float)
    if yaw_rate.size < 2:
        return np.zeros_like(yaw_rate, dtype=float)

    time = np.asarray(transient_profile.time, dtype=float)
    if time.shape != yaw_rate.shape:
        return np.zeros_like(yaw_rate, dtype=float)
    if np.any(~np.isfinite(time)) or np.any(~np.isfinite(yaw_rate)):
        return np.zeros_like(yaw_rate, dtype=float)

    dt = np.diff(time)
    if np.any(dt <= 0.0):
        return np.zeros_like(yaw_rate, dtype=float)

    yaw_inertia = _resolve_yaw_inertia(model)
    if yaw_inertia <= 0.0:
        return np.zeros_like(yaw_rate, dtype=float)

    edge_order: Literal[1, 2] = 2 if yaw_rate.size >= 3 else 1
    yaw_accel = np.gradient(yaw_rate, time, edge_order=edge_order)
    if np.any(~np.isfinite(yaw_accel)):
        return np.zeros_like(yaw_rate, dtype=float)
    return np.asarray(yaw_inertia * yaw_accel, dtype=float)


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
        torch_profile = solve_speed_profile_torch(track=track, model=torch_model, config=config)
        return torch_profile.to_numpy()

    if config.runtime.compute_backend == "numba":
        from apexsim.simulation.numba_profile import NumbaSpeedModel, solve_speed_profile_numba

        numba_model = cast(NumbaSpeedModel, model)
        return solve_speed_profile_numba(track=track, model=numba_model, config=config)

    return solve_speed_profile(track=track, model=model, config=config)


def _solve_transient_profile(
    track: TrackData,
    model: VehicleModel,
    config: SimulationConfig,
) -> TransientProfileResult:
    """Dispatch transient OC solve based on configured compute backend.

    Args:
        track: Track geometry and derived arc-length-domain quantities.
        model: Vehicle-model backend implementing ``VehicleModel``.
        config: Solver configuration containing runtime and numerical controls.

    Returns:
        Backend transient solve result.
    """
    if config.runtime.compute_backend == "torch":
        from apexsim.simulation.transient_torch import solve_transient_lap_torch

        torch_result = solve_transient_lap_torch(track=track, model=model, config=config)
        return torch_result.to_numpy()

    if config.runtime.compute_backend == "numba":
        from apexsim.simulation.transient_numba import solve_transient_lap_numba

        return solve_transient_lap_numba(track=track, model=model, config=config)

    from apexsim.simulation.transient_numpy import solve_transient_lap_numpy

    return solve_transient_lap_numpy(track=track, model=model, config=config)


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
    """Run lap simulation against a vehicle-model API backend.

    Args:
        track: Track geometry and derived arc-length-domain quantities.
        model: Vehicle-model backend implementing ``VehicleModel``.
        config: Solver configuration containing runtime and numerical controls.

    Returns:
        Full lap simulation result including profile arrays, diagnostics,
        and (for transient mode) time/state/control traces.

    Raises:
        apexsim.utils.exceptions.TrackDataError: If track data is invalid.
        apexsim.utils.exceptions.ConfigurationError: If model or solver
            configuration is invalid.
    """
    if config.runtime.solver_mode == "transient_oc":
        transient_profile = _solve_transient_profile(track=track, model=model, config=config)
        _diagnostic_yaw, front_axle_load, rear_axle_load, power = _compute_diagnostics(
            track=track,
            model=model,
            profile=SpeedProfileResult(
                speed=np.asarray(transient_profile.speed, dtype=float),
                longitudinal_accel=np.asarray(transient_profile.longitudinal_accel, dtype=float),
                lateral_accel=np.asarray(transient_profile.lateral_accel, dtype=float),
                lateral_envelope_iterations=0,
                lap_time=float(transient_profile.lap_time),
            ),
        )
        yaw_moment = _compute_transient_yaw_moment_residual(
            model=model,
            transient_profile=transient_profile,
        )
        energy = _compute_energy(power, transient_profile.speed, track.arc_length)
        return LapResult(
            track=track,
            speed=np.asarray(transient_profile.speed, dtype=float),
            longitudinal_accel=np.asarray(transient_profile.longitudinal_accel, dtype=float),
            lateral_accel=np.asarray(transient_profile.lateral_accel, dtype=float),
            yaw_moment=yaw_moment,
            front_axle_load=front_axle_load,
            rear_axle_load=rear_axle_load,
            power=power,
            energy=energy,
            lap_time=float(transient_profile.lap_time),
            solver_mode="transient_oc",
            time=np.asarray(transient_profile.time, dtype=float),
            vx=np.asarray(transient_profile.vx, dtype=float),
            vy=np.asarray(transient_profile.vy, dtype=float),
            yaw_rate=np.asarray(transient_profile.yaw_rate, dtype=float),
            steer_cmd=np.asarray(transient_profile.steer_cmd, dtype=float),
            ax_cmd=np.asarray(transient_profile.ax_cmd, dtype=float),
        )

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
        solver_mode="quasi_static",
    )
