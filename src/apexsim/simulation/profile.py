"""Forward/backward speed profile solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from apexsim.simulation._profile_core import (
    ProfileOps,
    SpeedProfileCoreCallbacks,
    SpeedProfileCoreInputs,
    resolve_profile_start_speed,
    solve_speed_profile_core,
)
from apexsim.simulation.config import SimulationConfig
from apexsim.simulation.model_api import VehicleModel
from apexsim.track.models import TrackData
from apexsim.utils.exceptions import ConfigurationError


@dataclass(frozen=True)
class SpeedProfileResult:
    """Speed profile along track arc length.

    `lateral_envelope_iterations` reports how many fixed-point iterations were
    required to converge the lateral speed envelope.

    Args:
        speed: Converged speed trace along track arc length [m/s].
        longitudinal_accel: Net longitudinal acceleration trace [m/s^2].
        lateral_accel: Lateral acceleration trace [m/s^2].
        lateral_envelope_iterations: Number of fixed-point iterations used for
            lateral envelope convergence.
        lap_time: Integrated lap time over one track traversal [s].
    """

    speed: np.ndarray
    longitudinal_accel: np.ndarray
    lateral_accel: np.ndarray
    lateral_envelope_iterations: int
    lap_time: float


_NUMPY_PROFILE_OPS = ProfileOps(
    full=lambda size, value: np.full((size,), value, dtype=float),
    copy=lambda value: np.copy(value),
    scalar=lambda value: float(value),
    abs=lambda value: np.asarray(np.abs(value), dtype=float),
    maximum=lambda left, right: np.maximum(left, right),
    minimum=lambda left, right: np.minimum(left, right),
    clip=lambda value, low, high: np.clip(value, low, high),
    sqrt=lambda value: np.sqrt(value),
    where=lambda condition, left, right: np.where(condition, left, right),
    stack=lambda values: np.asarray(values, dtype=float),
    cat_tail=lambda core: np.concatenate((core, core[-1:])),
    zeros_like=lambda ref: np.zeros_like(ref),
    max=lambda value: float(np.max(value)),
    sum=lambda value: float(np.sum(value)),
    to_float=lambda value: float(value),
)


def _lateral_accel_limit_batch(
    model: VehicleModel,
    speed: np.ndarray,
    banking: np.ndarray,
) -> np.ndarray:
    """Evaluate lateral acceleration limits over all track samples.

    The solver uses an optional vectorized model API when available. If a model
    does not provide `lateral_accel_limit_batch`, it falls back to scalar calls.

    Args:
        model: Vehicle model backend.
        speed: Speed samples [m/s].
        banking: Banking-angle samples [rad].

    Returns:
        Lateral acceleration limit samples [m/s^2].

    Raises:
        apexsim.utils.exceptions.ConfigurationError: If a vectorized model
            implementation returns an incompatible array shape.
    """
    batch_method: Any = getattr(model, "lateral_accel_limit_batch", None)
    if callable(batch_method):
        lateral_limit = np.asarray(batch_method(speed=speed, banking=banking), dtype=float)
        if lateral_limit.shape != speed.shape:
            msg = (
                "lateral_accel_limit_batch must return an array with shape "
                f"{speed.shape}, got {lateral_limit.shape}"
            )
            raise ConfigurationError(msg)
        return lateral_limit

    return np.array(
        [
            model.lateral_accel_limit(speed=float(speed[idx]), banking=float(banking[idx]))
            for idx in range(speed.size)
        ],
        dtype=float,
    )


def solve_speed_profile(
    track: TrackData,
    model: VehicleModel,
    config: SimulationConfig,
) -> SpeedProfileResult:
    """Solve lap speed profile with lateral and longitudinal constraints.

    The algorithm is a quasi-steady forward/backward solver in arc-length domain:
    1. Solve a fixed-point lateral speed envelope `v_lat[s]` via the model API.
    2. Forward pass enforces acceleration feasibility.
    3. Backward pass enforces braking feasibility.
    4. Integrate segment times to obtain lap time.

    See `docs/SOLVER.md` for the full mathematical derivation.

    Args:
        track: Track geometry and derived arc-length-domain quantities.
        model: Vehicle-model backend implementing ``VehicleModel``.
        config: Global simulation limits and solver iteration settings.

    Returns:
        Converged speed profile, derived accelerations, envelope iteration count,
        and integrated lap time.

    Raises:
        apexsim.utils.exceptions.TrackDataError: If ``track`` is invalid.
        apexsim.utils.exceptions.ConfigurationError: If model or solver
            configuration is invalid.
    """
    track.validate()
    model.validate()
    config.validate()

    inputs = SpeedProfileCoreInputs(
        ds=np.diff(track.arc_length),
        curvature=track.curvature,
        grade=track.grade,
        banking=track.banking,
        max_speed=float(config.runtime.max_speed),
        min_speed=float(config.numerics.min_speed),
        start_speed=resolve_profile_start_speed(
            max_speed=float(config.runtime.max_speed),
            initial_speed=config.runtime.initial_speed,
        ),
        lateral_envelope_max_iterations=int(config.numerics.lateral_envelope_max_iterations),
        lateral_envelope_convergence_tolerance=float(
            config.numerics.lateral_envelope_convergence_tolerance
        ),
    )
    callbacks = SpeedProfileCoreCallbacks(
        lateral_accel_limit=lambda *, speed, banking: _lateral_accel_limit_batch(
            model=model,
            speed=np.asarray(speed, dtype=float),
            banking=np.asarray(banking, dtype=float),
        ),
        max_longitudinal_accel=model.max_longitudinal_accel,
        max_longitudinal_decel=model.max_longitudinal_decel,
    )
    core = solve_speed_profile_core(
        inputs=inputs,
        callbacks=callbacks,
        ops=_NUMPY_PROFILE_OPS,
    )

    return SpeedProfileResult(
        speed=np.asarray(core.speed, dtype=float),
        longitudinal_accel=np.asarray(core.longitudinal_accel, dtype=float),
        lateral_accel=np.asarray(core.lateral_accel, dtype=float),
        lateral_envelope_iterations=int(core.lateral_envelope_iterations),
        lap_time=float(core.lap_time),
    )
