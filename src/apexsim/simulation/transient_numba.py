"""Numba transient OC lap solver wrapper."""

from __future__ import annotations

from typing import Any

from apexsim.simulation.config import SimulationConfig
from apexsim.simulation.transient_common import TransientProfileResult
from apexsim.simulation.transient_numpy import solve_transient_lap_numpy
from apexsim.track.models import TrackData
from apexsim.utils.exceptions import ConfigurationError


def solve_transient_lap_numba(
    *,
    track: TrackData,
    model: Any,
    config: SimulationConfig,
) -> TransientProfileResult:
    """Solve transient OC lap for numba backend.

    The numba transient path reuses the shared internal integration and SciPy
    optimization stack, while runtime validation still enforces that numba is
    available and selected.

    Args:
        track: Track geometry.
        model: Vehicle model.
        config: Simulation configuration.

    Returns:
        Transient profile result.

    Raises:
        apexsim.utils.exceptions.ConfigurationError: If numba backend is not
            selected in runtime config.
    """
    if config.runtime.compute_backend != "numba":
        msg = "solve_transient_lap_numba requires runtime.compute_backend='numba'"
        raise ConfigurationError(msg)
    return solve_transient_lap_numpy(track=track, model=model, config=config)

