"""Kinematic envelope helpers used by the speed-profile solver."""

from __future__ import annotations

import numpy as np

from lap_time_sim.utils.constants import SMALL_EPS


def lateral_speed_limit(
    curvature_1pm: float,
    ay_limit_mps2: float,
    vmax_mps: float,
) -> float:
    """Compute speed limit from curvature and lateral acceleration capability.

    Args:
        curvature_1pm: Signed path curvature in 1/m.
        ay_limit_mps2: Available lateral acceleration magnitude in m/s^2.
        vmax_mps: Global hard speed cap in m/s.

    Returns:
        Maximum feasible speed in m/s under curvature and lateral limits.
    """
    kappa = abs(curvature_1pm)
    if kappa < SMALL_EPS:
        return vmax_mps
    return float(min(float(np.sqrt(max(ay_limit_mps2 / kappa, 0.0))), vmax_mps))
