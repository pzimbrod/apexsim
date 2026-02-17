"""Kinematic envelope helpers used by the speed-profile solver."""

from __future__ import annotations

import numpy as np

from pylapsim.utils.constants import SMALL_EPS


def lateral_speed_limit(
    curvature: float,
    lateral_accel_limit: float,
    max_speed: float,
) -> float:
    """Compute speed limit from curvature and lateral acceleration capability.

    Args:
        curvature: Signed path curvature [1/m].
        lateral_accel_limit: Available lateral acceleration magnitude [m/s^2].
        max_speed: Global hard speed cap [m/s].

    Returns:
        Maximum feasible speed [m/s] under curvature and lateral limits.
    """
    kappa = abs(curvature)
    if kappa < SMALL_EPS:
        return max_speed
    return float(min(float(np.sqrt(max(lateral_accel_limit / kappa, 0.0))), max_speed))
