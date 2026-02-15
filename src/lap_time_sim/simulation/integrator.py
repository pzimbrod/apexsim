"""Numerical integration utilities."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def rk4_step(
    rhs: Callable[[float, FloatArray], FloatArray],
    t_s: float,
    state: FloatArray,
    dt_s: float,
) -> FloatArray:
    """Run a single explicit RK4 integration step.

    Args:
        rhs: Time-derivative function ``f(t, x)`` for the ODE ``dx/dt = f(t, x)``.
        t_s: Integration time in seconds for the current state.
        state: Current state vector.
        dt_s: Integration step width in seconds.

    Returns:
        Updated state vector after one RK4 step.
    """
    k1 = rhs(t_s, state)
    k2 = rhs(t_s + 0.5 * dt_s, state + 0.5 * dt_s * k1)
    k3 = rhs(t_s + 0.5 * dt_s, state + 0.5 * dt_s * k2)
    k4 = rhs(t_s + dt_s, state + dt_s * k3)
    return np.asarray(state + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), dtype=np.float64)
