"""Numerical integration utilities."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def rk4_step(
    rhs: Callable[[float, FloatArray], FloatArray],
    time: float,
    state: FloatArray,
    dtime: float,
) -> FloatArray:
    """Run a single explicit RK4 integration step.

    Args:
        rhs: Time-derivative function ``f(t, x)`` for the ODE ``dx/dt = f(t, x)``.
        time: Integration time [s] for the current state.
        state: Current state vector.
        dtime: Integration step width [s].

    Returns:
        Updated state vector after one RK4 step.
    """
    k1 = rhs(time, state)
    k2 = rhs(time + 0.5 * dtime, state + 0.5 * dtime * k1)
    k3 = rhs(time + 0.5 * dtime, state + 0.5 * dtime * k2)
    k4 = rhs(time + dtime, state + dtime * k3)
    return np.asarray(state + (dtime / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), dtype=np.float64)
