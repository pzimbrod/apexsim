"""Pacejka-style lateral tire model implementation."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from lap_time_sim.tire.models import AxleTireParameters, PacejkaParameters
from lap_time_sim.utils.constants import SMALL_EPS

FloatArray: TypeAlias = npt.NDArray[np.float64]


def _as_array(value: float | FloatArray) -> FloatArray:
    return np.asarray(value, dtype=np.float64)


def magic_formula_lateral(
    slip_angle_rad: float | FloatArray,
    normal_load_n: float | FloatArray,
    params: PacejkaParameters,
) -> float | FloatArray:
    """Evaluate lateral tire force with load-sensitive Pacejka coefficients.

    Args:
        slip_angle_rad: Tire slip angle in radians.
        normal_load_n: Tire normal load in Newton.
        params: Pacejka coefficient set.

    Returns:
        Lateral force in Newton. Sign follows slip-angle sign.
    """
    params.validate()
    alpha = _as_array(slip_angle_rad)
    fz = np.maximum(_as_array(normal_load_n), SMALL_EPS)

    slip_term = params.B * alpha
    nonlinear = slip_term - params.E * (slip_term - np.arctan(slip_term))
    shape = np.sin(params.C * np.arctan(nonlinear))

    load_delta = (fz - params.fz_reference_n) / params.fz_reference_n
    mu_scale = np.maximum(1.0 + params.load_sensitivity * load_delta, params.min_mu_scale)

    fy = params.D * mu_scale * fz * shape

    if np.isscalar(slip_angle_rad) and np.isscalar(normal_load_n):
        return float(np.asarray(fy).item())
    return np.asarray(fy, dtype=np.float64)


def axle_lateral_forces(
    front_slip_rad: float,
    rear_slip_rad: float,
    front_axle_load_n: float,
    rear_axle_load_n: float,
    axle_params: AxleTireParameters,
) -> tuple[float, float]:
    """Compute total front/rear axle lateral force for bicycle-equivalent tires."""
    axle_params.validate()
    front_tire_load = max(front_axle_load_n / 2.0, SMALL_EPS)
    rear_tire_load = max(rear_axle_load_n / 2.0, SMALL_EPS)

    fy_front = 2.0 * float(
        magic_formula_lateral(front_slip_rad, front_tire_load, axle_params.front)
    )
    fy_rear = 2.0 * float(magic_formula_lateral(rear_slip_rad, rear_tire_load, axle_params.rear))
    return fy_front, fy_rear
