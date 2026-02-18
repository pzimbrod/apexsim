"""Pacejka-style lateral tire model implementation."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from pylapsim.tire.models import AxleTireParameters, PacejkaParameters
from pylapsim.utils.constants import SMALL_EPS

FloatArray: TypeAlias = npt.NDArray[np.float64]


def _as_array(value: float | FloatArray) -> FloatArray:
    """Convert scalar or ndarray input to ``float64`` ndarray.

    Args:
        value: Scalar or ndarray input value.

    Returns:
        ``float64`` ndarray representation of ``value``.
    """
    return np.asarray(value, dtype=np.float64)


def magic_formula_lateral(
    slip_angle: float | FloatArray,
    normal_load: float | FloatArray,
    params: PacejkaParameters,
) -> float | FloatArray:
    """Evaluate lateral tire force with load-sensitive Pacejka coefficients.

    Args:
        slip_angle: Tire slip angle [rad].
        normal_load: Tire normal load [N].
        params: Pacejka coefficient set.

    Returns:
        Lateral force [N]. Sign follows slip-angle sign.
    """
    params.validate()
    alpha = _as_array(slip_angle)
    fz = np.maximum(_as_array(normal_load), SMALL_EPS)

    slip_term = params.B * alpha
    nonlinear = slip_term - params.E * (slip_term - np.arctan(slip_term))
    shape = np.sin(params.C * np.arctan(nonlinear))

    load_delta = (fz - params.reference_load) / params.reference_load
    mu_scale = np.maximum(1.0 + params.load_sensitivity * load_delta, params.min_mu_scale)

    fy = params.D * mu_scale * fz * shape

    if np.isscalar(slip_angle) and np.isscalar(normal_load):
        return float(np.asarray(fy).item())
    return np.asarray(fy, dtype=np.float64)


def axle_lateral_forces(
    front_slip_angle: float,
    rear_slip_angle: float,
    front_axle_load: float,
    rear_axle_load: float,
    axle_params: AxleTireParameters,
) -> tuple[float, float]:
    """Compute total front/rear axle lateral force for single_track-equivalent tires.

    Args:
        front_slip_angle: Front axle equivalent slip angle [rad].
        rear_slip_angle: Rear axle equivalent slip angle [rad].
        front_axle_load: Total front axle normal load [N].
        rear_axle_load: Total rear axle normal load [N].
        axle_params: Front/rear Pacejka parameter sets.

    Returns:
        Tuple ``(Fy_front, Fy_rear)`` with axle lateral forces [N].
    """
    axle_params.validate()
    front_tire_load = max(front_axle_load / 2.0, SMALL_EPS)
    rear_tire_load = max(rear_axle_load / 2.0, SMALL_EPS)

    fy_front = 2.0 * float(
        magic_formula_lateral(front_slip_angle, front_tire_load, axle_params.front)
    )
    fy_rear = 2.0 * float(magic_formula_lateral(rear_slip_angle, rear_tire_load, axle_params.rear))
    return fy_front, fy_rear
