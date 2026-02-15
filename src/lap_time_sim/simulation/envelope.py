"""Lateral acceleration envelope estimation."""

from __future__ import annotations

import numpy as np

from lap_time_sim.tire.models import AxleTireParameters
from lap_time_sim.tire.pacejka import magic_formula_lateral
from lap_time_sim.utils.constants import GRAVITY_MPS2, SMALL_EPS
from lap_time_sim.vehicle.load_transfer import estimate_normal_loads
from lap_time_sim.vehicle.params import VehicleParameters

# Quasi-steady peak slip-angle approximation for high-performance slick tires.
PEAK_SLIP_RAD = 0.12
MIN_LATERAL_ACCEL_LIMIT_MPS2 = 0.5
AY_LIMIT_SOLVER_MAX_ITERATIONS = 12
AY_LIMIT_SOLVER_TOL_MPS2 = 0.05


def lateral_accel_limit(
    vehicle: VehicleParameters,
    tires: AxleTireParameters,
    speed_mps: float,
    banking_rad: float,
) -> float:
    """Estimate lateral acceleration limit from tire peaks and banking."""
    ay_banking = GRAVITY_MPS2 * float(np.sin(banking_rad))
    ay_estimate = MIN_LATERAL_ACCEL_LIMIT_MPS2

    for _ in range(AY_LIMIT_SOLVER_MAX_ITERATIONS):
        loads = estimate_normal_loads(
            vehicle,
            speed_mps=speed_mps,
            longitudinal_accel_mps2=0.0,
            lateral_accel_mps2=ay_estimate,
        )
        fz_front_tire = max(loads.front_axle_n / 2.0, SMALL_EPS)
        fz_rear_tire = max(loads.rear_axle_n / 2.0, SMALL_EPS)

        fy_front = 2.0 * float(magic_formula_lateral(PEAK_SLIP_RAD, fz_front_tire, tires.front))
        fy_rear = 2.0 * float(magic_formula_lateral(PEAK_SLIP_RAD, fz_rear_tire, tires.rear))
        ay_tire = (fy_front + fy_rear) / vehicle.mass_kg
        ay_next = max(MIN_LATERAL_ACCEL_LIMIT_MPS2, ay_tire + ay_banking)

        if abs(ay_next - ay_estimate) <= AY_LIMIT_SOLVER_TOL_MPS2:
            ay_estimate = ay_next
            break
        ay_estimate = ay_next

    return float(ay_estimate)


def lateral_speed_limit(
    curvature_1pm: float,
    ay_limit_mps2: float,
    vmax_mps: float,
) -> float:
    """Compute speed limit from curvature and lateral acceleration capability."""
    kappa = abs(curvature_1pm)
    if kappa < SMALL_EPS:
        return vmax_mps
    return float(min(float(np.sqrt(max(ay_limit_mps2 / kappa, 0.0))), vmax_mps))
