"""KPI calculation from simulation results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lap_time_sim.simulation.runner import LapSimulationResult
from lap_time_sim.utils.constants import GRAVITY_MPS2


@dataclass(frozen=True)
class KpiSummary:
    """Summary metrics for a lap simulation."""

    lap_time_s: float
    avg_lateral_accel_g: float
    max_lateral_accel_g: float
    avg_longitudinal_accel_g: float
    max_longitudinal_accel_g: float
    energy_kwh: float


def compute_kpis(result: LapSimulationResult) -> KpiSummary:
    """Compute mandatory and energy KPIs from simulation output.

    Args:
        result: Full lap simulation output arrays and integrated metrics.

    Returns:
        Aggregated KPI summary containing lap time, acceleration metrics, and
        electrical-equivalent traction energy in kWh.
    """
    ay_g = result.ay_mps2 / GRAVITY_MPS2
    ax_g = result.ax_mps2 / GRAVITY_MPS2

    avg_lat = float(np.mean(np.abs(ay_g)))
    max_lat = float(np.max(np.abs(ay_g)))
    avg_long = float(np.mean(np.abs(ax_g)))
    max_long = float(np.max(np.abs(ax_g)))
    energy_kwh = result.energy_j / 3_600_000.0

    return KpiSummary(
        lap_time_s=float(result.lap_time_s),
        avg_lateral_accel_g=avg_lat,
        max_lateral_accel_g=max_lat,
        avg_longitudinal_accel_g=avg_long,
        max_longitudinal_accel_g=max_long,
        energy_kwh=float(energy_kwh),
    )
