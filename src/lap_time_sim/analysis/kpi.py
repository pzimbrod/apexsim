"""KPI calculation from simulation results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lap_time_sim.simulation.runner import LapSimulationResult
from lap_time_sim.utils.constants import GRAVITY


@dataclass(frozen=True)
class KpiSummary:
    """Summary metrics for a lap simulation.

    Args:
        lap_time: Total lap time [s].
        avg_lateral_accel_g: Mean absolute lateral acceleration (g).
        max_lateral_accel_g: Peak absolute lateral acceleration (g).
        avg_longitudinal_accel_g: Mean absolute longitudinal acceleration (g).
        max_longitudinal_accel_g: Peak absolute longitudinal acceleration (g).
        energy: Integrated positive traction energy [kWh].
    """

    lap_time: float
    avg_lateral_accel_g: float
    max_lateral_accel_g: float
    avg_longitudinal_accel_g: float
    max_longitudinal_accel_g: float
    energy: float


def compute_kpis(result: LapSimulationResult) -> KpiSummary:
    """Compute mandatory and energy KPIs from simulation output.

    Args:
        result: Full lap simulation output arrays and integrated metrics.

    Returns:
        Aggregated KPI summary containing lap time, acceleration metrics, and
        electrical-equivalent traction energy [kWh].
    """
    ay_g = result.lateral_accel / GRAVITY
    ax_g = result.longitudinal_accel / GRAVITY

    avg_lat = float(np.mean(np.abs(ay_g)))
    max_lat = float(np.max(np.abs(ay_g)))
    avg_long = float(np.mean(np.abs(ax_g)))
    max_long = float(np.max(np.abs(ax_g)))
    energy = result.energy / 3_600_000.0

    return KpiSummary(
        lap_time=float(result.lap_time),
        avg_lateral_accel_g=avg_lat,
        max_lateral_accel_g=max_lat,
        avg_longitudinal_accel_g=avg_long,
        max_longitudinal_accel_g=max_long,
        energy=float(energy),
    )
