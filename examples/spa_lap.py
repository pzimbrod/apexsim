"""Run an example lap on Spa-Francorchamps track data."""

from __future__ import annotations

import logging
from pathlib import Path

from lap_time_sim.analysis import compute_kpis, export_standard_plots
from lap_time_sim.analysis.export import export_kpi_json
from lap_time_sim.simulation import SimulationConfig, simulate_lap
from lap_time_sim.tire import default_axle_tire_parameters
from lap_time_sim.track import load_track_csv
from lap_time_sim.utils import configure_logging
from lap_time_sim.vehicle import default_vehicle_parameters


def main() -> None:
    configure_logging(logging.INFO)
    logger = logging.getLogger("spa_example")

    project_root = Path(__file__).resolve().parents[1]
    track = load_track_csv(project_root / "data" / "spa_francorchamps.csv")

    vehicle = default_vehicle_parameters()
    tires = default_axle_tire_parameters()
    config = SimulationConfig()

    result = simulate_lap(track=track, vehicle=vehicle, tires=tires, config=config)
    kpis = compute_kpis(result)

    output_dir = project_root / "examples" / "output"
    export_standard_plots(result, output_dir)
    export_kpi_json(kpis, output_dir / "kpis.json")

    logger.info("Lap time: %.2f s", kpis.lap_time_s)
    logger.info(
        "Avg lateral accel: %.2f g | Max lateral accel: %.2f g",
        kpis.avg_lateral_accel_g,
        kpis.max_lateral_accel_g,
    )
    logger.info(
        "Avg longitudinal accel: %.2f g | Max longitudinal accel: %.2f g",
        kpis.avg_longitudinal_accel_g,
        kpis.max_longitudinal_accel_g,
    )
    logger.info("Energy use: %.2f kWh", kpis.energy_kwh)


if __name__ == "__main__":
    main()
