"""Run an example lap on Spa-Francorchamps track data."""

from __future__ import annotations

import logging

from common import example_vehicle_parameters, spa_output_root, spa_track_path

from pylapsim.analysis import compute_kpis, export_standard_plots
from pylapsim.analysis.export import export_kpi_json
from pylapsim.simulation import build_simulation_config, simulate_lap
from pylapsim.tire import default_axle_tire_parameters
from pylapsim.track import load_track_csv
from pylapsim.utils import configure_logging
from pylapsim.vehicle import SingleTrackPhysics, build_single_track_model


def main() -> None:
    """Run one full Spa lap simulation and export plots plus KPI JSON."""
    configure_logging(logging.INFO)
    logger = logging.getLogger("spa_example")

    track = load_track_csv(spa_track_path())

    vehicle = example_vehicle_parameters()
    tires = default_axle_tire_parameters()
    model = build_single_track_model(
        vehicle=vehicle,
        tires=tires,
        physics=SingleTrackPhysics(),
    )
    config = build_simulation_config()

    result = simulate_lap(track=track, model=model, config=config)
    kpis = compute_kpis(result)

    output_dir = spa_output_root() / "single_track"
    export_standard_plots(result, output_dir)
    export_kpi_json(kpis, output_dir / "kpis.json")

    logger.info("Lap time: %.2f s", kpis.lap_time)
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
    logger.info("Energy use: %.2f kWh", kpis.energy)


if __name__ == "__main__":
    main()
