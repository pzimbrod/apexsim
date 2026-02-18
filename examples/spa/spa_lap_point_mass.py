"""Run an example lap on Spa-Francorchamps with the point-mass model."""

from __future__ import annotations

import logging

from common import example_vehicle_parameters, spa_output_root, spa_track_path

from apexsim.analysis import compute_kpis, export_standard_plots
from apexsim.analysis.export import export_kpi_json
from apexsim.simulation import build_simulation_config, simulate_lap
from apexsim.track import load_track_csv
from apexsim.utils import configure_logging
from apexsim.vehicle import PointMassPhysics, build_point_mass_model


def main() -> None:
    """Run one full Spa lap simulation and export plots plus KPI JSON."""
    configure_logging(logging.INFO)
    logger = logging.getLogger("spa_point_mass_example")

    track = load_track_csv(spa_track_path())
    vehicle = example_vehicle_parameters()

    model = build_point_mass_model(
        vehicle=vehicle,
        physics=PointMassPhysics(
            max_drive_accel=8.0,
            max_brake_accel=16.0,
            friction_coefficient=1.7,
        ),
    )
    config = build_simulation_config()
    result = simulate_lap(track=track, model=model, config=config)
    kpis = compute_kpis(result)

    output_dir = spa_output_root() / "point_mass"
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
