"""Run an example lap on Spa-Francorchamps track data."""

from __future__ import annotations

import logging
from pathlib import Path

from pylapsim.analysis import compute_kpis, export_standard_plots
from pylapsim.analysis.export import export_kpi_json
from pylapsim.simulation import build_simulation_config, simulate_lap
from pylapsim.tire import default_axle_tire_parameters
from pylapsim.track import load_track_csv
from pylapsim.utils import configure_logging
from pylapsim.utils.constants import STANDARD_AIR_DENSITY
from pylapsim.vehicle import BicyclePhysics, VehicleParameters, build_bicycle_model


def _example_vehicle_parameters() -> VehicleParameters:
    """Create explicit vehicle parameters used by the Spa example.

    Returns:
        Vehicle parameter set for the example run.
    """
    return VehicleParameters(
        mass=798.0,
        yaw_inertia=1120.0,
        cg_height=0.31,
        wheelbase=3.60,
        front_track=1.60,
        rear_track=1.55,
        front_weight_fraction=0.46,
        cop_position=0.10,
        lift_coefficient=3.20,
        drag_coefficient=0.90,
        frontal_area=1.50,
        roll_rate=4200.0,
        front_spring_rate=180000.0,
        rear_spring_rate=165000.0,
        front_arb_distribution=0.55,
        front_ride_height=0.030,
        rear_ride_height=0.050,
        air_density=STANDARD_AIR_DENSITY,
    )


def main() -> None:
    """Run one full Spa lap simulation and export plots plus KPI JSON."""
    configure_logging(logging.INFO)
    logger = logging.getLogger("spa_example")

    project_root = Path(__file__).resolve().parents[1]
    track = load_track_csv(project_root / "data" / "spa_francorchamps.csv")

    vehicle = _example_vehicle_parameters()
    tires = default_axle_tire_parameters()
    model = build_bicycle_model(
        vehicle=vehicle,
        tires=tires,
        physics=BicyclePhysics(),
    )
    config = build_simulation_config()

    result = simulate_lap(track=track, model=model, config=config)
    kpis = compute_kpis(result)

    output_dir = project_root / "examples" / "output"
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
