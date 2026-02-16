"""Run an example lap on Spa-Francorchamps track data."""

from __future__ import annotations

import logging
from pathlib import Path

from lap_time_sim.analysis import compute_kpis, export_standard_plots
from lap_time_sim.analysis.export import export_kpi_json
from lap_time_sim.simulation import build_simulation_config, simulate_lap
from lap_time_sim.tire import default_axle_tire_parameters
from lap_time_sim.track import load_track_csv
from lap_time_sim.utils import configure_logging
from lap_time_sim.utils.constants import AIR_DENSITY_KGPM3
from lap_time_sim.vehicle import BicyclePhysics, VehicleParameters, build_bicycle_model


def _example_vehicle_parameters() -> VehicleParameters:
    """Create explicit vehicle parameters used by the Spa example.

    Returns:
        Vehicle parameter set for the example run.
    """
    return VehicleParameters(
        mass_kg=798.0,
        yaw_inertia_kgm2=1120.0,
        h_cg_m=0.31,
        wheelbase_m=3.60,
        track_front_m=1.60,
        track_rear_m=1.55,
        static_front_weight_fraction=0.46,
        cop_position_m=0.10,
        c_l=3.20,
        c_d=0.90,
        frontal_area_m2=1.50,
        roll_rate_nm_per_deg=4200.0,
        spring_rate_front_npm=180000.0,
        spring_rate_rear_npm=165000.0,
        arb_distribution_front=0.55,
        ride_height_front_m=0.030,
        ride_height_rear_m=0.050,
        air_density_kgpm3=AIR_DENSITY_KGPM3,
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
