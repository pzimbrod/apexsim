"""Compare bicycle and point-mass model lap simulations on Spa."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lap_time_sim.analysis import compute_kpis, export_standard_plots
from lap_time_sim.analysis.export import export_kpi_json
from lap_time_sim.simulation import build_simulation_config, simulate_lap
from lap_time_sim.simulation.runner import LapSimulationResult
from lap_time_sim.tire import default_axle_tire_parameters
from lap_time_sim.track import load_track_csv
from lap_time_sim.utils import configure_logging
from lap_time_sim.utils.constants import STANDARD_AIR_DENSITY
from lap_time_sim.vehicle import (
    BicyclePhysics,
    PointMassPhysics,
    VehicleParameters,
    build_bicycle_model,
    build_point_mass_model,
    calibrate_point_mass_friction_to_bicycle,
)


def _example_vehicle_parameters() -> VehicleParameters:
    """Create explicit vehicle parameters used by the Spa examples.

    Returns:
        Vehicle parameter set for the example runs.
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


def _export_speed_comparison_plot(
    bicycle_result: LapSimulationResult,
    point_mass_result: LapSimulationResult,
    path: Path,
) -> None:
    """Export an overlay plot of speed traces for both vehicle models.

    Args:
        bicycle_result: Lap simulation result from bicycle backend.
        point_mass_result: Lap simulation result from point-mass backend.
        path: Output path for the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax.plot(
        bicycle_result.track.arc_length,
        bicycle_result.speed,
        lw=2.0,
        label="Bicycle model",
    )
    ax.plot(
        point_mass_result.track.arc_length,
        point_mass_result.speed,
        lw=2.0,
        label="Point-mass model (calibrated)",
    )
    ax.set_xlabel("Arc length [m]")
    ax.set_ylabel("Speed [m/s]")
    ax.set_title("Spa speed trace comparison")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _kpi_delta_dict(
    bicycle_result: LapSimulationResult,
    point_mass_result: LapSimulationResult,
) -> dict[str, float]:
    """Compute comparison deltas between bicycle and point-mass outputs.

    Args:
        bicycle_result: Lap simulation result from bicycle backend.
        point_mass_result: Lap simulation result from point-mass backend.

    Returns:
        Dictionary with absolute and relative delta metrics.
    """
    speed_delta = bicycle_result.speed - point_mass_result.speed
    mean_point_speed = max(float(np.mean(point_mass_result.speed)), 1e-9)
    return {
        "lap_time_delta": bicycle_result.lap_time - point_mass_result.lap_time,
        "lap_time_delta_pct_of_point_mass": (
            (bicycle_result.lap_time - point_mass_result.lap_time)
            / point_mass_result.lap_time
            * 100.0
        ),
        "mean_speed_delta": float(np.mean(speed_delta)),
        "mean_abs_speed_delta": float(np.mean(np.abs(speed_delta))),
        "max_abs_speed_delta": float(np.max(np.abs(speed_delta))),
        "mean_speed_delta_pct_of_point_mass": float(
            np.mean(speed_delta) / mean_point_speed * 100.0
        ),
    }


def main() -> None:
    """Run both models on Spa and export side-by-side comparison artifacts."""
    configure_logging(logging.INFO)
    logger = logging.getLogger("spa_model_comparison")

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "examples" / "output" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    track = load_track_csv(project_root / "data" / "spa_francorchamps.csv")
    vehicle = _example_vehicle_parameters()
    config = build_simulation_config()
    tires = default_axle_tire_parameters()
    bicycle_physics = BicyclePhysics()

    bicycle_model = build_bicycle_model(
        vehicle=vehicle,
        tires=tires,
        physics=bicycle_physics,
    )
    bicycle_result = simulate_lap(track=track, model=bicycle_model, config=config)
    calibration = calibrate_point_mass_friction_to_bicycle(
        vehicle=vehicle,
        tires=tires,
        bicycle_physics=bicycle_physics,
        speed_samples=bicycle_result.speed,
    )
    point_mass_model = build_point_mass_model(
        vehicle=vehicle,
        physics=PointMassPhysics(
            max_drive_accel=bicycle_physics.max_drive_accel,
            max_brake_accel=bicycle_physics.max_brake_accel,
            friction_coefficient=calibration.friction_coefficient,
        ),
    )
    point_mass_result = simulate_lap(track=track, model=point_mass_model, config=config)
    bicycle_kpis = compute_kpis(bicycle_result)
    point_mass_kpis = compute_kpis(point_mass_result)
    delta = _kpi_delta_dict(bicycle_result=bicycle_result, point_mass_result=point_mass_result)

    bicycle_dir = output_dir / "bicycle"
    point_mass_dir = output_dir / "point_mass_calibrated"
    export_standard_plots(bicycle_result, bicycle_dir)
    export_standard_plots(point_mass_result, point_mass_dir)
    export_kpi_json(bicycle_kpis, bicycle_dir / "kpis.json")
    export_kpi_json(point_mass_kpis, point_mass_dir / "kpis.json")
    _export_speed_comparison_plot(
        bicycle_result=bicycle_result,
        point_mass_result=point_mass_result,
        path=output_dir / "speed_trace_comparison.png",
    )

    comparison_payload = {
        "calibration": {
            "friction_coefficient": calibration.friction_coefficient,
            "mu_mean": float(np.mean(calibration.mu_samples)),
            "mu_min": float(np.min(calibration.mu_samples)),
            "mu_max": float(np.max(calibration.mu_samples)),
            "sample_count": int(calibration.speed_samples.size),
        },
        "bicycle_kpis": asdict(bicycle_kpis),
        "point_mass_kpis": asdict(point_mass_kpis),
        "delta_bicycle_minus_point_mass": delta,
    }
    (output_dir / "comparison_kpis.json").write_text(
        json.dumps(comparison_payload, indent=2),
        encoding="utf-8",
    )

    logger.info("Bicycle lap time: %.2f s", bicycle_kpis.lap_time)
    logger.info(
        "Point-mass (calibrated) lap time: %.2f s | fitted mu: %.3f",
        point_mass_kpis.lap_time,
        calibration.friction_coefficient,
    )
    logger.info(
        "Delta (bicycle - point-mass): %.2f s (%.2f %%)",
        delta["lap_time_delta"],
        delta["lap_time_delta_pct_of_point_mass"],
    )
    logger.info("Comparison artifacts written to %s", output_dir)


if __name__ == "__main__":
    main()
