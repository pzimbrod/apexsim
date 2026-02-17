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
from lap_time_sim.utils.constants import AIR_DENSITY_KGPM3
from lap_time_sim.vehicle import (
    BicyclePhysics,
    PointMassPhysics,
    VehicleParameters,
    build_bicycle_model,
    build_point_mass_model,
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
        air_density=AIR_DENSITY_KGPM3,
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
        bicycle_result.track.s_m,
        bicycle_result.speed_mps,
        lw=2.0,
        label="Bicycle model",
    )
    ax.plot(
        point_mass_result.track.s_m,
        point_mass_result.speed_mps,
        lw=2.0,
        label="Point-mass model",
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
    speed_delta = bicycle_result.speed_mps - point_mass_result.speed_mps
    mean_point_speed = max(float(np.mean(point_mass_result.speed_mps)), 1e-9)
    return {
        "lap_time_delta_s": bicycle_result.lap_time_s - point_mass_result.lap_time_s,
        "lap_time_delta_pct_of_point_mass": (
            (bicycle_result.lap_time_s - point_mass_result.lap_time_s)
            / point_mass_result.lap_time_s
            * 100.0
        ),
        "mean_speed_delta_mps": float(np.mean(speed_delta)),
        "mean_abs_speed_delta_mps": float(np.mean(np.abs(speed_delta))),
        "max_abs_speed_delta_mps": float(np.max(np.abs(speed_delta))),
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

    bicycle_model = build_bicycle_model(
        vehicle=vehicle,
        tires=default_axle_tire_parameters(),
        physics=BicyclePhysics(),
    )
    point_mass_model = build_point_mass_model(
        vehicle=vehicle,
        physics=PointMassPhysics(),
    )

    bicycle_result = simulate_lap(track=track, model=bicycle_model, config=config)
    point_mass_result = simulate_lap(track=track, model=point_mass_model, config=config)
    bicycle_kpis = compute_kpis(bicycle_result)
    point_mass_kpis = compute_kpis(point_mass_result)
    delta = _kpi_delta_dict(bicycle_result=bicycle_result, point_mass_result=point_mass_result)

    bicycle_dir = output_dir / "bicycle"
    point_mass_dir = output_dir / "point_mass"
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
        "bicycle_kpis": asdict(bicycle_kpis),
        "point_mass_kpis": asdict(point_mass_kpis),
        "delta_bicycle_minus_point_mass": delta,
    }
    (output_dir / "comparison_kpis.json").write_text(
        json.dumps(comparison_payload, indent=2),
        encoding="utf-8",
    )

    logger.info("Bicycle lap time: %.2f s", bicycle_kpis.lap_time_s)
    logger.info("Point-mass lap time: %.2f s", point_mass_kpis.lap_time_s)
    logger.info(
        "Delta (bicycle - point-mass): %.2f s (%.2f %%)",
        delta["lap_time_delta_s"],
        delta["lap_time_delta_pct_of_point_mass"],
    )
    logger.info("Comparison artifacts written to %s", output_dir)


if __name__ == "__main__":
    main()
