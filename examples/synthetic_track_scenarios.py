"""Run and compare synthetic benchmark track scenarios."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt

from pylapsim.analysis import compute_kpis, export_standard_plots
from pylapsim.analysis.export import export_kpi_json
from pylapsim.simulation import build_simulation_config, simulate_lap
from pylapsim.simulation.runner import LapResult
from pylapsim.tire import default_axle_tire_parameters
from pylapsim.track import (
    build_circular_track,
    build_figure_eight_track,
    build_straight_track,
)
from pylapsim.utils import configure_logging
from pylapsim.utils.constants import STANDARD_AIR_DENSITY
from pylapsim.vehicle import (
    SingleTrackPhysics,
    VehicleParameters,
    build_single_track_model,
)

STRAIGHT_LENGTH = 1_000.0
CIRCLE_RADIUS = 50.0
FIGURE_EIGHT_RADIUS = 80.0


def _example_vehicle_parameters() -> VehicleParameters:
    """Create explicit vehicle parameters used by synthetic-track examples.

    Returns:
        Vehicle parameter set for the scenario runs.
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


def _export_speed_trace_comparison(results: dict[str, LapResult], path: Path) -> None:
    """Export one speed-trace plot for all synthetic scenarios.

    Args:
        results: Scenario-name keyed simulation results.
        path: Output path for the figure.
    """
    fig, axis = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)

    for name, result in results.items():
        axis.plot(result.track.arc_length, result.speed, lw=2.0, label=name)

    axis.set_xlabel("Arc length [m]")
    axis.set_ylabel("Speed [m/s]")
    axis.set_title("Synthetic track speed-trace comparison")
    axis.grid(alpha=0.35)
    axis.legend()

    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    """Run synthetic track scenarios and export plots plus KPI summaries."""
    configure_logging(logging.INFO)
    logger = logging.getLogger("synthetic_track_scenarios")

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "examples" / "output" / "synthetic_tracks"
    output_dir.mkdir(parents=True, exist_ok=True)

    vehicle = _example_vehicle_parameters()
    tires = default_axle_tire_parameters()
    model = build_single_track_model(vehicle=vehicle, tires=tires, physics=SingleTrackPhysics())
    config = build_simulation_config(max_speed=115.0)

    tracks = {
        "straight_1km": build_straight_track(length=STRAIGHT_LENGTH),
        "circle_r50": build_circular_track(radius=CIRCLE_RADIUS),
        "figure_eight": build_figure_eight_track(lobe_radius=FIGURE_EIGHT_RADIUS),
    }

    results: dict[str, LapResult] = {}
    summary: dict[str, dict[str, float]] = {}

    for name, track in tracks.items():
        result = simulate_lap(track=track, model=model, config=config)
        kpis = compute_kpis(result)
        scenario_dir = output_dir / name

        export_standard_plots(result, scenario_dir)
        export_kpi_json(kpis, scenario_dir / "kpis.json")

        results[name] = result
        summary[name] = asdict(kpis)
        logger.info("%s lap time: %.2f s", name, kpis.lap_time)

    _export_speed_trace_comparison(results=results, path=output_dir / "speed_trace_comparison.png")
    (output_dir / "scenario_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    logger.info("Synthetic scenario artifacts written to %s", output_dir)


if __name__ == "__main__":
    main()
