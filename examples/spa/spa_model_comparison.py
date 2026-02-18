"""Compare single_track and point-mass model lap simulations on Spa."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from common import example_vehicle_parameters, spa_output_root, spa_track_path

from apexsim.analysis import compute_kpis, export_standard_plots
from apexsim.analysis.export import export_kpi_json
from apexsim.simulation import build_simulation_config, simulate_lap
from apexsim.simulation.runner import LapResult
from apexsim.tire import default_axle_tire_parameters
from apexsim.track import load_track_csv
from apexsim.utils import configure_logging
from apexsim.vehicle import (
    PointMassPhysics,
    SingleTrackPhysics,
    build_point_mass_model,
    build_single_track_model,
    calibrate_point_mass_friction_to_single_track,
)


def _export_speed_comparison_plot(
    single_track_result: LapResult,
    point_mass_result: LapResult,
    path: Path,
) -> None:
    """Export an overlay plot of speed traces for both vehicle models.

    Args:
        single_track_result: Lap simulation result from single_track backend.
        point_mass_result: Lap simulation result from point-mass backend.
        path: Output path for the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax.plot(
        single_track_result.track.arc_length,
        single_track_result.speed,
        lw=2.0,
        label="SingleTrack model",
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
    single_track_result: LapResult,
    point_mass_result: LapResult,
) -> dict[str, float]:
    """Compute comparison deltas between single_track and point-mass outputs.

    Args:
        single_track_result: Lap simulation result from single_track backend.
        point_mass_result: Lap simulation result from point-mass backend.

    Returns:
        Dictionary with absolute and relative delta metrics.
    """
    speed_delta = single_track_result.speed - point_mass_result.speed
    mean_point_speed = max(float(np.mean(point_mass_result.speed)), 1e-9)
    return {
        "lap_time_delta": single_track_result.lap_time - point_mass_result.lap_time,
        "lap_time_delta_pct_of_point_mass": (
            (single_track_result.lap_time - point_mass_result.lap_time)
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

    output_dir = spa_output_root() / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    track = load_track_csv(spa_track_path())
    vehicle = example_vehicle_parameters()
    config = build_simulation_config()
    tires = default_axle_tire_parameters()
    single_track_physics = SingleTrackPhysics()

    single_track_model = build_single_track_model(
        vehicle=vehicle,
        tires=tires,
        physics=single_track_physics,
    )
    single_track_result = simulate_lap(track=track, model=single_track_model, config=config)
    calibration = calibrate_point_mass_friction_to_single_track(
        vehicle=vehicle,
        tires=tires,
        single_track_physics=single_track_physics,
        speed_samples=single_track_result.speed,
    )
    point_mass_model = build_point_mass_model(
        vehicle=vehicle,
        physics=PointMassPhysics(
            max_drive_accel=single_track_physics.max_drive_accel,
            max_brake_accel=single_track_physics.max_brake_accel,
            friction_coefficient=calibration.friction_coefficient,
        ),
    )
    point_mass_result = simulate_lap(track=track, model=point_mass_model, config=config)
    single_track_kpis = compute_kpis(single_track_result)
    point_mass_kpis = compute_kpis(point_mass_result)
    delta = _kpi_delta_dict(
        single_track_result=single_track_result,
        point_mass_result=point_mass_result,
    )

    single_track_dir = output_dir / "single_track"
    point_mass_dir = output_dir / "point_mass_calibrated"
    export_standard_plots(single_track_result, single_track_dir)
    export_standard_plots(point_mass_result, point_mass_dir)
    export_kpi_json(single_track_kpis, single_track_dir / "kpis.json")
    export_kpi_json(point_mass_kpis, point_mass_dir / "kpis.json")
    _export_speed_comparison_plot(
        single_track_result=single_track_result,
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
        "single_track_kpis": asdict(single_track_kpis),
        "point_mass_kpis": asdict(point_mass_kpis),
        "delta_single_track_minus_point_mass": delta,
    }
    (output_dir / "comparison_kpis.json").write_text(
        json.dumps(comparison_payload, indent=2),
        encoding="utf-8",
    )

    logger.info("SingleTrack lap time: %.2f s", single_track_kpis.lap_time)
    logger.info(
        "Point-mass (calibrated) lap time: %.2f s | fitted mu: %.3f",
        point_mass_kpis.lap_time,
        calibration.friction_coefficient,
    )
    logger.info(
        "Delta (single_track - point-mass): %.2f s (%.2f %%)",
        delta["lap_time_delta"],
        delta["lap_time_delta_pct_of_point_mass"],
    )
    logger.info("Comparison artifacts written to %s", output_dir)


if __name__ == "__main__":
    main()
