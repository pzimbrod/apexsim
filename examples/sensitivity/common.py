"""Shared utilities for single-track sensitivity example studies."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from apexsim.analysis import (
    SensitivityStudyParameter,
    run_lap_sensitivity_study,
)
from apexsim.simulation import build_simulation_config
from apexsim.tire import default_axle_tire_parameters
from apexsim.track import TrackData
from apexsim.utils.constants import STANDARD_AIR_DENSITY
from apexsim.vehicle import SingleTrackPhysics, VehicleParameters, build_single_track_model

DEFAULT_STUDY_PARAMETERS = (
    SensitivityStudyParameter(
        name="mass",
        target="vehicle.mass",
        label="Vehicle mass",
    ),
    SensitivityStudyParameter(
        name="cg_height",
        target="vehicle.cg_height",
        label="Center of gravity height",
    ),
    SensitivityStudyParameter(
        name="yaw_inertia",
        target="vehicle.yaw_inertia",
        label="Yaw inertia",
    ),
    SensitivityStudyParameter(
        name="drag_coefficient",
        target="vehicle.drag_coefficient",
        label="Drag coefficient",
    ),
)


def sensitivity_output_root() -> Path:
    """Return canonical output root for sensitivity examples.

    Returns:
        Path to ``examples/output/sensitivity``.
    """
    return Path(__file__).resolve().parents[1] / "output" / "sensitivity"


def project_root() -> Path:
    """Return repository root path for scripts located in ``examples/sensitivity``.

    Returns:
        Project root directory path.
    """
    return Path(__file__).resolve().parents[2]


def spa_track_path() -> Path:
    """Return canonical Spa track CSV path.

    Returns:
        Absolute path to ``data/spa_francorchamps.csv``.
    """
    return project_root() / "data" / "spa_francorchamps.csv"


def example_vehicle_parameters() -> VehicleParameters:
    """Create explicit vehicle parameters used by sensitivity example scripts.

    Returns:
        Vehicle parameter set for sensitivity example runs.
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


def run_single_track_sensitivity_study(
    *,
    track: TrackData,
    track_label: str,
    output_dir: Path,
    max_speed: float = 115.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run local AD-first lap sensitivity study for the single-track model.

    Args:
        track: Track definition used for lap solves.
        track_label: Human-readable track name for exported artifacts.
        output_dir: Destination directory for CSV and plot files.
        max_speed: Global runtime speed cap [m/s].

    Returns:
        Tuple ``(long_table, pivot_table)`` where:
            - ``long_table`` contains one row per objective-parameter pair.
            - ``pivot_table`` contains a compact parameter-objective sensitivity map.
    """
    vehicle = example_vehicle_parameters()
    tires = default_axle_tire_parameters()
    physics = SingleTrackPhysics()
    simulation_config = build_simulation_config(
        compute_backend="torch",
        torch_device="cpu",
        torch_compile=False,
        max_speed=max_speed,
    )

    model = build_single_track_model(
        vehicle=vehicle,
        tires=tires,
        physics=physics,
    )

    study_result = run_lap_sensitivity_study(
        track=track,
        model=model,
        simulation_config=simulation_config,
        parameters=DEFAULT_STUDY_PARAMETERS,
        label=track_label,
    )

    long_table = study_result.to_dataframe().sort_values(
        by=["objective", "parameter"],
        kind="stable",
    )
    pivot_table = study_result.to_pivot().sort_index(kind="stable")

    output_dir.mkdir(parents=True, exist_ok=True)
    long_table.to_csv(output_dir / "sensitivities_long.csv", index=False)
    pivot_table.to_csv(output_dir / "sensitivities_pivot.csv")

    _plot_sensitivity_bars(long_table=long_table, path=output_dir / "sensitivity_bars.png")

    return long_table, pivot_table


def _plot_sensitivity_bars(long_table: pd.DataFrame, path: Path) -> None:
    """Create bar charts for absolute KPI deltas at a fixed parameter variation.

    Args:
        long_table: Long-form table returned by
            :func:`run_single_track_sensitivity_study`.
        path: Output image file path.
    """
    plus_variation_pct = float(long_table["variation_plus_pct"].iloc[0])
    plot_table = long_table[
        [
            "parameter_label",
            "objective",
            "objective_value",
            "predicted_objective_plus",
        ]
    ].copy()
    plot_table["absolute_delta_plus"] = (
        plot_table["predicted_objective_plus"] - plot_table["objective_value"]
    )

    pivot_delta = plot_table.pivot(
        index="parameter_label",
        columns="objective",
        values="absolute_delta_plus",
    )

    labels = pivot_delta.index.astype(str).tolist()
    x_positions = list(range(len(labels)))
    lap_values = pivot_delta["lap_time_s"].astype(float).tolist()
    energy_values_wh = (pivot_delta["energy_kwh"].astype(float) * 1000.0).tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), constrained_layout=True)
    axes[0].bar(x_positions, lap_values, color="#1565c0")
    axes[0].set_title(f"Lap-time delta for +{plus_variation_pct:.0f}% parameter variation")
    axes[0].set_ylabel("Delta lap time [s]")
    axes[0].set_xticks(x_positions, labels, rotation=20, ha="right")
    axes[0].grid(alpha=0.25, axis="y")

    axes[1].bar(x_positions, energy_values_wh, color="#2e7d32")
    axes[1].set_title(f"Energy delta for +{plus_variation_pct:.0f}% parameter variation")
    axes[1].set_ylabel("Delta energy [Wh]")
    axes[1].set_xticks(x_positions, labels, rotation=20, ha="right")
    axes[1].grid(alpha=0.25, axis="y")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
