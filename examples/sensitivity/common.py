"""Shared utilities for single-track sensitivity example studies."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from apexsim.analysis import (
    SensitivityStudyParameter,
    build_sensitivity_study_model,
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

    study_model = build_sensitivity_study_model(
        model_factory=build_single_track_model,
        model_inputs={
            "vehicle": vehicle,
            "tires": tires,
            "physics": physics,
        },
        label=track_label,
    )

    study_result = run_lap_sensitivity_study(
        track=track,
        study_model=study_model,
        simulation_config=simulation_config,
        parameters=DEFAULT_STUDY_PARAMETERS,
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
    """Create a bar-chart view for lap-time and energy sensitivities.

    Args:
        long_table: Long-form table returned by
            :func:`run_single_track_sensitivity_study`.
        path: Output image file path.
    """
    plot_table = long_table[
        ["parameter_label", "objective", "sensitivity_pct_per_pct"]
    ].copy()
    pivot = plot_table.pivot(
        index="parameter_label",
        columns="objective",
        values="sensitivity_pct_per_pct",
    )

    labels = pivot.index.astype(str).tolist()
    x_positions = list(range(len(labels)))
    lap_values = pivot["lap_time_s"].astype(float).tolist()
    energy_values = pivot["energy_kwh"].astype(float).tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), constrained_layout=True)
    axes[0].bar(x_positions, lap_values, color="#1565c0")
    axes[0].set_title("Lap-time sensitivity")
    axes[0].set_ylabel("Relative output change / relative input change")
    axes[0].set_xticks(x_positions, labels, rotation=20, ha="right")
    axes[0].grid(alpha=0.25, axis="y")

    axes[1].bar(x_positions, energy_values, color="#2e7d32")
    axes[1].set_title("Energy sensitivity")
    axes[1].set_ylabel("Relative output change / relative input change")
    axes[1].set_xticks(x_positions, labels, rotation=20, ha="right")
    axes[1].grid(alpha=0.25, axis="y")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
