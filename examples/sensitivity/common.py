"""Shared utilities for single-track sensitivity example studies."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from apexsim.analysis import (
    SensitivityNumerics,
    SensitivityRuntime,
    SensitivityStudyParameter,
    run_lap_sensitivity_study,
)
from apexsim.simulation import SimulationConfig, build_simulation_config
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
        arb_roll_stiffness_fraction=0.5,
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
    simulation_config: SimulationConfig | None = None,
    sensitivity_runtime: SensitivityRuntime | None = None,
    sensitivity_numerics: SensitivityNumerics | None = None,
    parameters: Sequence[SensitivityStudyParameter] = DEFAULT_STUDY_PARAMETERS,
    physics: SingleTrackPhysics | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run local lap sensitivity study for the single-track model.

    Args:
        track: Track definition used for lap solves.
        track_label: Human-readable track name for exported artifacts.
        output_dir: Destination directory for CSV and plot files.
        max_speed: Global runtime speed cap [m/s].
        simulation_config: Optional simulation config override. If omitted,
            uses torch quasi-static defaults.
        sensitivity_runtime: Optional sensitivity runtime override. If omitted,
            uses :func:`run_lap_sensitivity_study` defaults (autodiff-first).
        sensitivity_numerics: Optional sensitivity numerics override.
        parameters: Study parameters used in the local sensitivity analysis.
        physics: Optional single-track physics override.

    Returns:
        Tuple ``(long_table, pivot_table)`` where:
            - ``long_table`` contains one row per objective-parameter pair.
            - ``pivot_table`` contains a compact parameter-objective sensitivity map.
    """
    vehicle = example_vehicle_parameters()
    tires = default_axle_tire_parameters()
    resolved_physics = physics or SingleTrackPhysics(reference_mass=vehicle.mass)
    resolved_simulation_config = simulation_config or build_simulation_config(
        compute_backend="torch",
        torch_device="cpu",
        torch_compile=False,
        max_speed=max_speed,
    )

    model = build_single_track_model(
        vehicle=vehicle,
        tires=tires,
        physics=resolved_physics,
    )

    study_result = run_lap_sensitivity_study(
        track=track,
        model=model,
        simulation_config=resolved_simulation_config,
        parameters=list(parameters),
        label=track_label,
        runtime=sensitivity_runtime,
        numerics=sensitivity_numerics,
    )

    long_table = _add_absolute_delta_columns(study_result.to_dataframe()).sort_values(
        by=["objective", "parameter"],
        kind="stable",
    )
    pivot_table = study_result.to_pivot().sort_index(kind="stable")

    output_dir.mkdir(parents=True, exist_ok=True)
    long_table.to_csv(output_dir / "sensitivities_long.csv", index=False)
    pivot_table.to_csv(output_dir / "sensitivities_pivot.csv")

    _plot_sensitivity_bars(long_table=long_table, path=output_dir / "sensitivity_bars.png")

    return long_table, pivot_table


def build_solver_comparison_table(
    *,
    quasi_static_long: pd.DataFrame,
    transient_long: pd.DataFrame,
) -> pd.DataFrame:
    """Build per-parameter solver comparison table from two long-format studies.

    Args:
        quasi_static_long: Long-form sensitivity table for quasi-static solve.
        transient_long: Long-form sensitivity table for transient solve.

    Returns:
        Solver comparison table merged on objective and parameter.
    """
    quasi = _add_absolute_delta_columns(quasi_static_long).copy()
    transient = _add_absolute_delta_columns(transient_long).copy()

    quasi_view = quasi[
        [
            "objective",
            "parameter",
            "parameter_label",
            "objective_unit",
            "sensitivity_raw",
            "absolute_delta_plus",
        ]
    ].rename(
        columns={
            "sensitivity_raw": "sensitivity_raw_quasi_static",
            "absolute_delta_plus": "absolute_delta_plus_quasi_static",
        }
    )
    transient_view = transient[
        [
            "objective",
            "parameter",
            "sensitivity_raw",
            "absolute_delta_plus",
        ]
    ].rename(
        columns={
            "sensitivity_raw": "sensitivity_raw_transient_pid_ad",
            "absolute_delta_plus": "absolute_delta_plus_transient_pid_ad",
        }
    )

    comparison = quasi_view.merge(
        transient_view,
        on=["objective", "parameter"],
        how="inner",
    )
    comparison["sensitivity_raw_change"] = (
        comparison["sensitivity_raw_transient_pid_ad"]
        - comparison["sensitivity_raw_quasi_static"]
    )
    comparison["absolute_delta_plus_change"] = (
        comparison["absolute_delta_plus_transient_pid_ad"]
        - comparison["absolute_delta_plus_quasi_static"]
    )
    return comparison.sort_values(by=["objective", "parameter"], kind="stable")


def plot_yaw_inertia_solver_comparison(
    *,
    comparison_table: pd.DataFrame,
    path: Path,
) -> None:
    """Plot quasi-static vs transient +10% yaw-inertia KPI deltas.

    Args:
        comparison_table: Solver comparison table created by
            :func:`build_solver_comparison_table`.
        path: Output image file path.
    """
    subset = comparison_table[comparison_table["parameter"] == "yaw_inertia"]
    if subset.empty:
        msg = "comparison_table does not contain yaw_inertia rows"
        raise ValueError(msg)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    objectives = ("lap_time_s", "energy_kwh")
    for axis, objective in zip(axes, objectives, strict=True):
        row = subset[subset["objective"] == objective].iloc[0]
        quasi_value = float(row["absolute_delta_plus_quasi_static"])
        transient_value = float(row["absolute_delta_plus_transient_pid_ad"])
        ylabel = "Delta lap time [s]"
        title = "Lap-time delta for +10% yaw inertia"
        if objective == "energy_kwh":
            quasi_value *= 1000.0
            transient_value *= 1000.0
            ylabel = "Delta energy [Wh]"
            title = "Energy delta for +10% yaw inertia"

        axis.bar(
            [0, 1],
            [quasi_value, transient_value],
            color=["#1565c0", "#c62828"],
            tick_label=["Quasi-static AD", "Transient PID+AD"],
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25, axis="y")
        axis.tick_params(axis="x", rotation=15)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _add_absolute_delta_columns(long_table: pd.DataFrame) -> pd.DataFrame:
    """Return long table copy with absolute KPI delta columns added.

    Args:
        long_table: Long-format sensitivity output from study API.

    Returns:
        Copy of ``long_table`` with ``absolute_delta_plus`` and
        ``absolute_delta_minus`` columns.
    """
    output = long_table.copy()
    output["absolute_delta_plus"] = (
        output["predicted_objective_plus"] - output["objective_value"]
    )
    output["absolute_delta_minus"] = (
        output["predicted_objective_minus"] - output["objective_value"]
    )
    return output


def _plot_sensitivity_bars(long_table: pd.DataFrame, path: Path) -> None:
    """Create bar charts for absolute KPI deltas at a fixed parameter variation.

    Args:
        long_table: Long-form table returned by
            :func:`run_single_track_sensitivity_study`.
        path: Output image file path.
    """
    plus_variation_pct = float(long_table["variation_plus_pct"].iloc[0])
    plot_table = _add_absolute_delta_columns(long_table)[
        [
            "parameter_label",
            "objective",
            "absolute_delta_plus",
        ]
    ].copy()

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
