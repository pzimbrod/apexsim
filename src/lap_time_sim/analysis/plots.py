"""Plot generation for simulation analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from lap_time_sim.simulation.runner import LapSimulationResult
from lap_time_sim.utils.constants import GRAVITY

matplotlib.use("Agg")


def _save_dual_format(fig: Figure, out_base: Path) -> None:
    """Write a figure to PNG and PDF with a shared base path.

    Args:
        fig: Figure object to persist.
        out_base: Output path without suffix.
    """
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")


def plot_speed_trace(result: LapSimulationResult, out_base: Path) -> None:
    """Plot speed over arc length.

    Args:
        result: Simulation result containing `speed` and track arc length `arc_length`.
        out_base: Output path without suffix.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(result.track.arc_length, result.speed, lw=2.0)
    ax.set_xlabel("s [m]")
    ax.set_ylabel("Speed [m/s]")
    ax.set_title("Speed Trace")
    ax.grid(True, alpha=0.3)
    _save_dual_format(fig, out_base)
    plt.close(fig)


def plot_yaw_moment_vs_lateral_acc(result: LapSimulationResult, out_base: Path) -> None:
    """Plot yaw moment against lateral acceleration.

    Args:
        result: Simulation result containing yaw moment and lateral acceleration.
        out_base: Output path without suffix.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(result.lateral_accel / GRAVITY, result.yaw_moment, s=9, alpha=0.7)
    ax.set_xlabel("Lateral accel [g]")
    ax.set_ylabel("Yaw moment [Nm]")
    ax.set_title("Yaw Moment vs Lateral Acceleration")
    ax.grid(True, alpha=0.3)
    _save_dual_format(fig, out_base)
    plt.close(fig)


def plot_gg_diagram(result: LapSimulationResult, out_base: Path) -> None:
    """Plot longitudinal versus lateral acceleration in g-units.

    Args:
        result: Simulation result containing `longitudinal_accel` and `lateral_accel`.
        out_base: Output path without suffix.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(result.longitudinal_accel / GRAVITY, result.lateral_accel / GRAVITY, s=9, alpha=0.7)
    ax.set_xlabel("Longitudinal accel [g]")
    ax.set_ylabel("Lateral accel [g]")
    ax.set_title("G-G Diagram")
    ax.grid(True, alpha=0.3)
    _save_dual_format(fig, out_base)
    plt.close(fig)


def plot_tire_load_distribution(result: LapSimulationResult, out_base: Path) -> None:
    """Plot front and rear axle normal load over track distance.

    Args:
        result: Simulation result containing axle load traces and track distance.
        out_base: Output path without suffix.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(result.track.arc_length, result.front_axle_load, label="Front axle")
    ax.plot(result.track.arc_length, result.rear_axle_load, label="Rear axle")
    ax.set_xlabel("s [m]")
    ax.set_ylabel("Normal load [N]")
    ax.set_title("Axle Load Distribution")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_dual_format(fig, out_base)
    plt.close(fig)


def plot_power_trace(result: LapSimulationResult, out_base: Path) -> None:
    """Plot tractive power over arc length.

    Args:
        result: Simulation result containing `power` and track distance.
        out_base: Output path without suffix.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(result.track.arc_length, result.power / 1000.0, lw=2.0)
    ax.set_xlabel("s [m]")
    ax.set_ylabel("Power [kW]")
    ax.set_title("Power Trace")
    ax.grid(True, alpha=0.3)
    _save_dual_format(fig, out_base)
    plt.close(fig)


def export_standard_plots(result: LapSimulationResult, output_dir: str | Path) -> None:
    """Export all standard analysis plots in PNG and PDF format.

    Args:
        result: Simulation result used as plotting input.
        output_dir: Destination directory for all generated plots.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_speed_trace(result, out_dir / "speed_trace")
    plot_yaw_moment_vs_lateral_acc(result, out_dir / "yaw_moment_vs_ay")
    plot_gg_diagram(result, out_dir / "gg_diagram")
    plot_tire_load_distribution(result, out_dir / "tire_load_distribution")
    plot_power_trace(result, out_dir / "power_trace")
