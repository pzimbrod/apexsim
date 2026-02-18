"""Generate velocity-dependent performance-envelope artifacts for Spa studies."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from common import example_vehicle_parameters, spa_output_root

from apexsim.analysis import (
    PerformanceEnvelopeNumerics,
    PerformanceEnvelopePhysics,
    PerformanceEnvelopeResult,
    compute_performance_envelope,
)
from apexsim.tire import default_axle_tire_parameters
from apexsim.utils import configure_logging
from apexsim.utils.exceptions import ConfigurationError
from apexsim.vehicle import (
    PointMassPhysics,
    SingleTrackPhysics,
    build_point_mass_model,
    build_single_track_model,
    calibrate_point_mass_friction_to_single_track,
)

DEFAULT_ENVELOPE_SPEED_MIN = 20.0
DEFAULT_ENVELOPE_SPEED_MAX = 90.0
DEFAULT_ENVELOPE_SPEED_SAMPLES = 31
DEFAULT_ENVELOPE_LATERAL_SAMPLES = 51


def _export_envelope_npz(result: PerformanceEnvelopeResult, path: Path) -> None:
    """Export a performance-envelope result to compressed NumPy format.

    Args:
        result: Envelope result to serialize.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        speed=result.speed,
        lateral_accel_limit=result.lateral_accel_limit,
        lateral_accel_fraction=result.lateral_accel_fraction,
        lateral_accel=result.lateral_accel,
        max_longitudinal_accel=result.max_longitudinal_accel,
        min_longitudinal_accel=result.min_longitudinal_accel,
    )


def _try_export_envelope_csv(
    result: PerformanceEnvelopeResult,
    path: Path,
    logger: logging.Logger,
    label: str,
) -> None:
    """Export long-form CSV when pandas is available in the environment.

    Args:
        result: Envelope result to serialize.
        path: Output CSV path.
        logger: Logger used for status messages.
        label: Human-readable model label for logging.
    """
    try:
        frame = result.to_dataframe()
    except ConfigurationError:
        logger.info("pandas not installed; skipping CSV export for %s envelope", label)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _export_envelope_slice_plot(
    single_track_result: PerformanceEnvelopeResult,
    point_mass_result: PerformanceEnvelopeResult,
    path: Path,
) -> None:
    """Export a two-panel envelope plot over selected speed slices.

    Args:
        single_track_result: Envelope computed from single-track model.
        point_mass_result: Envelope computed from point-mass model.
        path: Output image path.
    """
    speed_index_samples = np.unique(
        np.linspace(0, single_track_result.speed.size - 1, 5, dtype=int)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True, sharey=True)
    panels = (
        ("SingleTrack envelope", single_track_result, axes[0]),
        ("Point-mass envelope", point_mass_result, axes[1]),
    )

    for title, result, axis in panels:
        for speed_index in speed_index_samples:
            speed_value = float(result.speed[speed_index])
            lateral = result.lateral_accel[speed_index]
            axis.plot(
                lateral,
                result.max_longitudinal_accel[speed_index],
                lw=1.5,
                label=f"{speed_value:.0f} m/s",
            )
            axis.plot(
                lateral,
                result.min_longitudinal_accel[speed_index],
                lw=1.5,
            )
        axis.set_title(title)
        axis.set_xlabel("Lateral acceleration [m/s^2]")
        axis.grid(alpha=0.3)

    axes[0].set_ylabel("Longitudinal acceleration [m/s^2]")
    axes[0].legend(title="Slice speed")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _summary_payload(
    single_track_result: PerformanceEnvelopeResult,
    point_mass_result: PerformanceEnvelopeResult,
) -> dict[str, dict[str, float]]:
    """Build compact summary statistics for envelope comparison output.

    Args:
        single_track_result: Envelope computed from single-track model.
        point_mass_result: Envelope computed from point-mass model.

    Returns:
        Nested summary dictionary with representative envelope magnitudes.
    """
    return {
        "single_track": {
            "max_lateral_accel_limit": float(np.max(single_track_result.lateral_accel_limit)),
            "max_longitudinal_accel": float(np.max(single_track_result.max_longitudinal_accel)),
            "max_brake_decel_magnitude": float(-np.min(single_track_result.min_longitudinal_accel)),
        },
        "point_mass": {
            "max_lateral_accel_limit": float(np.max(point_mass_result.lateral_accel_limit)),
            "max_longitudinal_accel": float(np.max(point_mass_result.max_longitudinal_accel)),
            "max_brake_decel_magnitude": float(-np.min(point_mass_result.min_longitudinal_accel)),
        },
    }


def main() -> None:
    """Run single-track and point-mass envelope generation for Spa studies."""
    configure_logging(logging.INFO)
    logger = logging.getLogger("spa_performance_envelope")

    output_dir = spa_output_root() / "performance_envelope"
    output_dir.mkdir(parents=True, exist_ok=True)

    vehicle = example_vehicle_parameters()
    tires = default_axle_tire_parameters()
    single_track_physics = SingleTrackPhysics()

    single_track_model = build_single_track_model(
        vehicle=vehicle,
        tires=tires,
        physics=single_track_physics,
    )
    calibration = calibrate_point_mass_friction_to_single_track(
        vehicle=vehicle,
        tires=tires,
        single_track_physics=single_track_physics,
    )
    point_mass_model = build_point_mass_model(
        vehicle=vehicle,
        physics=PointMassPhysics(
            max_drive_accel=single_track_physics.max_drive_accel,
            max_brake_accel=single_track_physics.max_brake_accel,
            friction_coefficient=calibration.friction_coefficient,
        ),
    )

    envelope_physics = PerformanceEnvelopePhysics(
        speed_min=DEFAULT_ENVELOPE_SPEED_MIN,
        speed_max=DEFAULT_ENVELOPE_SPEED_MAX,
    )
    envelope_numerics = PerformanceEnvelopeNumerics(
        speed_samples=DEFAULT_ENVELOPE_SPEED_SAMPLES,
        lateral_accel_samples=DEFAULT_ENVELOPE_LATERAL_SAMPLES,
    )

    single_track_result = compute_performance_envelope(
        model=single_track_model,
        physics=envelope_physics,
        numerics=envelope_numerics,
    )
    point_mass_result = compute_performance_envelope(
        model=point_mass_model,
        physics=envelope_physics,
        numerics=envelope_numerics,
    )

    _export_envelope_npz(single_track_result, output_dir / "single_track_envelope.npz")
    _export_envelope_npz(point_mass_result, output_dir / "point_mass_envelope.npz")
    _try_export_envelope_csv(
        single_track_result,
        output_dir / "single_track_envelope.csv",
        logger,
        "single_track",
    )
    _try_export_envelope_csv(
        point_mass_result,
        output_dir / "point_mass_envelope.csv",
        logger,
        "point_mass",
    )
    _export_envelope_slice_plot(
        single_track_result=single_track_result,
        point_mass_result=point_mass_result,
        path=output_dir / "envelope_family_comparison.png",
    )

    summary = _summary_payload(
        single_track_result=single_track_result,
        point_mass_result=point_mass_result,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Fitted point-mass friction coefficient: %.3f", calibration.friction_coefficient)
    logger.info("Performance-envelope artifacts written to %s", output_dir)


if __name__ == "__main__":
    main()
