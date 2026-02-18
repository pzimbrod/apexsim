"""Run a transient point-mass lap example from standing start."""

from __future__ import annotations

import argparse
import logging

from common import example_vehicle_parameters, export_transient_trace_csv, transient_output_root

from apexsim.analysis import compute_kpis, export_standard_plots
from apexsim.analysis.export import export_kpi_json
from apexsim.simulation import (
    TransientConfig,
    TransientNumericsConfig,
    TransientRuntimeConfig,
    build_simulation_config,
    simulate_lap,
)
from apexsim.track import build_straight_track
from apexsim.utils import configure_logging
from apexsim.utils.exceptions import ConfigurationError
from apexsim.vehicle import PointMassPhysics, build_point_mass_model


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for transient point-mass example.

    Returns:
        Parsed CLI namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("numpy", "numba", "torch"),
        default="numpy",
        help="Compute backend used for transient OC solving.",
    )
    parser.add_argument(
        "--integration-method",
        choices=("euler", "rk4"),
        default="rk4",
        help="Transient integration method.",
    )
    return parser.parse_args()


def main() -> None:
    """Run transient point-mass launch scenario and export artifacts."""
    args = _parse_args()
    configure_logging(logging.INFO)
    logger = logging.getLogger("transient_point_mass_example")

    track = build_straight_track(length=700.0, sample_count=161)
    model = build_point_mass_model(
        vehicle=example_vehicle_parameters(),
        physics=PointMassPhysics(
            max_drive_accel=7.5,
            max_brake_accel=16.0,
            friction_coefficient=1.8,
        ),
    )
    try:
        config = build_simulation_config(
            max_speed=80.0,
            initial_speed=0.0,
            compute_backend=args.backend,
            solver_mode="transient_oc",
            transient=TransientConfig(
                numerics=TransientNumericsConfig(
                    integration_method=args.integration_method,
                    max_iterations=50,
                ),
                runtime=TransientRuntimeConfig(verbosity=2)
            ),
        )
    except ConfigurationError as exc:
        logger.error("%s", exc)
        logger.error("Install or update dependencies with: pip install -e .")
        return

    result = simulate_lap(track=track, model=model, config=config)
    kpis = compute_kpis(result)

    output_dir = transient_output_root() / "point_mass_standing_start"
    export_standard_plots(result, output_dir)
    export_kpi_json(kpis, output_dir / "kpis.json")
    export_transient_trace_csv(
        arc_length=result.track.arc_length,
        time=result.time if result.time is not None else result.track.arc_length * 0.0,
        speed=result.speed,
        longitudinal_accel=result.longitudinal_accel,
        lateral_accel=result.lateral_accel,
        vx=result.vx if result.vx is not None else result.speed,
        vy=result.vy if result.vy is not None else result.speed * 0.0,
        yaw_rate=result.yaw_rate if result.yaw_rate is not None else result.speed * 0.0,
        steer_cmd=result.steer_cmd if result.steer_cmd is not None else result.speed * 0.0,
        ax_cmd=result.ax_cmd if result.ax_cmd is not None else result.longitudinal_accel,
        path=output_dir / "transient_trace.csv",
    )

    logger.info("Transient solver mode: %s", result.solver_mode)
    logger.info("Backend: %s", args.backend)
    logger.info("Lap time: %.2f s", result.lap_time)
    logger.info("Max speed: %.2f m/s", float(result.speed.max()))
    logger.info("Energy use: %.2f kWh", kpis.energy)


if __name__ == "__main__":
    main()
