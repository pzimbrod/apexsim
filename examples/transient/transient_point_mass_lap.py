"""Run a transient point-mass lap example from standing start."""

from __future__ import annotations

import argparse
import logging

from common import example_vehicle_parameters, export_transient_trace_csv, transient_output_root

from apexsim.analysis import compute_kpis, export_standard_plots
from apexsim.analysis.export import export_kpi_json
from apexsim.simulation import (
    PidSpeedSchedule,
    TransientConfig,
    TransientNumericsConfig,
    TransientPidGainSchedulingConfig,
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
        help="Compute backend used for transient solving.",
    )
    parser.add_argument(
        "--integration-method",
        choices=("euler", "rk4"),
        default="rk4",
        help="Transient integration method.",
    )
    parser.add_argument(
        "--driver-model",
        choices=("pid", "optimal_control"),
        default="pid",
        help="Driver/controller model inside transient solver.",
    )
    parser.add_argument(
        "--pid-scheduling-mode",
        choices=("off", "physics_informed", "custom"),
        default="off",
        help="Speed-dependent PID gain scheduling mode (PID driver only).",
    )
    parser.add_argument(
        "--custom-schedule-preset",
        choices=("balanced",),
        default="balanced",
        help="Preset used when --pid-scheduling-mode=custom.",
    )
    return parser.parse_args()


def _build_custom_pid_schedule_preset(
    *,
    max_speed: float,
    preset: str,
) -> TransientPidGainSchedulingConfig:
    """Return custom speed-dependent longitudinal PID schedules.

    Args:
        max_speed: Runtime speed cap [m/s].
        preset: Preset name.

    Returns:
        Valid custom scheduling config.
    """
    if preset != "balanced":
        raise ValueError(f"Unsupported custom schedule preset: {preset!r}")
    speed_nodes = (0.0, 10.0, 20.0, 35.0, 55.0, float(max_speed))
    return TransientPidGainSchedulingConfig(
        longitudinal_kp=PidSpeedSchedule(speed_nodes, (0.95, 0.90, 0.84, 0.77, 0.71, 0.68)),
        longitudinal_ki=PidSpeedSchedule(speed_nodes, (0.04, 0.03, 0.02, 0.015, 0.01, 0.008)),
        longitudinal_kd=PidSpeedSchedule(speed_nodes, (0.07, 0.065, 0.06, 0.055, 0.05, 0.045)),
    )


def main() -> None:
    """Run transient point-mass launch scenario and export artifacts."""
    args = _parse_args()
    configure_logging(logging.INFO)
    logger = logging.getLogger("transient_point_mass_example")

    track = build_straight_track(length=700.0, sample_count=161)
    max_speed = 80.0
    model = build_point_mass_model(
        vehicle=example_vehicle_parameters(),
        physics=PointMassPhysics(
            max_drive_accel=7.5,
            max_brake_accel=16.0,
            friction_coefficient=1.8,
        ),
    )
    custom_schedule = (
        _build_custom_pid_schedule_preset(
            max_speed=max_speed,
            preset=args.custom_schedule_preset,
        )
        if args.pid_scheduling_mode == "custom"
        else None
    )
    try:
        config = build_simulation_config(
            max_speed=max_speed,
            initial_speed=0.0,
            compute_backend=args.backend,
            solver_mode="transient_oc",
            transient=TransientConfig(
                numerics=TransientNumericsConfig(
                    integration_method=args.integration_method,
                    max_iterations=50,
                    pid_gain_scheduling_mode=args.pid_scheduling_mode,
                    pid_gain_scheduling=custom_schedule,
                ),
                runtime=TransientRuntimeConfig(
                    driver_model=args.driver_model,
                    verbosity=2,
                ),
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
    logger.info("Driver model: %s", args.driver_model)
    logger.info("PID scheduling mode: %s", args.pid_scheduling_mode)
    logger.info("Lap time: %.2f s", result.lap_time)
    logger.info("Max speed: %.2f m/s", float(result.speed.max()))
    logger.info("Energy use: %.2f kWh", kpis.energy)


if __name__ == "__main__":
    main()
