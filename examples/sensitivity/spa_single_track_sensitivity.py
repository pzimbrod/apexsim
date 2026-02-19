"""Run a single-track parameter sensitivity study on Spa-Francorchamps."""

from __future__ import annotations

import logging

from common import (
    build_solver_comparison_table,
    example_vehicle_parameters,
    plot_yaw_inertia_solver_comparison,
    run_single_track_sensitivity_study,
    sensitivity_output_root,
    spa_track_path,
)

from apexsim.simulation import (
    TransientConfig,
    TransientNumericsConfig,
    TransientRuntimeConfig,
    build_simulation_config,
)
from apexsim.track import load_track_csv
from apexsim.utils import configure_logging
from apexsim.vehicle import SingleTrackPhysics

TRACK_LABEL = "Spa-Francorchamps"


def main() -> None:
    """Run and export quasi-static and transient Spa sensitivity artifacts."""
    configure_logging(logging.INFO)
    logger = logging.getLogger("spa_single_track_sensitivity")

    track = load_track_csv(spa_track_path())
    root_output_dir = sensitivity_output_root() / "spa_single_track"
    quasi_output_dir = root_output_dir / "quasi_static"
    transient_output_dir = root_output_dir / "transient_pid_ad"
    study_physics = SingleTrackPhysics(
        max_steer_angle=0.3,
        max_steer_rate=2.0,
        reference_mass=example_vehicle_parameters().mass,
    )

    quasi_simulation_config = build_simulation_config(
        compute_backend="torch",
        torch_device="cpu",
        torch_compile=False,
        max_speed=115.0,
    )
    transient_simulation_config = build_simulation_config(
        compute_backend="torch",
        torch_device="cpu",
        torch_compile=False,
        max_speed=115.0,
        initial_speed=12.0,
        solver_mode="transient_oc",
        transient=TransientConfig(
            numerics=TransientNumericsConfig(max_time_step=1.0),
            runtime=TransientRuntimeConfig(driver_model="pid", verbosity=0),
        ),
    )

    quasi_long_table, quasi_pivot_table = run_single_track_sensitivity_study(
        track=track,
        track_label=TRACK_LABEL,
        output_dir=quasi_output_dir,
        simulation_config=quasi_simulation_config,
        physics=study_physics,
    )
    transient_long_table, transient_pivot_table = run_single_track_sensitivity_study(
        track=track,
        track_label=f"{TRACK_LABEL} (transient PID + autodiff)",
        output_dir=transient_output_dir,
        simulation_config=transient_simulation_config,
        physics=study_physics,
    )
    comparison_table = build_solver_comparison_table(
        quasi_static_long=quasi_long_table,
        transient_long=transient_long_table,
    )
    comparison_table.to_csv(root_output_dir / "solver_comparison.csv", index=False)
    plot_yaw_inertia_solver_comparison(
        comparison_table=comparison_table,
        path=root_output_dir / "solver_comparison_yaw_inertia.png",
    )

    logger.info(
        "Wrote quasi-static long-form rows (%d) to %s",
        len(quasi_long_table),
        quasi_output_dir / "sensitivities_long.csv",
    )
    logger.info(
        "Wrote transient long-form rows (%d) to %s",
        len(transient_long_table),
        transient_output_dir / "sensitivities_long.csv",
    )
    logger.info(
        "Wrote quasi-static pivot rows (%d) to %s",
        len(quasi_pivot_table),
        quasi_output_dir / "sensitivities_pivot.csv",
    )
    logger.info(
        "Wrote transient pivot rows (%d) to %s",
        len(transient_pivot_table),
        transient_output_dir / "sensitivities_pivot.csv",
    )
    logger.info("Solver comparison: %s", root_output_dir / "solver_comparison.csv")
    logger.info(
        "Yaw inertia comparison plot: %s",
        root_output_dir / "solver_comparison_yaw_inertia.png",
    )


if __name__ == "__main__":
    main()
