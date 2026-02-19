"""Benchmark solver runtime matrix across backends, models, and solver modes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from apexsim.simulation import (
    TransientConfig,
    TransientNumericsConfig,
    TransientRuntimeConfig,
    build_simulation_config,
    simulate_lap,
)
from apexsim.tire import default_axle_tire_parameters
from apexsim.track import build_circular_track, build_straight_track
from apexsim.track.io import load_track_csv
from apexsim.utils.constants import STANDARD_AIR_DENSITY
from apexsim.vehicle import (
    PointMassPhysics,
    SingleTrackPhysics,
    VehicleParameters,
    build_point_mass_model,
    build_single_track_model,
)

DEFAULT_WARMUP_RUNS = 2
DEFAULT_TIMED_RUNS = 5
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "output"
    / "solver_benchmarks"
    / "solver_matrix.json"
)


@dataclass(frozen=True)
class SolverBenchmarkCaseResult:
    """Timing summary for one solver benchmark case."""

    case_id: str
    backend: str
    model: str
    solver_mode: str
    track: str
    steady_mean_ms: float
    steady_median_ms: float
    lap_time_s: float


def _sample_vehicle_parameters() -> VehicleParameters:
    """Return representative baseline vehicle parameters.

    Returns:
        Baseline vehicle parameter set used for benchmark scenarios.
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


def _build_models() -> dict[str, object]:
    """Create point-mass and single-track benchmark models.

    Returns:
        Mapping from model label to initialized vehicle model instance.
    """
    vehicle = _sample_vehicle_parameters()
    point_mass = build_point_mass_model(
        vehicle=vehicle,
        physics=PointMassPhysics(
            max_drive_accel=8.0,
            max_brake_accel=16.0,
            friction_coefficient=1.65,
        ),
    )
    single_track = build_single_track_model(
        vehicle=vehicle,
        tires=default_axle_tire_parameters(),
        physics=SingleTrackPhysics(
            max_drive_accel=8.0,
            max_brake_accel=16.0,
            peak_slip_angle=0.12,
            max_steer_angle=0.55,
            max_steer_rate=2.5,
        ),
    )
    return {
        "point_mass": point_mass,
        "single_track": single_track,
    }


def _build_tracks(*, include_spa: bool) -> dict[str, object]:
    """Create synthetic and optional Spa benchmark tracks.

    Args:
        include_spa: Whether to include Spa-Francorchamps smoke case.

    Returns:
        Mapping from track label to track object.
    """
    tracks: dict[str, object] = {
        "straight": build_straight_track(length=700.0, sample_count=161),
        "circle": build_circular_track(radius=100.0, sample_count=241),
    }
    if include_spa:
        root = Path(__file__).resolve().parents[1]
        tracks["spa"] = load_track_csv(root / "data" / "spa_francorchamps.csv")
    return tracks


def _available_backends() -> list[str]:
    """Return available compute backends in the current environment.

    Returns:
        Ordered list of backend names available in current environment.
    """
    backends = ["numpy"]
    if importlib.util.find_spec("numba") is not None:
        backends.append("numba")
    if importlib.util.find_spec("torch") is not None:
        backends.append("torch")
    return backends


def _build_case_config(
    *,
    backend: str,
    solver_mode: str,
) -> object:
    """Build a validated simulation config for one benchmark case.

    Args:
        backend: Compute backend name.
        solver_mode: Solver mode name.

    Returns:
        Validated simulation config for the requested case.
    """
    if solver_mode == "quasi_static":
        return build_simulation_config(
            compute_backend=backend,
            max_speed=36.0,
            initial_speed=0.0,
            solver_mode="quasi_static",
        )
    return build_simulation_config(
        compute_backend=backend,
        max_speed=36.0,
        initial_speed=0.0,
        solver_mode="transient_oc",
        transient=TransientConfig(
            numerics=TransientNumericsConfig(
                integration_method="rk4",
                max_iterations=60,
            ),
            runtime=TransientRuntimeConfig(
                driver_model="pid",
                verbosity=0,
            ),
        ),
    )


def _time_case(
    *,
    backend: str,
    model_name: str,
    model: object,
    solver_mode: str,
    track_name: str,
    track: object,
    warmup_runs: int,
    timed_runs: int,
) -> SolverBenchmarkCaseResult:
    """Measure one benchmark case with warmup and timed runs.

    Args:
        backend: Compute backend name.
        model_name: Model label for reporting.
        model: Vehicle model instance.
        solver_mode: Solver mode name.
        track_name: Track label for reporting.
        track: Track object.
        warmup_runs: Number of untimed warmup runs.
        timed_runs: Number of timed benchmark runs.

    Returns:
        Aggregated timing summary for one benchmark case.
    """
    config = _build_case_config(backend=backend, solver_mode=solver_mode)

    for _ in range(warmup_runs):
        simulate_lap(track=track, model=model, config=config)

    timed_ms: list[float] = []
    lap_time = 0.0
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        result = simulate_lap(track=track, model=model, config=config)
        timed_ms.append((time.perf_counter() - t0) * 1_000.0)
        lap_time = float(result.lap_time)

    case_id = f"{backend}|{model_name}|{solver_mode}|{track_name}"
    return SolverBenchmarkCaseResult(
        case_id=case_id,
        backend=backend,
        model=model_name,
        solver_mode=solver_mode,
        track=track_name,
        steady_mean_ms=float(statistics.mean(timed_ms)),
        steady_median_ms=float(statistics.median(timed_ms)),
        lap_time_s=lap_time,
    )


def run_solver_benchmark_matrix(
    *,
    warmup_runs: int,
    timed_runs: int,
    include_spa: bool,
) -> list[SolverBenchmarkCaseResult]:
    """Run the full benchmark matrix and return case summaries.

    Args:
        warmup_runs: Number of warmup runs per case.
        timed_runs: Number of timed runs per case.
        include_spa: Whether to include Spa smoke case.

    Returns:
        Benchmark summaries for all matrix cases.
    """
    results: list[SolverBenchmarkCaseResult] = []
    models = _build_models()
    tracks = _build_tracks(include_spa=include_spa)
    for backend in _available_backends():
        for model_name, model in models.items():
            for solver_mode in ("quasi_static", "transient_oc"):
                for track_name, track in tracks.items():
                    results.append(
                        _time_case(
                            backend=backend,
                            model_name=model_name,
                            model=model,
                            solver_mode=solver_mode,
                            track_name=track_name,
                            track=track,
                            warmup_runs=warmup_runs,
                            timed_runs=timed_runs,
                        )
                    )
    return results


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--timed-runs", type=int, default=DEFAULT_TIMED_RUNS)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON path for benchmark matrix results.",
    )
    parser.add_argument(
        "--skip-spa",
        action="store_true",
        help="Skip Spa smoke case and benchmark only synthetic tracks.",
    )
    return parser.parse_args()


def main() -> None:
    """Run solver benchmark matrix and export JSON summary."""
    args = _parse_args()
    results = run_solver_benchmark_matrix(
        warmup_runs=int(args.warmup_runs),
        timed_runs=int(args.timed_runs),
        include_spa=not bool(args.skip_spa),
    )
    payload = {
        "metadata": {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "warmup_runs": int(args.warmup_runs),
            "timed_runs": int(args.timed_runs),
            "include_spa": not bool(args.skip_spa),
        },
        "cases": [asdict(item) for item in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Saved solver benchmark matrix to: {args.output}")
    print(f"Cases: {len(results)}")


if __name__ == "__main__":
    main()
