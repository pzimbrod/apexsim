"""Benchmark CPU/GPU backend options for point-mass lap simulation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from pylapsim.simulation import build_simulation_config, simulate_lap
from pylapsim.track.io import load_track_csv
from pylapsim.utils.constants import STANDARD_AIR_DENSITY
from pylapsim.vehicle import PointMassPhysics, VehicleParameters, build_point_mass_model

DEFAULT_WARMUP_RUNS = 5
DEFAULT_TIMED_RUNS = 20
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "output" / "backend_benchmarks.json"


@dataclass(frozen=True)
class BackendBenchmarkResult:
    """Timing summary for a single simulation backend configuration.

    Args:
        backend_id: Unique backend identifier string.
        first_call_ms: First lap call duration including startup/compile [ms].
        steady_mean_ms: Mean duration across timed steady-state runs [ms].
        steady_median_ms: Median duration across timed steady-state runs [ms].
        lap_time: Simulated lap time for consistency checks [s].
    """

    backend_id: str
    first_call_ms: float
    steady_mean_ms: float
    steady_median_ms: float
    lap_time: float


def _sample_vehicle_parameters() -> VehicleParameters:
    """Create a representative high-downforce vehicle parameter set.

    Returns:
        Vehicle parameter set used for backend benchmarking.
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


def _run_backend_benchmark(
    backend_id: str,
    warmup_runs: int,
    timed_runs: int,
) -> BackendBenchmarkResult:
    """Measure first-call and steady-state timing for one backend setup.

    Args:
        backend_id: Backend identifier used by ``build_simulation_config``.
        warmup_runs: Additional untimed warmup runs after first-call timing.
        timed_runs: Number of timed steady-state runs.

    Returns:
        Benchmark summary for the backend.

    Raises:
        ValueError: If ``backend_id`` is not supported by this benchmark script.
    """
    root = Path(__file__).resolve().parents[1]
    track = load_track_csv(root / "data" / "spa_francorchamps.csv")
    model = build_point_mass_model(
        vehicle=_sample_vehicle_parameters(),
        physics=PointMassPhysics(
            max_drive_accel=8.0,
            max_brake_accel=16.0,
            friction_coefficient=1.7,
        ),
    )

    if backend_id == "numpy":
        config = build_simulation_config(compute_backend="numpy")
    elif backend_id == "numba":
        config = build_simulation_config(compute_backend="numba")
    elif backend_id == "torch_cpu":
        config = build_simulation_config(compute_backend="torch", torch_device="cpu")
    elif backend_id == "torch_cpu_compile":
        config = build_simulation_config(
            compute_backend="torch",
            torch_device="cpu",
            torch_compile=True,
        )
    elif backend_id == "torch_cuda":
        config = build_simulation_config(compute_backend="torch", torch_device="cuda:0")
    elif backend_id == "torch_cuda_compile":
        config = build_simulation_config(
            compute_backend="torch",
            torch_device="cuda:0",
            torch_compile=True,
        )
    else:
        msg = f"Unsupported backend_id: {backend_id!r}"
        raise ValueError(msg)

    start = time.perf_counter()
    first_result = simulate_lap(track=track, model=model, config=config)
    first_call_ms = (time.perf_counter() - start) * 1_000.0

    for _ in range(warmup_runs):
        simulate_lap(track=track, model=model, config=config)

    timed_ms: list[float] = []
    lap_time = first_result.lap_time
    for _ in range(timed_runs):
        lap_start = time.perf_counter()
        lap_result = simulate_lap(track=track, model=model, config=config)
        timed_ms.append((time.perf_counter() - lap_start) * 1_000.0)
        lap_time = lap_result.lap_time

    return BackendBenchmarkResult(
        backend_id=backend_id,
        first_call_ms=float(first_call_ms),
        steady_mean_ms=float(statistics.mean(timed_ms)),
        steady_median_ms=float(statistics.median(timed_ms)),
        lap_time=float(lap_time),
    )


def benchmark_available_backends(
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    timed_runs: int = DEFAULT_TIMED_RUNS,
) -> list[BackendBenchmarkResult]:
    """Benchmark all available backends in the current environment.

    Args:
        warmup_runs: Number of untimed warmup runs per backend.
        timed_runs: Number of timed steady-state runs per backend.

    Returns:
        List of backend benchmark summaries.
    """
    results: list[BackendBenchmarkResult] = []
    results.append(_run_backend_benchmark("numpy", warmup_runs, timed_runs))

    if importlib.util.find_spec("numba") is not None:
        results.append(_run_backend_benchmark("numba", warmup_runs, timed_runs))

    if importlib.util.find_spec("torch") is not None:
        import torch

        results.append(_run_backend_benchmark("torch_cpu", warmup_runs, timed_runs))
        results.append(_run_backend_benchmark("torch_cpu_compile", warmup_runs, timed_runs))

        if torch.cuda.is_available():
            results.append(_run_backend_benchmark("torch_cuda", warmup_runs, timed_runs))
            results.append(_run_backend_benchmark("torch_cuda_compile", warmup_runs, timed_runs))

    return results


def _print_results_table(results: list[BackendBenchmarkResult]) -> None:
    """Print benchmark results as a markdown-style table.

    Args:
        results: Backend benchmark summaries.
    """
    print("| Backend | First Call [ms] | Steady Mean [ms] | Steady Median [ms] | Lap Time [s] |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for result in results:
        print(
            f"| {result.backend_id} | {result.first_call_ms:.2f} | "
            f"{result.steady_mean_ms:.2f} | {result.steady_median_ms:.2f} | "
            f"{result.lap_time:.6f} |"
        )


def _parse_args() -> argparse.Namespace:
    """Parse command-line options for backend benchmark execution.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=DEFAULT_WARMUP_RUNS,
        help="Untimed warmup runs per backend (default: 5).",
    )
    parser.add_argument(
        "--timed-runs",
        type=int,
        default=DEFAULT_TIMED_RUNS,
        help="Timed runs per backend for steady-state statistics (default: 20).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "JSON output path for benchmark data "
            "(default: examples/output/backend_benchmarks.json)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run backend benchmarks and export JSON output."""
    args = _parse_args()
    results = benchmark_available_backends(
        warmup_runs=args.warmup_runs,
        timed_runs=args.timed_runs,
    )
    _print_results_table(results)

    output_path = args.output
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps([asdict(result) for result in results], indent=2))
        print(f"\\nSaved benchmark data to: {output_path}")
    except OSError as exc:
        print(f"\\nWarning: could not write benchmark JSON to {output_path}: {exc}")


if __name__ == "__main__":
    main()
