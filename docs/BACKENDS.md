# Compute Backends

PyLapSim separates physical modeling from numerical execution backends.
This page helps you choose the right backend for your study.

## Backend policy

Supported backends are intentionally restricted to:

- `numpy`: CPU-only reference backend.
- `numba`: CPU-only JIT-accelerated backend.
- `torch`: CPU and GPU backend (device via `torch_device`).

The runtime validator enforces this policy:

- `numpy` and `numba` require `torch_device="cpu"`.
- `torch_compile=True` is only valid with `compute_backend="torch"`.

## Current model support

Backend support is model-dependent:

- `PointMassModel`: supports `numpy`, `numba`, `torch`.
- `BicycleModel`: supports `numpy`, `numba`, `torch`.

If you request a backend that a model does not implement, the solver raises a
clear `ConfigurationError` describing the missing model-side methods.

## Decision guide

Use this practical rule-set:

1. Use `numpy` when you need robust baseline behavior and easiest debugging.
2. Use `numba` for large CPU parameter sweeps with `PointMassModel` or `BicycleModel`.
3. Use `torch` when you need tensor-native workflows, GPU execution, or future
   differentiable optimization pipelines.

### Trade-offs at a glance

| Backend | Hardware | Typical strength | Typical cost |
| --- | --- | --- | --- |
| `numpy` | CPU | Stable baseline, easy to inspect | Slower for huge batch studies |
| `numba` | CPU | Very fast steady-state loops after JIT warmup | First call includes compile overhead |
| `torch` | CPU/GPU | Backend portability, tensor ecosystem | Higher overhead for single-lap CPU workloads |

## Configuration examples

### NumPy (CPU reference)

```python
from pylapsim.simulation import build_simulation_config

config = build_simulation_config(
    compute_backend="numpy",
    max_speed=115.0,
)
```

### Numba (CPU-optimized)

```python
from pylapsim.simulation import build_simulation_config

config = build_simulation_config(
    compute_backend="numba",
    max_speed=115.0,
)
```

### Torch (CPU or GPU)

```python
from pylapsim.simulation import build_simulation_config

# CPU
config_cpu = build_simulation_config(
    compute_backend="torch",
    torch_device="cpu",
    torch_compile=False,
)

# GPU
config_gpu = build_simulation_config(
    compute_backend="torch",
    torch_device="cuda:0",
    torch_compile=False,
)
```

## Benchmark methodology

Reference script:

```bash
python examples/backend_benchmarks.py --warmup-runs 5 --timed-runs 20
```

Notes:

- Benchmarks run full Spa point-mass laps (`data/spa_francorchamps.csv`).
- "First Call" includes startup/JIT/compile effects.
- "Steady" values are from repeated post-warmup runs.
- Use your own machine data for final backend decisions.
- With some torch versions, `torch_compile=True` may print graph-break warnings
  for scalar convergence checks; results remain valid, but timing includes this behavior.

## Benchmark snapshot (February 17, 2026)

Environment used for this snapshot:

- CPU: Intel Core i7-8550U (4C/8T)
- `numpy==2.3.5`
- `numba==0.63.1`
- `torch==2.10.0+cpu`
- CUDA unavailable in this run

| Backend | First Call [ms] | Steady Mean [ms] | Steady Median [ms] | Lap Time [s] |
| --- | ---: | ---: | ---: | ---: |
| `numpy` | 19.72 | 14.72 | 14.73 | 133.668234 |
| `numba` | 1303.90 | 0.69 | 0.68 | 133.668234 |
| `torch` (`cpu`) | 323.99 | 342.27 | 337.44 | 133.668234 |
| `torch` (`cpu`, `compile=True`) | 5177.77 | 230.14 | 226.74 | 133.668234 |

Interpretation of this snapshot:

- `numba` steady-state is about 21x faster than `numpy` after JIT warmup.
- `torch` on CPU is slower than `numpy` for single-lap workflows in this setup.
- `torch_compile` improved torch steady-state by about 1.5x, but remained slower
  than `numpy` in this CPU-only environment.
- Identical lap times across backends confirm numerical consistency for this case.

## Reproducibility tips

- Run each benchmark at least twice and compare medians.
- Avoid other heavy processes during timing runs.
- For GPU evaluation, include both `torch_device="cuda:0"` and
  `torch_compile=True/False` variants.
- Save your benchmark JSON using `--output` and keep it with your project notes.
