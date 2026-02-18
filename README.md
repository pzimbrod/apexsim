# PyLapSim

[![CI](https://github.com/pzimbrod/pylapsim/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/pzimbrod/pylapsim/actions/workflows/ci.yml)
[![Docs Dev](https://img.shields.io/badge/docs-dev-1f6feb)](https://pzimbrod.github.io/pylapsim/dev/)
[![Docs Stable](https://img.shields.io/badge/docs-stable-2ea043)](https://pzimbrod.github.io/pylapsim/stable/)

A modular lap time simulation library for race cars with interchangeable vehicle models.

## Features

- 3-DOF single-track dynamics with aerodynamic effects
- Point-mass vehicle model with isotropic friction-circle coupling
- Pacejka-style lateral tire model with load sensitivity
- Track CSV parsing (`x, y, elevation, banking`) and geometry processing
- Synthetic benchmark layouts (straight, circle, figure-eight) for model validation
- Quasi-steady lap simulation with forward/backward speed-profile optimization
- Compute backends: NumPy (CPU), Numba (CPU), and PyTorch (CPU/GPU)
- Vehicle-model API abstraction for plugging in multiple model complexities
- KPI and plot generation (lap time, g-levels, speed trace, yaw moment, G-G diagram)

Terminology note:

- `SingleTrack` in the API corresponds to the "bicycle model" naming often used in literature.

## Project Layout

- `src/pylapsim/vehicle`: Vehicle dynamics, aero, load transfer
- `src/pylapsim/tire`: Tire models
- `src/pylapsim/track`: Track parsing and geometry
- `src/pylapsim/simulation`: Integrators and lap simulation
- `src/pylapsim/analysis`: KPIs and visualizations
- `tests`: Unit, integration, and validation tests

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
python examples/spa/spa_lap_single_track.py
python examples/spa/spa_lap_point_mass.py
python examples/spa/spa_model_comparison.py
python examples/spa/spa_performance_envelope.py
python examples/synthetic_track_scenarios.py
```

## Example Scripts

- `examples/spa/spa_lap_single_track.py`: Single-track model end-to-end run with KPI/plot export.
- `examples/spa/spa_lap_point_mass.py`: Point-mass model end-to-end run with KPI/plot export.
- `examples/spa/spa_model_comparison.py`: Side-by-side single-track vs calibrated point-mass comparison with KPI deltas and speed-trace overlay.
- `examples/spa/spa_performance_envelope.py`: Velocity-dependent envelope generation and export (NumPy + optional CSV).
- `examples/synthetic_track_scenarios.py`: Straight, circle, and figure-eight benchmark runs for physical-consistency inspection.
- `examples/backend_benchmarks.py`: Quantitative timing comparison for NumPy, Numba, and Torch backends.

## Assumptions and Limitations

This initial version uses a quasi-steady speed profile and a simplified longitudinal force model.
See `docs/ASSUMPTIONS.md` for details.
For the mathematical derivation of the lap-time solver, see `docs/SOLVER.md`.
For point-mass model equations and assumptions, see `docs/POINT_MASS_MODEL.md`.
For backend-selection guidance and benchmark data, see `docs/BACKENDS.md`.

## Data Source

The Spa centerline dataset is based on the public TUMFTM racetrack database.
See `data/README.md` for provenance and conversion details.

## License

This project is licensed under the Apache License 2.0.
See `LICENSE` for the full text.
