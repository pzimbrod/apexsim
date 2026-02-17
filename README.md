# PyLapSim

[![CI](https://github.com/pzimbrod/pylapsim/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/pzimbrod/pylapsim/actions/workflows/ci.yml)

A modular lap time simulation library for race cars with interchangeable vehicle models.

## Features

- 3-DOF bicycle dynamics with aerodynamic effects
- Point-mass vehicle model with isotropic friction-circle coupling
- Pacejka-style lateral tire model with load sensitivity
- Track CSV parsing (`x, y, elevation, banking`) and geometry processing
- Quasi-steady lap simulation with forward/backward speed-profile optimization
- Vehicle-model API abstraction for plugging in multiple model complexities
- KPI and plot generation (lap time, g-levels, speed trace, yaw moment, G-G diagram)

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
python examples/spa_lap.py
python examples/spa_lap_point_mass.py
python examples/spa_model_comparison.py
```

## Example Scripts

- `examples/spa_lap.py`: Bicycle model end-to-end run with KPI/plot export.
- `examples/spa_lap_point_mass.py`: Point-mass model end-to-end run with KPI/plot export.
- `examples/spa_model_comparison.py`: Side-by-side bicycle vs calibrated point-mass comparison with KPI deltas and speed-trace overlay.

## Assumptions and Limitations

This initial version uses a quasi-steady speed profile and a simplified longitudinal force model.
See `docs/ASSUMPTIONS.md` for details.
For the mathematical derivation of the lap-time solver, see `docs/SOLVER.md`.
For point-mass model equations and assumptions, see `docs/POINT_MASS_MODEL.md`.

## Data Source

The Spa centerline dataset is based on the public TUMFTM racetrack database.
See `data/README.md` for provenance and conversion details.

## License

This project is licensed under the Apache License 2.0.
See `LICENSE` for the full text.
