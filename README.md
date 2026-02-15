# Lap Time Sim

A modular lap time simulation library for race cars using a 3-DOF bicycle model.

## Features

- 3-DOF bicycle dynamics with aerodynamic effects
- Pacejka-style lateral tire model with load sensitivity
- Track CSV parsing (`x, y, elevation, banking`) and geometry processing
- Quasi-steady lap simulation with forward/backward speed-profile optimization
- KPI and plot generation (lap time, g-levels, speed trace, yaw moment, G-G diagram)

## Project Layout

- `src/lap_time_sim/vehicle`: Vehicle dynamics, aero, load transfer
- `src/lap_time_sim/tire`: Tire models
- `src/lap_time_sim/track`: Track parsing and geometry
- `src/lap_time_sim/simulation`: Integrators and lap simulation
- `src/lap_time_sim/analysis`: KPIs and visualizations
- `tests`: Unit, integration, and validation tests

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
python examples/spa_lap.py
```

## Assumptions and Limitations

This initial version uses a quasi-steady speed profile and a simplified longitudinal force model.
See `docs/ASSUMPTIONS.md` for details.

## Data Source

The Spa centerline dataset is based on the public TUMFTM racetrack database.
See `data/README.md` for provenance and conversion details.
