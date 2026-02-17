# Spa Walkthrough

This page walks through the Spa example scripts in code order.

Use these scripts after validating your setup on synthetic tracks.

## 1. Common workflow skeleton

All Spa scripts follow the same core pattern:

1. Load track geometry from CSV.
2. Define vehicle and model parameters.
3. Configure solver numerics.
4. Run `simulate_lap(...)`.
5. Export KPIs and plots.

Only the vehicle model backend changes between scripts.

## 2. `examples/spa_lap.py` (Bicycle Model)

### 2.1 Imports

Main modules:

- `track.load_track_csv` for real track input,
- `vehicle.build_bicycle_model` + tire defaults,
- `simulation.build_simulation_config` + `simulate_lap`,
- `analysis` export functions.

### 2.2 Physical model setup

The script builds:

- `VehicleParameters` (car data),
- `AxleTireParameters` from `default_axle_tire_parameters()`,
- `BicyclePhysics()` for envelope and lateral approximation settings.

The bicycle backend adds axle-level lateral-force behavior and yaw diagnostics.

### 2.3 Track loading

```python
track = load_track_csv(project_root / "data" / "spa_francorchamps.csv")
```

Track preprocessing computes arc length, heading, curvature, and grade.

### 2.4 Numerics and solver

```python
config = build_simulation_config()
result = simulate_lap(track=track, model=model, config=config)
```

Defaults are chosen for stable convergence and can be refined later.

### 2.5 Postprocessing

```python
kpis = compute_kpis(result)
export_standard_plots(result, output_dir)
export_kpi_json(kpis, output_dir / "kpis.json")
```

This is the recommended baseline script for realistic lap studies.

## 3. `examples/spa_lap_point_mass.py` (Point-Mass Model)

This script uses the same pipeline but swaps the backend:

```python
model = build_point_mass_model(
    vehicle=vehicle,
    physics=PointMassPhysics(...),
)
```

Use cases:

- quick trade studies,
- parameter sweeps,
- baseline sanity checks before higher-fidelity runs.

## 4. `examples/spa_model_comparison.py` (Model Tradeoff)

This script demonstrates model-comparison workflow:

1. Run bicycle model.
2. Fit point-mass friction level to bicycle lateral envelope.
3. Run calibrated point-mass model.
4. Export KPI deltas and overlay plots.

This is useful for quantifying the value of additional model complexity.

## 5. What users should tune first

When adapting Spa scripts to a new car, tune in this order:

1. `VehicleParameters` (mass, aero, axle distribution).
2. Tire coefficients.
3. Model-physics limits (`BicyclePhysics` or `PointMassPhysics`).
4. Solver numerics only if convergence/robustness needs adjustment.

## 6. Common interpretation notes

1. Differences between bicycle and point-mass outputs are expected.
2. Yaw moment is meaningful only for models that represent yaw dynamics.
3. Lap-time changes from tiny parameter edits can be large on long tracks.

## Run commands

```bash
python examples/spa_lap.py
python examples/spa_lap_point_mass.py
python examples/spa_model_comparison.py
```
