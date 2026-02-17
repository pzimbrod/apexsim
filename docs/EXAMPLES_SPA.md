# Spa Walkthrough

This tutorial explains the Spa example scripts in execution order and in practical
engineering language.

Use this after the synthetic-track tutorial.

## What this tutorial helps you do

- Run realistic full-lap simulations on Spa.
- Understand the difference between bicycle and point-mass outputs.
- Build a reproducible model-comparison workflow.
- Interpret outputs within the limits of a quasi-steady solver.

## 1. Shared workflow across all Spa scripts

Each script follows the same structure:

1. Load Spa track CSV.
2. Create vehicle/model inputs.
3. Configure runtime/numerics.
4. Run `simulate_lap(...)`.
5. Export plots and KPI JSON.

Only the vehicle-model backend differs.

## 2. `examples/spa_lap_bicycle.py` (bicycle baseline)

### 2.1 Why start here?

This is the highest-fidelity backend currently available in the library’s
standard pipeline.

### 2.2 Code flow

- Load track:

```python
track = load_track_csv(project_root / "data" / "spa_francorchamps.csv")
```

- Build inputs:

```python
vehicle = _example_vehicle_parameters()
tires = default_axle_tire_parameters()
model = build_bicycle_model(vehicle=vehicle, tires=tires, physics=BicyclePhysics())
```

- Configure and run:

```python
config = build_simulation_config()
result = simulate_lap(track=track, model=model, config=config)
```

- Export:

```python
kpis = compute_kpis(result)
export_standard_plots(result, output_dir)
export_kpi_json(kpis, output_dir / "kpis.json")
```

### 2.3 What this script captures well

- lateral-force-limited behavior with load sensitivity,
- drag/downforce interaction,
- axle-load diagnostics,
- yaw-moment diagnostics.

### 2.4 What it still does not capture fully

- full transient driver/steering control effects,
- advanced powertrain/energy strategy effects,
- full multi-body suspension compliance.

## 3. `examples/spa_lap_point_mass.py` (fast baseline)

### 3.1 Purpose

Use this for rapid studies, initial sweeps, and baseline checks.

### 3.2 Core code difference

Only model backend changes:

```python
model = build_point_mass_model(
    vehicle=vehicle,
    physics=PointMassPhysics(
        max_drive_accel=8.0,
        max_brake_accel=16.0,
        friction_coefficient=1.7,
    ),
)
```

### 3.3 Interpretation

- faster and simpler than bicycle,
- no yaw-state dynamics,
- yaw moment reported as zero by construction.

## 4. `examples/spa_model_comparison.py` (tradeoff study)

### 4.1 Why this script matters

It demonstrates how to quantify value added by model complexity, not just compare lap times.

### 4.2 Sequence

1. Run bicycle model.
2. Calibrate point-mass friction to bicycle lateral envelope.
3. Run calibrated point-mass model.
4. Export KPI deltas and speed overlay.

### 4.3 Key outputs to inspect

- `comparison_kpis.json`
- `speed_trace_comparison.png`
- per-model KPI JSON and plots under comparison folders.

## 5. Parameter tuning order for a new vehicle

Recommended order:

1. Vehicle mass and aero parameters.
2. Tire parameters.
3. Model physics limits (`BicyclePhysics` / `PointMassPhysics`).
4. Numerical solver settings (only if required for convergence robustness).

This minimizes risk of hiding physical mis-modeling behind numerical tuning.

## 6. How to read Spa results correctly

1. Lap time alone is not enough; inspect speed trace shape and acceleration envelopes.
2. Compare where models differ along track distance, not only global KPIs.
3. Treat yaw-moment plots only as meaningful for models that represent yaw dynamics.
4. Validate magnitude ranges against known class-level expectations.

## 7. Practical limitations to keep in mind

Even with bicycle model, this remains a quasi-steady solver workflow.
Use caution when drawing conclusions about:

- highly transient driver actions,
- fine control-system behavior,
- sub-second event details.

## 8. Suggested study workflow

1. Start from `spa_lap_bicycle.py`.
2. Run `spa_lap_point_mass.py`.
3. Run `spa_model_comparison.py`.
4. Calibrate/adjust parameters and rerun all three for consistency.

## Run commands

```bash
python examples/spa_lap_bicycle.py
python examples/spa_lap_point_mass.py
python examples/spa_model_comparison.py
```
