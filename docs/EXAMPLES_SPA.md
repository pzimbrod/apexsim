# Spa Walkthrough

This tutorial explains the Spa example scripts in execution order and in practical
engineering language.

Use this after the synthetic-track tutorial.

## What this tutorial helps you do

- Run realistic full-lap simulations on Spa.
- Understand the difference between single-track and point-mass outputs.
- Build a reproducible model-comparison workflow.
- Interpret outputs within the limits of a quasi-steady solver.

## 1. Shared workflow across Spa lap scripts

The lap-focused scripts (`spa_lap_single_track.py`, `spa_lap_point_mass.py`,
`spa_model_comparison.py`) follow the same structure:

1. Load Spa track CSV.
2. Create vehicle/model inputs.
3. Configure runtime/numerics.
4. Run `simulate_lap(...)`.
5. Export plots and KPI JSON.

The performance-envelope script builds on the same vehicle setup but performs
envelope sampling instead of lap solving.

## 2. `examples/spa/spa_lap_single_track.py` (single-track baseline)

### 2.1 Why start here?

This is the highest-fidelity backend currently available in the library’s
standard pipeline.

### 2.2 Code flow

- Load track:

```python
track = load_track_csv(spa_track_path())
```

- Build inputs:

```python
vehicle = example_vehicle_parameters()
tires = default_axle_tire_parameters()
model = build_single_track_model(vehicle=vehicle, tires=tires, physics=SingleTrackPhysics())
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

## 3. `examples/spa/spa_lap_point_mass.py` (fast baseline)

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

- faster and simpler than single-track,
- no yaw-state dynamics,
- yaw moment reported as zero by construction.

## 4. `examples/spa/spa_model_comparison.py` (tradeoff study)

### 4.1 Why this script matters

It demonstrates how to quantify value added by model complexity, not just compare lap times.

### 4.2 Sequence

1. Run single-track model.
2. Calibrate point-mass friction to single-track lateral envelope.
3. Run calibrated point-mass model.
4. Export KPI deltas and speed overlay.

### 4.3 Key outputs to inspect

- `comparison_kpis.json`
- `speed_trace_comparison.png`
- per-model KPI JSON and plots under comparison folders.

## 5. `examples/spa/spa_performance_envelope.py` (speed-dependent G-G families)

### 5.1 Why this script matters

It demonstrates the new Performance-Envelope API on realistic vehicle setups:

1. Build single-track model.
2. Calibrate a point-mass model against single-track lateral limits.
3. Compute speed-dependent G-G envelopes for both models.
4. Export array-based and optional tabular artifacts.

### 5.2 Key outputs to inspect

- `single_track_envelope.npz`
- `point_mass_envelope.npz`
- `envelope_family_comparison.png`
- optional CSV exports (if pandas is installed)
- `summary.json`

## 6. Parameter tuning order for a new vehicle

Recommended order:

1. Vehicle mass and aero parameters.
2. Tire parameters.
3. Model physics limits (`SingleTrackPhysics` / `PointMassPhysics`).
4. Numerical solver settings (only if required for convergence robustness).

This minimizes risk of hiding physical mis-modeling behind numerical tuning.

## 7. How to read Spa results correctly

1. Lap time alone is not enough; inspect speed trace shape and acceleration envelopes.
2. Compare where models differ along track distance, not only global KPIs.
3. Treat yaw-moment plots only as meaningful for models that represent yaw dynamics.
4. Validate magnitude ranges against known class-level expectations.

## 8. Practical limitations to keep in mind

Even with single-track model, this remains a quasi-steady solver workflow.
Use caution when drawing conclusions about:

- highly transient driver actions,
- fine control-system behavior,
- sub-second event details.

## 9. Suggested study workflow

1. Start from `spa/spa_lap_single_track.py`.
2. Run `spa/spa_lap_point_mass.py`.
3. Run `spa/spa_model_comparison.py`.
4. Run `spa/spa_performance_envelope.py`.
5. Calibrate/adjust parameters and rerun for consistency.

## Run commands

```bash
python examples/spa/spa_lap_single_track.py
python examples/spa/spa_lap_point_mass.py
python examples/spa/spa_model_comparison.py
python examples/spa/spa_performance_envelope.py
```
