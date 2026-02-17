# Synthetic Track Walkthrough

This page walks through `examples/synthetic_track_scenarios.py` in the same order
as the code.

Purpose: isolate physics effects on simple geometry before using real tracks.

## 1. Imports and what they do

The script imports five functional blocks:

- `analysis`: KPI and plot export.
- `simulation`: solver setup and execution.
- `track`: synthetic layout builders.
- `vehicle`: point-mass model and its physics parameters.
- `utils/constants`: logging and air-density constant.

If you are new to Python, think of imports as selecting toolbox modules.

## 2. Define physical model inputs

### 2.1 Vehicle data

`_example_vehicle_parameters()` returns a fully defined `VehicleParameters` object.

What matters for this script:

- mass and aero terms affect acceleration and top-speed balance,
- axle split is used for diagnostic load outputs,
- remaining parameters keep interface compatibility with other models.

### 2.2 Vehicle model

The example uses:

```python
model = build_point_mass_model(vehicle=vehicle, physics=PointMassPhysics())
```

Interpretation:

- fast and robust for benchmark studies,
- no yaw-state dynamics,
- yaw moment diagnostic is therefore zero by design.

### 2.3 Solver numerics

```python
config = build_simulation_config(max_speed=115.0)
```

`max_speed` is a runtime bound, not a physical tire limit.
Physical limits still come from model + aero + friction.

## 3. Build the synthetic tracks

The script creates three layouts:

```python
tracks = {
    "straight_1km": build_straight_track(length=STRAIGHT_LENGTH),
    "circle_r50": build_circular_track(radius=CIRCLE_RADIUS),
    "figure_eight": build_figure_eight_track(lobe_radius=FIGURE_EIGHT_RADIUS),
}
```

Engineering interpretation:

- Straight: pure longitudinal behavior.
- Circle: steady cornering behavior.
- Figure-eight: turn-transition behavior with curvature sign change.

## 4. Run the simulation loop

Per scenario, the same three operations are executed:

```python
result = simulate_lap(track=track, model=model, config=config)
kpis = compute_kpis(result)
export_standard_plots(result, scenario_dir)
```

This pattern is the standard PyLapSim workflow and can be reused for custom tracks.

## 5. Export summary artifacts

The script writes:

- per-scenario plots and `kpis.json`,
- one cross-scenario speed overlay,
- one `scenario_summary.json` with KPI comparison.

## 6. What to check in results

### Straight (`straight_1km`)

Expected:

- lateral acceleration near zero,
- curvature near zero,
- speed evolves only from longitudinal force balance.

Note: speed may decrease if drag at chosen speed cap exceeds available drive acceleration.

### Circle (`circle_r50`)

Expected:

- near-constant interior speed,
- near-zero interior longitudinal acceleration,
- positive lateral acceleration around the loop.

### Figure-eight (`figure_eight`)

Expected:

- lateral acceleration changes sign,
- acceleration/deceleration phases around entry/exit,
- stronger speed variation than in constant-radius cornering.

## 7. Typical user pitfalls

1. Interpreting `max_speed` as guaranteed speed target.
2. Expecting nonzero yaw moment from the point-mass model.
3. Comparing edge samples on closed tracks without trimming seam points.

## Run command

```bash
python examples/synthetic_track_scenarios.py
```
