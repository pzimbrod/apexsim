# Synthetic Track Walkthrough

This tutorial walks through `examples/synthetic_track_scenarios.py` in code order.

Goal: build intuition for solver behavior on controlled geometries before moving
to real circuits.

## Why this tutorial is important

Synthetic tracks are your first physical consistency filter.
If behavior is implausible here, Spa results are likely hard to trust.

## 1. Imports: map code to engineering blocks

The script imports:

- `analysis`: KPI and figure export,
- `simulation`: lap solver,
- `track`: synthetic layout generators,
- `vehicle`: point-mass model and its physics settings,
- `utils`: logging and constants.

If you are new to Python: imports are simply selecting the tools you need.

## 2. Define vehicle and model

### 2.1 Vehicle parameter block

`_example_vehicle_parameters()` provides a complete car definition.

Most influential fields for this tutorial:

- `mass`, `lift_coefficient`, `drag_coefficient`, `frontal_area`
  - govern acceleration/drag balance and high-speed behavior,
- `front_weight_fraction`
  - influences axle-load diagnostics,
- remaining vehicle parameters
  - keep cross-model compatibility.

### 2.2 Model choice: point-mass

```python
model = build_point_mass_model(vehicle=vehicle, physics=PointMassPhysics())
```

Interpretation:

- very fast and stable for benchmark scenarios,
- no explicit yaw state,
- yaw moment output is structurally zero by design.

If your study requires yaw-moment dynamics, switch to single-track model.

### 2.3 Solver setup

```python
config = build_simulation_config(max_speed=115.0)
```

`max_speed` is a runtime cap, not a guarantee that the car can sustain this speed.
Actual speed comes from force balance and constraints.

## 3. Build three benchmark tracks

```python
tracks = {
    "straight_1km": build_straight_track(length=STRAIGHT_LENGTH),
    "circle_r50": build_circular_track(radius=CIRCLE_RADIUS),
    "figure_eight": build_figure_eight_track(lobe_radius=FIGURE_EIGHT_RADIUS),
}
```

### 3.1 Straight (1 km)

Tests pure longitudinal behavior with no curvature demand.

### 3.2 Circle (50 m radius)

Tests quasi-steady cornering with near-constant curvature.

### 3.3 Figure-eight

Tests left/right transition dynamics and sign changes in lateral acceleration.

## 4. Run simulation loop

Per scenario:

```python
result = simulate_lap(track=track, model=model, config=config)
kpis = compute_kpis(result)
export_standard_plots(result, scenario_dir)
export_kpi_json(kpis, scenario_dir / "kpis.json")
```

This is the canonical PyLapSim pattern and can be reused for custom studies.

## 5. Cross-scenario comparison output

After scenario runs, script generates:

- `speed_trace_comparison.png`
- `scenario_summary.json`

These files support quick A/B/C interpretation across geometries.

## 6. How to interpret each scenario

### Straight (`straight_1km`)

Expected:

- lateral acceleration near zero,
- curvature near zero,
- speed governed by net longitudinal balance.

Important nuance:

- speed can decrease if drag at current speed exceeds available drive acceleration.

### Circle (`circle_r50`)

Expected:

- interior speed close to steady value,
- interior longitudinal acceleration near zero,
- positive lateral acceleration around loop.

Practical tip:

- ignore seam-adjacent points for steady-state assessment.

### Figure-eight (`figure_eight`)

Expected:

- lateral acceleration changes sign,
- acceleration and braking phases near transitions,
- stronger speed variation than constant-radius cornering.

## 7. What this tutorial validates well

- solver stability,
- unit consistency and sign conventions,
- gross physical plausibility,
- basic model-behavior sanity.

## 8. What this tutorial does not validate alone

- absolute lap-time fidelity on a real circuit,
- transient steering/yaw control quality,
- powertrain energy strategy realism.

## 9. Common mistakes and fixes

1. Expecting nonzero yaw moment with point-mass model.
   - Use single-track model if yaw diagnostics are required.
2. Treating seam points as steady-state evidence.
   - trim boundary points before concluding.
3. Assuming `max_speed` is always reached.
   - check drag and available drive force first.

## 10. Suggested next step

After this tutorial, continue with [Spa Walkthrough](EXAMPLES_SPA.md).

## Run command

```bash
python examples/synthetic_track_scenarios.py
```
