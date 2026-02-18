# API Overview

This page maps the public API to the engineering workflow.

If you are new to the package, start with [How to Use](HOW_TO_USE.md).

## 1. Track inputs

### Load measured/real track data

- `pylapsim.track.load_track_csv(path) -> TrackData`

Required CSV columns:

- `x`
- `y`
- `elevation`
- `banking`

### Generate synthetic validation tracks

- `pylapsim.track.build_straight_track(...) -> TrackData`
- `pylapsim.track.build_circular_track(...) -> TrackData`
- `pylapsim.track.build_figure_eight_track(...) -> TrackData`

## 2. Vehicle and tire models

### Shared physical vehicle parameters

- `pylapsim.vehicle.VehicleParameters`

### Tire parameters

- `pylapsim.tire.default_axle_tire_parameters() -> AxleTireParameters`

### Single-track backend

- `pylapsim.vehicle.SingleTrackModel(vehicle, tires, physics, numerics)`
- `pylapsim.vehicle.SingleTrackPhysics`
- `pylapsim.vehicle.SingleTrackNumerics`
- `pylapsim.vehicle.build_single_track_model(vehicle, tires, physics=None, numerics=None)`

Terminology note:

- `SingleTrack` corresponds to the "bicycle model" terminology commonly used in literature.

### Point-mass backend

- `pylapsim.vehicle.PointMassModel(vehicle, physics)`
- `pylapsim.vehicle.PointMassPhysics`
- `pylapsim.vehicle.build_point_mass_model(vehicle, physics=None)`
- `pylapsim.vehicle.calibrate_point_mass_friction_to_single_track(vehicle, tires, ...)`

## 3. Simulation setup and run

- `pylapsim.simulation.RuntimeConfig`
- `pylapsim.simulation.NumericsConfig`
- `pylapsim.simulation.SimulationConfig`
- `pylapsim.simulation.build_simulation_config(...)`
- `pylapsim.simulation.simulate_lap(track, model, config) -> LapResult`

Backend runtime controls:

- `RuntimeConfig.compute_backend`: `"numpy"`, `"numba"`, or `"torch"`
- `RuntimeConfig.torch_device`: keep `"cpu"` for `numpy`/`numba`; use `"cpu"` or `"cuda:0"` for `torch`
- `RuntimeConfig.torch_compile`: optional compile acceleration for `torch` only

See [Compute Backends](BACKENDS.md) for selection guidance and benchmarks.

Vehicle-model solver contract:

- `validate()`
- `lateral_accel_limit(speed, banking)`
- `max_longitudinal_accel(speed, lateral_accel_required, grade, banking)`
- `max_longitudinal_decel(speed, lateral_accel_required, grade, banking)`
- `diagnostics(speed, longitudinal_accel, lateral_accel, curvature)`

## 4. Postprocessing

- `pylapsim.analysis.compute_kpis(result) -> KpiSummary`
- `pylapsim.analysis.export_standard_plots(result, output_dir)`
- `pylapsim.analysis.export.export_kpi_json(kpis, path)`

`LapResult` provides:

- lap time
- speed / longitudinal acceleration / lateral acceleration traces
- yaw moment
- front/rear axle loads
- power and energy

## 5. Minimal usage pattern

```python
track = load_track_csv("data/spa_francorchamps.csv")
model = build_single_track_model(vehicle=vehicle, tires=tires, physics=SingleTrackPhysics())
config = build_simulation_config(max_speed=115.0)
result = simulate_lap(track=track, model=model, config=config)
kpis = compute_kpis(result)
```

## 6. Related guides

- [How to Use](HOW_TO_USE.md)
- [Examples Overview](EXAMPLES.md)
- [Solver](SOLVER.md)
- [Single-Track Model](SINGLE_TRACK_MODEL.md)
- [Point-Mass Model](POINT_MASS_MODEL.md)
