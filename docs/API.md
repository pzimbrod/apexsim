# API Overview

This page maps the public API to the engineering workflow.

If you are new to the package, start with [How to Use](HOW_TO_USE.md).

## 1. Track inputs

### Load measured/real track data

- `apexsim.track.load_track_csv(path) -> TrackData`

Required CSV columns:

- `x`
- `y`
- `elevation`
- `banking`

### Generate synthetic validation tracks

- `apexsim.track.build_straight_track(...) -> TrackData`
- `apexsim.track.build_circular_track(...) -> TrackData`
- `apexsim.track.build_figure_eight_track(...) -> TrackData`

## 2. Vehicle and tire models

### Shared physical vehicle parameters

- `apexsim.vehicle.VehicleParameters`

### Tire parameters

- `apexsim.tire.default_axle_tire_parameters() -> AxleTireParameters`

### Single-track backend

- `apexsim.vehicle.SingleTrackModel(vehicle, tires, physics, numerics)`
- `apexsim.vehicle.SingleTrackPhysics`
- `apexsim.vehicle.SingleTrackNumerics`
- `apexsim.vehicle.build_single_track_model(vehicle, tires, physics=None, numerics=None)`

Terminology note:

- `SingleTrack` corresponds to the "bicycle model" terminology commonly used in literature.

### Point-mass backend

- `apexsim.vehicle.PointMassModel(vehicle, physics)`
- `apexsim.vehicle.PointMassPhysics`
- `apexsim.vehicle.build_point_mass_model(vehicle, physics=None)`
- `apexsim.vehicle.calibrate_point_mass_friction_to_single_track(vehicle, tires, ...)`

## 3. Simulation setup and run

- `apexsim.simulation.RuntimeConfig`
- `apexsim.simulation.NumericsConfig`
- `apexsim.simulation.SimulationConfig`
- `apexsim.simulation.build_simulation_config(...)`
- `apexsim.simulation.simulate_lap(track, model, config) -> LapResult`
- `apexsim.simulation.solve_speed_profile_torch(track, model, config) -> TorchSpeedProfileResult`

Backend runtime controls:

- `RuntimeConfig.compute_backend`: `"numpy"`, `"numba"`, or `"torch"`
- `RuntimeConfig.torch_device`: keep `"cpu"` for `numpy`/`numba`; use `"cpu"` or `"cuda:0"` for `torch`
- `RuntimeConfig.torch_compile`: reserved flag, must currently remain `False` for simulation
- `RuntimeConfig.initial_speed`: optional start speed at first track sample [m/s]
  (supports `0.0` for standing starts)

Constraint for differentiable solver use:

- `solve_speed_profile_torch(...)` is the differentiable torch solver and requires
  `RuntimeConfig.torch_compile = False`

See [Compute Backends](BACKENDS.md) for selection guidance and benchmarks.

Vehicle-model solver contract:

- `validate()`
- `lateral_accel_limit(speed, banking)`
- `max_longitudinal_accel(speed, lateral_accel_required, grade, banking)`
- `max_longitudinal_decel(speed, lateral_accel_required, grade, banking)`
- `diagnostics(speed, longitudinal_accel, lateral_accel, curvature)`

## 4. Postprocessing

- `apexsim.analysis.compute_kpis(result) -> KpiSummary`
- `apexsim.analysis.export_standard_plots(result, output_dir)`
- `apexsim.analysis.export.export_kpi_json(kpis, path)`
- `apexsim.analysis.compute_performance_envelope(model, ...) -> PerformanceEnvelopeResult`
- `apexsim.analysis.PerformanceEnvelopePhysics`
- `apexsim.analysis.PerformanceEnvelopeNumerics`
- `apexsim.analysis.PerformanceEnvelopeRuntime`
- `apexsim.analysis.build_performance_envelope_config(...) -> PerformanceEnvelopeConfig`
- `apexsim.analysis.SensitivityParameter`
- `apexsim.analysis.SensitivityNumerics`
- `apexsim.analysis.SensitivityRuntime`
- `apexsim.analysis.SensitivityConfig`
- `apexsim.analysis.build_sensitivity_config(...) -> SensitivityConfig`
- `apexsim.analysis.compute_sensitivities(objective, parameters, ...) -> SensitivityResult`

`LapResult` provides:

- lap time
- speed / longitudinal acceleration / lateral acceleration traces
- yaw moment
- front/rear axle loads
- power and energy

`PerformanceEnvelopeResult` provides:

- speed support points
- lateral-acceleration limits per speed
- sampled lateral-acceleration grid
- max/min longitudinal acceleration grid
- optional `.to_dataframe()` conversion for tabular studies

`SensitivityResult` provides:

- baseline scalar objective value
- local sensitivities per parameter
- method metadata (`autodiff` or `finite_difference`)
- baseline parameter values and parameter kind (`physical` / `numerical`)

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
