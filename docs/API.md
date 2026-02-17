# API Overview

## Track

- `pylapsim.track.load_track_csv(path) -> TrackData`
  - Input CSV columns: `x`, `y`, `elevation`, `banking`

## Vehicle and Tires

- `pylapsim.tire.default_axle_tire_parameters() -> AxleTireParameters`
- `pylapsim.vehicle.VehicleParameters` (provide explicit vehicle data)
- `pylapsim.vehicle.BicycleModel(vehicle, tires, physics, numerics)`
- `pylapsim.vehicle.BicyclePhysics` (sensible defaults included)
- `pylapsim.vehicle.BicycleNumerics` (sensible numerical defaults included)
- `pylapsim.vehicle.BicycleDynamicsModel` (state-space 3-DOF backend)
- `pylapsim.vehicle.build_bicycle_model(vehicle, tires, physics=None, numerics=None)`
- `pylapsim.vehicle.PointMassModel(vehicle, physics)`
- `pylapsim.vehicle.PointMassPhysics` (sensible defaults included)
- `pylapsim.vehicle.build_point_mass_model(vehicle, physics=None)`
- `pylapsim.vehicle.calibrate_point_mass_friction_to_bicycle(vehicle, tires, ...)`

## Simulation

- `pylapsim.simulation.SimulationConfig`
- `pylapsim.simulation.RuntimeConfig`
- `pylapsim.simulation.NumericsConfig` (sensible numerical defaults included)
- `pylapsim.simulation.VehicleModelBase` (optional OOP base class)
- `pylapsim.simulation.simulate_lap(track, model, config) -> LapResult`
- `pylapsim.simulation.build_simulation_config(max_speed=115.0, numerics=None, enable_transient_refinement=False)`

Relevant `SimulationConfig` knobs:
- `runtime.max_speed`
- `numerics.min_speed`
- `numerics.lateral_envelope_max_iterations`
- `numerics.lateral_envelope_convergence_tolerance`

Vehicle-model API required by the solver:
- `validate()`
- `lateral_accel_limit(speed, banking)`
- `max_longitudinal_accel(speed, lateral_accel_required, grade, banking)`
- `max_longitudinal_decel(speed, lateral_accel_required, grade, banking)`
- `diagnostics(speed, longitudinal_accel, lateral_accel, curvature)`

Solver math and derivation:
- `docs/SOLVER.md`
- `docs/POINT_MASS_MODEL.md` (point-mass backend equations)

`LapResult` contains:
- `lap_time`
- `speed`, `longitudinal_accel`, `lateral_accel`
- `yaw_moment`
- `front_axle_load`, `rear_axle_load`
- `power`, `energy`

## Analysis

- `pylapsim.analysis.compute_kpis(result) -> KpiSummary`
- `pylapsim.analysis.export_standard_plots(result, output_dir)`
- `pylapsim.analysis.export.export_kpi_json(kpis, path)`

`KpiSummary` includes mandatory KPIs:
- lap time
- average and max lateral acceleration
- average and max longitudinal acceleration
- energy usage

## Examples

- `examples/spa_lap.py`: Bicycle model baseline example.
- `examples/spa_lap_point_mass.py`: Point-mass model baseline example.
- `examples/spa_model_comparison.py`: Bicycle vs calibrated point-mass comparison with KPI deltas.
