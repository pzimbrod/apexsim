# API Overview

## Track

- `lap_time_sim.track.load_track_csv(path) -> TrackData`
  - Input CSV columns: `x`, `y`, `elevation`, `banking`

## Vehicle and Tires

- `lap_time_sim.tire.default_axle_tire_parameters() -> AxleTireParameters`
- `lap_time_sim.vehicle.VehicleParameters` (provide explicit vehicle data)
- `lap_time_sim.vehicle.BicycleModel(vehicle, tires, physics, numerics)`
- `lap_time_sim.vehicle.BicyclePhysics` (sensible defaults included)
- `lap_time_sim.vehicle.BicycleNumerics` (sensible numerical defaults included)
- `lap_time_sim.vehicle.BicycleDynamicsModel` (state-space 3-DOF backend)
- `lap_time_sim.vehicle.build_bicycle_model(vehicle, tires, physics=None, numerics=None)`

## Simulation

- `lap_time_sim.simulation.SimulationConfig`
- `lap_time_sim.simulation.RuntimeConfig`
- `lap_time_sim.simulation.NumericsConfig` (sensible numerical defaults included)
- `lap_time_sim.simulation.simulate_lap(track, model, config) -> LapSimulationResult`
- `lap_time_sim.simulation.build_simulation_config(max_speed=115.0, numerics=None, enable_transient_refinement=False)`

Relevant `SimulationConfig` knobs:
- `runtime.max_speed`
- `numerics.min_speed`
- `numerics.lateral_envelope_max_iterations`
- `numerics.lateral_envelope_convergence_tolerance`

Vehicle-model API required by the solver:
- `validate()`
- `lateral_accel_limit(speed_mps, banking_rad)`
- `max_longitudinal_accel(speed_mps, ay_required_mps2, grade, banking_rad)`
- `max_longitudinal_decel(speed_mps, ay_required_mps2, grade, banking_rad)`
- `diagnostics(speed_mps, ax_mps2, ay_mps2, curvature_1pm)`

Solver math and derivation:
- `docs/SOLVER.md`

`LapSimulationResult` contains:
- `lap_time_s`
- `speed_mps`, `ax_mps2`, `ay_mps2`
- `yaw_moment_nm`
- `front_axle_load_n`, `rear_axle_load_n`
- `power_w`, `energy_j`

## Analysis

- `lap_time_sim.analysis.compute_kpis(result) -> KpiSummary`
- `lap_time_sim.analysis.export_standard_plots(result, output_dir)`
- `lap_time_sim.analysis.export.export_kpi_json(kpis, path)`

`KpiSummary` includes mandatory KPIs:
- lap time
- average and max lateral acceleration
- average and max longitudinal acceleration
- energy usage
