# API Overview

## Track

- `lap_time_sim.track.load_track_csv(path) -> TrackData`
  - Input CSV columns: `x`, `y`, `elevation`, `banking`

## Vehicle and Tires

- `lap_time_sim.vehicle.default_vehicle_parameters() -> VehicleParameters`
- `lap_time_sim.tire.default_axle_tire_parameters() -> AxleTireParameters`

## Simulation

- `lap_time_sim.simulation.SimulationConfig`
- `lap_time_sim.simulation.simulate_lap(track, vehicle, tires, config=None) -> LapSimulationResult`

Relevant `SimulationConfig` knobs:
- `lateral_envelope_max_iterations`
- `lateral_envelope_convergence_tol_mps`

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
