# Assumptions and Limitations

- Base vehicle dynamics are 3-DOF (`vx`, `vy`, `yaw_rate`) with single-track abstraction.
- Point-mass backend is available as an alternative model and assumes zero yaw moment
  in diagnostics by construction.
- Tire model implements Pacejka-style lateral force only in this phase.
- Longitudinal force limits are represented by configurable accel/brake envelopes.
- Point-mass backend uses an isotropic friction-circle with speed-dependent normal
  load from aerodynamic downforce.
- Quasi-static mode solves a lateral speed envelope via fixed-point iteration with
  configurable tolerance and iteration cap in `SimulationConfig.numerics`.
- Transient mode solves a minimum-time optimal-control problem on the fixed centerline
  with bounded controls and dynamic-state propagation.
- In transient mode, `SingleTrackModel` steering limits are configured through
  `SingleTrackPhysics.max_steer_angle` and `SingleTrackPhysics.max_steer_rate`.
- Transient PID gain scheduling (when enabled) is speed-only in v1:
  no preview controller, no explicit curvature/load-state scheduling.
- `pid_gain_scheduling_mode="physics_informed"` uses deterministic heuristics
  based on flat-road longitudinal authority and clipped speed scaling.
- Aero model uses constant coefficients (`c_l`, `c_d`) and rigid ride height.
- The lap-time solver is decoupled from specific vehicle equations and only
  depends on the `VehicleModel` API contract.
- Track data expects closed-loop CSV with columns: `x`, `y`, `elevation`, `banking`.
- Current Spa import uses real centerline coordinates, while `elevation` and `banking` are
  set to `0.0` until higher-fidelity telemetry/map layers are integrated.

These simplifications are intentional to keep the architecture extensible for future modules:
full vehicle, powertrain, and energy management.
