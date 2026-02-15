# Assumptions and Limitations

- Base vehicle dynamics are 3-DOF (`vx`, `vy`, `yaw_rate`) with bicycle abstraction.
- Tire model implements Pacejka-style lateral force only in this phase.
- Longitudinal force limits are represented by configurable accel/brake envelopes.
- Lateral speed envelope is solved as a fixed-point iteration with configurable
  tolerance and iteration cap in `SimulationConfig`.
- Aero model uses constant coefficients (`c_l`, `c_d`) and rigid ride height.
- Track data expects closed-loop CSV with columns: `x`, `y`, `elevation`, `banking`.
- Current Spa import uses real centerline coordinates, while `elevation` and `banking` are
  set to `0.0` until higher-fidelity telemetry/map layers are integrated.

These simplifications are intentional to keep the architecture extensible for future modules:
full vehicle, powertrain, and energy management.
