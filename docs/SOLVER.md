# Solver Mathematics

This document explains the two solver modes available through
`simulate_lap(track, model, config)`:

- quasi-static speed-profile solver (`solver_mode="quasi_static"`)
- transient dynamic solver (`solver_mode="transient_oc"`)

Sections 1-10 cover the quasi-static formulation
(`src/apexsim/simulation/profile.py`). Section 11 summarizes the transient
formulations (`src/apexsim/simulation/transient_*.py`).

## 1. Discretization and State

The track is represented in arc-length domain by points
$i = 0,\dots,N-1$ with:

- position $(s_i)$ [m]
- curvature $\kappa_i$ [1/m]
- grade $\gamma_i = dz/ds$ [-]
- banking angle $\beta_i$ [rad]

Segment length:
$$
\Delta s_i = s_{i+1} - s_i, \quad i=0,\dots,N-2.
$$

The solver computes a speed profile $v_i$ [m/s], then derives:

- longitudinal acceleration $a_{x,i}$ [m/s²]
- lateral acceleration $a_{y,i}$ [m/s²]
- lap time $T$ [s]

## 2. Lateral Envelope (Cornering Limit)

A lateral speed limit $v_{\text{lat},i}$ is computed at each point. Core relation:
$$
|a_{y,i}| = v_i^2 |\kappa_i|.
$$

Given lateral acceleration capacity $a_{y,\text{lim},i}$:
$$
v_{\text{lat},i} =
\begin{cases}
\sqrt{a_{y,\text{lim},i}/|\kappa_i|}, & |\kappa_i| > \varepsilon \\
v_{\max}, & |\kappa_i| \le \varepsilon
\end{cases}
$$
with clipping to $[v_{\min}, v_{\max}]$.

### 2.1 Lateral Acceleration Capacity

`SingleTrackModel.lateral_accel_limit(...)` solves a fixed-point problem because tire force capacity depends
on normal load, and normal load depends on lateral acceleration through load transfer.

For fixed speed $v$:

1. Estimate axle loads from quasi-static vertical balance using current $a_y$ guess.
2. Compute front/rear lateral tire forces at a representative peak slip angle
   (Pacejka Magic Formula).
3. Update lateral limit:
$$
a_{y,\text{next}} = \max\left(a_{y,\min}, \frac{F_{y,f}+F_{y,r}}{m} + g\sin\beta\right).
$$
4. Repeat until $|a_{y,\text{next}}-a_{y,\text{current}}|\le\text{tol}$ or max iterations.

## 3. Longitudinal Coupling via Friction Circle

Available longitudinal capability is reduced by lateral usage:
$$
\lambda_i = \sqrt{\max\left(0, 1 - \left(\frac{|a_{y,\text{req},i}|}{a_{y,\text{lim},i}}\right)^2\right)},
\quad
|a_{y,\text{req},i}| = v_i^2 |\kappa_i|.
$$

- Drive limit: $a_{x,\text{drive},i} = a_{x,\text{drive,max}}\,\lambda_i$
- Brake limit: $a_{x,\text{brake},i} = a_{x,\text{brake,max}}\,\lambda_i$

This is a simplified isotropic friction-circle approximation.

## 4. Forward Pass (Acceleration-Limited)

Starting from $v_0$, propagate forward with kinematic relation:
$$
v_{i+1}^2 = v_i^2 + 2 a_{x,\text{net},i} \Delta s_i.
$$

Initial-condition rule:
$$
v_0 = \min\left(v_{\text{lat},0}, v_{\max}, v_{\text{init}}\right),
$$
where $v_{\text{init}}$ is:

- `RuntimeConfig.initial_speed`, if provided.
- otherwise the legacy fallback $v_{\max}$.

Net acceleration model:
$$
a_{x,\text{net},i} =
a_{x,\text{drive},i}
- \frac{D(v_i)}{m}
- g\,\gamma_i,
$$
where drag force is
$$
D(v) = \tfrac{1}{2}\rho c_d A v^2.
$$

Then enforce bounds:
$$
v_{i+1} \leftarrow \min(v_{i+1}, v_{\text{lat},i+1}, v_{\max}),
\quad v_{i+1} \ge v_{\min}.
$$

## 5. Backward Pass (Braking-Limited)

From the end of the lap backwards, enforce braking feasibility:
$$
v_i^2 = v_{i+1}^2 + 2 a_{x,\text{decel,avail},i} \Delta s_i,
$$
with
$$
a_{x,\text{decel,avail},i} =
a_{x,\text{brake},i}
+ \frac{D(v_{i+1})}{m}
+ g\,\gamma_{i+1}.
$$

Again clamp by lateral and global speed bounds.

## 6. Final Accelerations and Lap Time

After forward/backward constraints, the final profile is $v_i$. Longitudinal acceleration
is reconstructed by finite differences:
$$
a_{x,i} = \frac{v_{i+1}^2 - v_i^2}{2\Delta s_i}, \quad i=0,\dots,N-2.
$$

Lateral acceleration:
$$
a_{y,i} = v_i^2\kappa_i.
$$

Segment time is approximated with average segment speed:
$$
\Delta t_i = \frac{\Delta s_i}{\max\left(\tfrac{v_i+v_{i+1}}{2}, \varepsilon_v\right)}.
$$

Total lap time:
$$
T = \sum_{i=0}^{N-2} \Delta t_i.
$$

## 7. Numerical Convergence Controls

Lateral envelope convergence in `solve_speed_profile(...)` is configurable via
`SimulationConfig.numerics`:

- `lateral_envelope_max_iterations`
- `lateral_envelope_convergence_tolerance`

Stop criterion:
$$
\max_i |v^{(k)}_{\text{lat},i} - v^{(k-1)}_{\text{lat},i}| \le \text{tol}_v.
$$

The actual number of iterations used is reported as
`SpeedProfileResult.lateral_envelope_iterations`.

## 8. Physical Scope and Limitations

Quasi-static mode is intentionally envelope-based:

- No transient tire relaxation dynamics.
- No explicit driver/controller model in the speed-profile pass.
- Longitudinal limits are envelope-based constants (drive/brake maxima).
- Tire model uses fixed representative peak slip-angle for envelope estimation.

These simplifications keep the solver fast and stable, while preserving core constraints
for lap-time studies.

The same solver routine can be used with different backends implementing
`VehicleModel`, including the single-track and point-mass models.

## 9. Equation-to-Code Mapping

- $a_{y,i} = v_i^2\kappa_i$:
  - `src/apexsim/simulation/profile.py` (`solve_speed_profile`, `ay = ...`)
- $v_{\text{lat},i} = \sqrt{a_{y,\text{lim},i}/|\kappa_i|}$ (with clipping):
  - `src/apexsim/simulation/envelope.py` (`lateral_speed_limit`)
  - `src/apexsim/simulation/profile.py` (`solve_speed_profile`, `v_lat[idx] = ...`)
- Friction-circle scaling $\lambda_i$ (vehicle-model dependent):
  - `src/apexsim/vehicle/single_track_model.py` (`_friction_circle_scale`)
- Forward pass $v_{i+1}^2 = v_i^2 + 2a\Delta s$:
  - `src/apexsim/simulation/profile.py` (`solve_speed_profile`, `next_speed_sq = ...`)
- Backward pass braking feasibility:
  - `src/apexsim/simulation/profile.py` (`solve_speed_profile`, `entry_speed_sq = ...`)
- Lap-time accumulation $T = \sum \Delta t_i$:
  - `src/apexsim/simulation/profile.py` (`lap_time += _segment_dt(...)`)
- Segment time model $\Delta t_i = \Delta s_i / \bar v_i$:
  - `src/apexsim/simulation/profile.py` (`_segment_dt`)
- Lateral limit fixed-point update:
  - `src/apexsim/vehicle/single_track_model.py` (`lateral_accel_limit`)
- Lateral envelope fixed-point convergence in speed domain:
  - `src/apexsim/simulation/profile.py` (`for iteration_idx ...`, `max_delta_speed ...`)
- Vehicle-model API contract consumed by the solver:
  - `src/apexsim/simulation/model_api.py` (`VehicleModel`)
  - `src/apexsim/simulation/profile.py` (`solve_speed_profile`, calls on `model`)

## 10. Differentiable Torch Solver API

For gradient-based workflows (e.g. autodiff sensitivities), ApexSim exposes:

- `apexsim.simulation.solve_speed_profile_torch(track, model, config)`

This API returns tensor-valued outputs (`TorchSpeedProfileResult`) and keeps the
autograd graph intact for downstream `backward()` calls.

Current constraint:

- `RuntimeConfig.torch_compile` must be `False` for
  `solve_speed_profile_torch`.

## 11. Transient Solver Mathematics

Transient mode supports two driver/control strategies over fixed track
arc-length samples:

- `driver_model="pid"` (default): closed-loop PID driver tracking a quasi-static
  reference profile.
- `driver_model="optimal_control"`: minimum-time optimal-control formulation.

The OC objective is:
$$
\min_{u} \; T + w_{\text{lat}} J_{\text{lat}} + w_{\text{trk}} J_{\text{trk}} + w_{\text{sm}} J_{\text{sm}}.
$$

### 11.1 State and controls

- Point-mass transient state:
$$
x_i = [v_i]
$$
- Single-track transient state:
$$
x_i = [v_{x,i}, v_{y,i}, r_i]
$$
where $r$ is yaw rate.

Controls:

- Point-mass: longitudinal command $a_{x,\text{cmd},i}$.
- Single-track: $a_{x,\text{cmd},i}$ and steering command $\delta_i$.

Single-track control bounds are physical model inputs:

- `SingleTrackPhysics.max_steer_angle`
- `SingleTrackPhysics.max_steer_rate`

### 11.2 Dynamics propagation

Arc-length segments are converted to bounded time steps:
$$
\Delta t_i = \operatorname{clip}\left(\frac{\Delta s_i}{\max(|v_i|,\varepsilon_v)},\; \Delta t_{\min},\; \Delta t_{\max}\right).
$$

Point-mass update:
$$
v_{i+1} = \operatorname{clip}(v_i + a_{x,\text{net},i}\Delta t_i, 0, v_{\max}).
$$

Single-track update uses Euler or RK4 integration of:
$$
\dot{x} = f(x, u),
$$
with a 3-DOF single-track dynamic model.

### 11.3 Constraint penalties and objective terms

The transient objective combines:

- lap-time term $T$,
- lateral-feasibility penalty $J_{\text{lat}}$,
- single-track tracking penalty $J_{\text{trk}}$,
- control smoothness penalty $J_{\text{sm}}$.

Penalty weights come from `TransientNumericsConfig`:

- `lateral_constraint_weight`
- `tracking_weight`
- `control_smoothness_weight`

### 11.4 Backend strategies

- NumPy: SciPy optimizer with transient simulation core.
- Numba: numba backend dispatch with shared transient core semantics.
- Torch: differentiable transient graph plus torch optimizer and `torchdiffeq`.

All modes are accessed through the same public runner:

```python
result = simulate_lap(track=track, model=model, config=config)
```

Transient mode adds time/state/control traces in `LapResult`:

- `time`
- `vx`, `vy`, `yaw_rate`
- `steer_cmd`, `ax_cmd`

### 11.5 PID Gain Scheduling

The default transient driver is PID. ApexSim supports optional speed-dependent
gain scheduling with piecewise-linear (PWL) tables:
$$
k(v) = \operatorname{interp}(v;\, v_{\text{nodes}}, k_{\text{nodes}})
$$
with boundary clamping at the first/last node.

Scheduling modes in `TransientNumericsConfig`:

- `off`: scalar gains only (legacy-compatible behavior).
- `physics_informed`: deterministic schedule generated from vehicle physics.
- `custom`: user-provided `TransientPidGainSchedulingConfig`.

Physics-informed longitudinal scaling at node $v_j$:
$$
a_+(v_j) = a_{x,\max}(v_j, a_y=0, \theta=0, \beta=0), \quad
a_-(v_j) = a_{x,\min}(v_j, a_y=0, \theta=0, \beta=0),
$$
$$
a_{\text{eff}}(v_j) = \frac{a_+(v_j) + a_-(v_j)}{2}, \quad
s_a(v_j) = \frac{a_{\text{eff}}(v_j)}{a_{\text{eff}}(v_{\text{ref}})}.
$$
The scheduled gains are:
$$
k_{p,\text{long}}(v_j) = k_{p,0}\,\operatorname{clip}(s_a, s_{\min}, s_{\max}),
$$
$$
k_{i,\text{long}}(v_j) = k_{i,0}\,\operatorname{clip}(s_a, s_{\min}, s_{\max}),
$$
$$
k_{d,\text{long}}(v_j) = k_{d,0}\,\operatorname{clip}(\sqrt{s_a}, s_{d,\min}, s_{d,\max}).
$$

Single-track steering scaling uses speed normalization:
$$
s_v(v_j) = \frac{v_{\text{ref}}}{\max(v_j, v_{\min})}.
$$
This yields:
$$
k_{p,\delta}(v_j) = k_{p,0}\,\operatorname{clip}(s_v, s_{kp,\min}, s_{kp,\max}),
$$
$$
k_{i,\delta}(v_j) = k_{i,0}\,\operatorname{clip}(s_v, s_{ki,\min}, s_{ki,\max}),
$$
$$
k_{d,\delta}(v_j) = k_{d,0}\,\operatorname{clip}(\sqrt{s_v}, s_{kd,\min}, s_{kd,\max}),
$$
$$
k_{v_y}(v_j) = k_{v_y,0}\,\operatorname{clip}\!\left(
1 + c_{v_y}\frac{v_j}{v_{\text{ref}}},
s_{vy,\min},
s_{vy,\max}
\right).
$$

Default node set:
$$
v_{\text{nodes}} = (0,\,10,\,20,\,35,\,55,\,v_{\max})\ \text{m/s}
$$
(intermediate nodes above $v_{\max}$ are omitted).

Rationale:

- Longitudinal gains rise with available traction/braking authority.
- Steering gains decrease with speed because yaw response sensitivity grows.
- Lateral-velocity damping increases with speed to stabilize transient sideslip.
