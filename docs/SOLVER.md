# Solver Mathematics

This document explains how lap time is computed in the current quasi-steady solver
implementation (`src/lap_time_sim/simulation/profile.py`).

## 1. Discretization and State

The track is represented in arc-length domain by points
\(i = 0,\dots,N-1\) with:

- position \(s_i\) [m]
- curvature \(\kappa_i\) [1/m]
- grade \(\gamma_i = dz/ds\) [-]
- banking angle \(\beta_i\) [rad]

Segment length:
\[
\Delta s_i = s_{i+1} - s_i, \quad i=0,\dots,N-2.
\]

The solver computes a speed profile \(v_i\) [m/s], then derives:

- longitudinal acceleration \(a_{x,i}\) [m/s²]
- lateral acceleration \(a_{y,i}\) [m/s²]
- lap time \(T\) [s]

## 2. Lateral Envelope (Cornering Limit)

A lateral speed limit \(v_{\text{lat},i}\) is computed at each point. Core relation:
\[
|a_{y,i}| = v_i^2 |\kappa_i|.
\]

Given lateral acceleration capacity \(a_{y,\text{lim},i}\):
\[
v_{\text{lat},i} =
\begin{cases}
\sqrt{a_{y,\text{lim},i}/|\kappa_i|}, & |\kappa_i| > \varepsilon\\
v_{\max}, & |\kappa_i| \le \varepsilon
\end{cases}
\]
with clipping to \([v_{\min}, v_{\max}]\).

### 2.1 Lateral Acceleration Capacity

`lateral_accel_limit(...)` solves a fixed-point problem because tire force capacity depends
on normal load, and normal load depends on lateral acceleration through load transfer.

For fixed speed \(v\):

1. Estimate axle loads from quasi-static vertical balance using current \(a_y\) guess.
2. Compute front/rear lateral tire forces at a representative peak slip angle
   (Pacejka Magic Formula).
3. Update lateral limit:
\[
a_{y,\text{next}} = \max\left(a_{y,\min}, \frac{F_{y,f}+F_{y,r}}{m} + g\sin\beta\right).
\]
4. Repeat until \(|a_{y,\text{next}}-a_{y,\text{current}}|\le\text{tol}\) or max iterations.

## 3. Longitudinal Coupling via Friction Circle

Available longitudinal capability is reduced by lateral usage:
\[
\lambda_i = \sqrt{\max\left(0, 1 - \left(\frac{|a_{y,\text{req},i}|}{a_{y,\text{lim},i}}\right)^2\right)},
\quad
|a_{y,\text{req},i}| = v_i^2 |\kappa_i|.
\]

- Drive limit: \(a_{x,\text{drive},i} = a_{x,\text{drive,max}}\,\lambda_i\)
- Brake limit: \(a_{x,\text{brake},i} = a_{x,\text{brake,max}}\,\lambda_i\)

This is a simplified isotropic friction-circle approximation.

## 4. Forward Pass (Acceleration-Limited)

Starting from \(v_0\), propagate forward with kinematic relation:
\[
v_{i+1}^2 = v_i^2 + 2 a_{x,\text{net},i} \Delta s_i.
\]

Net acceleration model:
\[
a_{x,\text{net},i} =
a_{x,\text{drive},i}
- \frac{D(v_i)}{m}
- g\,\gamma_i,
\]
where drag force is
\[
D(v) = \tfrac{1}{2}\rho c_d A v^2.
\]

Then enforce bounds:
\[
v_{i+1} \leftarrow \min(v_{i+1}, v_{\text{lat},i+1}, v_{\max}),
\quad v_{i+1} \ge v_{\min}.
\]

## 5. Backward Pass (Braking-Limited)

From the end of the lap backwards, enforce braking feasibility:
\[
v_i^2 = v_{i+1}^2 + 2 a_{x,\text{decel,avail},i} \Delta s_i,
\]
with
\[
a_{x,\text{decel,avail},i} =
a_{x,\text{brake},i}
+ \frac{D(v_{i+1})}{m}
+ g\,\gamma_{i+1}.
\]

Again clamp by lateral and global speed bounds.

## 6. Final Accelerations and Lap Time

After forward/backward constraints, the final profile is \(v_i\). Longitudinal acceleration
is reconstructed by finite differences:
\[
a_{x,i} = \frac{v_{i+1}^2 - v_i^2}{2\Delta s_i}, \quad i=0,\dots,N-2.
\]

Lateral acceleration:
\[
a_{y,i} = v_i^2\kappa_i.
\]

Segment time is approximated with average segment speed:
\[
\Delta t_i = \frac{\Delta s_i}{\max\left(\tfrac{v_i+v_{i+1}}{2}, \varepsilon_v\right)}.
\]

Total lap time:
\[
T = \sum_{i=0}^{N-2} \Delta t_i.
\]

## 7. Numerical Convergence Controls

Lateral envelope convergence in `solve_speed_profile(...)` is configurable via
`SimulationConfig`:

- `lateral_envelope_max_iterations`
- `lateral_envelope_convergence_tol_mps`

Stop criterion:
\[
\max_i |v^{(k)}_{\text{lat},i} - v^{(k-1)}_{\text{lat},i}| \le \text{tol}_v.
\]

The actual number of iterations used is reported as
`SpeedProfileResult.lateral_envelope_iterations`.

## 8. Physical Scope and Limitations

Current solver is intentionally quasi-steady:

- No transient tire relaxation dynamics.
- No explicit driver/controller model in the speed-profile pass.
- Longitudinal limits are envelope-based constants (drive/brake maxima).
- Tire model uses fixed representative peak slip-angle for envelope estimation.

These simplifications keep the solver fast and stable, while preserving core constraints
for lap-time studies.
