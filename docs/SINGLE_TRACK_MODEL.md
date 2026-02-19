# Single-Track Model

This document defines the implemented single-track backend in
`src/apexsim/vehicle/single_track_model.py` and
`src/apexsim/vehicle/_single_track_physics.py`.

Terminology note:

- `SingleTrack` in this package corresponds to the "bicycle model" terminology
  frequently used in vehicle-dynamics literature.

The single-track backend is implemented as a physical extension of the
point-mass physical core (`PointMassPhysicalMixin`), with single-track-specific
lateral force and diagnostic equations layered on top.

## 1. Scope

The single-track model keeps the solver API contract and adds axle-level lateral
tire force modeling and quasi-static load transfer, while transient runs expose
dynamic yaw behavior through residual diagnostics.

State assumptions in the lap-time solver context:

- quasi-steady envelope evaluation (no transient tire relaxation in profile solve),
- lateral force from load-sensitive Pacejka at front and rear axle,
- friction-circle coupling between lateral demand and longitudinal capability,
- net along-track acceleration after drag and grade corrections.

## 2. Lateral Limit

At speed $v$ and banking angle $\beta$, the model solves lateral capacity by a
fixed-point iteration because normal load and lateral force depend on the
lateral-acceleration estimate itself.

Given current iterate $a_y^{(k)}$:

1. Estimate axle loads from quasi-static vertical balance and load transfer.
2. Evaluate axle lateral forces at representative peak slip angle
   $\alpha_\text{peak}$.
3. Update the lateral limit with banking contribution. The dependence on the
   previous iterate is carried by the axle normal loads used in tire-force
   evaluation:

\[
a_y^{(k+1)} = \max\left(
a_{y,\min},
\frac{2F_y\!\left(\alpha_\text{peak}, F_{z,f}\!\left(a_y^{(k)}\right)/2\right) + 2F_y\!\left(\alpha_\text{peak}, F_{z,r}\!\left(a_y^{(k)}\right)/2\right)}{m} + g\sin\beta
\right).
\]

Equivalently, writing axle-level lateral forces explicitly as
$F_{y,f}^{(k)}$ and $F_{y,r}^{(k)}$:

\[
a_y^{(k+1)} = \max\left(a_{y,\min}, \frac{F_{y,f}^{(k)} + F_{y,r}^{(k)}}{m} + g\sin\beta\right),
\]

with $F_{y,f}^{(k)}, F_{y,r}^{(k)}$ computed from $F_{z,f}(a_y^{(k)})$ and
$F_{z,r}(a_y^{(k)})$.

Stop criterion:

\[
\left|a_y^{(k+1)} - a_y^{(k)}\right| \le \text{tol}_y
\]

or max iteration count.

## 3. Lateral Tire Force Model (Pacejka)

Each axle force is the sum of two equivalent tires:

\[
F_{y,f} = 2\,F_y(\alpha_f, F_{z,f}/2), \quad
F_{y,r} = 2\,F_y(\alpha_r, F_{z,r}/2).
\]

Per-tire Pacejka-style lateral force:

\[
F_y = D\,\mu_\text{scale}(F_z)\,F_z\,\sin\left(C\,\arctan\left(\xi\right)\right),
\]

\[
\xi = B\alpha - E\left(B\alpha - \arctan(B\alpha)\right),
\]

with load sensitivity factor:

\[
\mu_\text{scale}(F_z) = \max\left(1 + s\,\frac{F_z - F_{z,\text{ref}}}{F_{z,\text{ref}}},\ \mu_{\min}\right).
\]

## 4. Quasi-Static Normal Loads

Total vertical load:

\[
F_{z,\text{tot}} = mg + F_\text{down}(v), \quad F_\text{down}(v)=\tfrac{1}{2}\rho C_L A v^2.
\]

Front axle raw load with longitudinal transfer:

\[
F_{z,f}^\text{raw} = mg\,\phi_f + F_{\text{down},f} - \frac{m a_x h}{L}.
\]

Rear axle load follows from equilibrium:

\[
F_{z,r} = F_{z,\text{tot}} - F_{z,f}.
\]

Lateral transfer is distributed by effective front roll-stiffness share and
split to left/right wheel loads while preserving axle totals.

## 5. Longitudinal Limits with Friction Circle

For required lateral acceleration magnitude $|a_{y,\text{req}}|$:

\[
\lambda = \sqrt{\max\left(0,\ 1 - \left(\frac{|a_{y,\text{req}}|}{a_{y,\text{lim}}}\right)^2\right)}.
\]

Drive and brake envelopes:

\[
a_{x,\text{drive}} = a_{x,\text{drive,max}}\,\lambda,\quad
a_{x,\text{brake}} = a_{x,\text{brake,max}}\,\lambda.
\]

Net along-track acceleration:

\[
a_{x,\text{net}} = a_{x,\text{drive}} - \frac{D(v)}{m} - g\gamma,
\]

available deceleration magnitude:

\[
a_{x,\text{decel,avail}} = \max\left(a_{x,\text{brake}} + \frac{D(v)}{m} + g\gamma,\ 0\right),
\]

with

\[
D(v)=\tfrac{1}{2}\rho C_D A v^2.
\]

## 6. Diagnostics

The backend reports at each operating point:

- yaw moment according to solver-mode semantics:
  - quasi-static mode: zero by steady-state model assumption,
  - transient mode: dynamic residual
\[
M_z = I_z \dot r,
\]
- front and rear axle normal loads,
- tractive power:

\[
P = \left(m a_x + D(v)\right)v.
\]

The solver uses quasi-steady envelopes for speed profile generation; the
3-DOF single-track dynamics model is used primarily for physically meaningful
analysis diagnostics (e.g., yaw-moment traces).

## 7. Equation-to-Code Mapping

- lateral limit fixed-point:
  `SingleTrackModel.lateral_accel_limit(...)`
- Pacejka lateral force:
  `magic_formula_lateral(...)`
- normal-load estimation:
  `estimate_normal_loads(...)`
- friction-circle scaling:
  `EnvelopeVehicleModel._friction_circle_scale(...)`
- longitudinal accel/decel:
  `SingleTrackModel.max_longitudinal_accel(...)`,
  `SingleTrackModel.max_longitudinal_decel(...)`
- diagnostics:
  `SingleTrackModel.diagnostics(...)`, `SingleTrackDynamicsModel.force_balance(...)`

## 8. Example

- Single-track standalone usage:
  `examples/spa/spa_lap_single_track.py`
- Side-by-side comparison against point-mass model:
  `examples/spa/spa_model_comparison.py`
