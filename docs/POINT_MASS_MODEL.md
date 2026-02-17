# Point-Mass Model

This document defines the implemented point-mass backend in
`src/pylapsim/vehicle/point_mass_model.py`.

## 1. Scope

The point-mass model keeps the solver API contract but replaces detailed
chassis/yaw dynamics with a scalar acceleration envelope.

State assumptions at each track point:

- no resolved yaw dynamics in diagnostics ($M_z = 0$),
- isotropic tire friction-circle coupling,
- normal load from gravity plus aerodynamic downforce.

## 2. Tire Normal-Acceleration Budget

At speed $v$:

$$
a_n(v) = g + \frac{F_\text{down}(v)}{m},
$$

with

$$
F_\text{down}(v) = \frac{1}{2}\rho C_L A v^2.
$$

The model applies a lower bound $a_n(v)\ge\varepsilon$ for numerical robustness.

## 3. Lateral Limit

With isotropic friction coefficient $\mu$:

$$
a_{y,\text{tire}}(v) = \mu a_n(v).
$$

Including banking contribution:

$$
a_{y,\text{lim}}(v,\beta) = \max\left(a_{y,\text{tire}}(v) + g\sin\beta,\ \varepsilon\right).
$$

## 4. Friction-Circle Coupling

For required lateral acceleration magnitude $|a_{y,\text{req}}|$:

$$
\lambda = \sqrt{\max\left(0,\ 1 - \left(\frac{|a_{y,\text{req}}|}{a_{y,\text{lim}}}\right)^2\right)}.
$$

## 5. Longitudinal Limits

Tire-limited longitudinal acceleration magnitude:

$$
a_{x,\text{tire,lim}}(v) = \mu a_n(v).
$$

Drive envelope:

$$
a_{x,\text{drive}}(v) =
\min\left(a_{x,\text{drive,max}},\ a_{x,\text{tire,lim}}(v)\right)\lambda.
$$

Brake envelope:

$$
a_{x,\text{brake}}(v) =
\min\left(a_{x,\text{brake,max}},\ a_{x,\text{tire,lim}}(v)\right)\lambda.
$$

Net along-track acceleration:

$$
a_{x,\text{net}} = a_{x,\text{drive}} - \frac{D(v)}{m} - g\,\gamma,
$$

available deceleration magnitude:

$$
a_{x,\text{decel,avail}} = \max\left(a_{x,\text{brake}} + \frac{D(v)}{m} + g\,\gamma,\ 0\right),
$$

with

$$
D(v)=\frac{1}{2}\rho C_D A v^2.
$$

## 6. Diagnostics

The backend reports:

- yaw moment: $0$,
- axle loads from static split plus aero split:
  - $F_{z,f} = mg\phi_f + F_{\text{down},f}$,
  - $F_{z,r} = mg(1-\phi_f) + F_{\text{down},r}$,
- tractive power:
  $$
  P = \left(m a_x + D(v)\right)v.
  $$

## 7. Equation-to-Code Mapping

- normal-acceleration budget:
  `PointMassModel._normal_accel_limit(...)`
- lateral limit:
  `PointMassModel.lateral_accel_limit(...)`
- friction-circle scaling:
  `PointMassModel._friction_circle_scale(...)`
- longitudinal accel/decel limits:
  `PointMassModel.max_longitudinal_accel(...)`,
  `PointMassModel.max_longitudinal_decel(...)`
- diagnostics:
  `PointMassModel.diagnostics(...)`

## 8. Cross-Model Calibration

To align the point-mass model with the bicycle model's lateral envelope, the
library provides:

- `calibrate_point_mass_friction_to_bicycle(vehicle, tires, ...)`

This identifies an effective isotropic $\mu$ by least-squares fitting:

$$
\mu^\star = \arg\min_\mu \sum_i \left(\mu a_n(v_i) - a_{y,\text{lim,bicycle}}(v_i)\right)^2.
$$

The comparison example uses this calibration before running the point-mass model.

## 9. Example

- Point-mass standalone usage:
  `examples/spa_lap_point_mass.py`
- Side-by-side comparison against bicycle model:
  `examples/spa_model_comparison.py`
