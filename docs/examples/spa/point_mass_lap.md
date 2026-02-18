# Spa: Point-Mass Lap

This page explains `examples/spa/spa_lap_point_mass.py`.

## Learning goal

Run a faster, lower-complexity baseline model and understand what is gained
and lost relative to single-track.

## What changes vs. single-track

The script keeps track and vehicle setup style identical, but swaps model backend:

```python
model = build_point_mass_model(
    vehicle=vehicle,
    physics=PointMassPhysics(
        max_drive_accel=8.0,
        max_brake_accel=16.0,
        friction_coefficient=1.7,
    ),
)
```

## Why this model is useful

- Fast parameter sweeps and sensitivity studies.
- Clean baseline for comparing model complexity.
- Lower numerical and conceptual overhead for first experiments.

## What this model cannot represent

- Yaw-state dynamics are not modeled.
- Yaw moment is zero by model structure.
- Tire behavior is represented as isotropic envelope simplification.

## Outputs to inspect first

1. `examples/output/spa/point_mass/kpis.json`
2. `examples/output/spa/point_mass/speed_trace.png`
3. `examples/output/spa/point_mass/gg_diagram.png`

## Theoretical foundation

The point-mass model uses an isotropic acceleration envelope with aero coupling.
Typical relations are:

Normal-acceleration budget:

$$
a_n(v) = g + \frac{F_{\mathrm{down}}(v)}{m}
$$

Lateral limit (flat road simplification):

$$
a_{y,\mathrm{lim}}(v) = \mu \, a_n(v)
$$

Friction-circle coupling:

$$
a_{x,\mathrm{avail}} =
a_{x,\max}\sqrt{1 - \left(\frac{a_{y,\mathrm{req}}}{a_{y,\mathrm{lim}}}\right)^2}
$$

This structure is intentionally compact and efficient for fast sweeps.

## Assumptions and limits

1. Yaw dynamics and axle-specific tire behavior are not represented.
2. Tire behavior is condensed into scalar friction/envelope parameters.
3. Model fidelity is lower, but computational throughput is high.

## Potential learnings from the data

1. Use this model for baseline trends and wide parameter scans.
2. Compare against single-track before drawing conclusions on yaw-sensitive effects.
3. Separate differences caused by calibration ($\mu$) from structural model differences.
