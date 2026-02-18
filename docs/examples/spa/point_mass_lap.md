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

## Interpretation hints for students

1. Treat this as a study baseline, not a final high-fidelity answer.
2. Compare trend directions first (faster/slower, more/less aggressive),
   then compare magnitudes.
3. If you need yaw diagnostics, switch to single-track.

## Suggested exercise

Vary `friction_coefficient` in three steps and document:

1. effect on lap time,
2. effect on max lateral acceleration,
3. sectors where speed profile changes the most.
