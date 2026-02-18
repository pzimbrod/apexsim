# Spa: Single-Track Lap

This page explains `examples/spa/spa_lap_single_track.py`.

## Learning goal

Use the highest-fidelity quasi-steady vehicle model in the current library
and interpret its outputs correctly.

## What the script does

1. Loads Spa track data.
2. Builds vehicle + tire + `SingleTrackModel`.
3. Runs `simulate_lap(...)` with default runtime config.
4. Exports KPI JSON and standard plots.

## Key code path

```python
track = load_track_csv(spa_track_path())
vehicle = example_vehicle_parameters()
tires = default_axle_tire_parameters()
model = build_single_track_model(vehicle=vehicle, tires=tires, physics=SingleTrackPhysics())
config = build_simulation_config()
result = simulate_lap(track=track, model=model, config=config)
```

## Why start with single-track

- It captures lateral tire behavior with load sensitivity.
- It provides yaw-moment and axle-load diagnostics.
- It is a strong baseline before simplifying to point-mass.

## Outputs to inspect first

1. `examples/output/spa/single_track/kpis.json`
2. `examples/output/spa/single_track/speed_trace.png`
3. `examples/output/spa/single_track/gg_diagram.png`
4. `examples/output/spa/single_track/yaw_moment_vs_ay.png`

## Interpretation hints for students

1. A low lap time alone is not enough; inspect the speed profile shape.
2. Check whether high lateral acceleration regions align with curved sectors.
3. Use yaw-moment plots qualitatively, not as direct controller targets.
4. If magnitudes look unrealistic, re-check physics inputs before numerics.

## Common beginner mistakes

1. Over-tuning numerical parameters to fix bad physical parameters.
2. Comparing absolute values without checking unit consistency.
3. Interpreting seam points at lap closure as physical events.

## Suggested exercise

Change only `max_speed` in `build_simulation_config(...)` and evaluate:

1. lap-time change,
2. max lateral acceleration change,
3. whether speed trace shape changes mostly on straights or corners.
