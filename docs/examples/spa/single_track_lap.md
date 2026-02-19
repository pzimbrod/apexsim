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

## Theoretical foundation

The single-track model is a reduced planar vehicle model with front/rear axle
representation. In quasi-steady use, the key balances are:

Lateral acceleration balance:

\[
a_y = \frac{F_{y,f} + F_{y,r}}{m}
\]

Yaw moment balance signal:

\[
M_z = l_f F_{y,f} - l_r F_{y,r}
\]

Path-kinematics coupling:

\[
a_y = v^2 \kappa
\]

Tire forces are generated from the Pacejka-style lateral model with load
sensitivity. This makes axle-load distribution and aero effects directly
relevant for cornering limits.

## Assumptions and limits

1. Quasi-steady envelope solving does not capture full transient tire relaxation.
2. The model is 3-DOF planar and omits full multibody compliance.
3. Powertrain and control strategy are represented through simplified envelopes.

## Potential learnings from the data

1. Check lap-time and speed-trace shape together, not separately.
2. Validate that high $|a_y|$ regions align with curved track sectors.
3. Use yaw-moment traces as consistency diagnostics for lateral balance.
4. If output magnitudes are implausible, re-check physical inputs before changing numerics.
