# Spa: Performance Envelope

This page explains `examples/spa/spa_performance_envelope.py`.

## Learning goal

Generate and interpret velocity-dependent G-G envelopes for both model families.

## What the script does

1. Builds single-track and calibrated point-mass models.
2. Computes `compute_performance_envelope(...)` for both.
3. Exports envelope arrays (`.npz`) and summary JSON.
4. Exports a comparison plot of envelope slices across speeds.
5. Optionally exports CSV when pandas is installed.

## Why this is useful

- Separates vehicle capability analysis from one specific lap profile.
- Supports parameter studies and sensitivity sweeps.
- Makes speed dependence explicit in longitudinal/lateral coupling.

## Main artifacts

1. `examples/output/spa/performance_envelope/single_track_envelope.npz`
2. `examples/output/spa/performance_envelope/point_mass_envelope.npz`
3. `examples/output/spa/performance_envelope/envelope_family_comparison.png`
4. `examples/output/spa/performance_envelope/summary.json`
5. optional CSV files (if pandas is installed)

## Interpreting the outputs (student view)

1. At each speed, read the feasible `(a_y, a_x)` boundary.
2. Compare max drive acceleration and max braking magnitude separately.
3. Identify speeds where model disagreement becomes largest.
4. Use these speeds as targeted cases for deeper model diagnostics.

## Common pitfalls

1. Comparing envelopes at different speed grids.
2. Mixing up signed `a_x` convention (braking is negative in exported minimum trace).
3. Assuming envelope superiority at one speed implies lap-time superiority everywhere.

## Suggested exercise

Increase the envelope speed range and report:

1. where drag starts dominating available forward acceleration,
2. where single-track and point-mass limits diverge most,
3. which result is most relevant for your intended study objective.
