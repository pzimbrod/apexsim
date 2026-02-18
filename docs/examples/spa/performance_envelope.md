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

## Theoretical definition

For each speed sample `v_j`, the script evaluates a feasible acceleration set:

`E(v_j) = {(a_y, a_x) | a_y in [-a_y,lim(v_j), +a_y,lim(v_j)], a_x in [a_x,min, a_x,max]}`

In the exported arrays, this becomes a discretized family of G-G slices indexed
by speed. The point-mass and single-track models provide `a_y,lim(v)` and
longitudinal bounds via their own physical assumptions.

## Main artifacts

1. `examples/output/spa/performance_envelope/single_track_envelope.npz`
2. `examples/output/spa/performance_envelope/point_mass_envelope.npz`
3. `examples/output/spa/performance_envelope/envelope_family_comparison.png`
4. `examples/output/spa/performance_envelope/summary.json`
5. optional CSV files (if pandas is installed)

## Potential learnings from the data

1. At each speed, interpret the upper/lower `a_x` bounds at equal `a_y`.
2. Compare drive and braking envelopes separately.
3. Identify speed ranges where model disagreement is systematically largest.
4. Use these ranges as focused inputs for lap-level model comparisons.

## Common pitfalls

1. Comparing envelopes at different speed grids.
2. Mixing up signed `a_x` convention (braking is negative in exported minimum trace).
3. Assuming envelope superiority at one speed implies lap-time superiority everywhere.
