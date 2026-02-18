# Spa Examples

Spa-Francorchamps is the first full-track study path in ApexSim.
These pages combine practical script usage with theoretical context so
model outputs can be interpreted consistently.

## Why Spa as tutorial track

- It contains long straights, high-speed corners, and strong elevation changes.
- It is complex enough to expose model differences clearly.
- It is still structured enough for reproducible onboarding studies.

## What you should know before starting

1. Run the [Synthetic Track Walkthrough](../../EXAMPLES_SYNTHETIC.md) first.
2. Be familiar with the 6-step workflow from [How to Use](../../HOW_TO_USE.md).
3. Use default parameters first, then tune one block at a time.

## Tutorial sequence

1. [Single-Track Lap](single_track_lap.md)
2. [Point-Mass Lap](point_mass_lap.md)
3. [Model Comparison](model_comparison.md)
4. [Performance Envelope](performance_envelope.md)

## Common output structure

All Spa scripts write into `examples/output/spa/`.
Subfolders are separated by script purpose (single-track, point-mass,
comparison, envelope), so results stay comparable.

## Theoretical context used across all Spa pages

1. Quasi-steady assumption:
   vehicle states are solved from local equilibrium in arc-length domain,
   without full transient tire-state dynamics.
2. Lateral/longitudinal coupling:
   available `a_x` is reduced as required `a_y` approaches the lateral limit.
3. Aero coupling:
   both drag and downforce scale approximately with `v^2`, which changes
   acceleration limits strongly at higher speeds.
4. Model hierarchy:
   point-mass is a reduced model; single-track adds yaw and axle-level effects.

## Quick run commands

```bash
python examples/spa/spa_lap_single_track.py
python examples/spa/spa_lap_point_mass.py
python examples/spa/spa_model_comparison.py
python examples/spa/spa_performance_envelope.py
```

## Neutral analysis checklist

1. Record physical assumptions and numerical settings for each run.
2. Compare both global KPIs and local trace behavior along arc length.
3. Separate model-structure effects from parameter-tuning effects.
4. State model limitations next to each engineering conclusion.
