# Spa Examples

Spa-Francorchamps is the first full-track study path in ApexSim.
These pages are written for engineers and students who want both
practical usage and interpretation guidance.

## Why Spa as tutorial track

- It contains long straights, high-speed corners, and strong elevation changes.
- It is complex enough to expose model differences clearly.
- It is still structured enough for reproducible onboarding exercises.

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

## Quick run commands

```bash
python examples/spa/spa_lap_single_track.py
python examples/spa/spa_lap_point_mass.py
python examples/spa/spa_model_comparison.py
python examples/spa/spa_performance_envelope.py
```

## Student-oriented study checklist

1. Record assumptions (tire setup, aero coefficients, solver backend).
2. Save all outputs for each run and label them by configuration.
3. Compare traces locally by track section, not only by global lap time.
4. Document one modeling limitation and one numerical limitation per run.
