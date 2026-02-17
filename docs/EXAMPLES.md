# Examples

This section is a guided path through the example scripts.
It is written for race engineers and student teams who want fast onboarding.

## Start here

If you are new to the package, read [How to Use](HOW_TO_USE.md) first.
Then use the examples in this order:

1. [Synthetic Tracks](EXAMPLES_SYNTHETIC.md)
2. [Spa-Francorchamps](EXAMPLES_SPA.md)

## Why this order?

- Synthetic tracks isolate single effects and make validation intuitive.
- Spa combines all effects into a realistic full-lap workflow.

## Which script answers which engineering question?

- `examples/synthetic_track_scenarios.py`
  - "Are my model settings physically consistent on simple geometry?"
- `examples/spa_lap.py`
  - "What does the higher-fidelity bicycle model predict on a real track?"
- `examples/spa_lap_point_mass.py`
  - "What does a faster low-complexity baseline predict?"
- `examples/spa_model_comparison.py`
  - "What do I gain by adding model complexity?"

## Output convention

All scripts export to `examples/output/`.
Typical artifacts:

- `kpis.json`
- speed trace plots
- G-G diagram
- yaw moment vs. lateral acceleration
- power and tire-load plots

This uniform output structure makes script results directly comparable.
