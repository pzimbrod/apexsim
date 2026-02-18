# Examples

This section is the tutorial path through the example scripts.
It is designed for race engineers and Students who want practical onboarding.

## What you will learn from the examples

After finishing all example pages, you should be able to:

- set up a complete lap-time run from scratch,
- choose between point-mass and single-track model for a study goal,
- interpret key plots and KPI outputs correctly,
- understand where model assumptions limit interpretation.

## Recommended tutorial sequence

1. [Synthetic Track Walkthrough](EXAMPLES_SYNTHETIC.md)
2. [Spa Walkthrough](examples/spa/index.md)

Reason:

- Synthetic tracks isolate single effects and simplify debugging.
- Spa introduces full-track interactions and realistic complexity.

## Mapping scripts to engineering questions

- `examples/synthetic_track_scenarios.py`
  - "Is my model setup physically consistent on canonical test geometries?"
- `examples/spa/spa_lap_single_track.py`
  - "What does the single-track model predict on a realistic circuit?"
- `examples/spa/spa_lap_point_mass.py`
  - "What is the low-complexity baseline on the same circuit?"
- `examples/spa/spa_model_comparison.py`
  - "What do I gain from additional model fidelity?"
- `examples/spa/spa_performance_envelope.py`
  - "How does the speed-dependent G-G envelope differ by model complexity?"
- `examples/backend_benchmarks.py`
  - "Which compute backend should I use for my workload?"

## What the example suite covers well

- End-to-end solver use on real and synthetic tracks.
- KPI and plot export workflows.
- Direct model-complexity comparison.
- Practical interpretation of speed/acceleration/yaw diagnostics.

## What the example suite does not claim

- It is not a final validation against fully instrumented race telemetry.
- It does not include a full transient driver + powertrain control stack.
- It does not replace model-calibration work on real vehicle data.

## Output structure

All scripts export into `examples/output/` with consistent folder logic
(`examples/output/spa/` for Spa-specific workflows).
Typical artifacts:

- `kpis.json`
- speed traces
- G-G diagram
- yaw moment vs. lateral acceleration
- tire-load and power traces

Because output format is standardized, comparisons across scripts are straightforward.

## Before applying results to decisions

Use this quick quality gate:

1. Check whether the selected model can represent your target effect.
2. Confirm basic sanity checks on synthetic tracks.
3. Check if KPI magnitudes are plausible for your vehicle class.
4. Document assumptions used in the run.
