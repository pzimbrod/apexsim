# Sensitivity Examples

These examples focus on practical interpretation of parameter sensitivities with
the single-track model on the torch backend and the high-level
`apexsim.analysis.run_lap_sensitivity_study(...)` API.

## Study scope

Both example studies use local derivatives around one operating point and report
an engineering-friendly `%/%` interpretation. The same four physical parameters
are included:

1. Vehicle mass
2. Center of gravity height
3. Yaw inertia
4. Drag coefficient

Two objective metrics are evaluated:

1. Lap time \([s]\)
2. Energy consumption \([kWh]\)

## Why this setup is useful

- It gives a consistent sensitivity baseline across different track classes.
- It highlights which parameters matter globally (lap time) vs. energetically.
- It uses one compact, model-agnostic API with clear parameter targets.
- The Spa notebook additionally compares quasi-static and transient sensitivities
  in an AD-first workflow to expose solver-path limitations (notably for yaw inertia).

## Output artifacts

Each study exports:

- `sensitivities_long.csv`: one row per `(objective, parameter)` pair
- `sensitivities_pivot.csv`: compact parameter × objective sensitivity map
- `sensitivity_bars.png`: compact comparison plot for both objectives

All outputs are written below:

- `examples/output/sensitivity/`

## Notebooks

1. [Synthetic Single-Track Sensitivity](synthetic_single_track_sensitivity.ipynb)
2. [Spa Single-Track Sensitivity](spa_single_track_sensitivity.ipynb)
