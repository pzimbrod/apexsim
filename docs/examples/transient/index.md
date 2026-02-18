# Transient Solver Examples

This section demonstrates the transient solver mode.

Core idea:

- `solver_mode="quasi_static"` computes an envelope-constrained speed profile.
- `solver_mode="transient_oc"` solves a dynamic, control-constrained lap problem.

In transient mode, ApexSim supports two driver models:

- `driver_model="pid"` (default): deterministic closed-loop driver with optional
  speed-dependent gain scheduling.
- `driver_model="optimal_control"`: full minimum-time control optimization.

Both enforce vehicle dynamics consistency, track-following feasibility,
tire/friction feasibility, and control bounds (longitudinal command, plus
steering angle/rate for `SingleTrackModel`).

Use this section when you need dynamic states (`vx`, `vy`, `yaw_rate`, controls)
and want to capture effects that are weak in quasi-static mode (for example yaw
inertia in maneuver-heavy scenarios).

## Pages

1. [Point-Mass Transient Lap](point_mass_lap.md)
2. [Single-Track Transient Lap](single_track_lap.md)

## Scripts

- `examples/transient/transient_point_mass_lap.py`
- `examples/transient/transient_single_track_lap.py`

Both scripts export standardized KPI/plot artifacts plus `transient_trace.csv`
with the full time/state/control traces for external analysis. Each script also
supports:

- `--driver-model pid|optimal_control`
- `--pid-scheduling-mode off|physics_informed|custom`
