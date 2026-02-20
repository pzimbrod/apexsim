# Single-Track Transient Lap

This example runs the transient solver with `SingleTrackModel` on a
figure-eight track.

Script: `examples/transient/transient_single_track_lap.py`

## Engineering intent

The figure-eight introduces repeated left/right transient maneuvers, so the
single-track dynamic states become informative:

- `vx`: longitudinal body speed,
- `vy`: lateral body speed,
- `yaw_rate`: yaw dynamics,
- `steer_cmd`: steering command trajectory,
- `ax_cmd`: longitudinal command trajectory.

Compared to quasi-static solves, this setup better exposes control-rate limits
and yaw-inertia effects.

## Configuration highlights

- solver mode: `transient_oc`
- default driver model: PID (`driver_model="pid"`)
- optional full optimizer: `driver_model="optimal_control"`
- model-level control limits:
  - `SingleTrackPhysics.max_steer_angle`
  - `SingleTrackPhysics.max_steer_rate`
- backend: selectable (`numpy`, `numba`, `torch`)
- integration method: `euler` or `rk4`
- PID scheduling mode: `off`, `physics_informed`, or `custom`

## Run

```bash
python examples/transient/transient_single_track_lap.py --backend numpy --integration-method rk4
```

Physics-informed PID scheduling:

```bash
python examples/transient/transient_single_track_lap.py --backend numpy --pid-scheduling-mode physics_informed
```

Full optimal-control mode:

```bash
python examples/transient/transient_single_track_lap.py --backend numpy --driver-model optimal_control
```

## Outputs

Artifacts are written to:

- `examples/output/transient/single_track_figure_eight/`

Key files:

- `kpis.json`
- standard plots
- `transient_trace.csv`

Use `transient_trace.csv` to inspect control/state transitions and to compare
backend behavior or configuration changes.
