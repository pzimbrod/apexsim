# Point-Mass Transient Lap

This example runs the transient solver with `PointMassModel` on a straight
track and a standing start (`initial_speed=0.0`).

Script: `examples/transient/transient_point_mass_lap.py`

## Engineering intent

This scenario isolates longitudinal behavior:

- no steering control,
- no lateral dynamics state evolution,
- direct visibility of launch and acceleration phases.

It is useful for validating traction/braking bottlenecks and for checking how
transient control limits affect the time trace from rest.

## Configuration highlights

- solver mode: `transient_oc`
- default driver model: PID (`driver_model="pid"`)
- optional full optimizer: `driver_model="optimal_control"`
- backend: selectable (`numpy`, `numba`, `torch`)
- integration method: `euler` or `rk4`
- PID scheduling mode: `off`, `physics_informed`, or `custom`
- objective: minimum lap time with smoothness and lateral-feasibility penalties

## Run

```bash
python examples/transient/transient_point_mass_lap.py --backend numpy --integration-method rk4
```

Physics-informed PID scheduling:

```bash
python examples/transient/transient_point_mass_lap.py --backend numpy --pid-scheduling-mode physics_informed
```

Full optimal-control mode:

```bash
python examples/transient/transient_point_mass_lap.py --backend numpy --driver-model optimal_control
```

## Outputs

Artifacts are written to:

- `examples/output/transient/point_mass_standing_start/`

Key files:

- `kpis.json`
- standard plots (speed/power/loads)
- `transient_trace.csv`

The trace CSV contains:

- arc length and time,
- speed and accelerations,
- `vx`, `vy`, `yaw_rate`,
- control signals (`steer_cmd`, `ax_cmd`).

For point-mass transient runs, `vy`, `yaw_rate`, and `steer_cmd` remain
structurally zero by model definition.
