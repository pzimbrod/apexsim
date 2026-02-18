# How to Use ApexSim

This guide is a practical, engineer-oriented walkthrough of the complete workflow.
It is written for race engineers and Students who may have strong dynamics knowledge
but limited software background.

## What ApexSim is designed to do

ApexSim is optimized for fast, physically grounded lap-time studies with modular
vehicle models.

Main strengths:

- Quasi-steady lap-time simulation on arbitrary track centerlines.
- Interchangeable vehicle models behind one solver API.
- Clear separation between physical inputs and numerical solver settings.
- Reproducible engineering outputs (KPIs + standardized plots).
- Fast iteration loops for setup changes and what-if studies.

## What ApexSim does not do (yet)

Current model boundaries are important for correct interpretation:

- No full transient lap solver in production path yet.
- No driver model or closed-loop control strategy model.
- No detailed powertrain/energy management model yet.
- No full multi-body chassis compliance model.
- No direct tire thermal/wear state evolution.

Interpretation rule:

- Use ApexSim for comparative setup studies and first-order lap-time sensitivity,
  not as final truth for every transient detail.

## The 6-step workflow (always the same)

1. Import modules.
2. Define physical model inputs.
3. Load or generate a track.
4. Configure numerics and runtime bounds.
5. Run the simulation.
6. Postprocess and review outputs.

This pattern is identical across all examples.

## Step 1: Imports

```python
from pathlib import Path

from apexsim.analysis import compute_kpis, export_standard_plots
from apexsim.analysis.export import export_kpi_json
from apexsim.simulation import build_simulation_config, simulate_lap
from apexsim.tire import default_axle_tire_parameters
from apexsim.track import load_track_csv
from apexsim.utils.constants import STANDARD_AIR_DENSITY
from apexsim.vehicle import SingleTrackPhysics, VehicleParameters, build_single_track_model
```

How to read these imports as an engineer:

- `track`: geometry and road profile.
- `vehicle` + `tire`: physical car model.
- `simulation`: numerical solver and runtime bounds.
- `analysis`: KPI and visualization outputs.

## Step 2: Define physical model inputs

### 2.1 Vehicle parameters (real-system inputs)

```python
vehicle = VehicleParameters(
    mass=798.0,
    yaw_inertia=1120.0,
    cg_height=0.31,
    wheelbase=3.60,
    front_track=1.60,
    rear_track=1.55,
    front_weight_fraction=0.46,
    cop_position=0.10,
    lift_coefficient=3.20,
    drag_coefficient=0.90,
    frontal_area=1.50,
    roll_rate=4200.0,
    front_spring_rate=180000.0,
    rear_spring_rate=165000.0,
    front_arb_distribution=0.55,
    front_ride_height=0.030,
    rear_ride_height=0.050,
    air_density=STANDARD_AIR_DENSITY,
)
```

This block should reflect the best available engineering estimate of the real car.

### 2.2 Tire data

```python
tires = default_axle_tire_parameters()
```

Recommendation:

- Start with defaults for initial integration.
- Replace with identified tire parameters for decision-quality studies.

### 2.3 Choose model complexity

Single-track model:

```python
model = build_single_track_model(
    vehicle=vehicle,
    tires=tires,
    physics=SingleTrackPhysics(),
)
```

Point-mass model:

```python
from apexsim.vehicle import PointMassPhysics, build_point_mass_model

model = build_point_mass_model(
    vehicle=vehicle,
    physics=PointMassPhysics(),
)
```

When to use which:

- Single-track (bicycle): better diagnostics (yaw moment, axle-load dynamics), better cornering interpretation.
- Point-mass: fast baseline and sensitivity sweeps.

## Step 3: Load or generate track

### Option A: Real track from CSV

```python
project_root = Path(__file__).resolve().parents[1]
track = load_track_csv(project_root / "data" / "spa_francorchamps.csv")
```

Required columns:

- `x`
- `y`
- `elevation`
- `banking`

Internally, ApexSim derives arc length, heading, curvature, and grade.

### Option B: Synthetic validation tracks

```python
from apexsim.track import build_straight_track, build_circular_track, build_figure_eight_track

straight = build_straight_track(length=1000.0)
circle = build_circular_track(radius=50.0)
figure_eight = build_figure_eight_track(lobe_radius=80.0)
```

Why synthetic tracks matter:

- You can verify single effects in isolation.
- Debugging is easier than on a full GP circuit.

## Step 4: Configure runtime and numerics

Simple setup:

```python
config = build_simulation_config(max_speed=115.0)
```

Explicit setup:

```python
from apexsim.simulation import NumericsConfig, RuntimeConfig, SimulationConfig

config = SimulationConfig(
    runtime=RuntimeConfig(max_speed=115.0),
    numerics=NumericsConfig(
        min_speed=8.0,
        lateral_envelope_max_iterations=20,
        lateral_envelope_convergence_tolerance=0.1,
        transient_step=0.01,
    ),
)
```

### Backend selection (numerical execution)

```python
config_numpy = build_simulation_config(compute_backend="numpy", max_speed=115.0)
config_numba = build_simulation_config(compute_backend="numba", max_speed=115.0)
config_torch = build_simulation_config(
    compute_backend="torch",
    torch_device="cpu",  # or "cuda:0"
    torch_compile=False,
    max_speed=115.0,
)
```

Selection rule:

- `numpy`: robust baseline and easiest debugging.
- `numba`: fastest CPU sweeps (currently with `PointMassModel` and `SingleTrackModel`).
- `torch`: CPU/GPU execution and tensor-native workflows.

For quantitative guidance, see [Compute Backends](BACKENDS.md).

Critical distinction:

- Physical parameters represent the car/track reality.
- Numerical parameters control solver stability and convergence.

Do not compensate wrong physics by over-tuning numerics.

## Step 5: Run the lap simulation

```python
result = simulate_lap(track=track, model=model, config=config)
```

`result` includes:

- lap time
- speed trace
- longitudinal/lateral accelerations
- yaw moment
- axle loads
- power trace
- integrated energy

## Step 6: Postprocess and export

```python
kpis = compute_kpis(result)

output_dir = project_root / "examples" / "output"
export_standard_plots(result, output_dir)
export_kpi_json(kpis, output_dir / "kpis.json")
```

Optional: generate a speed-dependent performance envelope (G-G map family):

```python
from apexsim.analysis import (
    PerformanceEnvelopeNumerics,
    PerformanceEnvelopePhysics,
    compute_performance_envelope,
)

envelope = compute_performance_envelope(
    model=model,
    physics=PerformanceEnvelopePhysics(speed_min=20.0, speed_max=90.0),
    numerics=PerformanceEnvelopeNumerics(speed_samples=31, lateral_accel_samples=41),
)
envelope_array = envelope.to_numpy()
```

Minimum review set:

- lap time
- max lateral acceleration
- max longitudinal acceleration/deceleration
- speed trace shape vs track layout

## Engineering interpretation checklist

Before using results for decisions:

1. Straight track check: near-zero lateral acceleration.
2. Constant-radius check: quasi-steady cornering in interior segments.
3. Figure-eight check: lateral acceleration sign change and entry/exit transitions.
4. Magnitude check: compare against realistic bounds for your car class.
5. Model check: verify whether selected model complexity can represent the effect you study.

## Typical pitfalls (and fixes)

1. "The solver should always accelerate to max speed."
   - Not necessarily. At high speed, drag can exceed available drive force.
2. "Yaw moment should always be visible."
   - Not with point-mass model (yaw moment is structurally zero).
3. "Changing numerics changed physics dramatically."
   - Re-check physical parameters first; numerics should refine stability, not redefine behavior.
4. "My results jump near lap closure."
   - For closed loops, inspect interior segments and avoid over-interpreting seam points.

## Recommended onboarding path

1. Read [Synthetic Track Walkthrough](EXAMPLES_SYNTHETIC.md).
2. Run [Spa Walkthrough](examples/spa/index.md).
3. Compare model complexity with `examples/spa/spa_model_comparison.py`.
4. Move from default tire/model settings to identified vehicle data.
