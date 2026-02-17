# How to Use PyLapSim

This guide is a practical, engineer-oriented walkthrough of the typical workflow.
It is designed for users with strong vehicle-dynamics background and limited
programming experience.

## What you build in every simulation

Every PyLapSim run has the same structure:

1. Import modules.
2. Define vehicle and model parameters.
3. Load or generate a track.
4. Configure simulation numerics.
5. Run the lap simulation.
6. Postprocess KPIs and plots.

If you remember this 6-step pattern, you can run all examples and extend them.

## Step 1: Imports

```python
from pathlib import Path

from pylapsim.analysis import compute_kpis, export_standard_plots
from pylapsim.analysis.export import export_kpi_json
from pylapsim.simulation import build_simulation_config, simulate_lap
from pylapsim.tire import default_axle_tire_parameters
from pylapsim.track import load_track_csv
from pylapsim.utils.constants import STANDARD_AIR_DENSITY
from pylapsim.vehicle import BicyclePhysics, VehicleParameters, build_bicycle_model
```

Why these imports matter:

- `track`: where geometry comes from.
- `vehicle` and `tire`: physical model inputs.
- `simulation`: solver configuration and execution.
- `analysis`: engineering outputs (KPIs and plots).

## Step 2: Define model inputs

### 2.1 Vehicle parameters (physical)

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

These parameters represent the real system (car + aero + suspension).

### 2.2 Tire parameters

```python
tires = default_axle_tire_parameters()
```

For first studies, start with defaults. Replace later with measured/identified tire data.

### 2.3 Choose a vehicle model

Bicycle model (higher fidelity):

```python
model = build_bicycle_model(
    vehicle=vehicle,
    tires=tires,
    physics=BicyclePhysics(),
)
```

Point-mass model (faster, simpler):

```python
from pylapsim.vehicle import PointMassPhysics, build_point_mass_model

model = build_point_mass_model(
    vehicle=vehicle,
    physics=PointMassPhysics(),
)
```

## Step 3: Load or generate track

### Option A: Real track from CSV

```python
project_root = Path(__file__).resolve().parents[1]
track = load_track_csv(project_root / "data" / "spa_francorchamps.csv")
```

Required CSV columns: `x`, `y`, `elevation`, `banking`.

### Option B: Synthetic validation tracks

```python
from pylapsim.track import build_straight_track, build_circular_track, build_figure_eight_track

straight = build_straight_track(length=1000.0)
circle = build_circular_track(radius=50.0)
figure_eight = build_figure_eight_track(lobe_radius=80.0)
```

Use synthetic tracks for quick physical sanity checks before running real tracks.

## Step 4: Configure simulation numerics

```python
config = build_simulation_config(max_speed=115.0)
```

Important distinction:

- Physical inputs: `VehicleParameters`, tire data, model physics classes.
- Numerical inputs: `SimulationConfig`, `NumericsConfig`, convergence tolerances,
  iteration limits.

If needed, create explicit numerics:

```python
from pylapsim.simulation import NumericsConfig, RuntimeConfig, SimulationConfig

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

## Step 5: Run simulation

```python
result = simulate_lap(track=track, model=model, config=config)
```

`result` contains speed trace, accelerations, yaw moment, axle loads, power, energy,
and lap time.

## Step 6: Postprocess and export

```python
kpis = compute_kpis(result)

output_dir = project_root / "examples" / "output"
export_standard_plots(result, output_dir)
export_kpi_json(kpis, output_dir / "kpis.json")
```

Key KPI outputs:

- lap time
- average/max lateral acceleration
- average/max longitudinal acceleration
- integrated traction energy

## Physical sanity checklist (recommended)

Before trusting a result:

1. Straight track: lateral acceleration should be near zero.
2. Constant-radius circle: speed and longitudinal acceleration should be nearly steady in the interior.
3. Figure-eight: lateral acceleration should change sign.
4. Magnitudes should be realistic for your vehicle class.

## Learning path

1. Run [Synthetic Track Examples](EXAMPLES_SYNTHETIC.md).
2. Run [Spa-Francorchamps Examples](EXAMPLES_SPA.md).
3. Compare models with `examples/spa_model_comparison.py`.
