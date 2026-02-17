# Development

## Environment

A local virtual environment is expected at `.venv`.

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Test Commands

Because this environment may run offline, tests can be executed via `unittest` directly:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m unittest discover -s tests -v
```

Coverage:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m coverage run -m unittest discover -s tests
python -m coverage report -m
```

Coverage policy:

- Keep overall coverage at or above `95%`.

If `pytest` is available:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest
```

## Docstring Policy

This repository enforces Google-style docstrings through `ruff` (`D` rules with
`pydocstyle` convention `google`) and an AST contract test.

Rules:

- Public classes, functions, and interfaces must include full Google docstrings.
- Private/technical helpers must always include a summary line.
- If a callable has parameters other than `self`/`cls`, include an `Args:` section.
- If a callable returns a non-`None` value, include a `Returns:` section.
- If a callable intentionally raises domain/validation errors, include `Raises:`.
- Keep units explicit for physics-facing values (`m/s`, `m/s^2`, `1/m`, `rad`, `W`).

Example:

```python
def lateral_speed_limit(curvature: float, lateral_accel_limit: float, max_speed: float) -> float:
    """Compute speed limit from curvature and lateral acceleration capability.

    Args:
        curvature: Signed path curvature [1/m].
        lateral_accel_limit: Available lateral acceleration magnitude [m/s^2].
        max_speed: Global hard speed cap [m/s].

    Returns:
        Maximum feasible speed [m/s] under curvature and lateral limits.
    """
```

## Parameter Layering

Keep parameter domains separate:

- Physical model inputs:
  - `VehicleParameters`
  - `PacejkaParameters` / `AxleTireParameters`
  - `BicyclePhysics`
- Numerical/discretization controls:
  - `NumericsConfig`
  - `BicycleNumerics`
- Simulation runtime controls (non-physical scenario bounds):
  - `RuntimeConfig`

Avoid mixing fixed-point tolerances, iteration limits, and numerical floors into
physical parameter dataclasses.

Defaulting guidance:

- Numerical controls may include stable defaults to keep quick-start simulations
  robust (`NumericsConfig`, `BicycleNumerics`).
- Physical parameter classes should stay explicit and scenario-specific.

## Vehicle Model Architecture

The solver contract is represented at two levels:

- `VehicleModel` (Protocol) in `src/pylapsim/simulation/model_api.py`
  keeps the simulation pipeline structurally open for external backends.
- `VehicleModelBase` plus
  `EnvelopeVehicleModel` in `src/pylapsim/vehicle/_model_base.py`
  provides inheritance-based code organization for built-in backends.

Built-in models (`BicycleModel`, `PointMassModel`) inherit the same base class
to share validation layering, friction-circle scaling, and net
drag/grade-corrected longitudinal limits.

For backend-enabled models, keep backend adapters out of the physics layer:

- `PointMassModel` composes private mixins from
  `src/pylapsim/vehicle/_point_mass_physics.py` and
  `src/pylapsim/vehicle/_point_mass_backends.py`.
- Physics equations stay backend-agnostic.
- Backend-specific methods (`numba`, `torch`) stay isolated in adapter mixins.
- `BicycleModel` composes `src/pylapsim/vehicle/_bicycle_physics.py`,
  which extends `PointMassPhysicalMixin` to reuse the shared physical core.

## Refreshing Spa Data

```bash
source .venv/bin/activate
python scripts/import_spa_from_tumftm.py
```
