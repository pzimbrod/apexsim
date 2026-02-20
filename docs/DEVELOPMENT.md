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
  - `SingleTrackPhysics`
- Numerical/discretization controls:
  - `NumericsConfig`
  - `SingleTrackNumerics`
- Simulation runtime controls (non-physical scenario bounds):
  - `RuntimeConfig`

Avoid mixing fixed-point tolerances, iteration limits, and numerical floors into
physical parameter dataclasses.

Defaulting guidance:

- Numerical controls may include stable defaults to keep quick-start simulations
  robust (`NumericsConfig`, `SingleTrackNumerics`).
- Physical parameter classes should stay explicit and scenario-specific.

## Vehicle Model Architecture

The solver contract is represented at two levels:

- `VehicleModel` (Protocol) in `src/apexsim/simulation/model_api.py`
  keeps the simulation pipeline structurally open for external backends.
- `VehicleModelBase` plus
  `EnvelopeVehicleModel` in `src/apexsim/vehicle/_model_base.py`
  provides inheritance-based code organization for built-in backends.

Built-in models (`SingleTrackModel`, `PointMassModel`) inherit the same base class
to share validation layering, friction-circle scaling, and net
drag/grade-corrected longitudinal limits.

For backend-enabled models, keep backend adapters out of the physics layer:

- `PointMassModel` composes private mixins from
  `src/apexsim/vehicle/_point_mass_physics.py` and
  `src/apexsim/vehicle/_point_mass_backends.py`.
- Physics equations stay backend-agnostic.
- Backend-specific methods (`numba`, `torch`) stay isolated in adapter mixins.
- `SingleTrackModel` composes `src/apexsim/vehicle/_single_track_physics.py`,
  which extends `PointMassPhysicalMixin` to reuse the shared physical core.
- Single-track backend adapters live in `src/apexsim/vehicle/_single_track_backends.py`
  and mirror the point-mass backend adapter structure.
- Single-track-specific dynamics/supporting components live in
  `src/apexsim/vehicle/single_track/` (for example `dynamics.py`,
  `load_transfer.py`) to keep model-specific code grouped together.

## Backend Unification Rules

When changing solver backend behavior:

- Do not duplicate core algorithmics per backend when a shared implementation
  is feasible.
- Extend shared solver cores first:
  - `src/apexsim/simulation/_profile_core.py`
  - `src/apexsim/simulation/_transient_pid_core.py`
  - `src/apexsim/simulation/_transient_controls_core.py`
  - `src/apexsim/simulation/_progress.py`
- Keep backend modules (`profile.py`, `torch_profile.py`, `transient_*`) as
  thin adapters around the shared core plus backend-specific primitives.
- Keep vehicle backend physics formulas centralized in
  `src/apexsim/vehicle/_backend_physics_core.py` and consume those helpers
  from point-mass and single-track backends.
- If a new backend-specific branch is unavoidable for performance, add a parity
  test against the shared semantics and document why specialization is needed.

Performance policy for refactors:

- Run `scripts/benchmark_solver_matrix.py` before and after significant backend
  refactors.
- Compare with `scripts/compare_solver_benchmarks.py`.
- Any case slower than 5% requires either optimization or explicit, documented
  justification.

## Refreshing Spa Data

```bash
source .venv/bin/activate
python scripts/import_spa_from_tumftm.py
```
