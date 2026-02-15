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
def lateral_speed_limit(curvature_1pm: float, ay_limit_mps2: float, vmax_mps: float) -> float:
    """Compute speed limit from curvature and lateral acceleration capability.

    Args:
        curvature_1pm: Signed path curvature in 1/m.
        ay_limit_mps2: Available lateral acceleration magnitude in m/s^2.
        vmax_mps: Global hard speed cap in m/s.

    Returns:
        Maximum feasible speed in m/s under curvature and lateral limits.
    """
```

## Refreshing Spa Data

```bash
source .venv/bin/activate
python scripts/import_spa_from_tumftm.py
```
