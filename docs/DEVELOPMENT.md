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

## Refreshing Spa Data

```bash
source .venv/bin/activate
python scripts/import_spa_from_tumftm.py
```
