"""Unit tests for solver benchmark utility scripts."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType


def _load_script_module(script_name: str) -> ModuleType:
    """Load one script module from ``scripts/`` for unit testing.

    Args:
        script_name: Script filename under ``scripts/``.

    Returns:
        Imported module object for the requested script.
    """
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load script module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SolverBenchmarkScriptTests(unittest.TestCase):
    """Validate benchmark matrix and comparison script helpers."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load benchmark scripts once for all tests."""
        cls.compare_module = _load_script_module("compare_solver_benchmarks.py")
        cls.benchmark_module = _load_script_module("benchmark_solver_matrix.py")

    def test_compare_computes_expected_slowdown(self) -> None:
        """Compute candidate slowdown percentage from median runtimes."""
        baseline = {
            "numpy|point_mass|quasi_static|straight": {
                "steady_median_ms": 10.0,
            }
        }
        candidate = {
            "numpy|point_mass|quasi_static|straight": {
                "steady_median_ms": 10.7,
            }
        }
        rows = self.compare_module._compare(baseline=baseline, candidate=candidate)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0].slowdown_pct, 7.0, places=6)

    def test_load_cases_reads_indexed_payload(self) -> None:
        """Read benchmark JSON payload and index by case id."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            payload_path = Path(tmp_dir) / "bench.json"
            payload_path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "case_id": "numpy|point_mass|quasi_static|straight",
                                "steady_median_ms": 11.2,
                            }
                        ]
                    }
                )
            )
            indexed = self.compare_module._load_cases(payload_path)
            self.assertIn("numpy|point_mass|quasi_static|straight", indexed)
            self.assertEqual(
                float(indexed["numpy|point_mass|quasi_static|straight"]["steady_median_ms"]),
                11.2,
            )

    def test_benchmark_helpers_expose_expected_matrix_dimensions(self) -> None:
        """Verify benchmark helper metadata without running simulations."""
        backends = self.benchmark_module._available_backends()
        self.assertIn("numpy", backends)

        models = self.benchmark_module._build_models()
        self.assertIn("point_mass", models)
        self.assertIn("single_track", models)

        tracks = self.benchmark_module._build_tracks(include_spa=False)
        self.assertEqual(set(tracks), {"straight", "circle"})
