"""Tests for analysis export and utility helpers."""

from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path

from lap_time_sim.analysis import compute_kpis, export_standard_plots
from lap_time_sim.analysis.export import export_kpi_json
from lap_time_sim.simulation.runner import simulate_lap
from lap_time_sim.tire.models import default_axle_tire_parameters
from lap_time_sim.track.io import load_track_csv
from lap_time_sim.utils.logging import configure_logging
from lap_time_sim.vehicle.params import default_vehicle_parameters


class AnalysisAndUtilsTests(unittest.TestCase):
    """Coverage tests for plotting, export and logging helpers."""

    def test_plot_and_json_exports_are_created(self) -> None:
        root = Path(__file__).resolve().parents[2]
        track = load_track_csv(root / "data" / "spa_francorchamps.csv")

        result = simulate_lap(track, default_vehicle_parameters(), default_axle_tire_parameters())
        kpis = compute_kpis(result)

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            export_standard_plots(result, out_dir)
            export_kpi_json(kpis, out_dir / "kpis.json")

            expected = [
                "speed_trace.png",
                "speed_trace.pdf",
                "yaw_moment_vs_ay.png",
                "yaw_moment_vs_ay.pdf",
                "gg_diagram.png",
                "gg_diagram.pdf",
                "tire_load_distribution.png",
                "tire_load_distribution.pdf",
                "power_trace.png",
                "power_trace.pdf",
                "kpis.json",
            ]
            for name in expected:
                self.assertTrue((out_dir / name).exists(), msg=f"missing {name}")

            data = json.loads((out_dir / "kpis.json").read_text(encoding="utf-8"))
            self.assertIn("lap_time_s", data)
            self.assertIn("max_lateral_accel_g", data)

    def test_logging_helper_runs(self) -> None:
        configure_logging(logging.INFO)
        logger = logging.getLogger("lap_time_sim_test")
        logger.info("smoke")


if __name__ == "__main__":
    unittest.main()
