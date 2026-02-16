"""Tests for analysis export and utility helpers."""

from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path

from lap_time_sim.analysis import compute_kpis, export_standard_plots
from lap_time_sim.analysis.export import export_kpi_json
from lap_time_sim.simulation.config import NumericsConfig, RuntimeConfig, SimulationConfig
from lap_time_sim.simulation.runner import simulate_lap
from lap_time_sim.tire.models import default_axle_tire_parameters
from lap_time_sim.track.io import load_track_csv
from lap_time_sim.utils.logging import configure_logging
from lap_time_sim.vehicle import BicycleModel, BicycleNumerics, BicyclePhysics
from tests.helpers import sample_vehicle_parameters


class AnalysisAndUtilsTests(unittest.TestCase):
    """Coverage tests for plotting, export and logging helpers."""

    def test_plot_and_json_exports_are_created(self) -> None:
        """Ensure standard plot files and KPI JSON export are generated."""
        root = Path(__file__).resolve().parents[2]
        track = load_track_csv(root / "data" / "spa_francorchamps.csv")
        model = BicycleModel(
            vehicle=sample_vehicle_parameters(),
            tires=default_axle_tire_parameters(),
            physics=BicyclePhysics(
                max_drive_accel_mps2=8.0,
                max_brake_accel_mps2=16.0,
                peak_slip_angle_rad=0.12,
            ),
            numerics=BicycleNumerics(
                min_lateral_accel_limit_mps2=0.5,
                lateral_limit_max_iterations=12,
                lateral_limit_convergence_tol_mps2=0.05,
            ),
        )

        result = simulate_lap(
            track=track,
            model=model,
            config=SimulationConfig(
                runtime=RuntimeConfig(
                    max_speed_mps=115.0,
                    enable_transient_refinement=False,
                ),
                numerics=NumericsConfig(
                    min_speed_mps=8.0,
                    lateral_envelope_max_iterations=20,
                    lateral_envelope_convergence_tol_mps=0.1,
                    transient_dt_s=0.01,
                ),
            ),
        )
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
        """Smoke-test logging helper configuration."""
        configure_logging(logging.INFO)
        logger = logging.getLogger("lap_time_sim_test")
        logger.info("smoke")


if __name__ == "__main__":
    unittest.main()
