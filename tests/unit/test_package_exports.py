"""Tests for lazy package export resolution."""

from __future__ import annotations

import unittest

import pylapsim.simulation as simulation_pkg
import pylapsim.track as track_pkg
import pylapsim.vehicle as vehicle_pkg


class PackageExportTests(unittest.TestCase):
    """Validate lazy export paths in package ``__getattr__`` handlers."""

    def test_simulation_lazy_exports_resolve(self) -> None:
        """Resolve simulation exports that are provided lazily."""
        self.assertIsNotNone(simulation_pkg.VehicleModelBase)
        self.assertIsNotNone(simulation_pkg.VehicleModel)
        self.assertIsNotNone(simulation_pkg.ModelDiagnostics)
        self.assertIsNotNone(simulation_pkg.LapResult)
        self.assertIsNotNone(simulation_pkg.simulate_lap)
        self.assertIsNotNone(simulation_pkg.RuntimeConfig)
        self.assertIsNotNone(simulation_pkg.NumericsConfig)
        self.assertIsNotNone(simulation_pkg.build_simulation_config)

    def test_track_exports_resolve(self) -> None:
        """Resolve track exports including synthetic layout builder helpers."""
        self.assertIsNotNone(track_pkg.TrackData)
        self.assertIsNotNone(track_pkg.load_track_csv)
        self.assertIsNotNone(track_pkg.build_straight_track)
        self.assertIsNotNone(track_pkg.build_circular_track)
        self.assertIsNotNone(track_pkg.build_figure_eight_track)

    def test_vehicle_lazy_exports_resolve(self) -> None:
        """Resolve vehicle exports that are provided lazily."""
        self.assertIsNotNone(vehicle_pkg.SingleTrackModel)
        self.assertIsNotNone(vehicle_pkg.SingleTrackDynamicsModel)
        self.assertIsNotNone(vehicle_pkg.SingleTrackPhysics)
        self.assertIsNotNone(vehicle_pkg.SingleTrackNumerics)
        self.assertIsNotNone(vehicle_pkg.build_single_track_model)
        self.assertIsNotNone(vehicle_pkg.PointMassModel)
        self.assertIsNotNone(vehicle_pkg.PointMassPhysics)
        self.assertIsNotNone(vehicle_pkg.PointMassCalibrationResult)
        self.assertIsNotNone(vehicle_pkg.build_point_mass_model)
        self.assertIsNotNone(vehicle_pkg.calibrate_point_mass_friction_to_single_track)

    def test_lazy_export_raises_for_missing_symbol(self) -> None:
        """Raise ``AttributeError`` for unknown lazy export names."""
        with self.assertRaises(AttributeError):
            _ = simulation_pkg.__getattr__("does_not_exist")
        with self.assertRaises(AttributeError):
            _ = vehicle_pkg.__getattr__("does_not_exist")


if __name__ == "__main__":
    unittest.main()
