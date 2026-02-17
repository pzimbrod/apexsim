"""Tests for lazy package export resolution."""

from __future__ import annotations

import unittest

import pylapsim.simulation as simulation_pkg
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

    def test_vehicle_lazy_exports_resolve(self) -> None:
        """Resolve vehicle exports that are provided lazily."""
        self.assertIsNotNone(vehicle_pkg.BicycleModel)
        self.assertIsNotNone(vehicle_pkg.BicycleDynamicsModel)
        self.assertIsNotNone(vehicle_pkg.BicyclePhysics)
        self.assertIsNotNone(vehicle_pkg.BicycleNumerics)
        self.assertIsNotNone(vehicle_pkg.build_bicycle_model)
        self.assertIsNotNone(vehicle_pkg.PointMassModel)
        self.assertIsNotNone(vehicle_pkg.PointMassPhysics)
        self.assertIsNotNone(vehicle_pkg.PointMassCalibrationResult)
        self.assertIsNotNone(vehicle_pkg.build_point_mass_model)
        self.assertIsNotNone(vehicle_pkg.calibrate_point_mass_friction_to_bicycle)

    def test_lazy_export_raises_for_missing_symbol(self) -> None:
        """Raise ``AttributeError`` for unknown lazy export names."""
        with self.assertRaises(AttributeError):
            _ = simulation_pkg.__getattr__("does_not_exist")
        with self.assertRaises(AttributeError):
            _ = vehicle_pkg.__getattr__("does_not_exist")


if __name__ == "__main__":
    unittest.main()
