"""Tests for lazy package export resolution."""

from __future__ import annotations

import unittest

import lap_time_sim.simulation as simulation_pkg
import lap_time_sim.vehicle as vehicle_pkg


class PackageExportTests(unittest.TestCase):
    """Validate lazy export paths in package ``__getattr__`` handlers."""

    def test_simulation_lazy_exports_resolve(self) -> None:
        """Resolve simulation exports that are provided lazily."""
        self.assertIsNotNone(simulation_pkg.LapTimeVehicleModel)
        self.assertIsNotNone(simulation_pkg.VehicleModelDiagnostics)
        self.assertIsNotNone(simulation_pkg.LapSimulationResult)
        self.assertIsNotNone(simulation_pkg.simulate_lap)
        self.assertIsNotNone(simulation_pkg.SimulationRuntime)
        self.assertIsNotNone(simulation_pkg.SimulationNumerics)

    def test_vehicle_lazy_exports_resolve(self) -> None:
        """Resolve vehicle exports that are provided lazily."""
        self.assertIsNotNone(vehicle_pkg.BicycleLapTimeModel)
        self.assertIsNotNone(vehicle_pkg.BicycleLapTimeModelPhysics)
        self.assertIsNotNone(vehicle_pkg.BicycleLapTimeModelNumerics)
        self.assertIsNotNone(vehicle_pkg.build_default_bicycle_lap_time_model)

    def test_lazy_export_raises_for_missing_symbol(self) -> None:
        """Raise ``AttributeError`` for unknown lazy export names."""
        with self.assertRaises(AttributeError):
            _ = simulation_pkg.__getattr__("does_not_exist")
        with self.assertRaises(AttributeError):
            _ = vehicle_pkg.__getattr__("does_not_exist")


if __name__ == "__main__":
    unittest.main()
