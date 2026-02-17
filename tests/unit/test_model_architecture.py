"""Tests for vehicle-model class hierarchy and shared OOP contracts."""

from __future__ import annotations

import unittest

from lap_time_sim.simulation.model_api import AbstractLapTimeVehicleModel
from lap_time_sim.tire.models import default_axle_tire_parameters
from lap_time_sim.vehicle._model_base import EnvelopeVehicleModel
from lap_time_sim.vehicle.bicycle_model import BicycleModel, BicycleNumerics, BicyclePhysics
from lap_time_sim.vehicle.point_mass_model import PointMassModel, PointMassPhysics
from tests.helpers import sample_vehicle_parameters


class ModelArchitectureTests(unittest.TestCase):
    """Validate inheritance-based model architecture for solver backends."""

    def test_models_derive_from_common_abstract_vehicle_model(self) -> None:
        """Ensure backends derive from the abstract solver model contract."""
        self.assertTrue(issubclass(BicycleModel, AbstractLapTimeVehicleModel))
        self.assertTrue(issubclass(PointMassModel, AbstractLapTimeVehicleModel))

    def test_models_share_common_envelope_vehicle_base(self) -> None:
        """Ensure backends share the envelope-limited base implementation."""
        self.assertTrue(issubclass(BicycleModel, EnvelopeVehicleModel))
        self.assertTrue(issubclass(PointMassModel, EnvelopeVehicleModel))

    def test_friction_circle_scaling_is_shared_via_base_class(self) -> None:
        """Use the shared base implementation for friction-circle scaling."""
        bicycle = BicycleModel(
            vehicle=sample_vehicle_parameters(),
            tires=default_axle_tire_parameters(),
            physics=BicyclePhysics(),
            numerics=BicycleNumerics(),
        )
        point_mass = PointMassModel(
            vehicle=sample_vehicle_parameters(),
            physics=PointMassPhysics(),
        )
        expected = EnvelopeVehicleModel._friction_circle_scale(5.0, 10.0)
        self.assertEqual(bicycle._friction_circle_scale(5.0, 10.0), expected)
        self.assertEqual(point_mass._friction_circle_scale(5.0, 10.0), expected)


if __name__ == "__main__":
    unittest.main()
