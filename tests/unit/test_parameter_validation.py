"""Validation tests for parameter dataclasses."""

from __future__ import annotations

import unittest

import numpy as np

from lap_time_sim.tire.models import PacejkaParameters
from lap_time_sim.track.models import TrackData
from lap_time_sim.utils.exceptions import ConfigurationError, TrackDataError
from lap_time_sim.vehicle.params import VehicleParameters


class ParameterValidationTests(unittest.TestCase):
    """Unit tests for parameter validation branches."""

    _BASE_TIRE_ARGS = dict(
        B=9.0,
        C=1.2,
        D=1.5,
        E=0.9,
        reference_load=3000.0,
        load_sensitivity=-0.08,
        min_mu_scale=0.4,
    )

    def test_pacejka_validation_rejects_invalid_values(self) -> None:
        """Raise configuration errors for invalid Pacejka coefficients."""
        with self.assertRaises(ConfigurationError):
            PacejkaParameters(**{**self._BASE_TIRE_ARGS, "B": 0.0}).validate()
        with self.assertRaises(ConfigurationError):
            PacejkaParameters(**{**self._BASE_TIRE_ARGS, "C": 0.0}).validate()
        with self.assertRaises(ConfigurationError):
            PacejkaParameters(**{**self._BASE_TIRE_ARGS, "D": 0.0}).validate()
        with self.assertRaises(ConfigurationError):
            PacejkaParameters(**{**self._BASE_TIRE_ARGS, "reference_load": 0.0}).validate()
        with self.assertRaises(ConfigurationError):
            PacejkaParameters(**{**self._BASE_TIRE_ARGS, "min_mu_scale": 0.0}).validate()

    def test_vehicle_validation_rejects_invalid_values(self) -> None:
        """Raise configuration errors for invalid vehicle parameters."""
        base = dict(
            mass=700.0,
            yaw_inertia=1000.0,
            cg_height=0.30,
            wheelbase=3.5,
            front_track=1.6,
            rear_track=1.6,
            front_weight_fraction=0.45,
            cop_position=0.0,
            lift_coefficient=3.0,
            drag_coefficient=0.9,
            frontal_area=1.4,
            roll_rate=4000.0,
            front_spring_rate=150000.0,
            rear_spring_rate=140000.0,
            front_arb_distribution=0.5,
            front_ride_height=0.03,
            rear_ride_height=0.05,
            air_density=1.225,
        )

        with self.assertRaises(ConfigurationError):
            VehicleParameters(**{**base, "mass": 0.0}).validate()
        with self.assertRaises(ConfigurationError):
            VehicleParameters(**{**base, "yaw_inertia": 0.0}).validate()
        with self.assertRaises(ConfigurationError):
            VehicleParameters(**{**base, "wheelbase": 0.0}).validate()
        with self.assertRaises(ConfigurationError):
            VehicleParameters(**{**base, "front_track": 0.0}).validate()
        with self.assertRaises(ConfigurationError):
            VehicleParameters(**{**base, "front_weight_fraction": 0.99}).validate()
        with self.assertRaises(ConfigurationError):
            VehicleParameters(**{**base, "frontal_area": 0.0}).validate()
        with self.assertRaises(ConfigurationError):
            VehicleParameters(**{**base, "front_arb_distribution": 1.5}).validate()
        with self.assertRaises(ConfigurationError):
            VehicleParameters(**{**base, "air_density": 0.0}).validate()

    def test_track_data_validation_rejects_inconsistent_arrays(self) -> None:
        """Raise track-data errors for mismatched array lengths."""
        with self.assertRaises(TrackDataError):
            TrackData(
                x=np.array([0.0, 1.0, 2.0, 3.0]),
                y=np.array([0.0, 0.0, 0.0, 0.0]),
                elevation=np.array([0.0, 0.0, 0.0]),
                banking=np.array([0.0, 0.0, 0.0, 0.0]),
                arc_length=np.array([0.0, 1.0, 2.0, 3.0]),
                heading=np.array([0.0, 0.0, 0.0, 0.0]),
                curvature=np.array([0.0, 0.0, 0.0, 0.0]),
                grade=np.array([0.0, 0.0, 0.0, 0.0]),
            ).validate()

    def test_track_data_validation_rejects_too_few_points(self) -> None:
        """Raise track-data errors for fewer than four points."""
        with self.assertRaises(TrackDataError):
            TrackData(
                x=np.array([0.0, 1.0, 2.0]),
                y=np.array([0.0, 0.0, 0.0]),
                elevation=np.array([0.0, 0.0, 0.0]),
                banking=np.array([0.0, 0.0, 0.0]),
                arc_length=np.array([0.0, 1.0, 2.0]),
                heading=np.array([0.0, 0.0, 0.0]),
                curvature=np.array([0.0, 0.0, 0.0]),
                grade=np.array([0.0, 0.0, 0.0]),
            ).validate()

    def test_track_data_validation_rejects_nonfinite_arc_length(self) -> None:
        """Raise track-data errors for non-finite arc-length entries."""
        with self.assertRaises(TrackDataError):
            TrackData(
                x=np.array([0.0, 1.0, 2.0, 3.0]),
                y=np.array([0.0, 0.0, 0.0, 0.0]),
                elevation=np.array([0.0, 0.0, 0.0, 0.0]),
                banking=np.array([0.0, 0.0, 0.0, 0.0]),
                arc_length=np.array([0.0, np.inf, 2.0, 3.0]),
                heading=np.array([0.0, 0.0, 0.0, 0.0]),
                curvature=np.array([0.0, 0.0, 0.0, 0.0]),
                grade=np.array([0.0, 0.0, 0.0, 0.0]),
            ).validate()

    def test_track_data_validation_rejects_nonmonotonic_arc_length(self) -> None:
        """Raise track-data errors for non-monotonic arc-length arrays."""
        with self.assertRaises(TrackDataError):
            TrackData(
                x=np.array([0.0, 1.0, 2.0, 3.0]),
                y=np.array([0.0, 0.0, 0.0, 0.0]),
                elevation=np.array([0.0, 0.0, 0.0, 0.0]),
                banking=np.array([0.0, 0.0, 0.0, 0.0]),
                arc_length=np.array([0.0, 1.0, 1.0, 3.0]),
                heading=np.array([0.0, 0.0, 0.0, 0.0]),
                curvature=np.array([0.0, 0.0, 0.0, 0.0]),
                grade=np.array([0.0, 0.0, 0.0, 0.0]),
            ).validate()


if __name__ == "__main__":
    unittest.main()
