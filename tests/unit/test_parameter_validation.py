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

    def test_pacejka_validation_rejects_invalid_values(self) -> None:
        """Raise configuration errors for invalid Pacejka coefficients."""
        with self.assertRaises(ConfigurationError):
            PacejkaParameters(B=0.0, C=1.2, D=1.5, E=0.9).validate()
        with self.assertRaises(ConfigurationError):
            PacejkaParameters(B=9.0, C=0.0, D=1.5, E=0.9).validate()
        with self.assertRaises(ConfigurationError):
            PacejkaParameters(B=9.0, C=1.2, D=0.0, E=0.9).validate()
        with self.assertRaises(ConfigurationError):
            PacejkaParameters(B=9.0, C=1.2, D=1.5, E=0.9, fz_reference_n=0.0).validate()

    def test_vehicle_validation_rejects_invalid_values(self) -> None:
        """Raise configuration errors for invalid vehicle parameters."""
        base = dict(
            mass_kg=700.0,
            yaw_inertia_kgm2=1000.0,
            h_cg_m=0.30,
            wheelbase_m=3.5,
            track_front_m=1.6,
            track_rear_m=1.6,
            static_front_weight_fraction=0.45,
            cop_position_m=0.0,
            c_l=3.0,
            c_d=0.9,
            frontal_area_m2=1.4,
            roll_rate_nm_per_deg=4000.0,
            spring_rate_front_npm=150000.0,
            spring_rate_rear_npm=140000.0,
            arb_distribution_front=0.5,
            ride_height_front_m=0.03,
            ride_height_rear_m=0.05,
        )

        with self.assertRaises(ConfigurationError):
            VehicleParameters(**{**base, "mass_kg": 0.0}).validate()
        with self.assertRaises(ConfigurationError):
            VehicleParameters(**{**base, "wheelbase_m": 0.0}).validate()
        with self.assertRaises(ConfigurationError):
            VehicleParameters(**{**base, "static_front_weight_fraction": 0.99}).validate()
        with self.assertRaises(ConfigurationError):
            VehicleParameters(**{**base, "arb_distribution_front": 1.5}).validate()

    def test_track_data_validation_rejects_inconsistent_arrays(self) -> None:
        """Raise track-data errors for mismatched array lengths."""
        with self.assertRaises(TrackDataError):
            TrackData(
                x_m=np.array([0.0, 1.0, 2.0, 3.0]),
                y_m=np.array([0.0, 0.0, 0.0, 0.0]),
                elevation_m=np.array([0.0, 0.0, 0.0]),
                banking_rad=np.array([0.0, 0.0, 0.0, 0.0]),
                s_m=np.array([0.0, 1.0, 2.0, 3.0]),
                heading_rad=np.array([0.0, 0.0, 0.0, 0.0]),
                curvature_1pm=np.array([0.0, 0.0, 0.0, 0.0]),
                grade=np.array([0.0, 0.0, 0.0, 0.0]),
            ).validate()


if __name__ == "__main__":
    unittest.main()
