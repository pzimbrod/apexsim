"""Unit tests for Pacejka tire model."""

from __future__ import annotations

import unittest

import numpy as np

from apexsim.tire.models import PacejkaParameters, default_axle_tire_parameters
from apexsim.tire.pacejka import axle_lateral_forces, magic_formula_lateral
from apexsim.utils.exceptions import ConfigurationError


class TireModelTests(unittest.TestCase):
    """Unit-level tire model checks."""

    _BASE_TIRE_ARGS = dict(
        B=10.0,
        C=1.3,
        D=5400.0,
        E=0.97,
        reference_load=3000.0,
        load_sensitivity=-0.08,
        min_mu_scale=0.4,
    )

    def test_zero_slip_produces_zero_lateral_force(self) -> None:
        """Produce zero lateral force at zero slip angle."""
        params = PacejkaParameters(**self._BASE_TIRE_ARGS)
        self.assertAlmostEqual(magic_formula_lateral(0.0, 3500.0, params), 0.0, places=9)

    def test_force_is_odd_function_of_slip(self) -> None:
        """Preserve odd symmetry with respect to slip-angle sign."""
        params = PacejkaParameters(**self._BASE_TIRE_ARGS)
        positive = magic_formula_lateral(0.08, 3500.0, params)
        negative = magic_formula_lateral(-0.08, 3500.0, params)
        self.assertAlmostEqual(positive, -negative, delta=1e-6)

    def test_load_sensitivity_reduces_force_per_newton(self) -> None:
        """Reduce force-per-load as normal load increases for negative sensitivity."""
        params = PacejkaParameters(
            **{
                **self._BASE_TIRE_ARGS,
                "D": 5700.0,
                "E": 0.96,
                "load_sensitivity": -0.2,
            }
        )
        low_load_force = magic_formula_lateral(0.10, 2000.0, params)
        high_load_force = magic_formula_lateral(0.10, 6000.0, params)

        self.assertGreater(high_load_force, low_load_force)
        self.assertLess(high_load_force / 6000.0, low_load_force / 2000.0)

    def test_axle_force_is_sum_of_two_tires(self) -> None:
        """Return positive axle forces at representative positive slip angles."""
        tires = default_axle_tire_parameters()
        fy_front, fy_rear = axle_lateral_forces(0.09, 0.08, 8000.0, 8500.0, tires)
        self.assertGreater(fy_front, 0.0)
        self.assertGreater(fy_rear, 0.0)

    def test_magic_formula_supports_vectorized_inputs(self) -> None:
        """Return array output when slip angle and load inputs are arrays."""
        params = PacejkaParameters(**self._BASE_TIRE_ARGS)
        slip = np.array([0.0, 0.05, -0.05], dtype=float)
        load = np.array([3000.0, 3200.0, 2800.0], dtype=float)
        output = magic_formula_lateral(slip, load, params)
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(output.shape, slip.shape)

    def test_axle_validation_rejects_invalid_rear_parameters(self) -> None:
        """Raise when rear axle tire parameters are invalid."""
        tires = default_axle_tire_parameters()
        bad_rear = PacejkaParameters(
            B=0.0,
            C=tires.rear.C,
            D=tires.rear.D,
            E=tires.rear.E,
            reference_load=tires.rear.reference_load,
            load_sensitivity=tires.rear.load_sensitivity,
            min_mu_scale=tires.rear.min_mu_scale,
        )
        with self.assertRaises(ConfigurationError):
            bad_axle_params = type(tires)(tires.front, bad_rear)
            axle_lateral_forces(0.05, 0.05, 7000.0, 7000.0, axle_params=bad_axle_params)


if __name__ == "__main__":
    unittest.main()
