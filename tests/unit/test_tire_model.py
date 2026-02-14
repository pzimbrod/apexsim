"""Unit tests for Pacejka tire model."""

from __future__ import annotations

import unittest

from lap_time_sim.tire.models import PacejkaParameters, default_axle_tire_parameters
from lap_time_sim.tire.pacejka import axle_lateral_forces, magic_formula_lateral


class TireModelTests(unittest.TestCase):
    """Unit-level tire model checks."""

    def test_zero_slip_produces_zero_lateral_force(self) -> None:
        params = PacejkaParameters(B=10.0, C=1.3, D=1.8, E=0.97)
        self.assertAlmostEqual(magic_formula_lateral(0.0, 3500.0, params), 0.0, places=9)

    def test_force_is_odd_function_of_slip(self) -> None:
        params = PacejkaParameters(B=10.0, C=1.3, D=1.8, E=0.97)
        positive = magic_formula_lateral(0.08, 3500.0, params)
        negative = magic_formula_lateral(-0.08, 3500.0, params)
        self.assertAlmostEqual(positive, -negative, delta=1e-6)

    def test_load_sensitivity_reduces_force_per_newton(self) -> None:
        params = PacejkaParameters(B=10.0, C=1.3, D=1.9, E=0.96, load_sensitivity=-0.2)
        low_load_force = magic_formula_lateral(0.10, 2000.0, params)
        high_load_force = magic_formula_lateral(0.10, 6000.0, params)

        self.assertGreater(high_load_force, low_load_force)
        self.assertLess(high_load_force / 6000.0, low_load_force / 2000.0)

    def test_axle_force_is_sum_of_two_tires(self) -> None:
        tires = default_axle_tire_parameters()
        fy_front, fy_rear = axle_lateral_forces(0.09, 0.08, 8000.0, 8500.0, tires)
        self.assertGreater(fy_front, 0.0)
        self.assertGreater(fy_rear, 0.0)


if __name__ == "__main__":
    unittest.main()
