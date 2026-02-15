"""Tests for integrator and simulation config validation."""

from __future__ import annotations

import unittest

import numpy as np

from lap_time_sim.simulation.config import SimulationConfig
from lap_time_sim.simulation.integrator import rk4_step
from lap_time_sim.utils.exceptions import ConfigurationError


class SimulationHelpersTests(unittest.TestCase):
    """Unit tests for numerical helper modules."""

    def test_rk4_step_matches_linear_solution(self) -> None:
        # dy/dt = -2y has exact solution y(t+dt) = y * exp(-2dt).
        def rhs(_: float, state: np.ndarray) -> np.ndarray:
            return -2.0 * state

        initial = np.array([1.0], dtype=float)
        dt = 0.1
        stepped = rk4_step(rhs, 0.0, initial, dt)
        expected = np.exp(-2.0 * dt)
        self.assertAlmostEqual(float(stepped[0]), float(expected), delta=1e-5)

    def test_simulation_config_validation_raises_for_invalid_values(self) -> None:
        with self.assertRaises(ConfigurationError):
            SimulationConfig(max_drive_accel_mps2=0.0).validate()
        with self.assertRaises(ConfigurationError):
            SimulationConfig(max_brake_accel_mps2=0.0).validate()
        with self.assertRaises(ConfigurationError):
            SimulationConfig(min_speed_mps=0.0).validate()
        with self.assertRaises(ConfigurationError):
            SimulationConfig(min_speed_mps=20.0, max_speed_mps=10.0).validate()
        with self.assertRaises(ConfigurationError):
            SimulationConfig(transient_dt_s=0.0).validate()
        with self.assertRaises(ConfigurationError):
            SimulationConfig(lateral_envelope_max_iterations=0).validate()
        with self.assertRaises(ConfigurationError):
            SimulationConfig(lateral_envelope_convergence_tol_mps=0.0).validate()


if __name__ == "__main__":
    unittest.main()
