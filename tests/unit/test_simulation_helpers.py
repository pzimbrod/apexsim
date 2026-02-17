"""Tests for integrator and simulation config validation."""

from __future__ import annotations

import unittest

import numpy as np

from pylapsim.simulation.config import (
    NumericsConfig,
    RuntimeConfig,
    SimulationConfig,
    build_simulation_config,
)
from pylapsim.simulation.integrator import rk4_step
from pylapsim.utils.exceptions import ConfigurationError


class SimulationHelpersTests(unittest.TestCase):
    """Unit tests for numerical helper modules."""

    def test_rk4_step_matches_linear_solution(self) -> None:
        """Match RK4 integration output against an exponential reference."""
        # dy/dt = -2y has exact solution y(t+dt) = y * exp(-2dt).
        def rhs(_: float, state: np.ndarray) -> np.ndarray:
            return -2.0 * state

        initial = np.array([1.0], dtype=float)
        dt = 0.1
        stepped = rk4_step(rhs, 0.0, initial, dt)
        expected = np.exp(-2.0 * dt)
        self.assertAlmostEqual(float(stepped[0]), float(expected), delta=1e-5)

    def test_simulation_config_validation_raises_for_invalid_values(self) -> None:
        """Raise configuration errors for invalid simulation settings."""
        with self.assertRaises(ConfigurationError):
            SimulationConfig(
                runtime=RuntimeConfig(max_speed=115.0, enable_transient_refinement=False),
                numerics=NumericsConfig(
                    min_speed=0.0,
                    lateral_envelope_max_iterations=20,
                    lateral_envelope_convergence_tolerance=0.1,
                    transient_step=0.01,
                ),
            ).validate()

    def test_build_simulation_config_uses_stable_default_numerics(self) -> None:
        """Build configuration with default numerics when omitted."""
        config = build_simulation_config(max_speed=115.0)
        self.assertEqual(config.numerics, NumericsConfig())
        self.assertFalse(config.runtime.enable_transient_refinement)
        with self.assertRaises(ConfigurationError):
            SimulationConfig(
                runtime=RuntimeConfig(max_speed=10.0, enable_transient_refinement=False),
                numerics=NumericsConfig(
                    min_speed=20.0,
                    lateral_envelope_max_iterations=20,
                    lateral_envelope_convergence_tolerance=0.1,
                    transient_step=0.01,
                ),
            ).validate()
        with self.assertRaises(ConfigurationError):
            SimulationConfig(
                runtime=RuntimeConfig(max_speed=115.0, enable_transient_refinement=False),
                numerics=NumericsConfig(
                    min_speed=8.0,
                    lateral_envelope_max_iterations=20,
                    lateral_envelope_convergence_tolerance=0.1,
                    transient_step=0.0,
                ),
            ).validate()
        with self.assertRaises(ConfigurationError):
            SimulationConfig(
                runtime=RuntimeConfig(max_speed=115.0, enable_transient_refinement=False),
                numerics=NumericsConfig(
                    min_speed=8.0,
                    lateral_envelope_max_iterations=0,
                    lateral_envelope_convergence_tolerance=0.1,
                    transient_step=0.01,
                ),
            ).validate()
        with self.assertRaises(ConfigurationError):
            SimulationConfig(
                runtime=RuntimeConfig(max_speed=115.0, enable_transient_refinement=False),
                numerics=NumericsConfig(
                    min_speed=8.0,
                    lateral_envelope_max_iterations=20,
                    lateral_envelope_convergence_tolerance=0.0,
                    transient_step=0.01,
                ),
            ).validate()


if __name__ == "__main__":
    unittest.main()
