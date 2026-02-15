"""Simulation configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass

from lap_time_sim.utils.exceptions import ConfigurationError


@dataclass(frozen=True)
class SimulationConfig:
    """Global simulation settings and envelope limits.

    Vehicle-specific force limits are intentionally *not* configured here.
    They belong to the selected `LapTimeVehicleModel` implementation.

    Attributes:
        min_speed_mps: Numerical floor for speed to avoid singular divisions.
        max_speed_mps: Hard speed cap for straight segments.
        lateral_envelope_max_iterations: Maximum fixed-point iterations for the
            lateral-speed envelope solver.
        lateral_envelope_convergence_tol_mps: Early-stop threshold for the
            lateral envelope fixed-point update (`max |v_k - v_{k-1}|`).
        transient_dt_s: Integration step for optional transient refinement.
        enable_transient_refinement: Flag for optional second-pass transient solve.
    """

    min_speed_mps: float = 8.0
    max_speed_mps: float = 115.0
    lateral_envelope_max_iterations: int = 20
    lateral_envelope_convergence_tol_mps: float = 0.1
    transient_dt_s: float = 0.01
    enable_transient_refinement: bool = False

    def validate(self) -> None:
        """Validate solver settings.

        Raises:
            lap_time_sim.utils.exceptions.ConfigurationError: If any solver
                configuration value violates its bound.
        """
        if self.min_speed_mps <= 0.0:
            msg = "min_speed_mps must be positive"
            raise ConfigurationError(msg)
        if self.max_speed_mps <= self.min_speed_mps:
            msg = "max_speed_mps must be greater than min_speed_mps"
            raise ConfigurationError(msg)
        if self.lateral_envelope_max_iterations < 1:
            msg = "lateral_envelope_max_iterations must be at least 1"
            raise ConfigurationError(msg)
        if self.lateral_envelope_convergence_tol_mps <= 0.0:
            msg = "lateral_envelope_convergence_tol_mps must be positive"
            raise ConfigurationError(msg)
        if self.transient_dt_s <= 0.0:
            msg = "transient_dt_s must be positive"
            raise ConfigurationError(msg)
