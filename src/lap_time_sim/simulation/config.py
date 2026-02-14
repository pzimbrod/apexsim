"""Simulation configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass

from lap_time_sim.utils.exceptions import ConfigurationError


@dataclass(frozen=True)
class SimulationConfig:
    """Global simulation settings and envelope limits."""

    max_drive_accel_mps2: float = 8.0
    max_brake_accel_mps2: float = 16.0
    min_speed_mps: float = 8.0
    max_speed_mps: float = 115.0
    transient_dt_s: float = 0.01
    enable_transient_refinement: bool = False

    def validate(self) -> None:
        """Validate solver settings."""
        if self.max_drive_accel_mps2 <= 0.0:
            msg = "max_drive_accel_mps2 must be positive"
            raise ConfigurationError(msg)
        if self.max_brake_accel_mps2 <= 0.0:
            msg = "max_brake_accel_mps2 must be positive"
            raise ConfigurationError(msg)
        if self.min_speed_mps <= 0.0:
            msg = "min_speed_mps must be positive"
            raise ConfigurationError(msg)
        if self.max_speed_mps <= self.min_speed_mps:
            msg = "max_speed_mps must be greater than min_speed_mps"
            raise ConfigurationError(msg)
        if self.transient_dt_s <= 0.0:
            msg = "transient_dt_s must be positive"
            raise ConfigurationError(msg)
