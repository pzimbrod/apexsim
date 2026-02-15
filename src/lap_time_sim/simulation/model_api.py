"""Interfaces between lap-time solver and vehicle models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class VehicleModelDiagnostics:
    """Per-track-point diagnostics exposed by a vehicle model."""

    yaw_moment_nm: float
    front_axle_load_n: float
    rear_axle_load_n: float
    power_w: float


class LapTimeVehicleModel(Protocol):
    """Protocol required by the quasi-steady lap-time solver.

    Any vehicle model (bicycle, point-mass, twin-track, ...) can be used by the
    simulation pipeline as long as it implements this interface.
    """

    def validate(self) -> None:
        """Validate model parameters and configuration."""
        ...

    def lateral_accel_limit(self, speed_mps: float, banking_rad: float) -> float:
        """Return lateral acceleration capability at the given operating point."""
        ...

    def max_longitudinal_accel(
        self,
        speed_mps: float,
        ay_required_mps2: float,
        grade: float,
        banking_rad: float,
    ) -> float:
        """Return net forward acceleration limit along the path tangent."""
        ...

    def max_longitudinal_decel(
        self,
        speed_mps: float,
        ay_required_mps2: float,
        grade: float,
        banking_rad: float,
    ) -> float:
        """Return available deceleration magnitude along the path tangent."""
        ...

    def diagnostics(
        self,
        speed_mps: float,
        ax_mps2: float,
        ay_mps2: float,
        curvature_1pm: float,
    ) -> VehicleModelDiagnostics:
        """Return diagnostic quantities used by analysis and plotting."""
        ...
