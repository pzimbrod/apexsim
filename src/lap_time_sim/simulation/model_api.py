"""Interfaces between lap-time solver and vehicle models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class VehicleModelDiagnostics:
    """Per-track-point diagnostics exposed by a vehicle model.

    Args:
        yaw_moment_nm: Net yaw moment at the operating point (N*m).
        front_axle_load_n: Front-axle normal load (N).
        rear_axle_load_n: Rear-axle normal load (N).
        power_w: Instantaneous tractive power (W).
    """

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
        """Validate model parameters and configuration.

        Raises:
            lap_time_sim.utils.exceptions.ConfigurationError: If model parameters
                violate required physical or numerical constraints.
        """
        ...

    def lateral_accel_limit(self, speed_mps: float, banking_rad: float) -> float:
        """Return lateral acceleration capability at the given operating point.

        Args:
            speed_mps: Vehicle speed at the queried operating point in m/s.
            banking_rad: Track banking angle at the queried point in rad.

        Returns:
            Maximum quasi-steady lateral acceleration magnitude in m/s^2.
        """
        ...

    def max_longitudinal_accel(
        self,
        speed_mps: float,
        ay_required_mps2: float,
        grade: float,
        banking_rad: float,
    ) -> float:
        """Return net forward acceleration limit along the path tangent.

        Args:
            speed_mps: Vehicle speed in m/s.
            ay_required_mps2: Required lateral acceleration magnitude in m/s^2.
            grade: Track grade defined as ``dz/ds``.
            banking_rad: Track banking angle in rad.

        Returns:
            Maximum net acceleration along the path tangent in m/s^2.
        """
        ...

    def max_longitudinal_decel(
        self,
        speed_mps: float,
        ay_required_mps2: float,
        grade: float,
        banking_rad: float,
    ) -> float:
        """Return available deceleration magnitude along the path tangent.

        Args:
            speed_mps: Vehicle speed in m/s.
            ay_required_mps2: Required lateral acceleration magnitude in m/s^2.
            grade: Track grade defined as ``dz/ds``.
            banking_rad: Track banking angle in rad.

        Returns:
            Maximum non-negative deceleration magnitude along path tangent in m/s^2.
        """
        ...

    def diagnostics(
        self,
        speed_mps: float,
        ax_mps2: float,
        ay_mps2: float,
        curvature_1pm: float,
    ) -> VehicleModelDiagnostics:
        """Return diagnostic quantities used by analysis and plotting.

        Args:
            speed_mps: Vehicle speed in m/s.
            ax_mps2: Net longitudinal acceleration along path tangent in m/s^2.
            ay_mps2: Lateral acceleration in m/s^2.
            curvature_1pm: Signed track curvature at the sample point in 1/m.

        Returns:
            Diagnostic signals for post-processing and visualization.
        """
        ...
