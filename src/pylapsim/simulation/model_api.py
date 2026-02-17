"""Interfaces between lap-time solver and vehicle models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ModelDiagnostics:
    """Per-track-point diagnostics exposed by a vehicle model.

    Args:
        yaw_moment: Net yaw moment at the operating point [N*m].
        front_axle_load: Front-axle normal load [N].
        rear_axle_load: Rear-axle normal load [N].
        power: Instantaneous tractive power [W].
    """

    yaw_moment: float
    front_axle_load: float
    rear_axle_load: float
    power: float


class VehicleModel(Protocol):
    """Protocol required by the quasi-steady lap-time solver.

    Any vehicle model (bicycle, point-mass, twin-track, ...) can be used by the
    simulation pipeline as long as it implements this interface.
    """

    def validate(self) -> None:
        """Validate model parameters and configuration.

        Raises:
            pylapsim.utils.exceptions.ConfigurationError: If model parameters
                violate required physical or numerical constraints.
        """
        ...

    def lateral_accel_limit(self, speed: float, banking: float) -> float:
        """Return lateral acceleration capability at the given operating point.

        Args:
            speed: Vehicle speed at the queried operating point [m/s].
            banking: Track banking angle at the queried point [rad].

        Returns:
            Maximum quasi-steady lateral acceleration magnitude [m/s^2].
        """
        ...

    def max_longitudinal_accel(
        self,
        speed: float,
        lateral_accel_required: float,
        grade: float,
        banking: float,
    ) -> float:
        """Return net forward acceleration limit along the path tangent.

        Args:
            speed: Vehicle speed [m/s].
            lateral_accel_required: Required lateral acceleration magnitude [m/s^2].
            grade: Track grade defined as ``dz/ds``.
            banking: Track banking angle [rad].

        Returns:
            Maximum net acceleration along the path tangent [m/s^2].
        """
        ...

    def max_longitudinal_decel(
        self,
        speed: float,
        lateral_accel_required: float,
        grade: float,
        banking: float,
    ) -> float:
        """Return available deceleration magnitude along the path tangent.

        Args:
            speed: Vehicle speed [m/s].
            lateral_accel_required: Required lateral acceleration magnitude [m/s^2].
            grade: Track grade defined as ``dz/ds``.
            banking: Track banking angle [rad].

        Returns:
            Maximum non-negative deceleration magnitude along path tangent [m/s^2].
        """
        ...

    def diagnostics(
        self,
        speed: float,
        longitudinal_accel: float,
        lateral_accel: float,
        curvature: float,
    ) -> ModelDiagnostics:
        """Return diagnostic quantities used by analysis and plotting.

        Args:
            speed: Vehicle speed [m/s].
            longitudinal_accel: Net longitudinal acceleration along path tangent [m/s^2].
            lateral_accel: Lateral acceleration [m/s^2].
            curvature: Signed track curvature at the sample point [1/m].

        Returns:
            Diagnostic signals for post-processing and visualization.
        """
        ...


class VehicleModelBase(ABC):
    """Nominal OOP base class for solver-compatible vehicle models.

    The solver continues to depend on :class:`VehicleModel` (Protocol)
    for structural flexibility. Concrete library backends can additionally
    subclass this abstract base class to make inheritance-based contracts and
    code sharing explicit.
    """

    @abstractmethod
    def validate(self) -> None:
        """Validate model parameters and configuration.

        Raises:
            pylapsim.utils.exceptions.ConfigurationError: If model parameters
                violate required physical or numerical constraints.
        """

    @abstractmethod
    def lateral_accel_limit(self, speed: float, banking: float) -> float:
        """Return lateral acceleration capability at the given operating point.

        Args:
            speed: Vehicle speed at the queried operating point [m/s].
            banking: Track banking angle at the queried point [rad].

        Returns:
            Maximum quasi-steady lateral acceleration magnitude [m/s^2].
        """

    @abstractmethod
    def max_longitudinal_accel(
        self,
        speed: float,
        lateral_accel_required: float,
        grade: float,
        banking: float,
    ) -> float:
        """Return net forward acceleration limit along the path tangent.

        Args:
            speed: Vehicle speed [m/s].
            lateral_accel_required: Required lateral acceleration magnitude [m/s^2].
            grade: Track grade defined as ``dz/ds``.
            banking: Track banking angle [rad].

        Returns:
            Maximum net acceleration along the path tangent [m/s^2].
        """

    @abstractmethod
    def max_longitudinal_decel(
        self,
        speed: float,
        lateral_accel_required: float,
        grade: float,
        banking: float,
    ) -> float:
        """Return available deceleration magnitude along the path tangent.

        Args:
            speed: Vehicle speed [m/s].
            lateral_accel_required: Required lateral acceleration magnitude [m/s^2].
            grade: Track grade defined as ``dz/ds``.
            banking: Track banking angle [rad].

        Returns:
            Maximum non-negative deceleration magnitude along path tangent [m/s^2].
        """

    @abstractmethod
    def diagnostics(
        self,
        speed: float,
        longitudinal_accel: float,
        lateral_accel: float,
        curvature: float,
    ) -> ModelDiagnostics:
        """Return diagnostic quantities used by analysis and plotting.

        Args:
            speed: Vehicle speed [m/s].
            longitudinal_accel: Net longitudinal acceleration along path tangent [m/s^2].
            lateral_accel: Lateral acceleration [m/s^2].
            curvature: Signed track curvature at the sample point [1/m].

        Returns:
            Diagnostic signals for post-processing and visualization.
        """
