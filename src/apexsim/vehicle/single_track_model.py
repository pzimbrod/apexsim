"""Solver-facing single-track vehicle model."""

from __future__ import annotations

from dataclasses import dataclass

from apexsim.tire.models import AxleTireParameters
from apexsim.utils.exceptions import ConfigurationError
from apexsim.vehicle._model_base import EnvelopeVehicleModel
from apexsim.vehicle._physics_primitives import EnvelopePhysics
from apexsim.vehicle._single_track_backends import (
    SingleTrackNumbaBackendMixin,
    SingleTrackTorchBackendMixin,
)
from apexsim.vehicle._single_track_physics import SingleTrackPhysicalMixin
from apexsim.vehicle.params import VehicleParameters
from apexsim.vehicle.single_track.dynamics import SingleTrackDynamicsModel

DEFAULT_MIN_LATERAL_ACCEL_LIMIT = 0.5
DEFAULT_LATERAL_LIMIT_MAX_ITERATIONS = 12
DEFAULT_LATERAL_LIMIT_CONVERGENCE_TOLERANCE = 0.05


@dataclass(frozen=True)
class _SingleTrackLateralPhysics:
    """Single-track-specific lateral-envelope approximation inputs.

    Args:
        peak_slip_angle: Quasi-steady peak slip angle used to evaluate tire
            lateral force capability in the envelope iteration [rad].
    """

    peak_slip_angle: float

    def validate(self) -> None:
        """Validate single-track-specific lateral approximation inputs.

        Raises:
            apexsim.utils.exceptions.ConfigurationError: If values are not
                strictly positive.
        """
        if self.peak_slip_angle <= 0.0:
            msg = "peak_slip_angle must be positive"
            raise ConfigurationError(msg)


@dataclass(frozen=True)
class SingleTrackPhysics:
    """Physical and model-level inputs for the single-track solver model.

    Args:
        max_drive_accel: Maximum forward tire acceleration on flat road and zero
            lateral demand, excluding drag and grade [m/s^2].
        max_brake_accel: Maximum braking deceleration magnitude on flat road and
            zero lateral demand, excluding drag and grade [m/s^2].
        peak_slip_angle: Quasi-steady peak slip angle used to evaluate tire
            lateral force capability in the envelope iteration [rad].
        max_steer_angle: Maximum absolute steering angle used in transient
            control optimization [rad].
        max_steer_rate: Maximum absolute steering-rate magnitude used in
            transient control optimization [rad/s].
    """

    max_drive_accel: float = 8.0
    max_brake_accel: float = 16.0
    peak_slip_angle: float = 0.12
    max_steer_angle: float = 0.6
    max_steer_rate: float = 4.0

    @property
    def envelope(self) -> EnvelopePhysics:
        """Return shared longitudinal envelope limits.

        Returns:
            Internal shared envelope-limit representation used by multiple
            solver model families.
        """
        return EnvelopePhysics(
            max_drive_accel=self.max_drive_accel,
            max_brake_accel=self.max_brake_accel,
        )

    @property
    def single_track_lateral(self) -> _SingleTrackLateralPhysics:
        """Return single-track-specific lateral approximation inputs.

        Returns:
            Internal single-track-specific lateral-envelope representation.
        """
        return _SingleTrackLateralPhysics(peak_slip_angle=self.peak_slip_angle)

    def validate(self) -> None:
        """Validate physical adapter parameters.

        Raises:
            apexsim.utils.exceptions.ConfigurationError: If limits are not
                strictly positive.
        """
        self.envelope.validate()
        self.single_track_lateral.validate()
        if self.max_steer_angle <= 0.0:
            msg = "max_steer_angle must be positive"
            raise ConfigurationError(msg)
        if self.max_steer_rate <= 0.0:
            msg = "max_steer_rate must be positive"
            raise ConfigurationError(msg)


@dataclass(frozen=True)
class SingleTrackNumerics:
    """Numerical controls for the single-track solver model.

    Args:
        min_lateral_accel_limit: Lower bound for lateral-acceleration iteration
            to avoid degenerate starts [m/s^2].
        lateral_limit_max_iterations: Maximum fixed-point iterations for lateral
            acceleration limit estimation.
        lateral_limit_convergence_tolerance: Convergence threshold for lateral
            acceleration fixed-point updates [m/s^2].
    """

    min_lateral_accel_limit: float = DEFAULT_MIN_LATERAL_ACCEL_LIMIT
    lateral_limit_max_iterations: int = DEFAULT_LATERAL_LIMIT_MAX_ITERATIONS
    lateral_limit_convergence_tolerance: float = DEFAULT_LATERAL_LIMIT_CONVERGENCE_TOLERANCE

    def validate(self) -> None:
        """Validate numerical settings for the adapter.

        Raises:
            apexsim.utils.exceptions.ConfigurationError: If numerical values
                violate bounds needed for robust convergence.
        """
        if self.min_lateral_accel_limit <= 0.0:
            msg = "min_lateral_accel_limit must be positive"
            raise ConfigurationError(msg)
        if self.lateral_limit_max_iterations < 1:
            msg = "lateral_limit_max_iterations must be at least 1"
            raise ConfigurationError(msg)
        if self.lateral_limit_convergence_tolerance <= 0.0:
            msg = "lateral_limit_convergence_tolerance must be positive"
            raise ConfigurationError(msg)


class SingleTrackModel(
    SingleTrackTorchBackendMixin,
    SingleTrackNumbaBackendMixin,
    SingleTrackPhysicalMixin,
    EnvelopeVehicleModel,
):
    """Vehicle-model API implementation for the single-track dynamics backend.

    The class is intentionally composed from separate OOP layers:

    - ``SingleTrackPhysicalMixin`` for backend-agnostic single-track physics,
    - ``SingleTrackNumbaBackendMixin`` for numba-specific solver adapter API,
    - ``SingleTrackTorchBackendMixin`` for torch-specific solver adapter API.
    """

    def __init__(
        self,
        vehicle: VehicleParameters,
        tires: AxleTireParameters,
        physics: SingleTrackPhysics,
        numerics: SingleTrackNumerics,
    ) -> None:
        """Initialize single-track-backed solver adapter.

        Args:
            vehicle: Vehicle parameterization for dynamics and aero.
            tires: Front/rear Pacejka tire coefficients.
            physics: Physical model inputs for the adapter.
            numerics: Numerical controls for iterative envelope solving.

        Raises:
            apexsim.utils.exceptions.ConfigurationError: If any provided
                parameter set is invalid.
        """
        self.tires = tires
        self.physics = physics
        self.numerics = numerics
        self._single_track_lateral_physics = physics.single_track_lateral
        self._dynamics = SingleTrackDynamicsModel(vehicle, tires)
        super().__init__(vehicle=vehicle, envelope_physics=physics.envelope)
        self.validate()

    def _validate_backend(self) -> None:
        """Validate single-track-specific model configuration.

        Raises:
            apexsim.utils.exceptions.ConfigurationError: If tire,
                single-track physics, or numerical settings violate constraints.
        """
        self.tires.validate()
        self._single_track_lateral_physics.validate()
        self.numerics.validate()


def build_single_track_model(
    vehicle: VehicleParameters,
    tires: AxleTireParameters,
    physics: SingleTrackPhysics | None = None,
    numerics: SingleTrackNumerics | None = None,
) -> SingleTrackModel:
    """Build a single-track solver model with sensible numerical defaults.

    Args:
        vehicle: Vehicle parameterization for dynamics and aero.
        tires: Front/rear Pacejka tire coefficients.
        physics: Optional physical model inputs for longitudinal and lateral
            limits. Defaults to :class:`SingleTrackPhysics`.
        numerics: Optional numerical controls. Defaults to :class:`SingleTrackNumerics`.

    Returns:
        Fully validated solver-facing single-track model.
    """
    return SingleTrackModel(
        vehicle=vehicle,
        tires=tires,
        physics=physics or SingleTrackPhysics(),
        numerics=numerics or SingleTrackNumerics(),
    )
