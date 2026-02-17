"""Solver-facing bicycle vehicle model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lap_time_sim.simulation.model_api import VehicleModelDiagnostics
from lap_time_sim.tire.models import AxleTireParameters
from lap_time_sim.tire.pacejka import magic_formula_lateral
from lap_time_sim.utils.constants import GRAVITY, SMALL_EPS
from lap_time_sim.utils.exceptions import ConfigurationError
from lap_time_sim.vehicle._physics_primitives import EnvelopePhysics
from lap_time_sim.vehicle.aero import aero_forces
from lap_time_sim.vehicle.bicycle_dynamics import BicycleDynamicsModel, ControlInput, VehicleState
from lap_time_sim.vehicle.load_transfer import estimate_normal_loads
from lap_time_sim.vehicle.params import VehicleParameters

DEFAULT_MIN_LATERAL_ACCEL_LIMIT = 0.5
DEFAULT_LATERAL_LIMIT_MAX_ITERATIONS = 12
DEFAULT_LATERAL_LIMIT_CONVERGENCE_TOLERANCE = 0.05


@dataclass(frozen=True)
class _BicycleLateralPhysics:
    """Bicycle-specific lateral-envelope approximation inputs.

    Args:
        peak_slip_angle: Quasi-steady peak slip angle used to evaluate tire
            lateral force capability in the envelope iteration [rad].
    """

    peak_slip_angle: float

    def validate(self) -> None:
        """Validate bicycle-specific lateral approximation inputs.

        Raises:
            lap_time_sim.utils.exceptions.ConfigurationError: If values are not
                strictly positive.
        """
        if self.peak_slip_angle <= 0.0:
            msg = "peak_slip_angle must be positive"
            raise ConfigurationError(msg)


@dataclass(frozen=True)
class BicyclePhysics:
    """Physical and model-level inputs for the bicycle solver model.

    Args:
        max_drive_accel: Maximum forward tire acceleration on flat road and zero
            lateral demand, excluding drag and grade [m/s^2].
        max_brake_accel: Maximum braking deceleration magnitude on flat road and
            zero lateral demand, excluding drag and grade [m/s^2].
        peak_slip_angle: Quasi-steady peak slip angle used to evaluate tire
            lateral force capability in the envelope iteration [rad].
    """

    max_drive_accel: float = 8.0
    max_brake_accel: float = 16.0
    peak_slip_angle: float = 0.12

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
    def bicycle_lateral(self) -> _BicycleLateralPhysics:
        """Return bicycle-specific lateral approximation inputs.

        Returns:
            Internal bicycle-specific lateral-envelope representation.
        """
        return _BicycleLateralPhysics(peak_slip_angle=self.peak_slip_angle)

    def validate(self) -> None:
        """Validate physical adapter parameters.

        Raises:
            lap_time_sim.utils.exceptions.ConfigurationError: If limits are not
                strictly positive.
        """
        self.envelope.validate()
        self.bicycle_lateral.validate()


@dataclass(frozen=True)
class BicycleNumerics:
    """Numerical controls for the bicycle solver model.

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
            lap_time_sim.utils.exceptions.ConfigurationError: If numerical values
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


class BicycleModel:
    """Vehicle-model API implementation for the bicycle dynamics backend."""

    def __init__(
        self,
        vehicle: VehicleParameters,
        tires: AxleTireParameters,
        physics: BicyclePhysics,
        numerics: BicycleNumerics,
    ) -> None:
        """Initialize bicycle-backed solver adapter.

        Args:
            vehicle: Vehicle parameterization for dynamics and aero.
            tires: Front/rear Pacejka tire coefficients.
            physics: Physical model inputs for the adapter.
            numerics: Numerical controls for iterative envelope solving.

        Raises:
            lap_time_sim.utils.exceptions.ConfigurationError: If any provided
                parameter set is invalid.
        """
        self.vehicle = vehicle
        self.tires = tires
        self.physics = physics
        self.numerics = numerics
        self._envelope_physics = physics.envelope
        self._bicycle_lateral_physics = physics.bicycle_lateral
        self._dynamics = BicycleDynamicsModel(vehicle, tires)
        self.validate()

    def validate(self) -> None:
        """Validate model configuration and parameterization.

        Raises:
            lap_time_sim.utils.exceptions.ConfigurationError: If vehicle, tire,
                or adapter configuration values violate constraints.
        """
        self.vehicle.validate()
        self.tires.validate()
        self.physics.validate()
        self.numerics.validate()

    def lateral_accel_limit(self, speed: float, banking: float) -> float:
        """Estimate lateral acceleration capacity for the operating point.

        Args:
            speed: Vehicle speed [m/s].
            banking: Track banking angle [rad].

        Returns:
            Quasi-steady lateral acceleration limit [m/s^2].
        """
        ay_banking = GRAVITY * float(np.sin(banking))
        ay_estimate = self.numerics.min_lateral_accel_limit

        for _ in range(self.numerics.lateral_limit_max_iterations):
            loads = estimate_normal_loads(
                self.vehicle,
                speed=speed,
                longitudinal_accel=0.0,
                lateral_accel=ay_estimate,
            )
            fz_front_tire = max(loads.front_axle_load / 2.0, SMALL_EPS)
            fz_rear_tire = max(loads.rear_axle_load / 2.0, SMALL_EPS)

            fy_front = 2.0 * float(
                magic_formula_lateral(
                    self._bicycle_lateral_physics.peak_slip_angle,
                    fz_front_tire,
                    self.tires.front,
                )
            )
            fy_rear = 2.0 * float(
                magic_formula_lateral(
                    self._bicycle_lateral_physics.peak_slip_angle,
                    fz_rear_tire,
                    self.tires.rear,
                )
            )
            ay_tire = (fy_front + fy_rear) / self.vehicle.mass
            ay_next = max(self.numerics.min_lateral_accel_limit, ay_tire + ay_banking)

            if (
                abs(ay_next - ay_estimate)
                <= self.numerics.lateral_limit_convergence_tolerance
            ):
                ay_estimate = ay_next
                break
            ay_estimate = ay_next

        return float(ay_estimate)

    def _friction_circle_scale(
        self,
        lateral_accel_required: float,
        lateral_accel_limit: float,
    ) -> float:
        """Compute remaining longitudinal utilization from friction-circle usage.

        Args:
            lateral_accel_required: Required lateral acceleration magnitude [m/s^2].
            lateral_accel_limit: Available lateral acceleration limit [m/s^2].

        Returns:
            Scalar in ``[0, 1]`` reducing longitudinal capability.
        """
        if lateral_accel_limit <= SMALL_EPS:
            return 0.0
        usage = min(abs(lateral_accel_required) / lateral_accel_limit, 1.0)
        return float(np.sqrt(max(0.0, 1.0 - usage * usage)))

    def max_longitudinal_accel(
        self,
        speed: float,
        lateral_accel_required: float,
        grade: float,
        banking: float,
    ) -> float:
        """Compute net forward acceleration limit along path tangent.

        Args:
            speed: Vehicle speed [m/s].
            lateral_accel_required: Required lateral acceleration magnitude [m/s^2].
            grade: Track grade defined as ``dz/ds``.
            banking: Track banking angle [rad].

        Returns:
            Net forward acceleration along path tangent [m/s^2].
        """
        ay_limit = self.lateral_accel_limit(speed, banking)
        circle_scale = self._friction_circle_scale(lateral_accel_required, ay_limit)

        tire_accel = self._envelope_physics.max_drive_accel * circle_scale
        drag_accel = aero_forces(self.vehicle, speed).drag / self.vehicle.mass
        grade_accel = GRAVITY * grade
        return float(tire_accel - drag_accel - grade_accel)

    def max_longitudinal_decel(
        self,
        speed: float,
        lateral_accel_required: float,
        grade: float,
        banking: float,
    ) -> float:
        """Compute available deceleration magnitude along path tangent.

        Args:
            speed: Vehicle speed [m/s].
            lateral_accel_required: Required lateral acceleration magnitude [m/s^2].
            grade: Track grade defined as ``dz/ds``.
            banking: Track banking angle [rad].

        Returns:
            Non-negative deceleration magnitude along path tangent [m/s^2].
        """
        ay_limit = self.lateral_accel_limit(speed, banking)
        circle_scale = self._friction_circle_scale(lateral_accel_required, ay_limit)

        tire_brake = self._envelope_physics.max_brake_accel * circle_scale
        drag_accel = aero_forces(self.vehicle, speed).drag / self.vehicle.mass
        grade_accel = GRAVITY * grade
        return float(max(tire_brake + drag_accel + grade_accel, 0.0))

    def diagnostics(
        self,
        speed: float,
        longitudinal_accel: float,
        lateral_accel: float,
        curvature: float,
    ) -> VehicleModelDiagnostics:
        """Evaluate yaw moment, axle loads, and power for analysis outputs.

        Args:
            speed: Vehicle speed [m/s].
            longitudinal_accel: Net longitudinal acceleration [m/s^2].
            lateral_accel: Lateral acceleration [m/s^2].
            curvature: Path curvature [1/m].

        Returns:
            Diagnostic values for plotting and KPI post-processing.
        """
        steer = float(np.arctan(self.vehicle.wheelbase * curvature))

        state = VehicleState(
            vx=speed,
            vy=0.0,
            yaw_rate=speed * curvature,
        )
        control = ControlInput(steer=steer, longitudinal_accel_cmd=longitudinal_accel)
        force_balance = self._dynamics.force_balance(state, control)

        loads = estimate_normal_loads(
            self.vehicle,
            speed=speed,
            longitudinal_accel=longitudinal_accel,
            lateral_accel=lateral_accel,
        )

        aero = aero_forces(self.vehicle, speed)
        tractive_force = self.vehicle.mass * longitudinal_accel + aero.drag
        power = tractive_force * speed

        return VehicleModelDiagnostics(
            yaw_moment=force_balance.yaw_moment,
            front_axle_load=loads.front_axle_load,
            rear_axle_load=loads.rear_axle_load,
            power=power,
        )


def build_bicycle_model(
    vehicle: VehicleParameters,
    tires: AxleTireParameters,
    physics: BicyclePhysics | None = None,
    numerics: BicycleNumerics | None = None,
) -> BicycleModel:
    """Build a bicycle solver model with sensible numerical defaults.

    Args:
        vehicle: Vehicle parameterization for dynamics and aero.
        tires: Front/rear Pacejka tire coefficients.
        physics: Optional physical model inputs for longitudinal and lateral
            limits. Defaults to :class:`BicyclePhysics`.
        numerics: Optional numerical controls. Defaults to :class:`BicycleNumerics`.

    Returns:
        Fully validated solver-facing bicycle model.
    """
    return BicycleModel(
        vehicle=vehicle,
        tires=tires,
        physics=physics or BicyclePhysics(),
        numerics=numerics or BicycleNumerics(),
    )
