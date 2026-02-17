"""Solver-facing point-mass vehicle model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from pylapsim.simulation.model_api import ModelDiagnostics
from pylapsim.utils.constants import GRAVITY, SMALL_EPS
from pylapsim.utils.exceptions import ConfigurationError
from pylapsim.vehicle._model_base import EnvelopeVehicleModel
from pylapsim.vehicle._physics_primitives import EnvelopePhysics
from pylapsim.vehicle.params import VehicleParameters

if TYPE_CHECKING:
    from pylapsim.simulation.numba_profile import NumbaProfileParameters
    from pylapsim.tire.models import AxleTireParameters
    from pylapsim.vehicle.bicycle_model import BicycleNumerics, BicyclePhysics

DEFAULT_CALIBRATION_SAMPLE_COUNT = 50


@dataclass(frozen=True)
class PointMassPhysics:
    """Physical inputs for the point-mass lap-time model.

    Args:
        max_drive_accel: Maximum forward tire acceleration on flat road and zero
            lateral demand, excluding drag and grade [m/s^2].
        max_brake_accel: Maximum braking deceleration magnitude on flat road and
            zero lateral demand, excluding drag and grade [m/s^2].
        friction_coefficient: Isotropic tire-road friction coefficient used for
            lateral limit and friction-circle coupling (-).
    """

    max_drive_accel: float = 8.0
    max_brake_accel: float = 16.0
    friction_coefficient: float = 1.70

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

    def validate(self) -> None:
        """Validate physical model parameters.

        Raises:
            pylapsim.utils.exceptions.ConfigurationError: If limits or
                friction settings violate required bounds.
        """
        self.envelope.validate()
        if self.friction_coefficient <= 0.0:
            msg = "friction_coefficient must be positive"
            raise ConfigurationError(msg)


@dataclass(frozen=True)
class PointMassCalibrationResult:
    """Calibration outputs for matching point-mass lateral limits to bicycle behavior.

    Args:
        friction_coefficient: Identified isotropic friction coefficient (-).
        speed_samples: Calibration speed samples used by the identification [m/s].
        bicycle_lateral_limit: Bicycle-model lateral limit evaluated at
            ``speed_samples`` [m/s^2].
        normal_accel_limit: Point-mass normal acceleration budget evaluated at
            ``speed_samples`` [m/s^2].
        mu_samples: Per-sample effective friction ratios computed as
            ``bicycle_lateral_limit / normal_accel_limit`` (-).
    """

    friction_coefficient: float
    speed_samples: np.ndarray
    bicycle_lateral_limit: np.ndarray
    normal_accel_limit: np.ndarray
    mu_samples: np.ndarray


class PointMassModel(EnvelopeVehicleModel):
    """Vehicle-model API implementation for a point-mass backend."""

    _cached_torch_module: Any | None = None

    def __init__(self, vehicle: VehicleParameters, physics: PointMassPhysics) -> None:
        """Initialize point-mass solver backend.

        Args:
            vehicle: Vehicle parameterization for mass, aero, and axle split.
            physics: Point-mass physical model settings.

        Raises:
            pylapsim.utils.exceptions.ConfigurationError: If any provided
                parameter set is invalid.
        """
        self.vehicle = vehicle
        self.physics = physics
        super().__init__(vehicle=vehicle, envelope_physics=physics.envelope)
        self.validate()

    def _validate_backend(self) -> None:
        """Validate point-mass-specific model configuration.

        Raises:
            pylapsim.utils.exceptions.ConfigurationError: If point-mass
                physical parameters violate constraints.
        """
        self.physics.validate()

    def numba_speed_profile_parameters(self) -> NumbaProfileParameters:
        """Return scalar coefficients required by the numba profile kernel.

        Returns:
            Tuple ``(mass, downforce_scale, drag_scale, friction_coefficient,
            max_drive_accel, max_brake_accel)`` used by
            :func:`pylapsim.simulation.numba_profile.solve_speed_profile_numba`.
        """
        return (
            float(self.vehicle.mass),
            float(self._downforce_scale),
            float(self._drag_force_scale),
            float(self.physics.friction_coefficient),
            float(self.envelope_physics.max_drive_accel),
            float(self.envelope_physics.max_brake_accel),
        )


    @classmethod
    def _torch_module(cls) -> Any:
        """Import torch lazily for optional tensor-backed execution.

        Returns:
            Imported `torch` module.

        Raises:
            pylapsim.utils.exceptions.ConfigurationError: If torch is not
                installed in the active environment.
        """
        if cls._cached_torch_module is not None:
            return cls._cached_torch_module
        try:
            import torch
        except ModuleNotFoundError as exc:
            msg = (
                "PointMassModel torch backend requires PyTorch. "
                "Install with `pip install -e '.[torch]'`."
            )
            raise ConfigurationError(msg) from exc
        cls._cached_torch_module = torch
        return torch

    def _downforce_total_batch(self, speed: np.ndarray) -> np.ndarray:
        """Compute total downforce for a vector of speeds.

        Args:
            speed: Speed samples [m/s].

        Returns:
            Total aerodynamic downforce samples [N].
        """
        speed_non_negative = np.maximum(speed, 0.0)
        downforce = self._downforce_scale * speed_non_negative * speed_non_negative
        return np.asarray(downforce, dtype=float)

    def _drag_force_torch(self, speed: Any) -> Any:
        """Compute aerodynamic drag force from a torch tensor speed input.

        Args:
            speed: Speed tensor [m/s].

        Returns:
            Drag force tensor [N].
        """
        torch = self._torch_module()
        speed_non_negative = torch.clamp(speed, min=0.0)
        return self._drag_force_scale * speed_non_negative * speed_non_negative

    def _downforce_total_torch(self, speed: Any) -> Any:
        """Compute total downforce from a torch tensor speed input.

        Args:
            speed: Speed tensor [m/s].

        Returns:
            Downforce tensor [N].
        """
        torch = self._torch_module()
        speed_non_negative = torch.clamp(speed, min=0.0)
        return self._downforce_scale * speed_non_negative * speed_non_negative

    def _normal_accel_limit(self, speed: float) -> float:
        """Compute available tire normal acceleration from weight and downforce.

        Args:
            speed: Vehicle speed [m/s].

        Returns:
            Positive normal-acceleration equivalent available at tire contact
            patch as ``(weight + downforce) / mass`` [m/s^2].
        """
        downforce = self._downforce_total(speed)
        return max(GRAVITY + downforce / self.vehicle.mass, SMALL_EPS)

    def _normal_accel_limit_batch(self, speed: np.ndarray) -> np.ndarray:
        """Compute normal-acceleration limits over a vector of speeds.

        Args:
            speed: Speed samples [m/s].

        Returns:
            Positive normal-acceleration limits [m/s^2].
        """
        downforce = self._downforce_total_batch(speed)
        return np.maximum(GRAVITY + downforce / self.vehicle.mass, SMALL_EPS)

    def _normal_accel_limit_torch(self, speed: Any) -> Any:
        """Compute normal-acceleration limits for torch tensor speeds.

        Args:
            speed: Speed tensor [m/s].

        Returns:
            Positive normal-acceleration tensor [m/s^2].
        """
        torch = self._torch_module()
        downforce = self._downforce_total_torch(speed)
        return torch.clamp(GRAVITY + downforce / self.vehicle.mass, min=SMALL_EPS)

    def _tire_accel_limit(self, speed: float) -> float:
        """Compute isotropic tire acceleration limit at a speed operating point.

        Args:
            speed: Vehicle speed [m/s].

        Returns:
            Tire acceleration magnitude limit from isotropic friction [m/s^2].
        """
        return self.physics.friction_coefficient * self._normal_accel_limit(speed)

    def _tire_accel_limit_batch(self, speed: np.ndarray) -> np.ndarray:
        """Compute isotropic tire acceleration limits over speed samples.

        Args:
            speed: Speed samples [m/s].

        Returns:
            Tire acceleration magnitude limits [m/s^2].
        """
        return self.physics.friction_coefficient * self._normal_accel_limit_batch(speed)

    def _tire_accel_limit_torch(self, speed: Any) -> Any:
        """Compute isotropic tire acceleration limits for torch tensor speeds.

        Args:
            speed: Speed tensor [m/s].

        Returns:
            Tire acceleration magnitude tensor [m/s^2].
        """
        return self.physics.friction_coefficient * self._normal_accel_limit_torch(speed)

    @staticmethod
    def _friction_circle_scale_torch(
        lateral_accel_required: Any,
        lateral_accel_limit: Any,
    ) -> Any:
        """Compute friction-circle longitudinal scale for torch tensors.

        Args:
            lateral_accel_required: Required lateral acceleration tensor [m/s^2].
            lateral_accel_limit: Available lateral acceleration tensor [m/s^2].

        Returns:
            Longitudinal utilization scale tensor in ``[0, 1]``.
        """
        torch = PointMassModel._torch_module()
        safe_limit = torch.clamp(lateral_accel_limit, min=SMALL_EPS)
        usage = torch.clamp(torch.abs(lateral_accel_required) / safe_limit, min=0.0, max=1.0)
        return torch.sqrt(torch.clamp(1.0 - usage * usage, min=0.0, max=1.0))

    def lateral_accel_limit(self, speed: float, banking: float) -> float:
        """Estimate lateral acceleration capacity for the operating point.

        Args:
            speed: Vehicle speed [m/s].
            banking: Track banking angle [rad].

        Returns:
            Quasi-steady lateral acceleration limit [m/s^2].
        """
        speed_non_negative = max(speed, 0.0)
        speed_squared = speed_non_negative * speed_non_negative
        downforce = self._downforce_scale * speed_squared
        normal_accel = max(GRAVITY + downforce / self.vehicle.mass, SMALL_EPS)
        ay_tire = self.physics.friction_coefficient * normal_accel
        ay_banking = GRAVITY * float(np.sin(banking))
        return float(max(ay_tire + ay_banking, SMALL_EPS))

    def lateral_accel_limit_batch(self, speed: np.ndarray, banking: np.ndarray) -> np.ndarray:
        """Estimate lateral acceleration limits over vectorized operating points.

        Args:
            speed: Speed samples [m/s].
            banking: Banking-angle samples [rad].

        Returns:
            Lateral acceleration limit samples [m/s^2].
        """
        speed_array, banking_array = np.broadcast_arrays(
            np.asarray(speed, dtype=float),
            np.asarray(banking, dtype=float),
        )
        ay_tire = self._tire_accel_limit_batch(speed_array)
        ay_banking = GRAVITY * np.sin(banking_array)
        lateral_limit = np.maximum(ay_tire + ay_banking, SMALL_EPS)
        return np.asarray(lateral_limit, dtype=float)

    def lateral_accel_limit_torch(self, speed: Any, banking: Any) -> Any:
        """Estimate lateral acceleration limits over torch tensor inputs.

        Args:
            speed: Speed tensor [m/s].
            banking: Banking-angle tensor [rad].

        Returns:
            Lateral acceleration limit tensor [m/s^2].
        """
        torch = self._torch_module()
        speed_non_negative = torch.clamp(speed, min=0.0)
        speed_squared = speed_non_negative * speed_non_negative
        downforce = self._downforce_scale * speed_squared
        normal_accel = torch.clamp(GRAVITY + downforce / self.vehicle.mass, min=SMALL_EPS)
        ay_tire = self.physics.friction_coefficient * normal_accel
        ay_banking = GRAVITY * torch.sin(banking)
        return torch.clamp(ay_tire + ay_banking, min=SMALL_EPS)

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
        speed_non_negative = max(speed, 0.0)
        speed_squared = speed_non_negative * speed_non_negative
        downforce = self._downforce_scale * speed_squared
        normal_accel = max(GRAVITY + downforce / self.vehicle.mass, SMALL_EPS)

        ay_tire = self.physics.friction_coefficient * normal_accel
        ay_limit = max(ay_tire + GRAVITY * float(np.sin(banking)), SMALL_EPS)
        usage = min(abs(lateral_accel_required) / ay_limit, 1.0)
        circle_scale = float(np.sqrt(max(0.0, 1.0 - usage * usage)))

        tire_limit = min(ay_tire, self.envelope_physics.max_drive_accel)
        tire_accel = tire_limit * circle_scale

        drag_accel = self._drag_force_scale * speed_squared / self.vehicle.mass
        grade_accel = GRAVITY * grade
        return float(tire_accel - drag_accel - grade_accel)

    def max_longitudinal_accel_torch(
        self,
        speed: Any,
        lateral_accel_required: Any,
        grade: Any,
        banking: Any,
    ) -> Any:
        """Compute net forward acceleration limit for torch tensor inputs.

        Args:
            speed: Speed tensor [m/s].
            lateral_accel_required: Lateral acceleration demand tensor [m/s^2].
            grade: Grade tensor ``dz/ds``.
            banking: Banking-angle tensor [rad].

        Returns:
            Net forward acceleration tensor [m/s^2].
        """
        torch = self._torch_module()

        speed_non_negative = torch.clamp(speed, min=0.0)
        speed_squared = speed_non_negative * speed_non_negative
        downforce = self._downforce_scale * speed_squared
        normal_accel = torch.clamp(GRAVITY + downforce / self.vehicle.mass, min=SMALL_EPS)

        ay_tire = self.physics.friction_coefficient * normal_accel
        ay_limit = torch.clamp(ay_tire + GRAVITY * torch.sin(banking), min=SMALL_EPS)
        usage = torch.clamp(torch.abs(lateral_accel_required) / ay_limit, min=0.0, max=1.0)
        circle_scale = torch.sqrt(torch.clamp(1.0 - usage * usage, min=0.0, max=1.0))

        tire_limit = torch.clamp(ay_tire, max=self.envelope_physics.max_drive_accel)
        tire_accel = tire_limit * circle_scale

        drag_accel = self._drag_force_scale * speed_squared / self.vehicle.mass
        grade_accel = GRAVITY * grade
        return tire_accel - drag_accel - grade_accel

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
        speed_non_negative = max(speed, 0.0)
        speed_squared = speed_non_negative * speed_non_negative
        downforce = self._downforce_scale * speed_squared
        normal_accel = max(GRAVITY + downforce / self.vehicle.mass, SMALL_EPS)

        ay_tire = self.physics.friction_coefficient * normal_accel
        ay_limit = max(ay_tire + GRAVITY * float(np.sin(banking)), SMALL_EPS)
        usage = min(abs(lateral_accel_required) / ay_limit, 1.0)
        circle_scale = float(np.sqrt(max(0.0, 1.0 - usage * usage)))

        tire_limit = min(ay_tire, self.envelope_physics.max_brake_accel)
        tire_brake = tire_limit * circle_scale

        drag_accel = self._drag_force_scale * speed_squared / self.vehicle.mass
        grade_accel = GRAVITY * grade
        return float(max(tire_brake + drag_accel + grade_accel, 0.0))

    def max_longitudinal_decel_torch(
        self,
        speed: Any,
        lateral_accel_required: Any,
        grade: Any,
        banking: Any,
    ) -> Any:
        """Compute deceleration magnitude limits for torch tensor inputs.

        Args:
            speed: Speed tensor [m/s].
            lateral_accel_required: Lateral acceleration demand tensor [m/s^2].
            grade: Grade tensor ``dz/ds``.
            banking: Banking-angle tensor [rad].

        Returns:
            Non-negative deceleration magnitude tensor [m/s^2].
        """
        torch = self._torch_module()

        speed_non_negative = torch.clamp(speed, min=0.0)
        speed_squared = speed_non_negative * speed_non_negative
        downforce = self._downforce_scale * speed_squared
        normal_accel = torch.clamp(GRAVITY + downforce / self.vehicle.mass, min=SMALL_EPS)

        ay_tire = self.physics.friction_coefficient * normal_accel
        ay_limit = torch.clamp(ay_tire + GRAVITY * torch.sin(banking), min=SMALL_EPS)
        usage = torch.clamp(torch.abs(lateral_accel_required) / ay_limit, min=0.0, max=1.0)
        circle_scale = torch.sqrt(torch.clamp(1.0 - usage * usage, min=0.0, max=1.0))

        tire_limit = torch.clamp(ay_tire, max=self.envelope_physics.max_brake_accel)
        tire_brake = tire_limit * circle_scale

        drag_accel = self._drag_force_scale * speed_squared / self.vehicle.mass
        grade_accel = GRAVITY * grade
        return torch.clamp(tire_brake + drag_accel + grade_accel, min=0.0)

    def diagnostics(
        self,
        speed: float,
        longitudinal_accel: float,
        lateral_accel: float,
        curvature: float,
    ) -> ModelDiagnostics:
        """Evaluate point-mass diagnostics for analysis outputs.

        Args:
            speed: Vehicle speed [m/s].
            longitudinal_accel: Net longitudinal acceleration [m/s^2].
            lateral_accel: Lateral acceleration [m/s^2].
            curvature: Path curvature [1/m].

        Returns:
            Diagnostic values for plotting and KPI post-processing.
        """
        del lateral_accel, curvature

        weight = self.vehicle.mass * GRAVITY
        front_downforce = self._downforce_front(speed)
        rear_downforce = self._downforce_rear(speed)

        front_axle = weight * self.vehicle.front_weight_fraction + front_downforce
        rear_axle = weight * (1.0 - self.vehicle.front_weight_fraction) + rear_downforce

        tractive_force = self.vehicle.mass * longitudinal_accel + self._drag_force(speed)
        power = tractive_force * speed

        return ModelDiagnostics(
            yaw_moment=0.0,
            front_axle_load=float(front_axle),
            rear_axle_load=float(rear_axle),
            power=float(power),
        )

    def diagnostics_batch(
        self,
        speed: np.ndarray,
        longitudinal_accel: np.ndarray,
        lateral_accel: np.ndarray,
        curvature: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate diagnostics for vectorized operating-point samples.

        Args:
            speed: Speed samples [m/s].
            longitudinal_accel: Net longitudinal-acceleration samples [m/s^2].
            lateral_accel: Lateral-acceleration samples [m/s^2].
            curvature: Curvature samples [1/m].

        Returns:
            Tuple ``(yaw_moment, front_axle_load, rear_axle_load, power)`` with
            arrays aligned to the input shape.
        """
        del lateral_accel, curvature

        speed_array = np.asarray(speed, dtype=float)
        accel_array = np.asarray(longitudinal_accel, dtype=float)

        downforce_total = self._downforce_total_batch(speed_array)
        front_downforce = downforce_total * self._front_downforce_share
        rear_downforce = downforce_total * (1.0 - self._front_downforce_share)

        weight = self.vehicle.mass * GRAVITY
        front_axle = weight * self.vehicle.front_weight_fraction + front_downforce
        rear_axle = weight * (1.0 - self.vehicle.front_weight_fraction) + rear_downforce

        drag_force = self._drag_force_scale * np.maximum(speed_array, 0.0) ** 2
        tractive_force = self.vehicle.mass * accel_array + drag_force
        power = tractive_force * speed_array

        yaw_moment = np.zeros_like(speed_array)
        return (
            np.asarray(yaw_moment, dtype=float),
            np.asarray(front_axle, dtype=float),
            np.asarray(rear_axle, dtype=float),
            np.asarray(power, dtype=float),
        )


def build_point_mass_model(
    vehicle: VehicleParameters,
    physics: PointMassPhysics | None = None,
) -> PointMassModel:
    """Build a point-mass solver model with sensible defaults.

    Args:
        vehicle: Vehicle parameterization for mass, aero, and axle split.
        physics: Optional physical model settings. Defaults to
            :class:`PointMassPhysics`.

    Returns:
        Fully validated solver-facing point-mass model.
    """
    return PointMassModel(vehicle=vehicle, physics=physics or PointMassPhysics())


def calibrate_point_mass_friction_to_bicycle(
    vehicle: VehicleParameters,
    tires: AxleTireParameters,
    bicycle_physics: BicyclePhysics | None = None,
    bicycle_numerics: BicycleNumerics | None = None,
    speed_samples: np.ndarray | None = None,
) -> PointMassCalibrationResult:
    """Calibrate point-mass friction coefficient to bicycle lateral capability.

    The calibration matches the point-mass isotropic lateral limit
    ``mu * (g + F_down(v)/m)`` to the bicycle model's quasi-steady lateral limit
    in a least-squares sense over provided speed samples.

    Args:
        vehicle: Vehicle parameterization shared by both models.
        tires: Tire parameters used by the bicycle model.
        bicycle_physics: Optional bicycle-physics settings for calibration.
            Defaults to :class:`pylapsim.vehicle.BicyclePhysics`.
        bicycle_numerics: Optional bicycle numerical settings for calibration.
            Defaults to :class:`pylapsim.vehicle.BicycleNumerics`.
        speed_samples: Optional calibration speeds [m/s]. If omitted, a linear
            sweep from 10 to 90 m/s is used.

    Returns:
        Calibration result with identified friction coefficient and
        intermediate traces.

    Raises:
        pylapsim.utils.exceptions.ConfigurationError: If provided speed
            samples are empty or contain non-positive entries.
    """
    from pylapsim.vehicle.bicycle_model import BicycleModel, BicycleNumerics, BicyclePhysics

    bicycle_model = BicycleModel(
        vehicle=vehicle,
        tires=tires,
        physics=bicycle_physics or BicyclePhysics(),
        numerics=bicycle_numerics or BicycleNumerics(),
    )
    bicycle_model.validate()

    if speed_samples is None:
        speeds = np.linspace(10.0, 90.0, DEFAULT_CALIBRATION_SAMPLE_COUNT, dtype=float)
    else:
        speeds = np.asarray(speed_samples, dtype=float)
        if speeds.size == 0:
            msg = "speed_samples must not be empty"
            raise ConfigurationError(msg)
        if np.any(speeds <= 0.0):
            msg = "speed_samples must be strictly positive"
            raise ConfigurationError(msg)

    bicycle_ay = np.array(
        [bicycle_model.lateral_accel_limit(float(v), 0.0) for v in speeds],
        dtype=float,
    )

    speed_sq = np.maximum(speeds, 0.0) ** 2
    downforce_scale = 0.5 * vehicle.air_density * vehicle.lift_coefficient * vehicle.frontal_area
    downforce = downforce_scale * speed_sq
    normal_accel = np.maximum(GRAVITY + downforce / vehicle.mass, SMALL_EPS)

    numerator = float(np.dot(normal_accel, bicycle_ay))
    denominator = float(np.dot(normal_accel, normal_accel))
    mu_fit = max(numerator / max(denominator, SMALL_EPS), SMALL_EPS)
    mu_samples = bicycle_ay / normal_accel

    return PointMassCalibrationResult(
        friction_coefficient=float(mu_fit),
        speed_samples=speeds,
        bicycle_lateral_limit=bicycle_ay,
        normal_accel_limit=normal_accel,
        mu_samples=mu_samples,
    )

