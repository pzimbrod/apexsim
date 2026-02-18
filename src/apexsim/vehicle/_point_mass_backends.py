"""Backend-specific adapters for the point-mass vehicle model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

from apexsim.utils.constants import GRAVITY, SMALL_EPS
from apexsim.utils.exceptions import ConfigurationError
from apexsim.vehicle._physics_primitives import EnvelopePhysics
from apexsim.vehicle._point_mass_physics import PointMassPhysicsProtocol
from apexsim.vehicle.params import VehicleParameters

if TYPE_CHECKING:
    from apexsim.simulation.numba_profile import NumbaProfileParameters


class PointMassBackendState(Protocol):
    """Protocol describing shared state required by backend adapters."""

    vehicle: VehicleParameters
    physics: PointMassPhysicsProtocol
    envelope_physics: EnvelopePhysics
    _drag_force_scale: float
    _downforce_scale: float


class PointMassNumbaBackendMixin:
    """Numba backend adapter methods for ``PointMassModel``."""

    def numba_speed_profile_parameters(self: PointMassBackendState) -> NumbaProfileParameters:
        """Return scalar coefficients consumed by the numba profile kernel.

        Returns:
            Tuple ``(mass, downforce_scale, drag_scale, friction_coefficient,
            max_drive_accel, max_brake_accel)``.
        """
        return (
            float(self.vehicle.mass),
            float(self._downforce_scale),
            float(self._drag_force_scale),
            float(self.physics.friction_coefficient),
            float(self.envelope_physics.max_drive_accel),
            float(self.envelope_physics.max_brake_accel),
        )


class PointMassTorchBackendMixin:
    """Torch backend adapter methods for ``PointMassModel``."""

    _cached_torch_module: Any | None = None

    if TYPE_CHECKING:
        vehicle: VehicleParameters
        physics: object
        envelope_physics: EnvelopePhysics
        _drag_force_scale: float
        _downforce_scale: float

    @classmethod
    def _torch_module(cls) -> Any:
        """Import torch lazily for optional tensor-backed execution.

        Returns:
            Imported ``torch`` module.

        Raises:
            apexsim.utils.exceptions.ConfigurationError: If torch is not
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

    def _normal_accel_limit_torch(self, speed: Any) -> Any:
        """Compute normal-acceleration limits for torch tensor speed inputs.

        Args:
            speed: Speed tensor [m/s].

        Returns:
            Available normal-acceleration tensor [m/s^2].
        """
        torch = self._torch_module()
        speed_non_negative = torch.clamp(speed, min=0.0)
        speed_squared = speed_non_negative * speed_non_negative
        downforce = self._downforce_scale * speed_squared
        return torch.clamp(GRAVITY + downforce / self.vehicle.mass, min=SMALL_EPS)

    def _tire_accel_limit_torch(self, speed: Any) -> Any:
        """Compute isotropic tire acceleration limits for torch tensors.

        Args:
            speed: Speed tensor [m/s].

        Returns:
            Isotropic tire acceleration-magnitude limit tensor [m/s^2].
        """
        physics = cast(PointMassPhysicsProtocol, self.physics)
        return physics.friction_coefficient * self._normal_accel_limit_torch(speed)

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
        torch = PointMassTorchBackendMixin._torch_module()
        safe_limit = torch.clamp(lateral_accel_limit, min=SMALL_EPS)
        usage = torch.clamp(torch.abs(lateral_accel_required) / safe_limit, min=0.0, max=1.0)
        return torch.sqrt(torch.clamp(1.0 - usage * usage, min=0.0, max=1.0))

    def tractive_power_torch(
        self,
        speed: Any,
        longitudinal_accel: Any,
    ) -> Any:
        """Compute tractive power for torch speed and acceleration tensors.

        Args:
            speed: Speed tensor [m/s].
            longitudinal_accel: Net longitudinal-acceleration tensor [m/s^2].

        Returns:
            Tractive power tensor [W].
        """
        torch = self._torch_module()
        speed_tensor = torch.as_tensor(speed)
        accel_tensor = torch.as_tensor(
            longitudinal_accel,
            dtype=speed_tensor.dtype,
            device=speed_tensor.device,
        )
        speed_non_negative = torch.clamp(speed_tensor, min=0.0)
        speed_squared = speed_non_negative * speed_non_negative
        drag_force = self._drag_force_scale * speed_squared
        tractive_force = self.vehicle.mass * accel_tensor + drag_force
        return tractive_force * speed_tensor

    def lateral_accel_limit_torch(
        self,
        speed: Any,
        banking: Any,
    ) -> Any:
        """Estimate lateral acceleration limits for torch tensor inputs.

        Args:
            speed: Speed tensor [m/s].
            banking: Banking-angle tensor [rad].

        Returns:
            Quasi-steady lateral acceleration limit tensor [m/s^2].
        """
        torch = self._torch_module()
        ay_tire = self._tire_accel_limit_torch(speed)
        ay_banking = GRAVITY * torch.sin(banking)
        return torch.clamp(ay_tire + ay_banking, min=SMALL_EPS)

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
            lateral_accel_required: Required lateral acceleration tensor [m/s^2].
            grade: Grade tensor ``dz/ds``.
            banking: Banking-angle tensor [rad].

        Returns:
            Net forward acceleration tensor [m/s^2].
        """
        torch = self._torch_module()

        ay_tire = self._tire_accel_limit_torch(speed)
        ay_limit = torch.clamp(ay_tire + GRAVITY * torch.sin(banking), min=SMALL_EPS)
        circle_scale = self._friction_circle_scale_torch(lateral_accel_required, ay_limit)

        tire_limit = torch.clamp(ay_tire, max=self.envelope_physics.max_drive_accel)
        tire_accel = tire_limit * circle_scale

        speed_non_negative = torch.clamp(speed, min=0.0)
        speed_squared = speed_non_negative * speed_non_negative
        drag_accel = self._drag_force_scale * speed_squared / self.vehicle.mass
        grade_accel = GRAVITY * grade
        return tire_accel - drag_accel - grade_accel

    def max_longitudinal_decel_torch(
        self,
        speed: Any,
        lateral_accel_required: Any,
        grade: Any,
        banking: Any,
    ) -> Any:
        """Compute available deceleration magnitudes for torch tensor inputs.

        Args:
            speed: Speed tensor [m/s].
            lateral_accel_required: Required lateral acceleration tensor [m/s^2].
            grade: Grade tensor ``dz/ds``.
            banking: Banking-angle tensor [rad].

        Returns:
            Non-negative deceleration magnitude tensor [m/s^2].
        """
        torch = self._torch_module()

        ay_tire = self._tire_accel_limit_torch(speed)
        ay_limit = torch.clamp(ay_tire + GRAVITY * torch.sin(banking), min=SMALL_EPS)
        circle_scale = self._friction_circle_scale_torch(lateral_accel_required, ay_limit)

        tire_limit = torch.clamp(ay_tire, max=self.envelope_physics.max_brake_accel)
        tire_brake = tire_limit * circle_scale

        speed_non_negative = torch.clamp(speed, min=0.0)
        speed_squared = speed_non_negative * speed_non_negative
        drag_accel = self._drag_force_scale * speed_squared / self.vehicle.mass
        grade_accel = GRAVITY * grade
        return torch.clamp(tire_brake + drag_accel + grade_accel, min=0.0)
