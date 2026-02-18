"""Backend adapters for the single-track vehicle model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pylapsim.utils.constants import GRAVITY, SMALL_EPS
from pylapsim.vehicle._point_mass_backends import PointMassTorchBackendMixin

if TYPE_CHECKING:
    from pylapsim.simulation.numba_profile import NumbaSingleTrackProfileParameters
    from pylapsim.tire.models import AxleTireParameters, PacejkaParameters
    from pylapsim.vehicle._physics_primitives import EnvelopePhysics
    from pylapsim.vehicle._single_track_physics import (
        SingleTrackLateralPhysicsProtocol,
        SingleTrackNumericsProtocol,
    )
    from pylapsim.vehicle.params import VehicleParameters


class SingleTrackNumbaBackendMixin:
    """Numba backend adapter methods for ``SingleTrackModel``."""

    if TYPE_CHECKING:
        vehicle: VehicleParameters
        tires: AxleTireParameters
        numerics: SingleTrackNumericsProtocol
        _single_track_lateral_physics: SingleTrackLateralPhysicsProtocol
        envelope_physics: EnvelopePhysics
        _drag_force_scale: float
        _downforce_scale: float
        _front_downforce_share: float

    def numba_speed_profile_parameters(self) -> NumbaSingleTrackProfileParameters:
        """Return scalar coefficients consumed by the single-track numba kernel.

        Returns:
            Tuple of single-track kernel coefficients used by
            :func:`pylapsim.simulation.numba_profile.solve_speed_profile_numba`.
        """
        front = self.tires.front
        rear = self.tires.rear
        return (
            float(self.vehicle.mass),
            float(self._downforce_scale),
            float(self._drag_force_scale),
            float(self._front_downforce_share),
            float(self.vehicle.front_weight_fraction),
            float(self.envelope_physics.max_drive_accel),
            float(self.envelope_physics.max_brake_accel),
            float(self._single_track_lateral_physics.peak_slip_angle),
            float(self.numerics.min_lateral_accel_limit),
            int(self.numerics.lateral_limit_max_iterations),
            float(self.numerics.lateral_limit_convergence_tolerance),
            float(front.B),
            float(front.C),
            float(front.D),
            float(front.E),
            float(front.reference_load),
            float(front.load_sensitivity),
            float(front.min_mu_scale),
            float(rear.B),
            float(rear.C),
            float(rear.D),
            float(rear.E),
            float(rear.reference_load),
            float(rear.load_sensitivity),
            float(rear.min_mu_scale),
        )


class SingleTrackTorchBackendMixin(PointMassTorchBackendMixin):
    """Torch backend adapter methods for ``SingleTrackModel``."""

    if TYPE_CHECKING:
        vehicle: VehicleParameters
        tires: AxleTireParameters
        numerics: SingleTrackNumericsProtocol
        _single_track_lateral_physics: SingleTrackLateralPhysicsProtocol
        envelope_physics: EnvelopePhysics
        _drag_force_scale: float
        _downforce_scale: float
        _front_downforce_share: float

    @staticmethod
    def _magic_formula_lateral_torch(
        torch: Any,
        slip_angle: Any,
        normal_load: Any,
        params: PacejkaParameters,
    ) -> Any:
        """Evaluate Pacejka lateral force for torch tensor inputs.

        Args:
            torch: Imported ``torch`` module.
            slip_angle: Slip-angle tensor [rad].
            normal_load: Normal-load tensor [N].
            params: Tire parameters.

        Returns:
            Lateral-force tensor [N].
        """
        fz = torch.clamp(normal_load, min=SMALL_EPS)
        slip = torch.as_tensor(slip_angle, dtype=fz.dtype, device=fz.device)
        slip_term = params.B * slip
        nonlinear = slip_term - params.E * (slip_term - torch.atan(slip_term))
        shape = torch.sin(params.C * torch.atan(nonlinear))

        load_delta = (fz - params.reference_load) / params.reference_load
        mu_scale = torch.clamp(
            1.0 + params.load_sensitivity * load_delta,
            min=params.min_mu_scale,
        )
        return params.D * mu_scale * fz * shape

    def _axle_tire_loads_torch(self, speed: Any) -> tuple[Any, Any]:
        """Estimate front/rear per-tire normal loads for torch speed inputs.

        Args:
            speed: Speed tensor [m/s].

        Returns:
            Tuple ``(front_tire_load, rear_tire_load)`` [N].
        """
        torch = self._torch_module()

        speed_non_negative = torch.clamp(speed, min=0.0)
        speed_squared = speed_non_negative * speed_non_negative
        downforce_total = self._downforce_scale * speed_squared
        front_downforce = downforce_total * self._front_downforce_share

        weight = self.vehicle.mass * GRAVITY
        total_vertical_load = weight + downforce_total
        front_static_load = weight * self.vehicle.front_weight_fraction

        min_axle_load = 2.0 * SMALL_EPS
        front_axle_raw = front_static_load + front_downforce
        min_axle_load_tensor = torch.full_like(total_vertical_load, min_axle_load)
        max_front_axle_load = torch.clamp(total_vertical_load - min_axle_load, min=min_axle_load)
        front_axle_load = torch.minimum(
            torch.maximum(front_axle_raw, min_axle_load_tensor),
            max_front_axle_load,
        )
        rear_axle_load = total_vertical_load - front_axle_load

        front_tire_load = torch.clamp(front_axle_load * 0.5, min=SMALL_EPS)
        rear_tire_load = torch.clamp(rear_axle_load * 0.5, min=SMALL_EPS)
        return front_tire_load, rear_tire_load

    def lateral_accel_limit_torch(self, speed: Any, banking: Any) -> Any:
        """Estimate lateral acceleration limits for torch tensor inputs.

        Args:
            speed: Speed tensor [m/s].
            banking: Banking-angle tensor [rad].

        Returns:
            Quasi-steady lateral acceleration limit tensor [m/s^2].
        """
        torch = self._torch_module()
        front_tire_load, rear_tire_load = self._axle_tire_loads_torch(speed)
        slip = self._single_track_lateral_physics.peak_slip_angle

        fy_front = 2.0 * self._magic_formula_lateral_torch(
            torch=torch,
            slip_angle=slip,
            normal_load=front_tire_load,
            params=self.tires.front,
        )
        fy_rear = 2.0 * self._magic_formula_lateral_torch(
            torch=torch,
            slip_angle=slip,
            normal_load=rear_tire_load,
            params=self.tires.rear,
        )
        ay_tire = (fy_front + fy_rear) / self.vehicle.mass
        ay_banking = GRAVITY * torch.sin(banking)
        return torch.clamp(
            ay_tire + ay_banking,
            min=self.numerics.min_lateral_accel_limit,
        )

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

        ay_limit = torch.clamp(self.lateral_accel_limit_torch(speed, banking), min=SMALL_EPS)
        circle_scale = self._friction_circle_scale_torch(lateral_accel_required, ay_limit)
        tire_accel = self.envelope_physics.max_drive_accel * circle_scale

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

        ay_limit = torch.clamp(self.lateral_accel_limit_torch(speed, banking), min=SMALL_EPS)
        circle_scale = self._friction_circle_scale_torch(lateral_accel_required, ay_limit)
        tire_brake = self.envelope_physics.max_brake_accel * circle_scale

        speed_non_negative = torch.clamp(speed, min=0.0)
        speed_squared = speed_non_negative * speed_non_negative
        drag_accel = self._drag_force_scale * speed_squared / self.vehicle.mass
        grade_accel = GRAVITY * grade
        return torch.clamp(tire_brake + drag_accel + grade_accel, min=0.0)
