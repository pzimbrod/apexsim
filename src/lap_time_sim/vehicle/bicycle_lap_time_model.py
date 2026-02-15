"""Lap-time vehicle-model adapter based on the 3-DOF bicycle model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lap_time_sim.simulation.envelope import lateral_accel_limit as bicycle_lateral_accel_limit
from lap_time_sim.simulation.model_api import VehicleModelDiagnostics
from lap_time_sim.tire.models import AxleTireParameters
from lap_time_sim.utils.constants import GRAVITY_MPS2, SMALL_EPS
from lap_time_sim.utils.exceptions import ConfigurationError
from lap_time_sim.vehicle.aero import aero_forces
from lap_time_sim.vehicle.bicycle import BicycleModel, ControlInput, VehicleState
from lap_time_sim.vehicle.load_transfer import estimate_normal_loads
from lap_time_sim.vehicle.params import VehicleParameters


@dataclass(frozen=True)
class BicycleLapTimeModelConfig:
    """Longitudinal performance limits for the bicycle lap-time adapter."""

    max_drive_accel_mps2: float = 8.0
    max_brake_accel_mps2: float = 16.0

    def validate(self) -> None:
        """Validate longitudinal envelope limits."""
        if self.max_drive_accel_mps2 <= 0.0:
            msg = "max_drive_accel_mps2 must be positive"
            raise ConfigurationError(msg)
        if self.max_brake_accel_mps2 <= 0.0:
            msg = "max_brake_accel_mps2 must be positive"
            raise ConfigurationError(msg)


class BicycleLapTimeModel:
    """Vehicle-model API implementation for the bicycle dynamics backend."""

    def __init__(
        self,
        vehicle: VehicleParameters,
        tires: AxleTireParameters,
        config: BicycleLapTimeModelConfig | None = None,
    ) -> None:
        self.vehicle = vehicle
        self.tires = tires
        self.config = config or BicycleLapTimeModelConfig()
        self._bicycle_model = BicycleModel(vehicle, tires)
        self.validate()

    def validate(self) -> None:
        """Validate model configuration and parameterization."""
        self.vehicle.validate()
        self.tires.validate()
        self.config.validate()

    def lateral_accel_limit(self, speed_mps: float, banking_rad: float) -> float:
        """Estimate lateral acceleration capacity for the operating point."""
        return bicycle_lateral_accel_limit(
            vehicle=self.vehicle,
            tires=self.tires,
            speed_mps=speed_mps,
            banking_rad=banking_rad,
        )

    def _friction_circle_scale(self, ay_required_mps2: float, ay_limit_mps2: float) -> float:
        if ay_limit_mps2 <= SMALL_EPS:
            return 0.0
        usage = min(abs(ay_required_mps2) / ay_limit_mps2, 1.0)
        return float(np.sqrt(max(0.0, 1.0 - usage * usage)))

    def max_longitudinal_accel(
        self,
        speed_mps: float,
        ay_required_mps2: float,
        grade: float,
        banking_rad: float,
    ) -> float:
        """Compute net forward acceleration limit along path tangent."""
        ay_limit = self.lateral_accel_limit(speed_mps, banking_rad)
        circle_scale = self._friction_circle_scale(ay_required_mps2, ay_limit)

        tire_accel = self.config.max_drive_accel_mps2 * circle_scale
        drag_accel = aero_forces(self.vehicle, speed_mps).drag_n / self.vehicle.mass_kg
        grade_accel = GRAVITY_MPS2 * grade
        return float(tire_accel - drag_accel - grade_accel)

    def max_longitudinal_decel(
        self,
        speed_mps: float,
        ay_required_mps2: float,
        grade: float,
        banking_rad: float,
    ) -> float:
        """Compute available deceleration magnitude along path tangent."""
        ay_limit = self.lateral_accel_limit(speed_mps, banking_rad)
        circle_scale = self._friction_circle_scale(ay_required_mps2, ay_limit)

        tire_brake = self.config.max_brake_accel_mps2 * circle_scale
        drag_accel = aero_forces(self.vehicle, speed_mps).drag_n / self.vehicle.mass_kg
        grade_accel = GRAVITY_MPS2 * grade
        return float(max(tire_brake + drag_accel + grade_accel, 0.0))

    def diagnostics(
        self,
        speed_mps: float,
        ax_mps2: float,
        ay_mps2: float,
        curvature_1pm: float,
    ) -> VehicleModelDiagnostics:
        """Evaluate yaw moment, axle loads, and power for analysis outputs."""
        steer_rad = float(np.arctan(self.vehicle.wheelbase_m * curvature_1pm))

        state = VehicleState(
            vx_mps=speed_mps,
            vy_mps=0.0,
            yaw_rate_rps=speed_mps * curvature_1pm,
        )
        control = ControlInput(steer_rad=steer_rad, longitudinal_accel_cmd_mps2=ax_mps2)
        force_balance = self._bicycle_model.force_balance(state, control)

        loads = estimate_normal_loads(
            self.vehicle,
            speed_mps=speed_mps,
            longitudinal_accel_mps2=ax_mps2,
            lateral_accel_mps2=ay_mps2,
        )

        aero = aero_forces(self.vehicle, speed_mps)
        tractive_force_n = self.vehicle.mass_kg * ax_mps2 + aero.drag_n
        power_w = tractive_force_n * speed_mps

        return VehicleModelDiagnostics(
            yaw_moment_nm=force_balance.yaw_moment_nm,
            front_axle_load_n=loads.front_axle_n,
            rear_axle_load_n=loads.rear_axle_n,
            power_w=power_w,
        )


def build_default_bicycle_lap_time_model() -> BicycleLapTimeModel:
    """Create a ready-to-use bicycle lap-time model with default parameters."""
    from lap_time_sim.tire.models import default_axle_tire_parameters
    from lap_time_sim.vehicle.params import default_vehicle_parameters

    return BicycleLapTimeModel(
        vehicle=default_vehicle_parameters(),
        tires=default_axle_tire_parameters(),
    )
