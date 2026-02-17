"""Vehicle models and parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lap_time_sim.vehicle.bicycle_dynamics import (
    BicycleDynamicsModel,
    ControlInput,
    ForceBalance,
    VehicleState,
)
from lap_time_sim.vehicle.params import VehicleParameters

if TYPE_CHECKING:
    from lap_time_sim.vehicle.bicycle_model import (
        BicycleModel,
        BicycleNumerics,
        BicyclePhysics,
        build_bicycle_model,
    )
    from lap_time_sim.vehicle.point_mass_model import (
        PointMassCalibrationResult,
        PointMassModel,
        PointMassPhysics,
        build_point_mass_model,
        calibrate_point_mass_friction_to_bicycle,
    )

__all__ = [
    "BicycleDynamicsModel",
    "BicycleModel",
    "BicycleNumerics",
    "BicyclePhysics",
    "ControlInput",
    "ForceBalance",
    "PointMassCalibrationResult",
    "PointMassModel",
    "PointMassPhysics",
    "VehicleParameters",
    "VehicleState",
    "build_bicycle_model",
    "build_point_mass_model",
    "calibrate_point_mass_friction_to_bicycle",
]


def __getattr__(name: str) -> Any:
    """Resolve lazily imported symbols for public package exports.

    Args:
        name: Attribute name requested from the package namespace.

    Returns:
        Exported class or function matching ``name``.

    Raises:
        AttributeError: If ``name`` is not part of the public export surface.
    """
    if name == "BicycleModel":
        from lap_time_sim.vehicle.bicycle_model import BicycleModel

        return BicycleModel
    if name == "BicyclePhysics":
        from lap_time_sim.vehicle.bicycle_model import BicyclePhysics

        return BicyclePhysics
    if name == "BicycleNumerics":
        from lap_time_sim.vehicle.bicycle_model import BicycleNumerics

        return BicycleNumerics
    if name == "build_bicycle_model":
        from lap_time_sim.vehicle.bicycle_model import build_bicycle_model

        return build_bicycle_model
    if name == "PointMassModel":
        from lap_time_sim.vehicle.point_mass_model import PointMassModel

        return PointMassModel
    if name == "PointMassPhysics":
        from lap_time_sim.vehicle.point_mass_model import PointMassPhysics

        return PointMassPhysics
    if name == "build_point_mass_model":
        from lap_time_sim.vehicle.point_mass_model import build_point_mass_model

        return build_point_mass_model
    if name == "PointMassCalibrationResult":
        from lap_time_sim.vehicle.point_mass_model import PointMassCalibrationResult

        return PointMassCalibrationResult
    if name == "calibrate_point_mass_friction_to_bicycle":
        from lap_time_sim.vehicle.point_mass_model import calibrate_point_mass_friction_to_bicycle

        return calibrate_point_mass_friction_to_bicycle
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
