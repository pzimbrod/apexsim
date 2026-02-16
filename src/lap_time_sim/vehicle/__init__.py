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

__all__ = [
    "BicycleDynamicsModel",
    "BicycleModel",
    "BicycleNumerics",
    "BicyclePhysics",
    "ControlInput",
    "ForceBalance",
    "VehicleParameters",
    "VehicleState",
    "build_bicycle_model",
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
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
