"""Vehicle models and parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lap_time_sim.vehicle.bicycle import BicycleModel, ControlInput, ForceBalance, VehicleState
from lap_time_sim.vehicle.params import VehicleParameters, default_vehicle_parameters

if TYPE_CHECKING:
    from lap_time_sim.vehicle.bicycle_lap_time_model import (
        BicycleLapTimeModel,
        BicycleLapTimeModelConfig,
        build_default_bicycle_lap_time_model,
    )

__all__ = [
    "BicycleModel",
    "BicycleLapTimeModel",
    "BicycleLapTimeModelConfig",
    "ControlInput",
    "ForceBalance",
    "VehicleParameters",
    "VehicleState",
    "build_default_bicycle_lap_time_model",
    "default_vehicle_parameters",
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
    if name == "BicycleLapTimeModel":
        from lap_time_sim.vehicle.bicycle_lap_time_model import BicycleLapTimeModel

        return BicycleLapTimeModel
    if name == "BicycleLapTimeModelConfig":
        from lap_time_sim.vehicle.bicycle_lap_time_model import BicycleLapTimeModelConfig

        return BicycleLapTimeModelConfig
    if name == "build_default_bicycle_lap_time_model":
        from lap_time_sim.vehicle.bicycle_lap_time_model import build_default_bicycle_lap_time_model

        return build_default_bicycle_lap_time_model
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
