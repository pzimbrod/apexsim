"""Vehicle models and parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pylapsim.vehicle.params import VehicleParameters
from pylapsim.vehicle.single_track.dynamics import (
    ControlInput,
    ForceBalance,
    SingleTrackDynamicsModel,
    VehicleState,
)

if TYPE_CHECKING:
    from pylapsim.vehicle.point_mass_model import (
        PointMassCalibrationResult,
        PointMassModel,
        PointMassPhysics,
        build_point_mass_model,
        calibrate_point_mass_friction_to_single_track,
    )
    from pylapsim.vehicle.single_track_model import (
        SingleTrackModel,
        SingleTrackNumerics,
        SingleTrackPhysics,
        build_single_track_model,
    )

__all__ = [
    "SingleTrackDynamicsModel",
    "SingleTrackModel",
    "SingleTrackNumerics",
    "SingleTrackPhysics",
    "ControlInput",
    "ForceBalance",
    "PointMassCalibrationResult",
    "PointMassModel",
    "PointMassPhysics",
    "VehicleParameters",
    "VehicleState",
    "build_single_track_model",
    "build_point_mass_model",
    "calibrate_point_mass_friction_to_single_track",
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
    if name == "SingleTrackModel":
        from pylapsim.vehicle.single_track_model import SingleTrackModel

        return SingleTrackModel
    if name == "SingleTrackPhysics":
        from pylapsim.vehicle.single_track_model import SingleTrackPhysics

        return SingleTrackPhysics
    if name == "SingleTrackNumerics":
        from pylapsim.vehicle.single_track_model import SingleTrackNumerics

        return SingleTrackNumerics
    if name == "build_single_track_model":
        from pylapsim.vehicle.single_track_model import build_single_track_model

        return build_single_track_model
    if name == "PointMassModel":
        from pylapsim.vehicle.point_mass_model import PointMassModel

        return PointMassModel
    if name == "PointMassPhysics":
        from pylapsim.vehicle.point_mass_model import PointMassPhysics

        return PointMassPhysics
    if name == "build_point_mass_model":
        from pylapsim.vehicle.point_mass_model import build_point_mass_model

        return build_point_mass_model
    if name == "PointMassCalibrationResult":
        from pylapsim.vehicle.point_mass_model import PointMassCalibrationResult

        return PointMassCalibrationResult
    if name == "calibrate_point_mass_friction_to_single_track":
        from pylapsim.vehicle.point_mass_model import calibrate_point_mass_friction_to_single_track

        return calibrate_point_mass_friction_to_single_track
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
