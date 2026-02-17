"""Simulation solvers and runners."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pylapsim.simulation.config import (
    NumericsConfig,
    RuntimeConfig,
    SimulationConfig,
    build_simulation_config,
)

if TYPE_CHECKING:
    from pylapsim.simulation.model_api import (
        ModelDiagnostics,
        VehicleModel,
        VehicleModelBase,
    )
    from pylapsim.simulation.runner import LapResult

__all__ = [
    "LapResult",
    "VehicleModelBase",
    "VehicleModel",
    "NumericsConfig",
    "RuntimeConfig",
    "SimulationConfig",
    "ModelDiagnostics",
    "build_simulation_config",
    "simulate_lap",
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
    if name == "VehicleModelBase":
        from pylapsim.simulation.model_api import VehicleModelBase

        return VehicleModelBase
    if name == "VehicleModel":
        from pylapsim.simulation.model_api import VehicleModel

        return VehicleModel
    if name == "RuntimeConfig":
        from pylapsim.simulation.config import RuntimeConfig

        return RuntimeConfig
    if name == "NumericsConfig":
        from pylapsim.simulation.config import NumericsConfig

        return NumericsConfig
    if name == "build_simulation_config":
        from pylapsim.simulation.config import build_simulation_config

        return build_simulation_config
    if name == "ModelDiagnostics":
        from pylapsim.simulation.model_api import ModelDiagnostics

        return ModelDiagnostics
    if name == "LapResult":
        from pylapsim.simulation.runner import LapResult

        return LapResult
    if name == "simulate_lap":
        from pylapsim.simulation.runner import simulate_lap

        return simulate_lap
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
