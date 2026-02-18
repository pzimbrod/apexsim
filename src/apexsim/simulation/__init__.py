"""Simulation solvers and runners."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from apexsim.simulation.config import (
    NumericsConfig,
    RuntimeConfig,
    SimulationConfig,
    build_simulation_config,
)

if TYPE_CHECKING:
    from apexsim.simulation.model_api import (
        ModelDiagnostics,
        VehicleModel,
        VehicleModelBase,
    )
    from apexsim.simulation.runner import LapResult
    from apexsim.simulation.torch_profile import TorchSpeedProfileResult

__all__ = [
    "LapResult",
    "TorchSpeedProfileResult",
    "VehicleModelBase",
    "VehicleModel",
    "NumericsConfig",
    "RuntimeConfig",
    "SimulationConfig",
    "ModelDiagnostics",
    "build_simulation_config",
    "simulate_lap",
    "solve_speed_profile_torch_autodiff",
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
        from apexsim.simulation.model_api import VehicleModelBase

        return VehicleModelBase
    if name == "VehicleModel":
        from apexsim.simulation.model_api import VehicleModel

        return VehicleModel
    if name == "ModelDiagnostics":
        from apexsim.simulation.model_api import ModelDiagnostics

        return ModelDiagnostics
    if name == "LapResult":
        from apexsim.simulation.runner import LapResult

        return LapResult
    if name == "simulate_lap":
        from apexsim.simulation.runner import simulate_lap

        return simulate_lap
    if name == "TorchSpeedProfileResult":
        from apexsim.simulation.torch_profile import TorchSpeedProfileResult

        return TorchSpeedProfileResult
    if name == "solve_speed_profile_torch_autodiff":
        from apexsim.simulation.torch_profile import solve_speed_profile_torch_autodiff

        return solve_speed_profile_torch_autodiff
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
