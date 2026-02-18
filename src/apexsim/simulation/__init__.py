"""Simulation solvers and runners."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from apexsim.simulation.config import (
    NumericsConfig,
    RuntimeConfig,
    SimulationConfig,
    build_simulation_config,
)
from apexsim.simulation.transient_common import (
    PidSpeedSchedule,
    TransientConfig,
    TransientNumericsConfig,
    TransientPidGainSchedulingConfig,
    TransientProfileResult,
    TransientRuntimeConfig,
)

if TYPE_CHECKING:
    from apexsim.simulation.model_api import (
        ModelDiagnostics,
        VehicleModel,
        VehicleModelBase,
    )
    from apexsim.simulation.runner import LapResult
    from apexsim.simulation.torch_profile import TorchSpeedProfileResult
    from apexsim.simulation.transient_torch import TorchTransientProfileResult

__all__ = [
    "LapResult",
    "TransientProfileResult",
    "TorchTransientProfileResult",
    "TorchSpeedProfileResult",
    "VehicleModelBase",
    "VehicleModel",
    "NumericsConfig",
    "RuntimeConfig",
    "SimulationConfig",
    "TransientConfig",
    "TransientNumericsConfig",
    "TransientPidGainSchedulingConfig",
    "TransientRuntimeConfig",
    "PidSpeedSchedule",
    "ModelDiagnostics",
    "build_simulation_config",
    "build_physics_informed_pid_gain_scheduling",
    "simulate_lap",
    "solve_transient_lap_torch",
    "solve_speed_profile_torch",
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
    if name == "TorchTransientProfileResult":
        from apexsim.simulation.transient_torch import TorchTransientProfileResult

        return TorchTransientProfileResult
    if name == "solve_speed_profile_torch":
        from apexsim.simulation.torch_profile import solve_speed_profile_torch

        return solve_speed_profile_torch
    if name == "solve_transient_lap_torch":
        from apexsim.simulation.transient_torch import solve_transient_lap_torch

        return solve_transient_lap_torch
    if name == "build_physics_informed_pid_gain_scheduling":
        from apexsim.simulation.transient_numpy import (
            build_physics_informed_pid_gain_scheduling,
        )

        return build_physics_informed_pid_gain_scheduling
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
