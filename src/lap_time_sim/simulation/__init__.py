"""Simulation solvers and runners."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lap_time_sim.simulation.config import SimulationConfig, SimulationNumerics, SimulationRuntime

if TYPE_CHECKING:
    from lap_time_sim.simulation.model_api import LapTimeVehicleModel, VehicleModelDiagnostics
    from lap_time_sim.simulation.runner import LapSimulationResult

__all__ = [
    "LapSimulationResult",
    "LapTimeVehicleModel",
    "SimulationConfig",
    "SimulationNumerics",
    "SimulationRuntime",
    "VehicleModelDiagnostics",
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
    if name == "LapTimeVehicleModel":
        from lap_time_sim.simulation.model_api import LapTimeVehicleModel

        return LapTimeVehicleModel
    if name == "SimulationRuntime":
        from lap_time_sim.simulation.config import SimulationRuntime

        return SimulationRuntime
    if name == "SimulationNumerics":
        from lap_time_sim.simulation.config import SimulationNumerics

        return SimulationNumerics
    if name == "VehicleModelDiagnostics":
        from lap_time_sim.simulation.model_api import VehicleModelDiagnostics

        return VehicleModelDiagnostics
    if name == "LapSimulationResult":
        from lap_time_sim.simulation.runner import LapSimulationResult

        return LapSimulationResult
    if name == "simulate_lap":
        from lap_time_sim.simulation.runner import simulate_lap

        return simulate_lap
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
